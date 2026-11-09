"""统一案例库的增量写入、状态管理和检索回归测试。"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest

from application.collection_service import SHANGHAI
from application.retrieval_service import CaseRetrievalService, CaseSearchFilters
from settings import AppSettings
from storage.case_repository import CaseRepository, DuplicateCaseError


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(value, ensure_ascii=False)}\n" for value in values),
        encoding="utf-8",
    )


def _settings(tmp_path: Path) -> AppSettings:
    root = tmp_path / "case_kb"
    data = root / "data"
    dictionary = root / "dictionary"
    dictionary.mkdir(parents=True)
    record = {
        "case_id": "C001",
        "case_title": "保证收益销售误导案例",
        "case_origin": "batch_build",
        "violation_behavior": "销售人员向客户承诺保本并保证年化收益。",
        "penalty_content": "责令改正并处罚款。",
        "party_name": "甲保险公司某分公司",
        "regulator": "国家金融监督管理总局甲监管分局",
        "institution_type": "保险公司分支机构",
        "penalty_date": "2026-06-18",
        "source_url": "https://example.test/C001",
        "risk_tags": ["承诺收益", "销售误导"],
        "risk_domains": ["销售行为"],
        "retrieval_status": "active",
        "evidence_status": "verified",
        "data_version": "test-v1",
    }
    chunk = {
        "chunk_id": "C001_fact_001",
        "case_id": "C001",
        "chunk_type": "fact",
        "chunk_text": "向客户承诺保本，保证年化收益百分之十。",
        "retrieval_enabled": True,
        "retrieval_weight": 1.0,
        "supports_risk_tags": ["承诺收益"],
        "is_verbatim": True,
    }
    _write_jsonl(data / "cases.jsonl", [record])
    _write_jsonl(data / "case_chunks.jsonl", [chunk])
    _write_jsonl(data / "case_expansions.jsonl", [])
    (root / "manifest.json").write_text(
        json.dumps({"data_version": "test-v1", "counts": {}, "files": {}}),
        encoding="utf-8",
    )
    (dictionary / "risk_type_dictionary.csv").write_text(
        "risk_type\n承诺收益\n销售误导\n", encoding="utf-8"
    )
    keyword_path = tmp_path / "risk.json"
    keyword_path.write_text('{"keywords": [], "regex_patterns": {}}', encoding="utf-8")
    return AppSettings(
        project_root=tmp_path,
        case_kb_root=root,
        risk_keyword_path=keyword_path,
        raw_case_upload_dir=tmp_path / "uploads",
    )


def test_repository_reads_and_retrieves_existing_case(tmp_path: Path) -> None:
    repository = CaseRepository(_settings(tmp_path))
    service = CaseRetrievalService(repository)

    assert repository.statistics().total_cases == 1
    assert repository.get_case("C001").evidence_chunks[0].is_verbatim is True
    matches = service.search("保证收益百分之十", top_k=3)
    assert matches
    assert matches[0].case_id == "C001"
    assert 0 < matches[0].score <= 1
    assert matches[0].bm25_score > 0
    assert matches[0].vector_score > 0
    assert matches[0].rrf_score > 0
    aligned = service.search("保证收益百分之十", top_k=3, risk_tags={"承诺收益"})
    assert aligned[0].matched_risk_tags == ["承诺收益"]
    assert service.search("保证收益百分之十", top_k=3, risk_tags={"赠送利益"}) == []


def test_release_partition_merges_sources_and_resolves_legacy_aliases(tmp_path):
    settings = _settings(tmp_path)
    root = settings.case_kb_root
    original = next(
        iter(
            __import__("json").loads(line)
            for line in settings.cases_path.read_text("utf-8").splitlines()
        )
    )
    original.update(file_id="F1")
    _write_jsonl(settings.cases_path, [original])
    static = root / "partitions/official_crawl"
    web = {
        **original,
        "case_id": "WEB1",
        "file_id": "W1",
        "source_group": "official_crawl",
        "case_origin": "official_crawl",
    }
    _write_jsonl(static / "cases.jsonl", [web])
    _write_jsonl(
        static / "case_chunks.jsonl",
        [{"chunk_id": "WU", "case_id": "WEB1", "chunk_type": "fact", "chunk_text": "保证收益"}],
    )
    _write_jsonl(
        settings.case_data_dir / "source_records.jsonl", [{**original, "case_id": "F1:SOURCE"}]
    )
    _write_jsonl(
        settings.source_root("official_crawl") / "cases.jsonl",
        [{**web, "case_id": "OLD", "file_id": "OLD-F"}],
    )
    (root / "source_aliases.json").write_text(
        json.dumps({"suppressed_file_ids": ["OLD-F"], "case_ids": {"OLD": "WEB1"}}),
        encoding="utf-8",
    )
    repository = CaseRepository(settings)
    assert repository.statistics().base_cases == 1
    assert repository.statistics().crawled_cases == 1
    assert repository.get_case("OLD").case_id == "WEB1"
    assert repository.get_case("OLD").evidence_chunks[0].chunk_id == "WU"
    before = (static / "cases.jsonl").read_bytes()
    repository.set_retrieval_status("OLD", active=False)
    assert (static / "cases.jsonl").read_bytes() == before
    assert CaseRepository(settings).get_case("WEB1").retrieval_status == "inactive"


def test_unreviewed_scope_cannot_be_enabled_with_a_status_override(tmp_path):
    settings = _settings(tmp_path)
    record = json.loads(settings.cases_path.read_text("utf-8"))
    record.update(retrieval_disposition="archive_only", retrieval_status="inactive")
    _write_jsonl(settings.cases_path, [record])
    repository = CaseRepository(settings)
    with pytest.raises(ValueError, match="范围审核"):
        repository.set_retrieval_status("C001", active=True)


def test_daily_collected_count_deduplicates_documents_and_uses_local_date(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    entries = [
        ("A1", "A", "2026-10-22T16:15:00+00:00"),
        ("A2", "A", "2026-10-22T16:15:00+00:00"),
        ("B", "B", "2026-10-23T15:59:59+00:00"),
        ("C", "C", "2026-10-23T16:00:00+00:00"),
        ("D", "D", "2026-10-22T23:59:59+08:00"),
        ("E", "E", ""),
        ("F", "F", "invalid"),
        ("G", "G", "2026-10-23T08:00:00"),
    ]
    records = [
        {"case_id": cid, "file_id": fid, "case_origin": "official_crawl", "collected_at": stamp}
        for cid, fid, stamp in entries
    ]
    official_path = settings.source_root("official_crawl") / "cases.jsonl"
    _write_jsonl(official_path, records)
    _write_jsonl(
        settings.source_root("web_ingestion") / "cases.jsonl",
        [{"case_id": "manual", "case_origin": "web_ingestion", "collected_at": entries[0][2]}],
    )
    repository = CaseRepository(settings)
    assert repository.count_collected_documents(date(2026, 10, 23), zone=SHANGHAI) == 3
    assert repository.count_collected_documents(date(2026, 10, 24), zone=SHANGHAI) == 1
    restarted = CaseRepository(settings)
    assert restarted.count_collected_documents(date(2026, 10, 23), zone=SHANGHAI) == 3
    records.append(
        {"case_id": "H", "case_origin": "official_crawl", "collected_at": "2026-10-23T10:00+08:00"}
    )
    _write_jsonl(official_path, records)
    restarted.reload()
    assert restarted.count_collected_documents(date(2026, 10, 23), zone=SHANGHAI) == 4


def test_batch_sources_keep_each_documents_own_literal_evidence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = CaseRepository(settings)
    artifacts = []
    texts = {"F_A": "甲公司给予合同外利益。", "F_B": "乙公司虚列费用。"}
    for fid, text in texts.items():
        artifacts.append(
            (
                {"case_id": fid, "file_id": fid, "violation_behavior": text},
                [{"chunk_id": fid + "_fact", "chunk_type": "fact", "chunk_text": text}],
                [],
            )
        )
    repository.add_cases(artifacts, source_texts=texts, origin="official_crawl")
    with sqlite3.connect(
        settings.source_root("official_crawl") / "evidence_registry.sqlite3"
    ) as db:
        for fid, text in texts.items():
            assert (
                db.execute("SELECT text FROM sources WHERE file_id=?", (fid,)).fetchone()[0] == text
            )
            unit = json.loads(
                db.execute("SELECT data FROM units WHERE file_id=?", (fid,)).fetchone()[0]
            )
            assert unit["source_text"] == text
            assert unit["source_spans"] == [{"start": 0, "end": len(text)}]


def test_batch_missing_source_rolls_back_whole_write(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = CaseRepository(settings)
    with pytest.raises(KeyError):
        repository.add_cases(
            [({"case_id": "new", "file_id": "missing"}, [], [])],
            source_texts={},
            origin="official_crawl",
        )
    assert CaseRepository(settings).statistics().total_cases == 1
    assert not (settings.source_root("official_crawl") / ".pending_write.json").exists()


def test_retrieval_and_repository_support_combined_metadata_filters(tmp_path: Path) -> None:
    repository = CaseRepository(_settings(tmp_path))
    service = CaseRetrievalService(repository)
    filters = CaseSearchFilters(
        risk_tags=frozenset({"承诺收益"}),
        regulator="甲监管分局",
        party_name="甲保险公司",
        institution_type="分支机构",
        penalty_date_from=date(2026, 1, 1),
        penalty_date_to=date(2026, 12, 31),
    )

    matches = service.search("保证收益百分之十", top_k=3, filters=filters)

    assert matches[0].party_name == "甲保险公司某分公司"
    assert matches[0].institution_type == "保险公司分支机构"
    assert matches[0].source_url == "https://example.test/C001"
    assert "收益承诺" in matches[0].match_reason
    assert "保证收益" in matches[0].match_reason
    assert (
        service.search(
            "保证收益百分之十",
            filters=CaseSearchFilters(regulator="不存在的监管机构"),
        )
        == []
    )
    page = repository.list_cases(
        regulator="甲监管分局",
        party_name="甲保险公司",
        institution_type="分支机构",
        penalty_date_from=date(2026, 6, 1),
        penalty_date_to=date(2026, 6, 30),
    )
    assert page.total == 1
    assert page.items[0].institution_type == "保险公司分支机构"


def test_incremental_case_is_separate_and_can_be_deactivated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = CaseRepository(settings)
    record = {
        "case_id": "U002",
        "case_title": "Web 新增案例",
        "case_origin": "web_ingestion",
        "violation_behavior": "通过赠送礼品诱导投保。",
        "risk_tags": ["赠送利益"],
        "risk_domains": ["销售行为"],
        "retrieval_status": "active",
        "evidence_status": "limited",
        "data_version": "test-v1",
    }
    chunks = [
        {
            "chunk_id": "U002_fact_001",
            "case_id": "U002",
            "chunk_type": "fact",
            "chunk_text": "赠送礼品诱导投保",
            "retrieval_enabled": True,
        }
    ]
    expansions = [
        {
            "expansion_id": "U002_exp_001",
            "expansion_text": "送礼投保",
            "case_links": [{"case_id": "U002"}],
            "retrieval_enabled": True,
        }
    ]

    repository.add_case(record, chunks, expansions)

    assert repository.statistics().total_cases == 2
    assert repository.statistics().user_cases == 1
    assert sum(1 for _ in settings.cases_path.open(encoding="utf-8")) == 1
    assert sum(1 for _ in settings.chunks_path.open(encoding="utf-8")) == 1
    assert sum(1 for _ in settings.expansions_path.open(encoding="utf-8")) == 0
    root = settings.source_root("web_ingestion")
    assert len((root / "cases.jsonl").read_text("utf-8").splitlines()) == 1
    assert CaseRepository(settings).get_case("U002").source_group == "web_ingestion"
    assert repository.set_retrieval_status("U002", active=False).retrieval_status == "inactive"
    assert all(
        "U002" not in document.case_ids for document in repository.iter_retrieval_documents()
    )
    with pytest.raises(DuplicateCaseError):
        repository.add_case(record, chunks, expansions)


def test_three_source_partitions_are_combined_without_changing_originals(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    originals = {
        p: p.read_bytes()
        for p in (
            settings.cases_path,
            settings.chunks_path,
            settings.expansions_path,
            settings.manifest_path,
        )
    }
    repository = CaseRepository(settings)
    stale = CaseRepository(settings)
    for cid, origin in (("N001", "official_crawl"), ("W001", "web_ingestion")):
        text = f"{cid}保险销售欺骗投保人"
        record = {
            "case_id": cid,
            "file_id": f"F_{cid}",
            "case_origin": origin,
            "violation_behavior": text,
            "retrieval_status": "active",
            "risk_tags": ["销售误导"],
        }
        chunks = [
            {
                "chunk_id": f"{cid}_fact",
                "case_id": cid,
                "chunk_type": "fact",
                "chunk_text": text,
                "is_verbatim": True,
            }
        ]
        stale.add_cases([(record, chunks, [])], source_text=text, origin=origin)
    repository.reload()
    stats = repository.statistics()
    assert (stats.total_cases, stats.base_cases, stats.crawled_cases, stats.user_cases) == (
        3,
        1,
        1,
        1,
    )
    for origin, cid in (
        ("original", "C001"),
        ("official_crawl", "N001"),
        ("web_ingestion", "W001"),
    ):
        assert repository.list_cases(origin=origin).items[0].case_id == cid
    assert {c for doc in repository.iter_retrieval_documents() for c in doc.case_ids} == {
        "C001",
        "N001",
        "W001",
    }
    repository.set_retrieval_status("N001", active=False)
    assert CaseRepository(settings).get_case("N001").retrieval_status == "inactive"
    repository.set_retrieval_status("C001", active=False)
    assert CaseRepository(settings).get_case("C001").retrieval_status == "inactive"
    assert all(path.read_bytes() == data for path, data in originals.items())
    for origin in ("official_crawl", "web_ingestion"):
        root = settings.source_root(origin)
        assert (root / "manifest.json").is_file()
        with sqlite3.connect(root / "evidence_registry.sqlite3") as connection:
            assert connection.execute("SELECT COUNT(*) FROM units").fetchone()[0] == 1


def test_readers_notice_a_different_process_partition_commit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    reader, writer = CaseRepository(settings), CaseRepository(settings)
    reader.statistics()
    writer.add_case(
        {"case_id": "new", "case_origin": "official_crawl", "violation_behavior": "销售误导"},
        [],
        [],
    )
    reader._last_check = 0
    assert reader.statistics().crawled_cases == 1


def test_cross_source_content_and_document_party_dedup(tmp_path: Path) -> None:
    repository = CaseRepository(_settings(tmp_path))
    record = {
        "case_id": "official1",
        "case_origin": "official_crawl",
        "violation_behavior": "欺骗投保人",
        "penalty_doc_no": "金罚〔2026〕1号",
        "party_name": "甲保险公司",
        "source_content_sha256": "abc",
    }
    repository.add_case(record, [], [])
    with pytest.raises(DuplicateCaseError):
        repository.add_case(
            {**record, "case_id": "upload1", "case_origin": "web_ingestion"}, [], []
        )
    with pytest.raises(DuplicateCaseError):
        repository.add_case(
            {**record, "case_id": "upload2", "source_content_sha256": "changed-format"}, [], []
        )
    assert repository.statistics().total_cases == 2


def test_evidence_reader_federates_original_and_new_sources(tmp_path: Path) -> None:
    from application.evidence_service import KnowledgeEvidenceService

    settings = _settings(tmp_path)
    text = "原始保险处罚事实"
    original_db = settings.case_kb_root / "evidence_registry.sqlite3"
    with sqlite3.connect(original_db) as connection:
        connection.executescript(
            "CREATE TABLE sources(file_id TEXT,text TEXT,sha256 TEXT);"
            "CREATE TABLE units(id TEXT,file_id TEXT,data TEXT);"
            "CREATE TABLE evidence(id TEXT,file_id TEXT,data TEXT);"
        )
        connection.execute(
            "INSERT INTO sources VALUES (?,?,?)", ("old", text, sha256(text.encode()).hexdigest())
        )
    repository = CaseRepository(settings)
    repository.add_cases(
        [
            (
                {"case_id": "new", "file_id": "newfile", "violation_behavior": text},
                [{"chunk_id": "newunit", "chunk_text": text}],
                [],
            )
        ],
        source_text=text,
    )
    reader = object.__new__(KnowledgeEvidenceService)
    reader.database, reader.additional_databases = (
        original_db,
        settings.additional_evidence_databases,
    )
    reader.p0, reader.articles, reader.scope_units, reader.versions = None, {}, {}, {}
    evidence = reader.read("newunit")
    assert evidence["literal_verified"] and evidence["file_id"] == "newfile"
    assert evidence["segments"][0]["text"] == text


def test_interrupted_multi_file_append_is_recovered_before_reading(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = settings.source_root("official_crawl")
    root.mkdir(parents=True)
    (root / "cases.jsonl").write_text('{"case_id":"incomplete', encoding="utf-8")
    (root / ".pending_write.json").write_text(
        json.dumps(
            {
                "id": "uncommitted",
                "files": {
                    name: [False, 0]
                    for name in ("cases.jsonl", "case_chunks.jsonl", "case_expansions.jsonl")
                },
                "manifest": None,
            }
        ),
        encoding="utf-8",
    )
    repository = CaseRepository(settings)
    assert repository.statistics().total_cases == 1
    assert not (root / "cases.jsonl").exists()
    assert not (root / ".pending_write.json").exists()


def test_committed_append_survives_crash_before_journal_cleanup(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = CaseRepository(settings)
    repository.add_case(
        {"case_id": "new", "case_origin": "official_crawl", "violation_behavior": "保险处罚"},
        [],
        [],
    )
    root = settings.source_root("official_crawl")
    with sqlite3.connect(root / "evidence_registry.sqlite3") as connection:
        token = connection.execute("SELECT id FROM ingestion_commits").fetchone()[0]
    (root / ".pending_write.json").write_text(
        json.dumps({"id": token, "files": {"cases.jsonl": [False, 0]}, "manifest": None}),
        encoding="utf-8",
    )
    recovered = CaseRepository(settings)
    assert recovered.statistics().crawled_cases == 1
    assert recovered.get_case("new").source_group == "official_crawl"
    assert not (root / ".pending_write.json").exists()
