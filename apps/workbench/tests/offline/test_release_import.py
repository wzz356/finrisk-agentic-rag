"""Already structured imports validate evidence and never invoke a chat model."""

import asyncio
import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from application.case_service import CaseManagementService
from offline.penalty_crawler.polite_http import CollectionHTTPError
from offline.release_import import case_payload, load_new_sources, validate_source
from risk.detection import RuleRiskDetector
from settings import AppSettings
from storage.case_repository import CaseRepository, DuplicateCaseError

TEXT = (
    "当事人: 甲财产保险公司\n金罚〔2026〕18号\n"
    "给予投保人保险合同约定以外的保险费回扣或者其他利益\n"
    "行政处罚内容\n罚款14万元\n甲金融监管分局\n2026年8月17日"
)
FACT = "给予投保人保险合同约定以外的保险费回扣或者其他利益"


def group():
    digest = hashlib.sha256(TEXT.encode()).hexdigest()
    case = {
        "case_id": "remote-1",
        "official_file_id": "new-file",
        "parties": [{"name": "甲财产保险公司", "party_type": "institution"}],
        "penalty_doc_no": "金罚〔2026〕18号",
        "regulator": "甲金融监管分局",
        "penalty_date": "2026年8月17日",
        "case_summary": "给予合同外利益被罚款14万元。",
        "penalties": [{"type": "罚款", "quote": "行政处罚内容\n罚款14万元", "amount": "14万元"}],
        "legal_refs": [],
    }
    behavior = {
        "case_id": "remote-1",
        "regulatory_finding": FACT,
        "risk_tags": ["合同外利益", "回扣返佣"],
        "risk_tag_basis": [
            {"tag": tag, "reason": "原文明确记载", "quotes": [FACT]}
            for tag in ("合同外利益", "回扣返佣")
        ],
    }
    return {
        "file_id": "new-file",
        "source": {
            "official_file_id": "new-file",
            "insurance_relevance": True,
            "original_mapping_metadata": {
                "source_url": "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=123&itemId=4115"
            },
        },
        "document": {"official_file_id": "new-file", "full_text": TEXT, "text_sha256": digest},
        "evidence": [
            {
                "official_file_id": "new-file",
                "text_sha256": digest,
                "offset_unit": "unicode_codepoint",
                "start_char": TEXT.index(FACT),
                "end_char": TEXT.index(FACT) + len(FACT),
                "verbatim": FACT,
            }
        ],
        "cases": [{"case": case, "behaviors": [behavior]}],
    }


def test_release_loading_skips_existing_ids_before_reading_auxiliary_tables(tmp_path):
    archive = tmp_path / "only-existing.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("data/cases.jsonl", json.dumps({"case_id": "known"}) + "\n")
    groups, stats = load_new_sources(archive, {"known"})
    assert not groups and stats["existing_case_ids"] == 1


def test_release_selects_only_new_source_and_preserves_fields(tmp_path):
    data = group()
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as z:
        tables = {
            "cases": [{"case_id": "known"}, data["cases"][0]["case"]],
            "source_inventory": [data["source"]],
            "documents": [data["document"]],
            "behaviors": data["cases"][0]["behaviors"],
            "evidence": data["evidence"],
        }
        for name, values in tables.items():
            z.writestr(
                f"data/{name}.jsonl",
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values),
            )
    groups, stats = load_new_sources(archive, {"known"})
    assert len(groups) == 1 and stats["candidate_cases"] == 1
    validate_source(groups[0])
    payload = case_payload(groups[0]["cases"][0], TEXT, {"合同外利益", "回扣返佣"})
    assert payload["penalty_content"] == "罚款14万元"
    assert payload["penalty_date"] == "2026-08-17"
    assert payload["fact_quotes"] == [FACT]


@pytest.mark.parametrize("mode", ["hash", "span", "quote", "host"])
def test_import_rejects_invalid_source_or_evidence(mode):
    data = group()
    if mode == "hash":
        data["document"]["text_sha256"] = "bad"
    if mode == "span":
        data["evidence"][0]["start_char"] += 1
    if mode == "quote":
        data["evidence"][0]["verbatim"] = "没有的事实"
    if mode == "host":
        data["source"]["original_mapping_metadata"]["source_url"] = "https://unregistered.test/case"
    with pytest.raises((ValueError, CollectionHTTPError)):
        validate_source(data)


def test_import_refuses_unregistered_or_ungrounded_labels():
    data = group()
    item = deepcopy(data["cases"][0])
    with pytest.raises(ValueError, match="27类"):
        case_payload(item, TEXT, {"合同外利益"})
    item["behaviors"][0]["risk_tag_basis"] = []
    with pytest.raises(ValueError, match="原文依据"):
        case_payload(item, TEXT, {"合同外利益", "回扣返佣"})


def test_prestructured_ingestion_never_calls_model_and_stays_in_official_partition(tmp_path):
    base = tmp_path / "base"
    (base / "data").mkdir(parents=True)
    (base / "dictionary").mkdir()
    for name in ("cases.jsonl", "case_chunks.jsonl", "case_expansions.jsonl"):
        (base / "data" / name).write_text("", encoding="utf-8")
    (base / "manifest.json").write_text('{"data_version":"test"}', encoding="utf-8")
    (base / "dictionary/risk_type_dictionary.csv").write_text(
        "risk_tag\n合同外利益\n回扣返佣\n", encoding="utf-8"
    )
    (base / "dictionary/risk_domain_mapping.json").write_text("{}", encoding="utf-8")
    keywords = tmp_path / "keywords.json"
    keywords.write_text('{"keywords":[],"regex_patterns":{}}', encoding="utf-8")
    settings = AppSettings(
        project_root=tmp_path,
        case_kb_root=base,
        risk_keyword_path=keywords,
        raw_case_upload_dir=tmp_path / "uploads",
        embedding_provider="local",
    )
    repository = CaseRepository(settings)

    class NoModel:
        model_name = "must-not-be-used"

        async def complete_json(self, request):
            raise AssertionError("structured import must never request a model")

    service = CaseManagementService(
        settings=settings,
        repository=repository,
        rule_detector=RuleRiskDetector(keywords),
        llm_client=NoModel(),
    )
    path = tmp_path / "source.txt"
    path.write_text(TEXT, encoding="utf-8")
    payload = case_payload(group()["cases"][0], TEXT, {"合同外利益", "回扣返佣"})
    result = asyncio.run(
        service.ingest_file(
            path,
            original_filename="保险公开文书" * 30 + ".txt",
            origin="official_crawl",
            structured_cases=[payload],
            source_metadata={"source_url": validate_source(group())},
        )
    )
    assert result.llm_used is False and result.case.source_group == "official_crawl"
    assert result.case.penalty_content == "罚款14万元"
    assert result.case.risk_tags == ["合同外利益", "回扣返佣"]
    assert result.case.field_sources["party_name"]["origin"] == "prestructured_release"
    assert "rule_fallback" not in result.case.quality_flags
    assert (base / "data/cases.jsonl").read_text("utf-8") == ""
    assert repository.statistics().crawled_cases == 1
    assert len(Path(result.case.raw_source_file).name) < 40
    with pytest.raises(DuplicateCaseError):
        asyncio.run(
            service.ingest_file(
                path,
                original_filename="source.txt",
                origin="official_crawl",
                structured_cases=[payload],
            )
        )
