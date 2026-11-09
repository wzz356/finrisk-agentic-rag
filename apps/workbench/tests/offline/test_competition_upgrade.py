import asyncio
import csv
import json
import sqlite3
from pathlib import Path

import pytest

from domain.common import FilePurpose
from extraction import FileExtractor
from llm import JsonChatResponse
from offline.case_ingestion import IncrementalCaseBuilder
from offline.knowledge_projection import normalize_date, project_metadata
from risk.detection import RuleRiskDetector
from settings import AppSettings
from storage.case_repository import CaseRepository

ROOT = Path(__file__).resolve().parents[2]


def test_projection_keeps_amounts_dates_and_field_sources():
    case = {
        "parties": [{"name": "甲人寿保险股份有限公司", "party_type": "institution"}],
        "regulator": "某监管局",
        "penalty_date": "2024年2月9日",
        "penalties": [
            {
                "type": "罚款",
                "amount_yuan": 250000,
                "amount_quote": "罚款25万元",
                "evidence_ids": ["e1"],
            }
        ],
        "field_evidence": {"party": ["e2"], "regulator": ["e3"]},
    }
    result = project_metadata(case, {"full_text": "发布日期：2024年3月1日"})
    assert result["penalty_content"] == "罚款25万元"
    assert result["penalty_date"] == "2024-02-09"
    assert result["publish_date"] == "2024-03-01"
    assert result["field_sources"]["party_name"] == ["e2"]
    assert result["institution_type"] == "人身险公司"
    assert normalize_date("2024-02-30") is None


def test_new_file_splits_parties_and_excludes_non_insurance(tmp_path):
    text = (
        "甲人寿保险公司虚列培训费用套取资金，罚款十万元。\n"
        "张三作为该公司代理人代客户签字。\n乙商业银行违规发放贷款。"
    )
    path = tmp_path / "处罚.txt"
    path.write_text(text, "utf-8")
    extraction = FileExtractor().extract(path, FilePurpose.CASE)
    taxonomy_path = AppSettings.from_env().risk_dictionary_path
    with taxonomy_path.open(encoding="utf-8-sig") as f:
        taxonomy = tuple(r["risk_tag"] for r in csv.DictReader(f))

    class Client:
        model_name = "offline-test"

        async def complete_json(self, request):
            cases = []
            for name, quote, related in [
                ("甲人寿保险公司", "虚列培训费用套取资金", True),
                ("张三", "代客户签字", True),
                ("乙商业银行", "违规发放贷款", False),
            ]:
                cases.append(
                    {
                        "party_name": name,
                        "is_insurance_related": related,
                        "fact_quotes": [quote],
                        "violation_behavior": quote + "。",
                        "case_summary": quote,
                        "confidence": 0.8,
                        "risk_tags": ["虚列费用套取资金"]
                        if name.startswith("甲")
                        else ["代理人管理不到位"],
                    }
                )
            return JsonChatResponse(data={"cases": cases}, model=self.model_name)

    builder = IncrementalCaseBuilder(
        taxonomy=taxonomy,
        risk_domain_mapping={},
        rule_detector=RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json"),
        llm_client=Client(),
        data_version="test",
    )
    result = asyncio.run(builder.build_many(extraction, source_file="处罚.txt"))
    assert [r.record["party_name"] for r in result] == ["甲人寿保险公司", "张三"]
    assert len({r.record["case_id"] for r in result}) == 2
    quotes = [{c["chunk_text"] for c in r.chunks if c["chunk_type"] == "evidence"} for r in result]
    assert quotes[0] == {"虚列培训费用套取资金"}
    assert quotes[1] == {"代客户签字"}


def test_batch_ingestion_rolls_back_files_and_registry(tmp_path, monkeypatch):
    base = tmp_path / "kb"
    (base / "data").mkdir(parents=True)
    for name in ["cases.jsonl", "case_chunks.jsonl", "case_expansions.jsonl"]:
        (base / "data" / name).write_text("", "utf-8")
    (base / "manifest.json").write_text(json.dumps({"counts": {}, "files": {}}), "utf-8")
    with sqlite3.connect(base / "evidence_registry.sqlite3") as con:
        con.executescript(
            "CREATE TABLE sources(file_id TEXT PRIMARY KEY,text TEXT,sha256 TEXT);"
            "CREATE TABLE units(id TEXT PRIMARY KEY,file_id TEXT,data TEXT);"
        )
    settings = AppSettings(
        project_root=tmp_path,
        case_kb_root=base,
        risk_keyword_path=tmp_path / "unused",
        raw_case_upload_dir=tmp_path / "uploads",
        full_corpus=False,
    )
    repository = CaseRepository(settings)

    def fail(*args):
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(repository, "_refresh_manifest", fail)
    with pytest.raises(OSError):
        repository.add_cases(
            [({"case_id": "one", "file_id": "F1"}, [{"chunk_id": "u1", "chunk_text": "原文"}], [])],
            source_text="原文",
        )
    assert settings.cases_path.read_text("utf-8") == ""
    assert settings.chunks_path.read_text("utf-8") == ""
    root = settings.source_root("web_ingestion")
    assert not (root / "cases.jsonl").exists()
    assert not (root / "manifest.json").exists()
    with sqlite3.connect(root / "evidence_registry.sqlite3") as con:
        assert con.execute("SELECT COUNT(*) FROM units").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0
    with sqlite3.connect(base / "evidence_registry.sqlite3") as con:
        assert con.execute("SELECT COUNT(*) FROM units").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0
