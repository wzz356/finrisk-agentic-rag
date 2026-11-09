"""用户提供的真实处罚 PDF 在不写主库前提下验证增量构建流程。"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import pytest

from offline.case_ingestion import IncrementalCaseBuilder, load_risk_domain_mapping
from offline.extraction import extract_case_file
from risk.detection import RuleRiskDetector
from settings import AppSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_KB_ROOT = AppSettings.from_env().case_kb_root


def test_real_pdf_can_build_three_table_artifacts_without_writing_database() -> None:
    pdf_files = sorted((PROJECT_ROOT / "data" / "raw" / "cases").glob("*.pdf"))
    if not pdf_files:
        pytest.skip("未提供真实案例 PDF")
    extraction = extract_case_file(pdf_files[0])
    with (CASE_KB_ROOT / "dictionary" / "risk_type_dictionary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        taxonomy = tuple(row["risk_tag"] for row in csv.DictReader(file))
    builder = IncrementalCaseBuilder(
        taxonomy=taxonomy,
        risk_domain_mapping=load_risk_domain_mapping(
            str(CASE_KB_ROOT / "dictionary" / "risk_domain_mapping.json")
        ),
        rule_detector=RuleRiskDetector(
            PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
        ),
        llm_client=None,
        data_version="ingestion-regression-fixture",
    )

    artifacts = asyncio.run(
        builder.build(extraction, source_file=f"data/raw/cases/{pdf_files[0].name}")
    )

    assert str(artifacts.record["case_id"]).startswith("U")
    assert artifacts.record["case_origin"] == "web_ingestion"
    assert artifacts.record["source_sha256"] == extraction.checksum_sha256
    assert artifacts.chunks
    assert all(chunk["case_id"] == artifacts.record["case_id"] for chunk in artifacts.chunks)
    assert artifacts.llm_used is False
