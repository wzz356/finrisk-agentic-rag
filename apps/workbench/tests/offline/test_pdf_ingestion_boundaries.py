"""PDF 排版空白可回绑定；改写、错主体与虚构金额不能通过。"""

import asyncio
from pathlib import Path

import pytest

from domain.common import FilePurpose
from extraction import FileExtractor
from extraction.errors import ExtractionError
from llm import JsonChatResponse, LLMProviderError
from offline.case_ingestion import IncrementalCaseBuilder, _literal_quote
from risk.detection import RuleRiskDetector

ROOT = Path(__file__).resolve().parents[2]


def test_broken_pdf_reports_a_readable_extraction_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(ExtractionError, match="PDF 无法读取"):
        FileExtractor().extract(path, FilePurpose.CASE)


@pytest.mark.parametrize("quote", ["甲保险公司四川分公司", "甲保险 公司四川分公司"])
def test_wrapped_quote_returns_the_original_contiguous_source(quote):
    text = "处罚对象：甲保险公司四川\n分公司。"
    result = _literal_quote(quote, text)
    assert result == "甲保险公司四川\n分公司"
    assert result in text


@pytest.mark.parametrize("quote", ["罚款10万元", "虚列费用", "甲保险公司并被罚款"])
def test_changed_numbers_invented_facts_and_spliced_quotes_are_rejected(quote):
    assert _literal_quote(quote, "甲保险公司因未履行代理人管理职责，被罚款1 万元。") is None


def test_invalid_model_evidence_repairs_once_then_rebinds_literal_text(tmp_path):
    text = "甲保险公司四川\n分公司因未履行代理人行为管理职责，被罚款1 万元。"
    source = tmp_path / "处罚.txt"
    source.write_text(text, encoding="utf-8")
    extraction = FileExtractor().extract(source, FilePurpose.CASE)

    class Reviewer:
        model_name = "ingestion-fixture"

        def __init__(self, repair):
            self.calls = 0
            self.repair = repair

        async def complete_json(self, request):
            self.calls += 1
            quote = "被罚款10万元" if self.calls == 1 or not self.repair else "被罚款1万元"
            return JsonChatResponse(
                model=self.model_name,
                data={
                    "cases": [
                        {
                            "is_insurance_related": True,
                            "party_name": "甲保险公司四川分公司",
                            "violation_behavior": "未履行代理人行为管理职责。",
                            "fact_quotes": [quote],
                            "case_summary": "代理人管理违规。",
                            "confidence": 0.9,
                            "risk_tags": ["代理人管理不到位"],
                            "risk_tag_basis": [
                                {
                                    "risk_tag": "代理人管理不到位",
                                    "reason": "见原文",
                                    "quote": "未履行代理人行为管理职责",
                                }
                            ],
                        }
                    ]
                },
            )

    def build(client):
        builder = IncrementalCaseBuilder(
            taxonomy=("代理人管理不到位",),
            risk_domain_mapping={},
            rule_detector=RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json"),
            llm_client=client,
            data_version="test",
        )
        return asyncio.run(builder.build_many(extraction, source_file="处罚.txt"))

    repaired = Reviewer(True)
    artifacts = build(repaired)
    assert repaired.calls == 2
    assert artifacts[0].record["party_name"] == "甲保险公司四川分公司"
    assert any(chunk["chunk_text"] == "被罚款1 万元" for chunk in artifacts[0].chunks)
    broken = Reviewer(False)
    with pytest.raises(LLMProviderError, match="对应原文"):
        build(broken)
    assert broken.calls == 2
