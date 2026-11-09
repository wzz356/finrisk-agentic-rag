"""Scope triage uses subject and facts, preserving literal evidence and pending identity."""

import pytest

from offline.insurance_relevance import evaluate_insurance_scope
from offline.source_identity import source_identity


@pytest.mark.parametrize(
    "text,relevance,route",
    [
        (
            "当事人：甲财险宁县支公司\n主要违法违规事实：虚构中介业务套取费用",
            "insurance",
            "candidate_review",
        ),
        (
            "当事人：刘某（时任甲人寿支公司经理）\n主要违法违规事实：编制虚假资料",
            "insurance",
            "candidate_review",
        ),
        ("当事人：甲银行支行\n主要违法违规事实：违规销售保险产品", "insurance", "candidate_review"),
        (
            "当事人：甲银行支行\n主要违法违规事实：贷款管理不审慎\n处罚机关：中国银行保险监督管理委员会",
            "non_insurance",
            "scope_review",
        ),
        (
            "当事人：甲银行\n主要违法违规事实：资料不真实\n处罚依据：《保险法》",
            "uncertain",
            "scope_review",
        ),
        ("当事人：张某\n处罚机关：中国银行保险监督管理委员会", "uncertain", "scope_review"),
        ("当事人：甲人寿公司\n主要违法违规事实：换领保险许可证公告", "insurance", "scope_review"),
        (
            "当事人：甲科技公司（甲保险公司客户）\n主要违法违规事实：贷款管理不审慎",
            "non_insurance",
            "scope_review",
        ),
    ],
)
def test_scope_routes_with_literal_codepoint_evidence(text, relevance, route):
    result = evaluate_insurance_scope(text)
    assert result["relevance"] == relevance and result["route"] == route
    assert result["publication_allowed"] is False and result["review_status"] == "pending"
    for evidence in result["evidence"]:
        assert text[evidence["start"] : evidence["end"]] == evidence["quote"]


def test_source_identity_normalizes_layout_but_keeps_distinct_parties_and_facts():
    text = "当事人：甲公司\r\n罚款１０万元。"
    assert source_identity(text) == source_identity("当事人:甲公司\n罚款10万元。")
    assert source_identity(text) != source_identity("当事人：乙公司\n罚款10万元。")
    assert source_identity(text) != source_identity("当事人：甲公司\n罚款1万元。")
