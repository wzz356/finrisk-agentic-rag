"""Regressions from direct development/source review, without evaluation Query IDs."""

import asyncio
from pathlib import Path

from application.query_service import QueryProcessingService
from query.classification import QuerySceneClassifier
from query.review_frame import carry_document_frame, rejected_proposal_only
from risk.detection import RuleRiskDetector
from risk.pipeline import _current_fact_gaps

ROOT = Path(__file__).resolve().parents[2]


def detector():
    return RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json")


def test_financial_false_data_is_not_fake_expense_or_client_identity():
    result = detector().detect("检查确认，公司编制虚假财务资料，财务数据不真实。")
    regex = [e.description for e in result.evidence if e.source == "rule_regex"]
    assert any("false_financial_reporting" in evidence for evidence in regex)
    assert "其他" in result.risk_labels
    assert not any("false_expense" in evidence for evidence in regex)


def test_actual_cash_flow_or_missing_receipt_does_not_prove_fictitious_expense():
    for text in ("活动支出没有签到记录，需补凭证。", "款项转入营销团队账户，业务背景待核。"):
        result = detector().detect(text)
        assert not any(
            "false_expense" in e.description for e in result.evidence if e.source == "rule_regex"
        )


def test_unlicensed_individual_is_distinct_from_unlicensed_agency():
    result = detector().detect("委托未持有展业证的人员从事保险销售。")
    assert "代理人管理不到位" in result.risk_labels
    assert not any(
        "unauthorized_sales" in e.description for e in result.evidence if e.source == "rule_regex"
    )


def test_meeting_support_without_qualification_fact_does_not_trigger_agency_rule():
    result = detector().detect("会务公司负责场地布置和流程支持，保险责任由本公司人员讲解。")
    assert not any(
        "unauthorized_sales" in e.description for e in result.evidence if e.source == "rule_regex"
    )


def test_missing_documents_in_corrective_teaching_are_not_a_fact_gap():
    assert not _current_fact_gaps("本次反面教学没有合同凭证，无法判断旧案例责任层级。")
    assert not _current_fact_gaps("条款解释不清楚时，应完整说明，不能确定的转人工。")
    assert _current_fact_gaps("未提供这笔业务的代理资格证明，无法判断机构资质是否满足要求。")


def test_document_frame_survives_blank_paragraphs_without_changing_target_stance():
    text = "【反面教材，仅供警示】\n\n讲师转述：保证年化回报9%。\n\n本例明确禁止，当前不得照做。"
    query = asyncio.run(
        QueryProcessingService(classifier=QuerySceneClassifier(None)).process_text(text)
    )
    contexts = {r.sentence_id: r for r in query.sentence_contexts}
    framed = carry_document_frame(query, contexts)
    target = next(r for r in query.sentences if "年化" in r.text)
    context = framed[target.sentence_id]
    assert context.stance == contexts[target.sentence_id].stance
    assert context.requires_review
    assert any("反面教材" in r.text for r in context.source_sentences)
    for source in context.source_sentences:
        assert text[source.document_start_offset : source.document_end_offset] == source.text


def test_plain_marketing_deadline_alone_is_not_strong_coercion():
    result = detector().detect("活动仅限今天，名额先到先得。")
    assert not any(
        "urgency_pressure" in e.description for e in result.evidence if e.source == "rule_regex"
    )


def test_only_explicitly_rejected_unexecuted_proposal_is_a_corrective_scope():
    text = "客户活动方案（待审）：签单赠油卡。请先停止该方案执行，删除赠礼安排。"
    assert rejected_proposal_only(text)
    assert not rejected_proposal_only(text + "但团队已开始执行，目前仍在执行。")
    assert not rejected_proposal_only("客户活动方案（待审）：签单赠油卡。请各组照此继续。")
