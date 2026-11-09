"""单句上下文参与实际判断和检索的行为回归。"""

import asyncio
import json
from pathlib import Path

import pytest

from domain.common import QueryInputType
from domain.knowledge_models import MatchedCase
from domain.query_models import QueryScene
from domain.risk_models import QueryAnalysisResult
from llm import JsonChatRequest, JsonChatResponse, LLMProviderError
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query
from risk.detection import RuleRiskDetector
from risk.pipeline import RiskAssessmentPipeline

ROOT = Path(__file__).resolve().parents[2]
TAGS = ("承诺收益", "保证收益", "销售误导", "夸大收益")


class EmptyRetrieval:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def search(self, query: str, **kwargs: object) -> list[MatchedCase]:
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("test index unavailable")
        return []


class ContextReviewer:
    model_name = "context-review-fixture"

    def __init__(self, *, fail: bool = False, forged_quote: bool = False) -> None:
        self.requests: list[JsonChatRequest] = []
        self.fail = fail
        self.forged_quote = forged_quote

    async def complete_json(self, request: JsonChatRequest) -> JsonChatResponse:
        self.requests.append(request)
        if self.fail:
            raise LLMProviderError("模拟语义服务超时")
        records = json.loads(json.dumps(request.review_context["sentences"]))
        targets = request.review_context["target_ids"]
        items = []
        for value in records:
            if value["sentence_id"] not in targets:
                continue
            risky = "是的" in value["text"]
            evidence = ["并不存在的逐字原文"] if self.forged_quote else [value["text"]]
            items.append(
                {
                    "sentence_id": value["sentence_id"],
                    "risk_score": 0.9 if risky else 0.05,
                    "risk_labels": ["承诺收益"] if risky else [],
                    "evidence": evidence,
                    "reasoning": "结合说话人和问答关系给出判断。",
                }
            )
        return JsonChatResponse(data={"items": items}, model=self.model_name)


def run(
    text: str,
    reviewer: ContextReviewer | None = None,
    *,
    fail_retrieval: bool = False,
    small_batches: bool = False,
):
    query = asyncio.run(
        prepare_query(
            text,
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
            scene_override=QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        )
    )
    retrieval = EmptyRetrieval(fail=fail_retrieval)
    pipeline = RiskAssessmentPipeline(
        rule_detector=RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json"),
        retrieval_service=retrieval,  # type: ignore[arg-type]
        taxonomy=TAGS,
        llm_client=reviewer,
        whole_query_llm_chars=1 if small_batches else 6000,
        llm_batch_items=1 if small_batches else 12,
    )
    result = asyncio.run(pipeline.analyze(query))
    return query, result, retrieval


def test_short_answer_uses_question_for_risk_and_retrieval_but_does_not_blame_customer() -> None:
    query, result, retrieval = run("客户：这款保险保证年化收益8%吗？\n顾问：是的，可以这样理解。")
    question, answer = result.risk_assessments
    assert question.risk_score < 0.2
    assert "提问" in question.reasoning
    assert answer.risk_score >= 0.4
    assert answer.review_status == "needs_review"
    assert retrieval.calls and query.sentences[0].text in retrieval.calls[0]
    assert query.sentences[1].text in retrieval.calls[0]
    assert question.retrieval_status == "not_requested"
    assert answer.retrieval_status == "no_sufficient_match"


def test_duplicate_answer_quotes_prefer_the_assessed_sentence_not_the_first_occurrence() -> None:
    query, result, _ = run("客户：保证收益吗？\n顾问：是的。\n顾问：是的。", ContextReviewer())
    evidence = next(
        item
        for item in result.risk_assessments[-1].evidence
        if item.source == "llm_semantic_evidence"
    )
    assert evidence.sentence_id == query.sentences[-1].sentence_id
    assert evidence.document_start_offset == query.source_mappings[-1].document_start_offset


@pytest.mark.parametrize("fault", ["wrong_target", "unknown_source", "outside_source"])
def test_result_rejects_judgment_or_evidence_detached_from_source(fault: str) -> None:
    from pydantic import ValidationError

    _, result, _ = run("客户：保证收益8%吗？\n顾问：是的。")
    data = result.model_dump()
    answer = data["risk_assessments"][-1]
    evidence = next(item for item in answer["evidence"] if item["sentence_id"])
    if fault == "wrong_target":
        answer["text"] = "被替换的原句。"
    elif fault == "unknown_source":
        evidence["sentence_id"] = "another_query_s1"
    else:
        evidence["document_end_offset"] = len(data["source_text"]) + 1
    with pytest.raises(ValidationError, match="原句|原文偏移"):
        QueryAnalysisResult.model_validate(data)


def test_prohibited_quote_with_internal_period_is_one_compliance_expression() -> None:
    query, result, _ = run("培训要求：不得使用“保证收益。稳赚不赔！”这样的说法。")
    assert len(query.sentences) == 1
    assert result.risk_assessments[0].risk_score < 0.2


@pytest.mark.parametrize("answer", ["不是，不能这样理解。", "不能保证。", "不一定。"])
def test_denial_does_not_inherit_guarantee_from_customers_question(answer: str) -> None:
    _, result, retrieval = run("客户：保证年化收益8%吗？\n顾问：" + answer)
    assert all(item.risk_score < 0.2 for item in result.risk_assessments)
    assert not retrieval.calls
    assert result.risk_assessments[-1].action != "放行"


def test_explicit_negative_example_stays_nonaffirmative_but_actual_conduct_is_flagged() -> None:
    _, result, _ = run(
        "培训材料：以下是反面话术，严禁使用。\n“本金绝对安全，保证年化收益8%。”"
        "\n\n实际销售中仍宣传保证年化收益8%，本金绝对安全。"
    )
    assert result.risk_assessments[1].risk_score < 0.2
    assert result.risk_assessments[-1].risk_score >= 0.4


def test_description_of_actual_use_of_bad_script_is_not_treated_as_negative_example() -> None:
    query, result, _ = run("销售使用违规话术：保证年化收益8%，本金绝对安全。")
    assert query.sentence_contexts[0].stance != "negative_example"
    assert result.risk_assessments[0].risk_score >= 0.4


def test_neighboring_disclaimer_does_not_clear_independent_guarantee_or_cross_products() -> None:
    _, result, retrieval = run(
        "产品A：分红不保证，可能为零。\n\n产品B：保证年化收益8%，本金绝对安全。"
    )
    assert result.risk_assessments[-1].risk_score >= 0.4
    assert all("产品A" not in value for value in retrieval.calls)


def test_long_document_batch_still_receives_speakers_and_source_bound_context() -> None:
    reviewer = ContextReviewer()
    query, result, _ = run(
        "客户：保证年化收益8%吗？\n顾问：是的，可以这样理解。",
        reviewer,
        small_batches=True,
    )
    assert len(reviewer.requests) == 2
    assert result.risk_assessments[0].risk_score < 0.2
    assert result.risk_assessments[1].llm_reviewed
    answer_request = next(
        value
        for value in reviewer.requests
        if query.sentences[-1].sentence_id in value.review_context["target_ids"]
    )
    serialized = json.dumps(answer_request.review_context, ensure_ascii=False)
    assert query.sentences[0].text in serialized
    assert '"speaker_role": "sales"' in serialized
    evidence = [
        item
        for item in result.risk_assessments[1].evidence
        if item.source == "llm_semantic_evidence"
    ][0]
    assert evidence.document_start_offset is not None
    assert evidence.document_end_offset is not None
    assert (
        query.source_text[evidence.document_start_offset : evidence.document_end_offset]
        == query.sentences[1].text
    )


def test_forged_llm_quote_is_not_displayed_as_original_evidence() -> None:
    _, result, _ = run("客户：保证年化收益8%吗？", ContextReviewer(forged_quote=True))
    assert not result.risk_assessments[0].llm_reviewed
    assert result.risk_summary.warnings
    assert all(
        "并不存在的逐字原文" not in item.description
        for assessment in result.risk_assessments
        for item in assessment.evidence
    )


def test_semantic_review_does_not_close_review_when_source_context_is_truncated() -> None:
    _, result, _ = run(
        "客户：" + "需要说明的条件" * 400 + "。\n顾问：是的，可以这样理解。",
        ContextReviewer(),
    )
    answer = result.risk_assessments[-1]
    assert answer.llm_reviewed
    assert answer.review_status == "needs_review"
    assert result.risk_summary.pending_review_sentence_count >= 1
    assert any("窗口超出预算" in warning for warning in result.risk_summary.warnings)


def test_agent_failure_and_index_failure_do_not_turn_empty_matches_into_clearance() -> None:
    _, result, _ = run(
        "客户：保证年化收益8%吗？\n顾问：是的，可以这样理解。",
        ContextReviewer(fail=True),
        fail_retrieval=True,
    )
    answer = result.risk_assessments[-1]
    assert answer.risk_score >= 0.4
    assert answer.retrieval_status == "unavailable"
    assert answer.review_status == "needs_review"
    assert result.risk_summary.pending_review_sentence_count == 2
    assert any("案例检索不可用" in warning for warning in result.risk_summary.warnings)
