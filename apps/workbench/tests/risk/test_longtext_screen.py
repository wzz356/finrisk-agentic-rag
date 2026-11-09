"""Long-document screening keeps full coverage and cannot discard local candidates."""

import asyncio
from pathlib import Path

import pytest

from domain.common import QueryInputType
from domain.query_models import QueryScene
from llm import JsonChatRequest, JsonChatResponse, LLMProviderError
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query
from risk.detection import RuleRiskDetector
from risk.longtext import parse_screen
from risk.pipeline import RiskAssessmentPipeline

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "data",
    [
        {"complete": False, "review_indexes": [], "normal_ranges": [[0, 2]]},
        {"complete": True, "review_indexes": [], "normal_ranges": [[0, 1]]},
        {"complete": True, "review_indexes": [1], "normal_ranges": [[0, 2]]},
        {"complete": True, "review_indexes": [1, 1], "normal_ranges": [[0, 0], [2, 2]]},
        {"complete": True, "review_indexes": [4], "normal_ranges": [[0, 2]]},
        {"complete": True, "review_indexes": [True], "normal_ranges": [[0, 0], [2, 2]]},
    ],
)
def test_invalid_partition_never_implicitly_marks_missing_sentences_normal(data):
    with pytest.raises(LLMProviderError):
        parse_screen(data, {0, 1, 2})


def test_valid_explicit_partition_accepts_ranges_and_independent_candidates():
    normal, review = parse_screen(
        {"complete": True, "review_indexes": [2], "normal_ranges": [[0, 1], [3, 4]]},
        {0, 1, 2, 3, 4},
    )
    assert normal == {0, 1, 3, 4}
    assert review == {2}


class EmptyRetrieval:
    def search(self, query, **kwargs):
        return []


class ScreenReviewer:
    model_name = "test-cascade"

    def __init__(self, *, fail_screen=False):
        self.fail_screen = fail_screen
        self.requests = []

    async def complete_json(self, request: JsonChatRequest):
        self.requests.append(request)
        if request.task == "risk_screen":
            if self.fail_screen:
                raise LLMProviderError("test invalid partition")
            indexes = request.review_context["expected_indexes"]
            return JsonChatResponse(
                model=self.model_name,
                data={
                    "complete": True,
                    "review_indexes": [],
                    "normal_ranges": [[min(indexes), max(indexes)]],
                },
            )
        targets = set(request.review_context["target_ids"])
        return JsonChatResponse(
            model=self.model_name,
            data={
                "items": [
                    {
                        "sentence_id": item["sentence_id"],
                        "risk_score": 0.7,
                        "decision_state": "risk",
                        "risk_labels": ["诱导投保"],
                        "evidence": [item["text"]],
                        "reasoning": "详细核查已保留本地疑似句。",
                    }
                    for item in request.review_context["sentences"]
                    if item["sentence_id"] in targets
                ]
            },
        )


def run_cascade(*, fail_screen=False):
    text = (
        "\n\n".join(
            f"第{i}节目录编排：本节介绍日常服务窗口的办理步骤、材料清单和页面位置，"
            "工作人员记录服务日期并将签到页按顺序放入档案，会议日程列明开始时间和场地。"
            for i in range(30)
        )
        + "\n申请名额有限，今天完成办理。"
    )
    query = asyncio.run(
        prepare_query(
            text,
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
            scene_override=QueryScene.ST3_PRODUCT_INTRO_DETAIL,
        )
    )
    reviewer = ScreenReviewer(fail_screen=fail_screen)
    pipeline = RiskAssessmentPipeline(
        rule_detector=RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json"),
        retrieval_service=EmptyRetrieval(),  # type: ignore[arg-type]
        taxonomy=("诱导投保",),
        llm_client=reviewer,
    )
    result = asyncio.run(pipeline.analyze(query))
    return query, result, reviewer


def test_local_candidate_overrides_normal_screen_without_detailing_every_normal_sentence():
    query, result, reviewer = run_cascade()
    candidate = next(a for a in result.risk_assessments if "名额有限" in a.text)
    assert candidate.llm_reviewed
    assert candidate.review_depth == "detail"
    assert any(item.review_depth == "screen" for item in result.risk_assessments)
    assert len(result.risk_assessments) == len(query.sentences)
    detailed = [r for r in reviewer.requests if r.task == "risk_review"]
    assert sum(len(r.review_context["target_ids"]) for r in detailed) < len(query.sentences) // 2


def test_failed_screen_returns_to_full_review_and_preserves_every_original_sentence():
    query, result, reviewer = run_cascade(fail_screen=True)
    assert all(item.review_depth != "screen" for item in result.risk_assessments)
    assert len(result.risk_assessments) == len(query.sentences)
    assert any("逐句完整复核" in warning for warning in result.risk_summary.warnings)
    detailed = [r for r in reviewer.requests if r.task == "risk_review"]
    assert sum(len(r.review_context["target_ids"]) for r in detailed) == len(query.sentences)
