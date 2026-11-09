"""风险复核和案例检索顺序的回归测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from domain.common import QueryInputType
from domain.knowledge_models import MatchedCase
from llm.types import JsonChatRequest, JsonChatResponse
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query
from risk.detection import RuleRiskDetector
from risk.pipeline import RiskAssessmentPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = (
    "销售误导",
    "夸大收益",
    "虚假宣传",
    "绝对化表述",
    "承诺收益",
    "保证收益",
)


class FakeLLM:
    def __init__(self) -> None:
        self.requests: list[JsonChatRequest] = []

    @property
    def model_name(self) -> str:
        return "fake-json-model"

    async def complete_json(self, request: JsonChatRequest) -> JsonChatResponse:
        self.requests.append(request)
        user_content = request.messages[-1].content
        target_line = user_content.split("只需输出这些待复核句子的结果：", 1)[1].splitlines()[0]
        target_ids = json.loads(target_line)
        return JsonChatResponse(
            data={
                "items": [
                    {
                        "sentence_id": sentence_id,
                        "risk_score": 0.1,
                        "risk_labels": [],
                        "evidence": [],
                        "reasoning": "存在明确合同限定，不构成收益承诺。",
                    }
                    for sentence_id in target_ids
                ]
            },
            model=self.model_name,
        )


class RecordingRetrieval:
    def __init__(self) -> None:
        self.calls: list[tuple[str, set[str]]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        risk_tags: set[str] | None = None,
    ) -> list[MatchedCase]:
        del top_k
        tags = risk_tags or set()
        self.calls.append((query, tags))
        return [
            MatchedCase(
                case_id="C001",
                violation_behavior="保证收益误导销售",
                risk_tags=["承诺收益", "销售误导"],
                score=0.8,
                bm25_score=0.7,
                vector_score=0.6,
                rrf_score=0.8,
                rerank_score=0.8,
                matched_risk_tags=sorted(tags.intersection({"承诺收益", "销售误导"})),
                matched_text="保证收益误导销售",
                matched_source="case_chunk",
                matched_unit_id="C001_fact_001",
            )
        ]


def detector() -> RuleRiskDetector:
    return RuleRiskDetector(PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json")


def test_llm_prompt_has_full_short_context_but_no_case_payload() -> None:
    llm = FakeLLM()
    retrieval = RecordingRetrieval()
    query = asyncio.run(
        prepare_query(
            "产品详情：本金绝对安全，保证年化收益8%。具体责任与收益需要结合完整材料判断。",
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
        )
    )
    pipeline = RiskAssessmentPipeline(
        rule_detector=detector(),
        retrieval_service=retrieval,  # type: ignore[arg-type]
        taxonomy=TAXONOMY,
        llm_client=llm,
    )

    result = asyncio.run(pipeline.analyze(query))

    assert len(llm.requests) == 1
    prompt = llm.requests[0].messages[-1].content
    assert all(sentence.sentence_id in prompt for sentence in query.sentences)
    assert "matched_cases" not in prompt
    assert "case_id" not in prompt
    assert result.risk_assessments[0].llm_reviewed is False
    assert result.risk_assessments[1].llm_reviewed is True
    assert len(retrieval.calls) == 1
    assert result.risk_assessments[0].matched_cases
    assert result.risk_assessments[1].matched_cases == []


def test_low_risk_sentence_never_triggers_case_search() -> None:
    llm = FakeLLM()
    retrieval = RecordingRetrieval()
    query = asyncio.run(
        prepare_query(
            "产品详情：收益存在不确定性，实际结果以合同为准。",
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
        )
    )
    pipeline = RiskAssessmentPipeline(
        rule_detector=detector(),
        retrieval_service=retrieval,  # type: ignore[arg-type]
        taxonomy=TAXONOMY,
        llm_client=llm,
    )

    result = asyncio.run(pipeline.analyze(query))

    assert result.risk_summary.overall_score < result.risk_summary.retrieval_threshold
    assert retrieval.calls == []
    assert result.risk_summary.top_matched_cases == []
