"""Evidence-based label selection and repair targets stay separate from broad hints."""

import asyncio
from pathlib import Path

from domain.common import QueryInputType
from domain.query_models import QueryScene
from llm import JsonChatResponse
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query
from risk.detection import RuleRiskDetector
from risk.pipeline import RiskAssessmentPipeline, _explicit_content_labels
from risk.taxonomy import LABEL_BOUNDARIES

ROOT = Path(__file__).resolve().parents[2]


class NoCases:
    def search(self, query, **kwargs):
        return []


class PartialReviewer:
    model_name = "label-test"

    def __init__(self):
        self.targets = []

    async def complete_json(self, request):
        targets = request.review_context["target_ids"]
        self.targets.append(list(targets))
        items = []
        for source in request.review_context["sentences"]:
            if source["sentence_id"] not in targets:
                continue
            forged = len(self.targets) == 1 and "归档" in source["text"]
            items.append(
                {
                    "sentence_id": source["sentence_id"],
                    "risk_score": 0.03,
                    "decision_state": "no_clear_risk",
                    "risk_labels": [],
                    "evidence": ["invalid quotation" if forged else source["text"]],
                    "reasoning": "正常工作安排。",
                }
            )
        return JsonChatResponse(model=self.model_name, data={"items": items})


def test_repair_resubmits_only_the_failed_target_not_valid_neighbors():
    async def run():
        query = await prepare_query(
            "请按目录整理材料。请归档本次服务记录。会议日程见附件。",
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
            scene_override=QueryScene.ST6_PENALTY_BACKEND_FACT,
        )
        client = PartialReviewer()
        pipeline = RiskAssessmentPipeline(
            rule_detector=RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json"),
            retrieval_service=NoCases(),  # type: ignore[arg-type]
            taxonomy=("其他",),
            llm_client=client,
            retrieval_timing="adaptive",
        )
        result = await pipeline.analyze(query)
        assert all(a.llm_reviewed for a in result.risk_assessments)
        assert len(client.targets) == 2
        assert len(client.targets[0]) == 3 and len(client.targets[1]) == 1
        assert client.targets[1] == [query.sentences[1].sentence_id]

    asyncio.run(run())


def test_projected_number_does_not_independently_prove_fabricated_yield():
    assert "虚构收益率" not in _explicit_content_labels("预计复利可达3.5%，实际利益以合同为准。")


def test_label_boundary_cards_cover_exactly_27_types():
    assert len(LABEL_BOUNDARIES) == 27
    assert "其他" in LABEL_BOUNDARIES
