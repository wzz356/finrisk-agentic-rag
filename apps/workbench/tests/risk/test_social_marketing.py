"""社交文案的具体行为边界；产品设计与违法行为不能只按词面混同。"""

import asyncio
from pathlib import Path

import pytest

from domain.common import QueryInputType
from domain.query_models import QueryScene
from llm import JsonChatResponse
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query
from risk.detection import RuleRiskDetector
from risk.label_guards import social_label_supported
from risk.pipeline import RiskAssessmentPipeline

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setenv("FINRISK_LOCAL_CALIBRATION", "false")
    return RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json")


@pytest.mark.parametrize(
    "text",
    [
        "50-85岁符合条件的可以投保，还不用健康告知。",
        "这是免健康告知医疗险，责任内的一般既往症也能赔。",
        "不记名雇主险不用提供员工名单、身份证信息和劳动合同，风险名额内谁在岗就保谁。",
        "AI不会骗人！",
        "客服可以快速回复，不用等电话。",
    ],
)
def test_product_features_and_ai_are_not_concealment_or_competitor_attacks(detector, text):
    result = detector.detect(text, QueryScene.ST1_POSTER_SMS_SOCIAL)
    assert not {"隐瞒重要信息", "弱化风险提示", "贬低竞品"}.intersection(result.risk_labels)


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("AI不会骗人！AI的推荐就是最好的保障。", "绝对化表述"),
        ("A公司理赔慢，别买他们的。", "贬低竞品"),
        ("销售人员不用说明健康告知，先让客户付款。", "隐瞒重要信息"),
        ("不要告知客户免责条款。", "隐瞒重要信息"),
        ("今天下单，有机会获得一个1000元的夏日出游大礼包。", "赠送利益"),
        ("点击链接投保，就会赠送360元保额的门诊报销金。", "赠送利益"),
        ("左下角下单送360元门诊保额。", "赠送利益"),
        ("我们的保险有国家背书资金保障。", "虚假宣传"),
    ],
)
def test_explicit_promotional_risks_remain_detectable(detector, text, label):
    assert label in detector.detect(text, QueryScene.ST1_POSTER_SMS_SOCIAL).risk_labels


def test_family_rate_discount_is_not_a_gift_rule(detector):
    result = detector.detect("2人投保95折、3人9折，4人及以上85折优惠。")
    assert not {"赠送利益", "回扣返佣", "未按规定使用费率条款"}.intersection(result.risk_labels)
    assert "赠送利益" not in detector.detect("购买即可获得100万保额的保障。").risk_labels


@pytest.mark.parametrize(
    ("label", "text", "supported"),
    [
        ("夸大收益", "保额800万，保费20元起", False),
        ("夸大收益", "预计年化回报高达15%", True),
        ("隐瞒重要信息", "不用健康告知，符合条件即可投保", False),
        ("隐瞒重要信息", "不要告知客户免责条款", True),
        ("弱化风险提示", "健康告知宽松，不用担心理赔被卡", False),
        ("弱化风险提示", "不用看风险提示", True),
        ("绝对化表述", "责任内医疗费用全年最高400万，不限社保，100%报销", False),
        ("绝对化表述", "责任内100%报销，所有疾病全都能赔", True),
        ("销售误导", "✅0健康告知！", False),
        ("销售误导", "✅既往症可赔！", False),
        ("销售误导", "任何疾病都能赔", True),
        ("销售误导", "不实名制投保，很省钱！", False),
        ("销售误导", "不用提供员工名单，风险名额内谁在岗就保谁", False),
        ("销售误导", "不记名承保，不限人数和任何职业", True),
        ("赠送利益", "体检卡、健康管理服务，成为官方会员可享", False),
        ("赠送利益", "成为会员后投保送体检卡", True),
        ("避债避税", "税优险每年最高节税45%", False),
        ("避债避税", "买保险即可逃债避税", True),
    ],
)
def test_social_labels_require_their_own_current_fact(label, text, supported):
    assert social_label_supported(label, text) is supported


@pytest.mark.parametrize("adaptive", [False, True])
def test_semantic_label_guard_keeps_independent_supported_risk(detector, adaptive):
    class Reviewer:
        model_name = "boundary-fixture"

        async def complete_json(self, request):
            items = []
            for source in request.review_context["sentences"]:
                if source["sentence_id"] not in request.review_context["target_ids"]:
                    continue
                labels = ["夸大收益", "绝对化表述"]
                items.append(
                    {
                        "sentence_id": source["sentence_id"],
                        "risk_score": 0.8,
                        "decision_state": "risk",
                        "risk_labels": labels,
                        "evidence": [source["text"]],
                        "reasoning": "核查具体标签边界。",
                        "label_assessments": [
                            {"label": label, "decision_state": "risk", "evidence": [source["text"]]}
                            for label in labels
                        ]
                        if adaptive
                        else [],
                    }
                )
            return JsonChatResponse(data={"items": items}, model=self.model_name)

    class NoCases:
        def search(self, *args, **kwargs):
            return []

    async def run():
        query = await prepare_query(
            "责任内医疗费用全年最高400万，不限社保，100%报销。\n\n全网唯一的医疗保障。",
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
            scene_override=QueryScene.ST1_POSTER_SMS_SOCIAL,
        )
        pipeline = RiskAssessmentPipeline(
            rule_detector=detector,
            retrieval_service=NoCases(),
            taxonomy=("夸大收益", "绝对化表述"),
            llm_client=Reviewer(),
            retrieval_timing="adaptive" if adaptive else "post",
        )
        result = await pipeline.analyze(query)
        qualified, absolute = result.risk_assessments
        assert qualified.decision_state == "no_clear_risk"
        assert not qualified.risk_labels
        assert qualified.risk_score < 0.2
        assert absolute.decision_state == "risk"
        assert absolute.risk_labels == ["绝对化表述"]

    asyncio.run(run())
