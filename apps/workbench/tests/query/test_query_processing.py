"""六类 Query 场景分类、切句和原文映射测试。"""

import asyncio

import pytest

from domain.common import QueryInputType
from domain.query_models import QueryClassificationMethod, QueryScene
from llm.types import JsonChatRequest, JsonChatResponse
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query, split_sentences


class FakeLLMClient:
    """返回固定六类 JSON 结果的假大模型。"""

    def __init__(self, scene: QueryScene) -> None:
        self.scene = scene
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-local-model"

    async def complete_json(self, request: JsonChatRequest) -> JsonChatResponse:
        self.call_count += 1
        assert request.messages
        return JsonChatResponse(
            data={
                "scene": self.scene.value,
                "confidence": 0.88,
                "reason": "材料主要用于会议活动",
            },
            model=self.model_name,
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "营销海报：限时活动，立即扫码领取专属权益。",
            QueryScene.ST1_POSTER_SMS_SOCIAL,
        ),
        (
            "直播脚本：主播欢迎家人们来到直播间，三二一上链接，点击小黄车。",
            QueryScene.ST2_LIVE_SHORT_VIDEO,
        ),
        (
            "产品详情：产品名称稳健一号，风险等级R2，投资期限一年，起购金额一万元，支持申购和赎回。",
            QueryScene.ST3_PRODUCT_INTRO_DETAIL,
        ),
        (
            "客户答谢会材料。会议议程包括签到、开场致辞、嘉宾介绍和茶歇。",
            QueryScene.ST4_MEETING_MATERIAL,
        ),
        (
            "电销话术培训材料：坐席：您好，本次致电是进行客户回访；客户：好的。随后进入异议处理流程。",
            QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        ),
        (
            "说服语言越精炼越好。推销话术示例：如果您认为薪水赚来不易，请考虑家人需要的保障。",
            QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        ),
        (
            "行政处罚决定书：经查，该机构违反相关规定，决定给予警告并处罚款。",
            QueryScene.ST6_PENALTY_BACKEND_FACT,
        ),
    ],
)
def test_high_confidence_rules_cover_six_scenes(text: str, expected: QueryScene) -> None:
    """六类明确材料应由高置信规则直接输出。"""

    fake_llm = FakeLLMClient(QueryScene.ST1_POSTER_SMS_SOCIAL)
    classifier = QuerySceneClassifier(fake_llm)

    result = asyncio.run(classifier.classify(text))

    assert result.scene is expected
    assert result.method is QueryClassificationMethod.RULE
    assert result.evidence
    assert fake_llm.call_count == 0


def test_ambiguous_rule_result_uses_llm() -> None:
    """规则重复命中且差距不足时必须交给 LLM 唯一分类。"""

    fake_llm = FakeLLMClient(QueryScene.ST4_MEETING_MATERIAL)
    classifier = QuerySceneClassifier(fake_llm)

    result = asyncio.run(classifier.classify("直播间产品说明会材料，主播负责开场致辞。"))

    assert result.scene is QueryScene.ST4_MEETING_MATERIAL
    assert result.method is QueryClassificationMethod.LLM
    assert result.llm_model == "fake-local-model"
    assert fake_llm.call_count == 1


def test_low_confidence_without_llm_uses_local_fallback() -> None:
    """无 LLM 时必须返回可识别的本地回退结果，保证离线流程可运行。"""

    classifier = QuerySceneClassifier(None)
    result = asyncio.run(classifier.classify("这是今天整理的一份材料。"))

    assert result.scene is QueryScene.ST1_POSTER_SMS_SOCIAL
    assert result.method is QueryClassificationMethod.RULE_FALLBACK
    assert result.evidence


def test_user_scene_override_skips_rule_and_llm() -> None:
    """用户显式选择场景时应原样采用，并跳过模型调用。"""

    fake_llm = FakeLLMClient(QueryScene.ST1_POSTER_SMS_SOCIAL)
    classifier = QuerySceneClassifier(fake_llm)
    result = asyncio.run(
        classifier.classify(
            "这是一份很难仅靠文本判断载体的材料。",
            scene_override=QueryScene.ST4_MEETING_MATERIAL,
        )
    )

    assert result.scene is QueryScene.ST4_MEETING_MATERIAL
    assert result.method is QueryClassificationMethod.USER
    assert result.confidence == 1.0
    assert fake_llm.call_count == 0


def test_sentence_split_preserves_text_and_offsets() -> None:
    """切句后应能用偏移从原文精确还原每句话。"""

    text = "您好！\n第二句话； 第三句话。"
    sentences = split_sentences(text)

    assert [sentence.text for sentence in sentences] == ["您好！", "第二句话；", "第三句话。"]
    assert [text[sentence.start_offset : sentence.end_offset] for sentence in sentences] == [
        sentence.text for sentence in sentences
    ]


def test_soft_line_break_does_not_split_ocr_sentence() -> None:
    """OCR 行尾软换行应留在句内，项目符号仍可形成新句。"""

    text = "我们不知道明天会\n发生什么事。\n·下一条完整话术。"
    sentences = split_sentences(text)

    assert [sentence.text for sentence in sentences] == [
        "我们不知道明天会\n发生什么事。",
        "·下一条完整话术。",
    ]
    assert all(text[item.start_offset : item.end_offset] == item.text for item in sentences)


def test_risk_sentence_has_exact_four_fields_and_text_mapping() -> None:
    """Risk 输入严格四字段，定位表可以还原文字输入中的原句。"""

    text = "营销海报，限时参与！立即扫码领取权益。"
    result = asyncio.run(
        prepare_query(
            text,
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
        )
    )

    assert len(result.sentences) == 2
    assert set(result.sentences[0].model_dump()) == {
        "query_id",
        "sentence_id",
        "scene",
        "text",
    }
    for sentence, mapping in zip(result.sentences, result.source_mappings, strict=True):
        assert sentence.sentence_id == mapping.sentence_id
        assert (
            result.source_text[mapping.document_start_offset : mapping.document_end_offset]
            == sentence.text
        )
