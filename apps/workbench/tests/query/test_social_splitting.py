"""社交平台多行文案的切分回归测试。"""

import pytest

from domain.query_models import QueryScene
from query.classification import RuleBasedSceneClassifier
from query.enhancement import split_sentences


def test_social_media_bullets_start_new_risk_sentences() -> None:
    text = "产品介绍\n✅ 投保门槛\n⭐3人起投\n⭐保障有效期内可增减人员\n👉 点击立即咨询"

    sentences = split_sentences(text)

    assert [item.text for item in sentences] == [
        "产品介绍",
        "✅ 投保门槛",
        "⭐3人起投",
        "⭐保障有效期内可增减人员",
        "👉 点击立即咨询",
    ]
    assert all(text[item.start_offset : item.end_offset] == item.text for item in sentences)


@pytest.mark.parametrize(
    "text",
    [
        "企业主看过来👇\n✅投保门槛\n⭐3人起投\n⭐保费74元起\n👉点击立即咨询",
        "医疗险内卷升级！\n✨核心王牌\n❤可选0免赔\n✅二十年保证续保\n✅配齐全家更省钱",
        "家人们，AI不会骗人！\n✅官方正品\n✅支付加密\n💡健康告知\n🎁会员权益（银保监会盯着呢）",
    ],
)
def test_social_carrier_outweighs_product_parameters_and_regulator_mentions(text):
    result = RuleBasedSceneClassifier().evaluate(text)
    assert result.candidates[0].scene == QueryScene.ST1_POSTER_SMS_SOCIAL


def test_additional_social_bullets_keep_literal_unicode_offsets():
    text = "旅行保障\n❤核心责任\n🔸药品清单\n💡投保说明\n🎁会员服务"
    spans = split_sentences(text)
    assert len(spans) == 5
    assert all(text[span.start_offset:span.end_offset] == span.text for span in spans)
