"""切分以完整表达为单位，字符位置始终能回到原文。"""

import asyncio

import pytest
from pydantic import ValidationError

from domain.common import QueryInputType
from domain.query_models import QueryProcessingResult, QueryScene
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query, split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "培训要求：不得使用“保证收益。稳赚不赔！”这样的说法。请如实披露。",
            ["培训要求：不得使用“保证收益。稳赚不赔！”这样的说法。", "请如实披露。"],
        ),
        ('"保本。保证8%。"属于禁止话术。', ['"保本。保证8%。"属于禁止话术。']),
        (
            "不得使用“客户：保本吗 顾问：是的。”这样的说法。",
            ["不得使用“客户：保本吗 顾问：是的。”这样的说法。"],
        ),
        (
            "说明（非保证利益；可能为零）以合同为准。下一项。",
            ["说明（非保证利益；可能为零）以合同为准。", "下一项。"],
        ),
        (
            "说明：“禁止使用‘保本。’的说法。”\n顾问：好的。",
            ["说明：“禁止使用‘保本。’的说法。”", "顾问：好的。"],
        ),
        ("客户：是否保本 顾问：不能保证", ["客户：是否保本", "顾问：不能保证"]),
        ("客户：是否保本，顾问：不能保证。", ["客户：是否保本，", "顾问：不能保证。"]),
        ("消费者：收益如何\n代理人：不能保证", ["消费者：收益如何", "代理人：不能保证"]),
        ("q: 是否保证收益\na: 不能保证", ["q: 是否保证收益", "a: 不能保证"]),
        (
            "1. 演示收益3.5%。\n2. 网址 https://example.com/terms 以合同为准。",
            ["1. 演示收益3.5%。", "2. 网址 https://example.com/terms 以合同为准。"],
        ),
        ("客户：“是否保本？”\n顾问：“不保证。”", ["客户：“是否保本？”", "顾问：“不保证。”"]),
        (
            "非保证利益的实际金额\r\n以公司经营状况为准。",
            ["非保证利益的实际金额\r\n以公司经营状况为准。"],
        ),
        ("  📢 原句。\r\n\r\n👉 另一句𠀀。  ", ["📢 原句。", "👉 另一句𠀀。"]),
    ],
)
def test_complete_expression_boundaries_and_exact_offsets(text: str, expected: list[str]) -> None:
    spans = split_sentences(text)
    assert [span.text for span in spans] == expected
    cursor = 0
    for span in spans:
        assert not text[cursor : span.start_offset].strip()
        assert text[span.start_offset : span.end_offset] == span.text
        cursor = span.end_offset
    assert not text[cursor:].strip()


def _prepare() -> QueryProcessingResult:
    text = "  📢 客户：是否保证收益？\n代理人：不能保证。  "
    return asyncio.run(
        prepare_query(
            text,
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
            scene_override=QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        )
    )


def test_emoji_role_prefix_and_code_point_offset_contract() -> None:
    query = _prepare()
    assert query.source_text.startswith("  📢")
    assert query.source_offset_unit == "unicode_code_point"
    assert query.source_mappings[0].document_start_offset == 2
    assert [context.speaker_role for context in query.sentence_contexts] == ["customer", "sales"]


@pytest.mark.parametrize("fault", ["out_of_bounds", "overlap", "wrong_text", "missing_text"])
def test_invalid_source_mapping_is_rejected_before_rendering(fault: str) -> None:
    data = _prepare().model_dump()
    if fault == "out_of_bounds":
        data["source_mappings"][1]["document_end_offset"] = len(data["source_text"]) + 1
    elif fault == "overlap":
        data["source_mappings"][1]["document_start_offset"] = 2
    elif fault == "wrong_text":
        data["sentences"][0]["text"] = "另一句话。"
    else:
        data["source_text"] += "不能遗漏这句话。"
    with pytest.raises(ValidationError, match="原文|目标句"):
        QueryProcessingResult.model_validate(data)
