"""上下文窗口的边界、说话人继承与原文可逆性。"""

import asyncio

import pytest
from pydantic import ValidationError

from domain.common import QueryInputType
from domain.query_models import QueryProcessingResult, QueryScene
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query


def prepare(text: str) -> QueryProcessingResult:
    return asyncio.run(
        prepare_query(
            text,
            QuerySceneClassifier(None),
            input_type=QueryInputType.TEXT,
            scene_override=QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        )
    )


def test_unpunctuated_dialogue_turns_preserve_speakers_and_source_offsets() -> None:
    text = "客户：是否保证8%的收益\n顾问：不能保证。具体权益以合同为准。"
    query = prepare(text)
    assert len(query.sentences) == 3
    assert [value.speaker_role for value in query.sentence_contexts] == [
        "customer",
        "sales",
        "sales",
    ]
    for context in query.sentence_contexts:
        for source in context.source_sentences:
            assert text[source.document_start_offset : source.document_end_offset] == source.text
    assert query.sentence_contexts[1].context_before == query.sentences[0].text
    assert query.sentence_contexts[1].stance == "denial"


@pytest.mark.parametrize("separator", ["\n\n", "\n", "\r\n"])
def test_explicit_product_switch_never_borrows_other_product_conditions(separator: str) -> None:
    query = prepare("产品A：分红不保证，可能为零。" + separator + "产品B：保证年化收益8%。")
    left, right = query.sentence_contexts
    assert left.scope_id != right.scope_id
    assert "产品A" not in right.context_before
    assert "产品B" not in left.context_after
    assert len(right.source_sentences) == 1


def test_independent_paragraph_resets_inherited_speaker_and_example_scope() -> None:
    query = prepare("培训材料：反面话术，禁止使用。保证年化收益8%。\n\n实际销售中仍承诺收益8%。")
    assert query.sentence_contexts[1].stance == "negative_example"
    assert query.sentence_contexts[2].stance != "negative_example"
    assert query.sentence_contexts[2].context_before == ""


def test_window_budget_warns_without_rewriting_or_silently_clipping_source() -> None:
    query = prepare("客户：" + "需要说明的条件" * 400 + "。\n顾问：是的，可以这样理解。")
    context = query.sentence_contexts[-1]
    assert context.context_truncated
    assert context.warnings
    assert context.query_text == query.sentences[-1].text
    assert [source.text for source in context.source_sentences] == [context.query_text]


def test_context_quote_with_forged_offsets_is_rejected_by_output_contract() -> None:
    query = prepare("客户：有收益保证吗？\n顾问：不能保证。")
    data = query.model_dump()
    data["sentence_contexts"][0]["source_sentences"][0]["document_start_offset"] = 1
    with pytest.raises(ValidationError, match="原文偏移"):
        QueryProcessingResult.model_validate(data)
