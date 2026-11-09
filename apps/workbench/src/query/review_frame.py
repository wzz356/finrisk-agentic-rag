"""Carry literal document-level quotation/correction framing into split review batches."""

from __future__ import annotations

import re

from domain.query_models import QueryProcessingResult, RiskSentenceContext
from query.context import has_adopted_instruction

_FRAME = re.compile(
    r"反面(?:教材|示例|案例)|错误(?:示例|示范)|教学立场|"
    r"仅(?:作|作为|供).{0,8}(?:警示|教学|反面)|"
    r"(?:做法|方案|提案).{0,16}(?:已被否决|已经否决|未被采纳)|"
    r"请(?:先)?停止.{0,8}方案执行|(?:一旦执行|如果执行).{0,40}属于|"
    r"(?:错误|不当)(?:说法|口径|示例).{0,16}(?:不能照讲|不得模仿|不得照做)"
)


def rejected_proposal_only(text: str) -> bool:
    """A rejected, unexecuted proposal is a corrective review, not an active instruction."""
    return bool(
        re.search(r"方案[（(]待审[）)]|(?:方案|提案).{0,10}尚未执行", text)
        and re.search(r"请先停止该方案执行|该方案已被否决|该提案未被采纳", text)
        and not re.search(
            r"已开始(?:执行|投放)|正在执行|已经实施|仍在执行|照此继续|已查实|已核实|已发生", text
        )
    )


def carry_document_frame(
    query: QueryProcessingResult,
    contexts: dict[str, RiskSentenceContext],
) -> dict[str, RiskSentenceContext]:
    """Add bounded, verbatim framing as context, without changing any risk decision.

    Blank paragraphs still isolate products and speakers. These additional source
    sentences only explain whether a passage is a quotation or a corrective task;
    they do not transfer another paragraph's business facts to the target.
    """
    # A rejected proposal in one scope cannot clear an independently reported fact.
    scope_texts: dict[str, list[str]] = {}
    for sentence in query.sentences:
        scope_texts.setdefault(contexts[sentence.sentence_id].scope_id, []).append(sentence.text)
    rejected_scopes = {
        scope for scope, parts in scope_texts.items() if rejected_proposal_only("\n".join(parts))
    }
    anchors = [sentence for sentence in query.sentences if _FRAME.search(sentence.text)]
    if not anchors:
        return contexts
    chosen = [anchors[0]] if len(anchors) == 1 else [anchors[0], anchors[-1]]
    selected = []
    used = 0
    for sentence in chosen:
        if used + len(sentence.text) > 650:
            continue
        selected.append(sentence)
        used += len(sentence.text)
    mappings = {mapping.sentence_id: mapping for mapping in query.source_mappings}
    from domain.query_models import ContextSourceSentence

    framed = {}
    for sid, context in contexts.items():
        scope_text = "\n".join(scope_texts[context.scope_id])
        rejected = context.scope_id in rejected_scopes
        actual_fact_scope = bool(
            re.search(
                r"已(?:查实|核实|实施|发生)|检查(?:确认|发现)|独立.{0,8}事实|"
                r"(?:业务员|销售|代理人).{0,20}(?:已向|实际|已经)",
                scope_text,
            )
        )
        corrective = bool(
            re.search(
                r"(?:以上|上述).{0,30}(?:错误示范|错误示例|反面教材).{0,30}"
                r"(?:不得|不能|严禁).{0,8}(?:照此|照做|照讲|使用)",
                scope_text,
            )
            and not actual_fact_scope
            and not has_adopted_instruction(scope_text)
        )
        existing = {source.sentence_id for source in context.source_sentences}
        additions = [
            ContextSourceSentence(
                sentence_id=sentence.sentence_id,
                text=sentence.text,
                document_start_offset=mappings[sentence.sentence_id].document_start_offset,
                document_end_offset=mappings[sentence.sentence_id].document_end_offset,
                speaker_role="unknown",
                is_target=False,
            )
            for sentence in selected
            if sentence.sentence_id not in existing
            and (
                not actual_fact_scope or contexts[sentence.sentence_id].scope_id == context.scope_id
            )
        ]
        framed[sid] = context.model_copy(
            update={
                "source_sentences": sorted(
                    [*context.source_sentences, *additions],
                    key=lambda source: source.document_start_offset,
                ),
                "requires_review": False if rejected else True,
                "stance": "negative_example" if rejected or corrective else context.stance,
                "review_reasons": [
                    *context.review_reasons,
                    "全文含引述或纠错立场；仅解释目标句立场，不借用其他段落事实",
                    *(["rejected_proposal_only"] if rejected else []),
                ],
            }
        )
    return framed
