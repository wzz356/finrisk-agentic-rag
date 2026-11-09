"""构建有原文绑定的句子窗口，保留问答角色并隔离段落和产品作用域。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from domain.query_models import (
    ContextSourceSentence,
    RiskQuerySentence,
    RiskSentenceContext,
    SentenceSourceMapping,
)

SPEAKER_NAMES_PATTERN = (
    r"客户|投保人|消费者|销售人员|销售|业务员|营销员|代理人|坐席|客服|顾问|"
    r"主播|主持人|讲师|主管|老师|学员|问|答|Q|A"
)
_SPEAKER = re.compile(
    r"^(?:【材料载体[:：][^】]+】\s*)?"
    r"(?:[·•●▪✅⭐👏👉🔥✔📢🙋🌈🏆]\ufe0f?\s*)?"
    rf"(?P<name>{SPEAKER_NAMES_PATTERN})\s*[:：]\s*",
    re.IGNORECASE,
)
_PRODUCT_START = re.compile(
    r"^(?:【(?:产品|方案)[^】]+】|(?:产品|方案)\s*[A-Za-z0-9甲乙丙丁一二三四五六七八九十]+"
    r"\s*[:：]|(?:另一款|另一个|切换到|换到).{0,8}产品)"
)
_EXAMPLE = re.compile(
    r"(?:^|[:：。；;\n])\s*(?:(?:以下|下列|下面|这些).{0,8}|(?:这是|示例)\s*)?"
    r"(?:反面|错误|禁止)(?:话术|示例|说法|表述)|"
    r"(?:以下|下列|这些).{0,12}(?:话术|说法|表述).{0,12}(?:禁止|不得|严禁)"
)
_NEW_EXAMPLE = re.compile(r"^(?:正确|合规|正面)(?:示例|话术|说法|表述)")
_ACTUAL_BEHAVIOR = re.compile(
    r"(?:实际|现场|仍然|仍|却).{0,20}(?:承诺|宣传|执行|照做|使用|销售)|"
    r"(?:销售|业务员|营销员|代理人).{0,8}(?:称|说|承诺|向客户|使用|采用|宣传|宣称)"
)
_DENIAL = re.compile(r"^(?:不可以|不能|不是|并非|不一定|不保证|无法保证|没有保证|不能这样理解)")
_AFFIRMATIVE = re.compile(
    r"^(?:是的|当然|没错|对的|对[，,。！!]|确实如此|就是这个意思|可以这样理解)"
)
_REFERENCE = re.compile(r"^(?:这个|这款|该产品|上述|前述|它|这样|那就|同样|照这个)")
_PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n")
_ADOPT_INSTRUCTION = re.compile(
    r"(?:要求|指导|让)(?![^,，。;；\n]{0,24}(?:不得|不能|禁止|严禁|不要))"
    r"[^,，。;；\n]{0,24}(?:照讲|照说|按这句讲|照此销售)"
)


def has_adopted_instruction(text: str) -> bool:
    """Recognize affirmative reuse of quoted instructions, retaining prohibition scope."""
    for match in _ADOPT_INSTRUCTION.finditer(text):
        prefix = re.split(r"[,，。;；\n]", text[: match.start()])[-1]
        if not re.search(r"(?:不得|严禁|禁止|不能|不应|不要|不再).{0,8}$", prefix):
            return True
    return False


@dataclass(frozen=True, slots=True)
class _ScopedSentence:
    scope: int
    role: str
    name: str | None
    body: str
    negative_example: bool


def sentence_body(text: str) -> str:
    match = _SPEAKER.match(text)
    return text[match.end() :].strip() if match else text.strip()


def is_affirmative_answer(text: str) -> bool:
    return _AFFIRMATIVE.search(sentence_body(text)) is not None


def _role(name: str) -> str:
    if name in {"客户", "投保人", "消费者"}:
        return "customer"
    if name in {"销售人员", "销售", "业务员", "营销员", "代理人", "顾问", "坐席", "客服", "主播"}:
        return "sales"
    if name in {"讲师", "主管", "老师", "学员"}:
        return "training"
    if name.upper() == "Q" or name == "问":
        return "question"
    if name.upper() == "A" or name == "答":
        return "answer"
    return "host"


def build_sentence_contexts(
    source_text: str,
    sentences: list[RiskQuerySentence],
    mappings: list[SentenceSourceMapping],
    *,
    neighbor_count: int = 2,
    max_neighbor_chars: int = 1600,
) -> list[RiskSentenceContext]:
    scoped: list[_ScopedSentence] = []
    scope = 0
    name: str | None = None
    role = "unknown"
    negative_example = False
    for index, (sentence, mapping) in enumerate(zip(sentences, mappings, strict=True)):
        gap = (
            source_text[mappings[index - 1].document_end_offset : mapping.document_start_offset]
            if index
            else ""
        )
        if index and (
            _PARAGRAPH_BREAK.search(gap)
            or _PRODUCT_START.match(sentence.text)
            or _NEW_EXAMPLE.match(sentence.text)
        ):
            scope += 1
            name, role, negative_example = None, "unknown", False
        speaker = _SPEAKER.match(sentence.text)
        if speaker:
            name = speaker.group("name")
            role = _role(name)
        body = sentence_body(sentence.text)
        if _ACTUAL_BEHAVIOR.search(body) or has_adopted_instruction(body):
            negative_example = False
        elif _EXAMPLE.search(body[:120]):
            negative_example = True
        scoped.append(_ScopedSentence(scope, role, name, body, negative_example))

    contexts: list[RiskSentenceContext] = []
    for index, sentence in enumerate(sentences):
        target = scoped[index]
        candidates = [
            position
            for position in range(
                max(0, index - neighbor_count), min(len(sentences), index + neighbor_count + 1)
            )
            if position != index and scoped[position].scope == target.scope
        ]
        selected = [index]
        remaining = max_neighbor_chars
        truncated = False
        for position in sorted(candidates, key=lambda value: (abs(value - index), value)):
            if len(sentences[position].text) <= remaining:
                selected.append(position)
                remaining -= len(sentences[position].text)
            else:
                truncated = True
        sources = [
            ContextSourceSentence(
                sentence_id=sentences[position].sentence_id,
                text=sentences[position].text,
                document_start_offset=mappings[position].document_start_offset,
                document_end_offset=mappings[position].document_end_offset,
                speaker_role=scoped[position].role,
                speaker_name=scoped[position].name,
                is_target=position == index,
            )
            for position in sorted(selected)
        ]
        before = "\n".join(
            sentences[position].text for position in sorted(selected) if position < index
        )
        after = "\n".join(
            sentences[position].text for position in sorted(selected) if position > index
        )
        stance: str = "statement"
        reasons: list[str] = []
        if has_adopted_instruction(target.body):
            reasons.append("当前要求实际采纳引用话术，需核查其内容，不能仅按反面示例放行")
        if target.negative_example:
            stance = "negative_example"
            reasons.append("当前段落明确标为反面或禁止话术，引用不等于实施")
        elif target.role == "question" or re.search(r"[?？][”’」』）)]*$", target.body):
            stance = "question"
            reasons.append("本句是提问，需结合回答区分询问与销售承诺")
        elif _DENIAL.match(target.body):
            stance = "denial"
            reasons.append("本句是否定或纠正，需要核对其指向的前文")
        elif re.match(r"^(?:不得|严禁|禁止|不能|切勿)", target.body):
            stance = "prohibition"
        if is_affirmative_answer(sentence.text):
            reasons.append("本句是简短肯定回答，需保留前一问题的对象和条件")
        if _REFERENCE.match(target.body):
            reasons.append("本句含指代或承接，孤立单句可能缺少对象")
        warnings = ["邻句窗口超出预算，部分上下文未纳入，需查看完整原文"] if truncated else []
        contexts.append(
            RiskSentenceContext(
                query_id=sentence.query_id,
                sentence_id=sentence.sentence_id,
                query_text=sentence.text,
                scene=sentence.scene,
                scope_id=f"{sentence.query_id}_scope{target.scope + 1:04d}",
                context_before=before,
                context_after=after,
                speaker_role=target.role,
                speaker_name=target.name,
                stance=stance,  # type: ignore[arg-type]
                requires_review=bool(reasons) or truncated,
                review_reasons=reasons,
                context_truncated=truncated,
                warnings=warnings,
                source_sentences=sources,
            )
        )
    return contexts


def build_retrieval_query(context: RiskSentenceContext) -> str:
    """以目标句为主，再补充同一作用域的对象和条件；不生成新事实。"""
    neighbors = [source.text for source in context.source_sentences if not source.is_target]
    return "\n".join([context.query_text, *neighbors])
