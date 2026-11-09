"""Query 切句、Risk 四字段输入和原文映射。"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from domain.common import QueryInputType
from domain.file_models import FileExtractionResult, PageContent, TextBlock
from domain.query_models import (
    QueryExtractionSummary,
    QueryProcessingResult,
    QueryScene,
    QuerySourceRegion,
    RiskQuerySentence,
    SentenceSourceMapping,
)
from query.classification import QueryClassificationContext, QuerySceneClassifier
from query.context import SPEAKER_NAMES_PATTERN, build_sentence_contexts

_STRUCTURAL_BOUNDARY = re.compile(
    r"(?:\r?\n[ \t]*\r?\n+)"
    r"|(?:\r?\n(?=[ \t]*(?:[·•●▪✅⭐👏👉🔥✔📢🙋🌈🏆❤✨🔸💁💡🎁💔🏠🔒#]"
    r"|[-*＊][ \t]*|\d+[.、])))"
    rf"|(?:\r?\n(?=[ \t]*(?:{SPEAKER_NAMES_PATTERN})\s*[:：]))"
    r"|(?:\r?\n(?=[ \t]*(?:【(?:产品|方案)|(?:产品|方案)\s*"
    r"[A-Za-z0-9甲乙丙丁一二三四五六七八九十]+\s*[:：])))"
    rf"|(?<=[ \t，,；;。！？!?])(?=(?:{SPEAKER_NAMES_PATTERN})\s*[:：])",
    re.IGNORECASE,
)
_OPEN_TO_CLOSE = {
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "（": "）",
    "(": ")",
    "[": "]",
    "【": "】",
    "《": "》",
}
_CLOSERS = frozenset((*_OPEN_TO_CLOSE.values(), '"'))
_TERMINATORS = frozenset("。！？!?；;")


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """从完整 Query 中切分出、保留全文偏移的一句话。"""

    text: str
    start_offset: int
    end_offset: int


def split_sentences(text: str) -> list[SentenceSpan]:
    """以完整表达和对话轮次为单元；引用、括号及 OCR 软换行留在原句中。"""

    if not text.strip():
        raise ValueError("Query 文本不能为空")

    sentences: list[SentenceSpan] = []
    structural = set()
    inline_turns = set()
    for match in _STRUCTURAL_BOUNDARY.finditer(text):
        if match.start() == match.end():
            line_prefix = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
            if re.fullmatch(r"[ \t]*(?:[·•●▪✅⭐👏👉🔥✔📢🙋🌈🏆]\ufe0f?\s*)?", line_prefix):
                continue  # “📢 客户：”属于同一个句首，不能把装饰符号切成一句。
            inline_turns.add(match.end())
        structural.add(match.end())
    segment_start = 0
    stack: list[str] = []
    index = 0
    while index < len(text):
        if index in structural and (index not in inline_turns or not stack):
            _append_sentence(sentences, text, segment_start, index)
            segment_start = index
            stack.clear()  # 明确的新说话轮次可终止上一轮未闭合的引号。
        char = text[index]
        if char in _OPEN_TO_CLOSE:
            stack.append(_OPEN_TO_CLOSE[char])
        elif char == '"' and (index == 0 or text[index - 1] != "\\"):
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        elif stack and char == stack[-1]:
            stack.pop()
        is_period = char == "." and (
            index + 1 == len(text) or text[index + 1].isspace() or text[index + 1] in _CLOSERS
        )
        # 行首的“1. 项目”是列表编号；3.5%、域名与网址中的点也不是句末。
        if is_period and re.fullmatch(r"[ \t]*\d+", text[text.rfind("\n", 0, index) + 1 : index]):
            is_period = False
        if char not in _TERMINATORS and not is_period:
            index += 1
            continue
        end = index + 1
        remaining = list(stack)
        while end < len(text):
            next_char = text[end]
            if next_char in _TERMINATORS:
                end += 1
            elif remaining and next_char == remaining[-1]:
                remaining.pop()
                end += 1
            elif not remaining and next_char in _CLOSERS:
                end += 1
            else:
                break
        # 引号/括号里的标点不拆开“‘保证收益。’是错误话术”这一完整判断。
        can_close = not stack or (not remaining and (end == len(text) or text[end].isspace()))
        if can_close:
            _append_sentence(sentences, text, segment_start, end)
            segment_start = end
            stack.clear()
            index = end
        else:
            index += 1
    _append_sentence(sentences, text, segment_start, len(text))
    if not sentences:
        raise ValueError("Query 文本未包含可处理句子")
    return sentences


async def prepare_query(
    text: str,
    classifier: QuerySceneClassifier,
    *,
    input_type: QueryInputType,
    extraction: FileExtractionResult | None = None,
    metadata: Mapping[str, object] | None = None,
    scene_override: QueryScene | None = None,
) -> QueryProcessingResult:
    """整份材料分类后，生成逐句 Risk 输入和原文定位表。"""

    classification_context = (
        QueryClassificationContext(
            media_type=extraction.media_type,
            page_count=extraction.page_count,
        )
        if extraction
        else None
    )
    classification = await classifier.classify(
        text,
        classification_context,
        scene_override=scene_override,
    )
    query_id = _query_id(text, extraction.document_id if extraction else None)
    spans = split_sentences(text)
    sentences: list[RiskQuerySentence] = []
    mappings: list[SentenceSourceMapping] = []

    for index, span in enumerate(spans, start=1):
        sentence_id = f"{query_id}_s{index:04d}"
        sentences.append(
            RiskQuerySentence(
                query_id=query_id,
                sentence_id=sentence_id,
                scene=classification.scene,
                text=span.text,
            )
        )
        mappings.append(
            _build_source_mapping(
                query_id=query_id,
                sentence_id=sentence_id,
                input_type=input_type,
                span=span,
                context_before=spans[index - 2].text if index > 1 else "",
                context_after=spans[index].text if index < len(spans) else "",
                extraction=extraction,
            )
        )

    contexts = build_sentence_contexts(text, sentences, mappings)
    for mapping, context in zip(mappings, contexts, strict=True):
        mapping.context_before = context.context_before
        mapping.context_after = context.context_after
    return QueryProcessingResult(
        query_id=query_id,
        input_type=input_type,
        source_text=text,
        source_document_id=extraction.document_id if extraction else None,
        source_filename=extraction.source_filename if extraction else None,
        extraction=_extraction_summary(text, extraction),
        classification=classification,
        sentences=sentences,
        source_mappings=mappings,
        sentence_contexts=contexts,
        created_at=datetime.now(UTC),
        metadata=dict(metadata or {}),
    )


def _append_sentence(
    sentences: list[SentenceSpan],
    full_text: str,
    raw_start: int,
    raw_end: int,
) -> None:
    raw_segment = full_text[raw_start:raw_end]
    if not raw_segment.strip():
        return
    leading_space_count = len(raw_segment) - len(raw_segment.lstrip())
    trailing_space_count = len(raw_segment) - len(raw_segment.rstrip())
    start_offset = raw_start + leading_space_count
    end_offset = raw_end - trailing_space_count
    sentences.append(
        SentenceSpan(
            text=full_text[start_offset:end_offset],
            start_offset=start_offset,
            end_offset=end_offset,
        )
    )


def _build_source_mapping(
    *,
    query_id: str,
    sentence_id: str,
    input_type: QueryInputType,
    span: SentenceSpan,
    context_before: str,
    context_after: str,
    extraction: FileExtractionResult | None,
) -> SentenceSourceMapping:
    regions: list[QuerySourceRegion] = []
    if extraction:
        for page in extraction.pages:
            for block in page.blocks:
                if not _overlaps(span, block):
                    continue
                overlap_start = max(span.start_offset, block.document_start_offset)
                overlap_end = min(span.end_offset, block.document_end_offset)
                regions.append(
                    QuerySourceRegion(
                        page_number=page.page_number,
                        block_id=block.block_id,
                        word_ids=_overlapping_word_ids(page, overlap_start, overlap_end),
                        text=extraction.full_text[overlap_start:overlap_end],
                        document_start_offset=overlap_start,
                        document_end_offset=overlap_end,
                        polygon=block.polygon,
                        confidence=block.confidence,
                        extraction_method=block.extraction_method.value,
                    )
                )

    return SentenceSourceMapping(
        query_id=query_id,
        sentence_id=sentence_id,
        input_type=input_type,
        source_document_id=extraction.document_id if extraction else None,
        source_filename=extraction.source_filename if extraction else None,
        document_start_offset=span.start_offset,
        document_end_offset=span.end_offset,
        context_before=context_before,
        context_after=context_after,
        regions=regions,
    )


def _extraction_summary(
    text: str,
    extraction: FileExtractionResult | None,
) -> QueryExtractionSummary:
    if extraction is None:
        encoded = text.encode("utf-8")
        return QueryExtractionSummary(
            media_type="text/plain",
            checksum_sha256=sha256(encoded).hexdigest(),
            file_size_bytes=len(encoded),
            page_count=1,
            ocr_page_count=0,
            extraction_method="native_text",
            quality_score=1.0,
            block_count=1,
            word_count=0,
        )
    return QueryExtractionSummary(
        media_type=extraction.media_type,
        checksum_sha256=extraction.checksum_sha256,
        file_size_bytes=extraction.file_size_bytes,
        page_count=extraction.page_count,
        ocr_page_count=extraction.ocr_page_count,
        extraction_method=extraction.extraction_method.value,
        quality_score=extraction.quality_score,
        block_count=sum(len(page.blocks) for page in extraction.pages),
        word_count=sum(len(page.words) for page in extraction.pages),
        warnings=extraction.warnings,
    )


def _overlaps(span: SentenceSpan, block: TextBlock) -> bool:
    return (
        block.document_start_offset < span.end_offset
        and block.document_end_offset > span.start_offset
    )


def _overlapping_word_ids(
    page: PageContent,
    start_offset: int,
    end_offset: int,
) -> list[str]:
    return [
        word.word_id
        for word in page.words
        if word.document_start_offset is not None
        and word.document_end_offset is not None
        and word.document_start_offset < end_offset
        and word.document_end_offset > start_offset
    ]


def _query_id(text: str, source_document_id: str | None) -> str:
    identity_source = f"{source_document_id or ''}\0{text}".encode()
    return f"query_{sha256(identity_source).hexdigest()[:20]}"


__all__ = ["SentenceSpan", "prepare_query", "split_sentences"]
