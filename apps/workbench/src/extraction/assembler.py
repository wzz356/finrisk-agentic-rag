"""把读取草稿组装成带页内和全文偏移的领域对象。"""

from domain.file_models import NormalizedPoint, PageContent, TextBlock, WordContent
from extraction.types import ElementDraft, PageDraft, Polygon


def assemble_pages(drafts: list[PageDraft]) -> tuple[list[PageContent], str]:
    """组装全部页面，并返回分页对象和完整文档文本。"""

    pages: list[PageContent] = []
    page_texts: list[str] = []
    document_offset = 0

    for page_index, draft in enumerate(drafts):
        page_number = page_index + 1
        page_text, block_ranges = _join_blocks(draft.blocks)
        blocks = [
            _build_block(
                element=element,
                page_number=page_number,
                reading_order=index,
                page_range=block_ranges[index],
                document_offset=document_offset,
            )
            for index, element in enumerate(draft.blocks)
        ]
        words = _build_words(
            elements=draft.words,
            page_number=page_number,
            page_text=page_text,
            document_offset=document_offset,
        )
        pages.append(
            PageContent(
                page_number=page_number,
                width=draft.width,
                height=draft.height,
                text=page_text,
                blocks=blocks,
                words=words,
                used_ocr=draft.used_ocr,
                average_confidence=draft.average_confidence,
                quality_score=draft.quality_score,
                extraction_method=draft.extraction_method,
                warnings=draft.warnings,
            )
        )
        page_texts.append(page_text)
        document_offset += len(page_text)
        if page_index < len(drafts) - 1:
            document_offset += 2

    return pages, "\n\n".join(page_texts)


def _join_blocks(blocks: list[ElementDraft]) -> tuple[str, list[tuple[int, int]]]:
    """使用换行连接文本块并记录每个块的页内偏移。"""

    parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0

    for index, block in enumerate(blocks):
        if index:
            cursor += 1
        start = cursor
        parts.append(block.text)
        cursor += len(block.text)
        ranges.append((start, cursor))

    return "\n".join(parts), ranges


def _build_block(
    element: ElementDraft,
    page_number: int,
    reading_order: int,
    page_range: tuple[int, int],
    document_offset: int,
) -> TextBlock:
    """把文本块草稿转换为带完整偏移的领域模型。"""

    start, end = page_range
    return TextBlock(
        block_id=f"p{page_number}-b{reading_order}",
        page_number=page_number,
        text=element.text,
        confidence=element.confidence,
        polygon=_to_points(element.polygon),
        reading_order=reading_order,
        page_start_offset=start,
        page_end_offset=end,
        document_start_offset=document_offset + start,
        document_end_offset=document_offset + end,
        extraction_method=element.extraction_method,
    )


def _build_words(
    elements: list[ElementDraft],
    page_number: int,
    page_text: str,
    document_offset: int,
) -> list[WordContent]:
    """按顺序把字词对齐到页文本，并补充全文偏移。"""

    words: list[WordContent] = []
    search_cursor = 0

    for index, element in enumerate(elements):
        start = page_text.find(element.text, search_cursor)
        if start < 0:
            start = page_text.find(element.text)
        end = start + len(element.text) if start >= 0 else None
        if end is not None:
            search_cursor = end

        words.append(
            WordContent(
                word_id=f"p{page_number}-w{index}",
                page_number=page_number,
                text=element.text,
                confidence=element.confidence,
                polygon=_to_points(element.polygon),
                reading_order=index,
                page_start_offset=start if start >= 0 else None,
                page_end_offset=end,
                document_start_offset=document_offset + start if start >= 0 else None,
                document_end_offset=document_offset + end if end is not None else None,
                extraction_method=element.extraction_method,
            )
        )

    return words


def _to_points(polygon: Polygon | None) -> list[NormalizedPoint] | None:
    """把内部坐标元组转换为领域坐标对象。"""

    if polygon is None:
        return None
    return [NormalizedPoint(x=x, y=y) for x, y in polygon]
