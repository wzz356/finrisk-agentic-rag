"""PDF 文件读取器，原生文本层优先，RapidOCR 按页降级。"""

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pymupdf
from PIL import Image

from domain.common import ExtractionMethod
from domain.file_models import ExtractedTable
from extraction.config import ExtractionConfig
from extraction.errors import ExtractionError
from extraction.geometry import normalize_rectangle
from extraction.ocr_reader import RapidOcrReader
from extraction.types import DocumentDraft, ElementDraft, PageDraft


def read_pdf(
    path: Path,
    config: ExtractionConfig,
    ocr_reader: RapidOcrReader,
) -> DocumentDraft:
    """逐页读取 PDF，并仅对低质量文本层页面执行 OCR。"""

    page_drafts: list[PageDraft] = []
    tables: list[ExtractedTable] = []
    try:
        document = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    except (RuntimeError, ValueError) as error:
        raise ExtractionError("PDF 无法读取，请检查文件是否损坏或重新导出") from error
    with document:
        if document.needs_pass:
            raise ExtractionError("PDF 已加密，请先移除打开密码后上传")
        for page in document:  # type: ignore[attr-defined]
            tables.extend(_extract_native_tables(page, page.number + 1))
            native_page = _read_native_page(page, config)
            if _native_text_is_usable(native_page, config):
                page_drafts.append(native_page)
                continue

            reason = (
                "原生文本层为空或质量不足，已使用 RapidOCR；"
                f"文本字符数={sum(len(block.text) for block in native_page.blocks)}，"
                f"质量={native_page.quality_score:.4f}"
            )
            ocr_page = _read_ocr_page(page, config, ocr_reader)
            if ocr_page.blocks:
                page_drafts.append(
                    replace(
                        ocr_page,
                        warnings=[reason, *ocr_page.warnings],
                    )
                )
            elif native_page.blocks:
                page_drafts.append(
                    replace(
                        native_page,
                        warnings=[reason, "RapidOCR 未识别到文本，保留原生文本层"],
                    )
                )
            else:
                page_drafts.append(
                    replace(
                        ocr_page,
                        warnings=[reason, *ocr_page.warnings],
                    )
                )

    return DocumentDraft(pages=page_drafts, tables=merge_cross_page_tables(tables))


def calculate_text_quality(text: str, minimum_chars: int) -> float:
    """根据长度、可打印字符和有效文字比例估算原生文本质量。"""

    compact = "".join(character for character in text if not character.isspace())
    if not compact:
        return 0.0

    length_score = 1.0 if minimum_chars == 0 else min(len(compact) / minimum_chars, 1.0)
    printable_ratio = sum(
        character.isprintable() and character != "\ufffd" for character in compact
    ) / len(compact)
    meaningful_ratio = sum(
        character.isalnum() or "\u4e00" <= character <= "\u9fff" for character in compact
    ) / len(compact)

    return min(
        max(0.2 * length_score + 0.5 * printable_ratio + 0.3 * meaningful_ratio, 0.0),
        1.0,
    )


def _read_native_page(page: pymupdf.Page, config: ExtractionConfig) -> PageDraft:
    """读取单页原生文本块和字词坐标。"""

    width = float(page.rect.width)
    height = float(page.rect.height)
    blocks: list[ElementDraft] = []

    raw_blocks = cast(list[tuple[Any, ...]], page.get_text("blocks", sort=True))  # type: ignore[no-untyped-call]
    for raw_block in raw_blocks:
        if len(raw_block) < 7 or int(raw_block[6]) != 0:
            continue
        text = str(raw_block[4]).strip()
        if not text:
            continue
        blocks.append(
            ElementDraft(
                text=text,
                confidence=None,
                polygon=normalize_rectangle(
                    float(raw_block[0]),
                    float(raw_block[1]),
                    float(raw_block[2]),
                    float(raw_block[3]),
                    width,
                    height,
                ),
                extraction_method=ExtractionMethod.PDF_TEXT,
            )
        )

    words: list[ElementDraft] = []
    raw_words = cast(list[tuple[Any, ...]], page.get_text("words", sort=True))  # type: ignore[no-untyped-call]
    for raw_word in raw_words:
        if len(raw_word) < 5:
            continue
        text = str(raw_word[4]).strip()
        if not text:
            continue
        words.append(
            ElementDraft(
                text=text,
                confidence=None,
                polygon=normalize_rectangle(
                    float(raw_word[0]),
                    float(raw_word[1]),
                    float(raw_word[2]),
                    float(raw_word[3]),
                    width,
                    height,
                ),
                extraction_method=ExtractionMethod.PDF_TEXT,
            )
        )

    combined_text = "\n".join(block.text for block in blocks)
    return PageDraft(
        width=width,
        height=height,
        blocks=blocks,
        words=words,
        used_ocr=False,
        average_confidence=None,
        quality_score=calculate_text_quality(
            combined_text,
            config.min_native_text_chars,
        ),
        extraction_method=ExtractionMethod.PDF_TEXT,
    )


def _native_text_is_usable(
    page: PageDraft,
    config: ExtractionConfig,
) -> bool:
    """判断原生文本是否达到长度和质量阈值。"""

    character_count = sum(
        len("".join(character for character in block.text if not character.isspace()))
        for block in page.blocks
    )
    return (
        character_count >= config.min_native_text_chars
        and page.quality_score >= config.min_native_text_quality
    )


def _read_ocr_page(
    page: pymupdf.Page,
    config: ExtractionConfig,
    ocr_reader: RapidOcrReader,
) -> PageDraft:
    """把 PDF 页面渲染为图片并交给 RapidOCR。"""

    pixmap = page.get_pixmap(dpi=config.pdf_render_dpi, alpha=False)
    png_bytes = pixmap.tobytes("png")  # type: ignore[no-untyped-call]
    with Image.open(BytesIO(png_bytes)) as image:
        return ocr_reader.read(image)


def _extract_native_tables(page: pymupdf.Page, page_number: int) -> list[ExtractedTable]:
    """使用 PyMuPDF 的线框/文本表格识别保留单元格结构。"""

    try:
        finder = page.find_tables()  # type: ignore[no-untyped-call]
    except (AttributeError, RuntimeError, ValueError):
        return []
    extracted: list[ExtractedTable] = []
    for table in finder.tables:
        raw_rows = table.extract()
        rows = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in raw_rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            continue
        extracted.append(
            ExtractedTable(
                page_number=page_number,
                page_numbers=[page_number],
                title=None,
                rows=rows,
                confidence=0.92,
            )
        )
    return extracted


def merge_cross_page_tables(tables: list[ExtractedTable]) -> list[ExtractedTable]:
    """按连续页和重复表头合并跨页表格，并去掉续页重复表头。"""

    merged: list[ExtractedTable] = []
    for table in tables:
        if not merged or not _is_table_continuation(merged[-1], table):
            merged.append(table)
            continue
        previous = merged[-1]
        page_numbers = [
            *previous.page_numbers,
            *(table.page_numbers or ([table.page_number] if table.page_number else [])),
        ]
        merged[-1] = previous.model_copy(
            update={
                "rows": [*previous.rows, *table.rows[1:]],
                "page_numbers": list(dict.fromkeys(page_numbers)),
                "is_cross_page": True,
                "confidence": min(previous.confidence or 1.0, table.confidence or 1.0),
            }
        )
    return merged


def _is_table_continuation(previous: ExtractedTable, current: ExtractedTable) -> bool:
    if not previous.rows or not current.rows:
        return False
    previous_pages = previous.page_numbers or (
        [previous.page_number] if previous.page_number is not None else []
    )
    current_pages = current.page_numbers or (
        [current.page_number] if current.page_number is not None else []
    )
    if not previous_pages or not current_pages or current_pages[0] != previous_pages[-1] + 1:
        return False
    previous_header = tuple(cell.casefold() for cell in previous.rows[0])
    current_header = tuple(cell.casefold() for cell in current.rows[0])
    return bool(any(previous_header)) and previous_header == current_header
