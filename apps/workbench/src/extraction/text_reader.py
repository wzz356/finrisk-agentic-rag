"""纯文本文件读取器。"""

from pathlib import Path

from domain.common import ExtractionMethod
from extraction.types import DocumentDraft, ElementDraft, PageDraft

_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def read_text(path: Path) -> DocumentDraft:
    """按常见中文编码读取 TXT 或 Markdown 文件。"""

    warnings: list[str] = []
    text: str | None = None
    used_encoding: str | None = None

    for encoding in _ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if text is None or used_encoding is None:
        raise UnicodeError(f"无法识别文本文件编码：{path.name}")
    if used_encoding not in {"utf-8", "utf-8-sig"}:
        warnings.append(f"文本文件使用兼容编码读取：{used_encoding}")

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = [
        ElementDraft(
            text=paragraph.strip(),
            confidence=None,
            polygon=None,
            extraction_method=ExtractionMethod.PLAIN_TEXT,
        )
        for paragraph in cleaned.split("\n")
        if paragraph.strip()
    ]
    quality = 1.0 if blocks else 0.0
    return DocumentDraft(
        pages=[
            PageDraft(
                width=None,
                height=None,
                blocks=blocks,
                words=[],
                used_ocr=False,
                average_confidence=None,
                quality_score=quality,
                extraction_method=ExtractionMethod.PLAIN_TEXT,
                warnings=warnings,
            )
        ],
        warnings=warnings,
    )
