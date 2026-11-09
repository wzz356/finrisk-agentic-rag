"""文件读取器内部使用的草稿数据结构。"""

from dataclasses import dataclass, field
from typing import TypeAlias

from domain.common import ExtractionMethod
from domain.file_models import ExtractedTable

Point: TypeAlias = tuple[float, float]
Polygon: TypeAlias = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class ElementDraft:
    """尚未补齐字符偏移的文本元素。"""

    text: str
    confidence: float | None
    polygon: Polygon | None
    extraction_method: ExtractionMethod


@dataclass(frozen=True, slots=True)
class PageDraft:
    """单页读取结果草稿。"""

    width: float | None
    height: float | None
    blocks: list[ElementDraft] = field(default_factory=list)
    words: list[ElementDraft] = field(default_factory=list)
    used_ocr: bool = False
    average_confidence: float | None = None
    quality_score: float = 0.0
    extraction_method: ExtractionMethod = ExtractionMethod.PLAIN_TEXT
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DocumentDraft:
    """具体格式读取器返回的文档草稿。"""

    pages: list[PageDraft] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
