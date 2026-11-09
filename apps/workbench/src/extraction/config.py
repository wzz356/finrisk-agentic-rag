"""文件读取模块配置。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """控制 PDF 文本层质量判断和 OCR 行为。"""

    pdf_render_dpi: int = 200
    min_native_text_chars: int = 20
    min_native_text_quality: float = 0.65
    min_ocr_confidence: float = 0.0
    include_word_boxes: bool = True

    def __post_init__(self) -> None:
        """检查配置范围，避免运行中出现无意义参数。"""

        if self.pdf_render_dpi <= 0:
            raise ValueError("pdf_render_dpi 必须大于 0")
        if self.min_native_text_chars < 0:
            raise ValueError("min_native_text_chars 不能小于 0")
        if not 0 <= self.min_native_text_quality <= 1:
            raise ValueError("min_native_text_quality 必须在 0 到 1 之间")
        if not 0 <= self.min_ocr_confidence <= 1:
            raise ValueError("min_ocr_confidence 必须在 0 到 1 之间")
