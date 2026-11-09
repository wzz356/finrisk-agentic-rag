"""图片文件读取器。"""

from pathlib import Path

from PIL import Image

from extraction.ocr_reader import RapidOcrReader
from extraction.types import DocumentDraft


def read_image(path: Path, ocr_reader: RapidOcrReader) -> DocumentDraft:
    """打开图片并使用 RapidOCR 读取。"""

    with Image.open(path) as image:
        page = ocr_reader.read(image)
    return DocumentDraft(pages=[page])
