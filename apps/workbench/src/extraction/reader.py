"""案例和 Query 共用的文件读取统一入口。"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean

from domain.common import (
    ExtractionMethod,
    ExtractionStatus,
    FilePurpose,
)
from domain.file_models import FileExtractionResult, PageContent
from extraction.assembler import assemble_pages
from extraction.config import ExtractionConfig
from extraction.docx_reader import read_docx
from extraction.errors import UnsupportedFileTypeError
from extraction.image_reader import read_image
from extraction.ocr_reader import OcrEngine, RapidOcrReader
from extraction.pdf_reader import read_pdf
from extraction.spreadsheet_reader import read_spreadsheet
from extraction.text_reader import read_text
from extraction.types import DocumentDraft

EXTRACTOR_VERSION = "0.2.0"

_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_TEXT_EXTENSIONS = {".txt", ".md"}
_SPREADSHEET_EXTENSIONS = {".csv", ".xlsx"}


class FileExtractor:
    """按文件格式读取文本，并返回统一 FileExtractionResult。"""

    def __init__(
        self,
        config: ExtractionConfig | None = None,
        ocr_engine: OcrEngine | None = None,
    ) -> None:
        """创建读取器，并允许测试或部署环境注入 OCR 引擎。"""

        self.config = config or ExtractionConfig()
        self.ocr_reader = RapidOcrReader(self.config, ocr_engine)

    def extract(
        self,
        file_path: str | Path,
        file_purpose: FilePurpose,
        *,
        source_uri: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> FileExtractionResult:
        """读取一个案例或 Query 文件。"""

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")

        extension = path.suffix.lower()
        if extension not in _MEDIA_TYPES:
            raise UnsupportedFileTypeError(f"暂不支持文件格式：{extension or '无扩展名'}")

        draft = self._dispatch(path, extension)
        pages, full_text = assemble_pages(draft.pages)
        warnings = [*draft.warnings]
        for page in pages:
            warnings.extend(f"第 {page.page_number} 页：{warning}" for warning in page.warnings)

        empty_page_count = sum(not page.text.strip() for page in pages)
        if not full_text.strip():
            status = ExtractionStatus.FAILED
        elif empty_page_count:
            status = ExtractionStatus.PARTIAL
        else:
            status = ExtractionStatus.SUCCEEDED

        quality_values = [page.quality_score for page in pages]
        checksum = _sha256(path)
        output_metadata: dict[str, object] = dict(metadata or {})
        output_metadata.update(
            {
                "extension": extension,
                "empty_page_count": empty_page_count,
            }
        )

        return FileExtractionResult(
            document_id=checksum,
            file_purpose=file_purpose,
            source_filename=path.name,
            source_uri=source_uri or path.name,
            media_type=_MEDIA_TYPES[extension],
            checksum_sha256=checksum,
            file_size_bytes=path.stat().st_size,
            full_text=full_text,
            page_count=len(pages),
            ocr_page_count=sum(page.used_ocr for page in pages),
            pages=pages,
            tables=draft.tables,
            extraction_method=_document_method(pages, extension),
            extraction_status=status,
            quality_score=fmean(quality_values) if quality_values else None,
            warnings=warnings,
            extracted_at=datetime.now(UTC),
            extractor_version=EXTRACTOR_VERSION,
            metadata=output_metadata,
        )

    def _dispatch(self, path: Path, extension: str) -> DocumentDraft:
        """根据扩展名选择具体格式读取器。"""

        if extension == ".pdf":
            return read_pdf(path, self.config, self.ocr_reader)
        if extension in _IMAGE_EXTENSIONS:
            return read_image(path, self.ocr_reader)
        if extension in _TEXT_EXTENSIONS:
            return read_text(path)
        if extension == ".docx":
            return read_docx(path)
        if extension in _SPREADSHEET_EXTENSIONS:
            return read_spreadsheet(path)
        raise UnsupportedFileTypeError(f"暂不支持文件格式：{extension}")


def extract_file(
    file_path: str | Path,
    file_purpose: FilePurpose,
    *,
    source_uri: str | None = None,
    metadata: Mapping[str, object] | None = None,
    config: ExtractionConfig | None = None,
) -> FileExtractionResult:
    """使用默认读取器读取一个文件。"""

    return FileExtractor(config=config).extract(
        file_path,
        file_purpose,
        source_uri=source_uri,
        metadata=metadata,
    )


def _sha256(path: Path) -> str:
    """分块计算文件 SHA-256，避免大文件一次性载入内存。"""

    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_method(
    pages: Sequence[PageContent],
    extension: str,
) -> ExtractionMethod:
    """根据各页方法确定文件级读取方式。"""

    methods: set[ExtractionMethod] = {page.extraction_method for page in pages}
    if len(methods) > 1:
        return ExtractionMethod.HYBRID
    if methods:
        return next(iter(methods))
    if extension == ".docx":
        return ExtractionMethod.DOCX
    if extension in _SPREADSHEET_EXTENSIONS:
        return ExtractionMethod.SPREADSHEET
    if extension in _TEXT_EXTENSIONS:
        return ExtractionMethod.PLAIN_TEXT
    if extension in _IMAGE_EXTENSIONS:
        return ExtractionMethod.RAPIDOCR
    return ExtractionMethod.PDF_TEXT
