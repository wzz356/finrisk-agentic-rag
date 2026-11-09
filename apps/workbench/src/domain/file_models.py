"""文件读取模块的统一输出字段。

案例文件和 Query 文件都使用 FileExtractionResult，利用 file_purpose 区分用途。
页面坐标统一归一化为 0 到 1，便于网页按任意显示尺寸还原风险证据位置。
"""

from datetime import datetime

from pydantic import Field

from domain.common import (
    ContractModel,
    ExtractionMethod,
    ExtractionStatus,
    FilePurpose,
)


class NormalizedPoint(ContractModel):
    """页面内的归一化坐标点。"""

    x: float = Field(ge=0, le=1, description="相对于页面宽度的横坐标")
    y: float = Field(ge=0, le=1, description="相对于页面高度的纵坐标")


class TextBlock(ContractModel):
    """可独立定位的一行或一个段落文本。"""

    block_id: str = Field(description="页面内稳定文本块标识")
    page_number: int = Field(ge=1, description="所在页码，从 1 开始")
    text: str = Field(description="文本块内容")
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="OCR 识别置信度，原生文本层为空",
    )
    polygon: list[NormalizedPoint] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="文本块四边形的归一化页面坐标",
    )
    reading_order: int = Field(ge=0, description="页面内阅读顺序")
    page_start_offset: int = Field(ge=0, description="在当前页文本中的起始位置")
    page_end_offset: int = Field(ge=0, description="在当前页文本中的结束位置")
    document_start_offset: int = Field(ge=0, description="在完整文档文本中的起始位置")
    document_end_offset: int = Field(ge=0, description="在完整文档文本中的结束位置")
    extraction_method: ExtractionMethod = Field(description="该文本块的读取方式")


class WordContent(ContractModel):
    """OCR 或原生文本层识别出的可定位字词。"""

    word_id: str = Field(description="页面内稳定字词标识")
    page_number: int = Field(ge=1, description="所在页码，从 1 开始")
    text: str = Field(description="字或词内容")
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="OCR 字词置信度，原生文本层为空",
    )
    polygon: list[NormalizedPoint] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="字词四边形的归一化页面坐标",
    )
    reading_order: int = Field(ge=0, description="页面内阅读顺序")
    page_start_offset: int | None = Field(
        default=None,
        ge=0,
        description="在当前页文本中的起始位置，无法对齐时为空",
    )
    page_end_offset: int | None = Field(
        default=None,
        ge=0,
        description="在当前页文本中的结束位置，无法对齐时为空",
    )
    document_start_offset: int | None = Field(
        default=None,
        ge=0,
        description="在完整文档文本中的起始位置，无法对齐时为空",
    )
    document_end_offset: int | None = Field(
        default=None,
        ge=0,
        description="在完整文档文本中的结束位置，无法对齐时为空",
    )
    extraction_method: ExtractionMethod = Field(description="该字词的读取方式")


class PageContent(ContractModel):
    """文件单页的读取结果。"""

    page_number: int = Field(ge=1, description="页码，从 1 开始")
    width: float | None = Field(default=None, gt=0, description="原始页面宽度")
    height: float | None = Field(default=None, gt=0, description="原始页面高度")
    text: str = Field(description="该页按阅读顺序合并的文本")
    blocks: list[TextBlock] = Field(default_factory=list, description="行或段落级文本块")
    words: list[WordContent] = Field(default_factory=list, description="字词级定位结果")
    used_ocr: bool = Field(default=False, description="该页是否使用了 OCR")
    average_confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="该页 OCR 平均置信度",
    )
    quality_score: float = Field(ge=0, le=1, description="该页综合读取质量")
    extraction_method: ExtractionMethod = Field(description="该页最终采用的读取方式")
    warnings: list[str] = Field(default_factory=list, description="该页读取警告")


class ExtractedTable(ContractModel):
    """文件中提取出的一个表格。"""

    page_number: int | None = Field(default=None, ge=1, description="表格所在页码")
    title: str | None = Field(default=None, description="表格标题")
    rows: list[list[str]] = Field(default_factory=list, description="按行保存的单元格文本")
    page_numbers: list[int] = Field(default_factory=list, description="跨页表格涉及的全部页码")
    confidence: float | None = Field(default=None, ge=0, le=1, description="表格抽取置信度")
    is_cross_page: bool = Field(default=False, description="是否由连续页面同表合并")


class FileExtractionResult(ContractModel):
    """文件读取完成后交给后续模块的统一结果。"""

    schema_version: str = Field(default="1.2", description="当前字段结构版本")
    document_id: str = Field(description="文件稳定标识，当前使用文件 SHA-256")
    file_purpose: FilePurpose = Field(description="文件用途：案例或 Query")
    source_filename: str = Field(description="上传或批量导入时的原始文件名")
    source_uri: str = Field(description="原始文件的项目内相对位置或逻辑 URI")
    media_type: str = Field(description="文件 MIME 类型，例如 application/pdf")
    checksum_sha256: str = Field(description="原始文件的 SHA-256")
    file_size_bytes: int = Field(ge=0, description="原始文件大小")
    full_text: str = Field(description="按页码和阅读顺序合并后的完整文本")
    page_count: int = Field(ge=0, description="文件总页数")
    ocr_page_count: int = Field(ge=0, description="实际使用 OCR 的页数")
    pages: list[PageContent] = Field(default_factory=list, description="分页文本及位置")
    tables: list[ExtractedTable] = Field(default_factory=list, description="提取出的表格")
    extraction_method: ExtractionMethod = Field(description="整个文件的主要读取方式")
    extraction_status: ExtractionStatus = Field(description="读取成功、部分成功或失败")
    quality_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="各页读取质量平均值，无法计算时为空",
    )
    warnings: list[str] = Field(default_factory=list, description="缺页、乱码等读取警告")
    extracted_at: datetime = Field(description="完成读取的时间，必须包含时区")
    extractor_version: str = Field(description="文件读取模块版本")
    metadata: dict[str, object] = Field(default_factory=dict, description="可选来源信息")
