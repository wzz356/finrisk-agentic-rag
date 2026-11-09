"""领域模型共同使用的基础类型和枚举。"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """所有共享字段模型的基类，禁止接收未约定字段。"""

    model_config = ConfigDict(extra="forbid")


class FilePurpose(StrEnum):
    """文件进入系统后的业务用途。"""

    CASE = "case"
    QUERY = "query"


class ExtractionStatus(StrEnum):
    """文件读取结果状态。"""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class ExtractionMethod(StrEnum):
    """文件或页面实际使用的读取方式。"""

    PDF_TEXT = "pdf_text"
    RAPIDOCR = "rapidocr"
    HYBRID = "hybrid"
    PLAIN_TEXT = "plain_text"
    DOCX = "docx"
    SPREADSHEET = "spreadsheet"


class LabelSource(StrEnum):
    """案例风险标签的产生来源。"""

    CLASSIFIER = "classifier"
    LLM_REVIEWER = "llm_reviewer"
    RESOLVER = "resolver"


class LabelResolutionStatus(StrEnum):
    """独立分类模型和大模型复核结果的融合状态。"""

    AGREED = "agreed"
    RESOLVED = "resolved"
    HUMAN_REVIEW = "human_review"


class QueryInputType(StrEnum):
    """Query 的原始输入形式。"""

    TEXT = "text"
    FILE = "file"


class EvidenceSpan(ContractModel):
    """字段或标签对应的原文证据位置。"""

    text: str = Field(description="作为证据的原文片段")
    page_number: int | None = Field(default=None, ge=1, description="原文件页码，从 1 开始")
    start_offset: int | None = Field(default=None, ge=0, description="在完整文本中的起始位置")
    end_offset: int | None = Field(default=None, ge=0, description="在完整文本中的结束位置")
    field_name: str | None = Field(default=None, description="该证据支持的字段名称")
