"""Query 六类场景、逐句 Risk 输入与原文定位字段。"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from domain.common import ContractModel, QueryInputType
from domain.file_models import NormalizedPoint


class QueryScene(StrEnum):
    """Query 文件级使用场景。"""

    ST1_POSTER_SMS_SOCIAL = "ST1_poster_sms_social"
    ST2_LIVE_SHORT_VIDEO = "ST2_live_short_video"
    ST3_PRODUCT_INTRO_DETAIL = "ST3_product_intro_detail"
    ST4_MEETING_MATERIAL = "ST4_meeting_material"
    ST5_TRAINING_TELEMARKETING_CALLBACK = "ST5_training_telemarketing_callback"
    ST6_PENALTY_BACKEND_FACT = "ST6_penalty_backend_fact"


class QueryClassificationMethod(StrEnum):
    """文件级场景分类的产生方式。"""

    RULE = "rule"
    RULE_FALLBACK = "rule_fallback"
    LLM = "llm"
    USER = "user"


class SceneRuleCandidate(ContractModel):
    """规则阶段的一个候选场景及可解释证据。"""

    scene: QueryScene
    score: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class QuerySceneClassification(ContractModel):
    """整份 Query 的统一场景分类结果。"""

    scene: QueryScene
    confidence: float = Field(ge=0, le=1)
    method: QueryClassificationMethod
    evidence: list[str] = Field(default_factory=list)
    rule_candidates: list[SceneRuleCandidate] = Field(default_factory=list)
    classifier_version: str
    taxonomy_version: str
    llm_model: str | None = None


class RiskQuerySentence(ContractModel):
    """交给 Risk 模块的严格四字段单句输入。"""

    query_id: str
    sentence_id: str
    scene: QueryScene
    text: str = Field(min_length=1)


class ContextSourceSentence(ContractModel):
    """上下文中的逐字原文与全文位置；角色来自显式标记或同一段落继承。"""

    sentence_id: str
    text: str = Field(min_length=1)
    document_start_offset: int = Field(ge=0)
    document_end_offset: int = Field(ge=1)
    speaker_role: str = "unknown"
    speaker_name: str | None = None
    is_target: bool = False


class RiskSentenceContext(ContractModel):
    """不改变四字段单句接口的上下文侧表，供 Risk、检索和页面共用。"""

    query_id: str
    sentence_id: str
    query_text: str = Field(min_length=1)
    scene: QueryScene
    scope_id: str
    context_before: str = ""
    context_after: str = ""
    speaker_role: str = "unknown"
    speaker_name: str | None = None
    stance: Literal[
        "statement", "question", "denial", "prohibition", "negative_example", "unknown"
    ] = "unknown"
    requires_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    context_truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    source_sentences: list[ContextSourceSentence] = Field(min_length=1)


class QuerySourceRegion(ContractModel):
    """句子与 PDF/图片页面文本块的相交区域。"""

    page_number: int = Field(ge=1)
    block_id: str
    word_ids: list[str] = Field(default_factory=list)
    text: str
    document_start_offset: int = Field(ge=0)
    document_end_offset: int = Field(ge=1)
    polygon: list[NormalizedPoint] | None = Field(default=None, min_length=4, max_length=4)
    confidence: float | None = Field(default=None, ge=0, le=1)
    extraction_method: str | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> "QuerySourceRegion":
        if self.document_end_offset <= self.document_start_offset:
            raise ValueError("document_end_offset 必须大于 document_start_offset")
        return self


class SentenceSourceMapping(ContractModel):
    """通过 sentence_id 回到原文；偏移为 Unicode 码点的左闭右开区间。"""

    query_id: str
    sentence_id: str
    input_type: QueryInputType
    source_document_id: str | None = None
    source_filename: str | None = None
    document_start_offset: int = Field(ge=0)
    document_end_offset: int = Field(ge=1)
    context_before: str = ""
    context_after: str = ""
    regions: list[QuerySourceRegion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_offsets(self) -> "SentenceSourceMapping":
        if self.document_end_offset <= self.document_start_offset:
            raise ValueError("document_end_offset 必须大于 document_start_offset")
        return self


class QueryExtractionSummary(ContractModel):
    """Query 内容抽取层的可审计摘要。"""

    media_type: str
    checksum_sha256: str
    file_size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=0)
    ocr_page_count: int = Field(ge=0)
    extraction_method: str
    quality_score: float | None = Field(default=None, ge=0, le=1)
    block_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class QueryProcessingResult(ContractModel):
    """Query 分类、切句及原文映射的完整输出。"""

    schema_version: str = "2.0"
    query_id: str
    input_type: QueryInputType
    source_text: str = Field(description="文件抽取后的全文或文字输入，用于偏移映射")
    source_offset_unit: Literal["unicode_code_point"] = "unicode_code_point"
    segmentation_version: str = "source-spans-2.0"
    source_document_id: str | None = None
    source_filename: str | None = None
    extraction: QueryExtractionSummary
    classification: QuerySceneClassification
    sentences: list[RiskQuerySentence] = Field(min_length=1)
    source_mappings: list[SentenceSourceMapping] = Field(min_length=1)
    sentence_contexts: list[RiskSentenceContext] = Field(default_factory=list)
    created_at: datetime
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sentence_mappings(self) -> "QueryProcessingResult":
        sentence_ids = [sentence.sentence_id for sentence in self.sentences]
        mapping_ids = [mapping.sentence_id for mapping in self.source_mappings]
        if sentence_ids != mapping_ids:
            raise ValueError("sentences 与 source_mappings 必须按 sentence_id 一一对应")
        if len(set(sentence_ids)) != len(sentence_ids):
            raise ValueError("sentence_id 必须唯一")
        if any(sentence.query_id != self.query_id for sentence in self.sentences):
            raise ValueError("句子的 query_id 必须与结果 query_id 一致")
        previous_end = 0
        for sentence, mapping in zip(self.sentences, self.source_mappings, strict=True):
            if mapping.query_id != self.query_id:
                raise ValueError("原文映射的 query_id 必须与结果一致")
            start, end = mapping.document_start_offset, mapping.document_end_offset
            if start < previous_end or end > len(self.source_text):
                raise ValueError("原文偏移必须有序、不重叠且位于全文范围内")
            if self.source_text[start:end] != sentence.text:
                raise ValueError("目标句与原文偏移不一致")
            if self.source_text[previous_end:start].strip():
                raise ValueError("原文映射不得遗漏非空白文字")
            previous_end = end
        if self.source_text[previous_end:].strip():
            raise ValueError("原文映射不得遗漏尾部文字")
        mapping_by_id = {mapping.sentence_id: mapping for mapping in self.source_mappings}
        if self.sentence_contexts:
            if [value.sentence_id for value in self.sentence_contexts] != sentence_ids:
                raise ValueError("sentence_contexts 必须与 sentences 一一对应")
            for sentence, context in zip(self.sentences, self.sentence_contexts, strict=True):
                if context.query_id != self.query_id or context.query_text != sentence.text:
                    raise ValueError("上下文必须属于当前 Query 并保留目标原句")
                targets = [source for source in context.source_sentences if source.is_target]
                if len(targets) != 1 or targets[0].sentence_id != sentence.sentence_id:
                    raise ValueError("每个上下文必须包含且仅包含一个目标句")
                for source in context.source_sentences:
                    if source.sentence_id not in mapping_by_id:
                        raise ValueError("上下文引文必须来自当前 Query 的句子")
                    mapping = mapping_by_id[source.sentence_id]
                    if (source.document_start_offset, source.document_end_offset) != (
                        mapping.document_start_offset,
                        mapping.document_end_offset,
                    ):
                        raise ValueError("上下文引文与原文偏移不一致")
                    actual = self.source_text[
                        source.document_start_offset : source.document_end_offset
                    ]
                    if source.text != actual:
                        raise ValueError("上下文引文与原文偏移不一致")
        return self
