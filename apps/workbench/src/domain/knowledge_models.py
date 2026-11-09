"""案例知识库、案例管理与检索结果的共享字段。"""

from __future__ import annotations

from datetime import date

from pydantic import Field

from domain.common import ContractModel


class CaseSummary(ContractModel):
    """案例列表和检索结果共用的稳定摘要。"""

    case_id: str
    case_title: str | None = None
    party_name: str | None = None
    penalty_doc_no: str | None = None
    violation_behavior: str
    penalty_content: str | None = None
    regulator: str | None = None
    penalty_date: date | None = None
    publish_date: date | None = None
    institution_type: str | None = None
    source_url: str | None = None
    risk_tags: list[str] = Field(default_factory=list)
    risk_domains: list[str] = Field(default_factory=list)
    case_origin: str
    source_group: str = "original"
    retrieval_status: str
    evidence_status: str
    is_user_managed: bool = False
    document_id: str | None = None
    subject_case_count: int = Field(default=1, ge=1)
    document_kind: str = "penalty"
    decision_status: str = "not_individually_verified"
    insurance_relevance: bool | None = None
    review_status: str = "pending"
    retrieval_disposition: str | None = None
    scope_assessment: dict[str, object] = Field(default_factory=dict)


class EvidenceChunk(ContractModel):
    """案例详情中的可追溯知识库切片。"""

    chunk_id: str
    chunk_type: str
    chunk_text: str
    supports_risk_tags: list[str] = Field(default_factory=list)
    is_verbatim: bool = False
    source_span: dict[str, object] | None = None


class CaseDetail(CaseSummary):
    """案例详情；保留权威 JSONL 中对日常管理有用的字段。"""

    case_summary: str | None = None
    related_parties: list[dict[str, object]] = Field(default_factory=list)
    entity_relations: list[dict[str, object]] = Field(default_factory=list)
    legal_basis_refs: list[dict[str, object]] = Field(default_factory=list)
    source_file: str | None = None
    source_sha256: str | None = None
    raw_source_file: str | None = None
    original_pdf_available: bool = False
    party_type: str | None = None
    penalty_amounts: list[dict[str, object]] = Field(default_factory=list)
    field_sources: dict[str, object] = Field(default_factory=dict)
    field_quality: dict[str, str] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    data_version: str
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    related_cases: list[CaseSummary] = Field(default_factory=list)


class CaseListPage(ContractModel):
    """案例分页结果。"""

    items: list[CaseSummary]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_pages: int = Field(ge=0)


class CaseStatistics(ContractModel):
    """Web 首页和案例管理页使用的数据概览。"""

    total_cases: int = Field(ge=0)
    base_cases: int = Field(ge=0)
    user_cases: int = Field(ge=0)
    crawled_cases: int = Field(default=0, ge=0)
    active_cases: int = Field(ge=0)
    risk_tag_count: int = Field(ge=0)
    data_version: str | None = None
    latest_crawl_at: str | None = None
    latest_penalty_date: str | None = None
    subject_case_count: int = Field(default=0, ge=0)
    listing_unit: str = "case"


class MatchedCase(ContractModel):
    """一条 Query 命中的案例证据。"""

    case_id: str
    document_kind: str = "penalty"
    decision_status: str = "not_individually_verified"
    source_group: str = "original"
    human_approved: bool = False
    case_title: str | None = None
    penalty_doc_no: str | None = None
    violation_behavior: str
    penalty_content: str | None = None
    regulator: str | None = None
    party_name: str | None = None
    institution_type: str | None = None
    penalty_date: date | None = None
    source_url: str | None = None
    risk_tags: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=1)
    bm25_score: float = Field(default=0.0, ge=0, le=1)
    vector_score: float = Field(default=0.0, ge=0, le=1)
    rrf_score: float = Field(default=0.0, ge=0, le=1)
    rerank_score: float = Field(default=0.0, ge=0, le=1)
    dense_score: float = Field(default=0.0, ge=0, le=1)
    type_bridge_score: float = Field(default=0.0, ge=0, le=1)
    retrieval_channel: str = "bm25+tfidf"
    matched_risk_tags: list[str] = Field(default_factory=list)
    matched_text: str
    matched_source: str
    matched_unit_id: str
    match_reason: str = ""
    retrieval_queries: list[str] = Field(default_factory=list)
    file_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_sha256: str | None = None
    source_spans: list[dict[str, int]] = Field(default_factory=list)
    evidence_origin: str = "candidate_retrieval"
    literal_verified: bool = False
    knowledge_version: str | None = None


class CaseIngestionResult(ContractModel):
    """Web 案例录入结果。"""

    case: CaseDetail
    cases: list[CaseDetail] = Field(default_factory=list)
    extraction_method: str
    llm_used: bool
    warnings: list[str] = Field(default_factory=list)
