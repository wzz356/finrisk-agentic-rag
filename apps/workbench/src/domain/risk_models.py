"""Query 风险判断与匹配案例的共享输出契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from domain.agent_models import AgentExecutionTrace
from domain.common import ContractModel
from domain.knowledge_models import MatchedCase
from domain.query_models import QueryProcessingResult


class RiskLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    REVIEW = "review"
    LOW = "low"
    NONE = "none"


class RiskEvidence(ContractModel):
    """规则、检索或大模型提供的一条判断依据。"""

    source: str
    description: str
    sentence_id: str | None = None
    document_start_offset: int | None = Field(default=None, ge=0)
    document_end_offset: int | None = Field(default=None, ge=1)


class MechanismDecision(ContractModel):
    """Independent current-input decision; a missing fact cannot erase another risk."""

    mechanism_id: str = Field(pattern=r"^M(?:0[1-9]|1[0-2])$")
    decision_state: Literal["risk", "no_clear_risk", "insufficient_information"]
    risk_labels: list[str] = Field(default_factory=list, max_length=27)
    candidate_labels: list[str] = Field(default_factory=list, max_length=27)
    evidence: list[str] = Field(default_factory=list, max_length=5)
    missing_facts: list[str] = Field(default_factory=list, max_length=8)
    reasoning: str = ""
    condition_assessments: list[dict[str, object]] = Field(default_factory=list)
    decision_origin: Literal["model", "deterministic_guard"] = "model"


class SentenceRiskAssessment(ContractModel):
    """与 RiskQuerySentence 通过 sentence_id 一一对应的判断结果。"""

    query_id: str
    sentence_id: str
    text: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    risk_level_name: str
    action: str
    risk_labels: list[str] = Field(default_factory=list)
    reasoning: str
    method: str
    local_score: float = Field(default=0.0, ge=0, le=1)
    local_confidence: float = Field(default=0.0, ge=0, le=1)
    llm_reviewed: bool = False
    review_depth: Literal["rule", "screen", "detail"] = "rule"
    retrieval_performed: bool = False
    evidence: list[RiskEvidence] = Field(default_factory=list)
    matched_cases: list[MatchedCase] = Field(default_factory=list)
    context_review_required: bool = False
    review_status: Literal["reviewed", "local", "needs_review"] = "local"
    retrieval_status: Literal[
        "not_requested", "candidates", "no_sufficient_match", "unavailable"
    ] = "not_requested"
    retrieval_query: str | None = None
    decision_state: Literal["risk", "no_clear_risk", "insufficient_information"] = "no_clear_risk"
    candidate_labels: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    condition_assessments: list[dict[str, object]] = Field(default_factory=list)
    mechanism_assessments: list[MechanismDecision] = Field(default_factory=list)
    label_assessments: list[dict[str, object]] = Field(default_factory=list)
    checked_mechanism_ids: list[str] = Field(default_factory=list)
    remediation_suggestions: list[str] = Field(default_factory=list)


class RiskLabelFinding(ContractModel):
    """整篇材料中一个超过展示阈值的风险标签。"""

    label: str
    max_score: float = Field(ge=0, le=1)
    sentence_count: int = Field(ge=1)
    sentence_ids: list[str] = Field(min_length=1)


class QueryRiskSummary(ContractModel):
    """整份 Query 的风险概览。"""

    overall_score: float = Field(ge=0, le=1)
    overall_level: RiskLevel
    overall_level_name: str
    action: str
    risk_labels: list[str] = Field(default_factory=list)
    risk_findings: list[RiskLabelFinding] = Field(default_factory=list)
    top_matched_cases: list[MatchedCase] = Field(default_factory=list)
    risk_display_threshold: float = Field(default=0.2, ge=0, le=1)
    retrieval_threshold: float = Field(default=0.4, ge=0, le=1)
    max_sentence_score: float = Field(default=0.0, ge=0, le=1)
    sentence_count: int = Field(ge=1)
    risky_sentence_count: int = Field(ge=0)
    llm_used: bool
    llm_model: str | None = None
    llm_reviewed_sentence_count: int = Field(default=0, ge=0)
    local_direct_sentence_count: int = Field(default=0, ge=0)
    pending_review_sentence_count: int = Field(default=0, ge=0)
    retrieval_sentence_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    agent_runs: list[AgentExecutionTrace] = Field(default_factory=list)
    decision_state: Literal["risk", "no_clear_risk", "insufficient_information"] = "no_clear_risk"
    missing_facts: list[str] = Field(default_factory=list)
    evidence_tool_events: list[dict[str, object]] = Field(default_factory=list)
    mechanism_coverage: dict[str, object] = Field(default_factory=dict)
    retrieval_strategy: str = "post"


class QueryAnalysisResult(QueryProcessingResult):
    """分类、切句、风险判断、案例匹配的一体化 Web/API 输出。"""

    schema_version: str = "4.0"
    risk_summary: QueryRiskSummary
    risk_assessments: list[SentenceRiskAssessment] = Field(min_length=1)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_risk_assessments(self) -> QueryAnalysisResult:
        sentence_ids = [sentence.sentence_id for sentence in self.sentences]
        assessment_ids = [item.sentence_id for item in self.risk_assessments]
        if sentence_ids != assessment_ids:
            raise ValueError("risk_assessments 必须与 sentences 按 sentence_id 一一对应")
        mappings = {item.sentence_id: item for item in self.source_mappings}
        for sentence, assessment in zip(self.sentences, self.risk_assessments, strict=True):
            if assessment.query_id != self.query_id or assessment.text != sentence.text:
                raise ValueError("风险判断必须绑定当前 Query 的目标原句")
            for evidence in assessment.evidence:
                if evidence.sentence_id is None:
                    continue
                mapping = mappings.get(evidence.sentence_id)
                if mapping is None:
                    raise ValueError("判断依据必须引用当前 Query 的原句")
                start, end = evidence.document_start_offset, evidence.document_end_offset
                if start is None and end is None:
                    continue
                if (
                    start is None
                    or end is None
                    or not (
                        mapping.document_start_offset <= start < end <= mapping.document_end_offset
                    )
                ):
                    raise ValueError("判断依据的原文偏移必须落在引用句内")
        return self
