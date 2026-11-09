"""Query 人工复核留痕模型。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from domain.common import ContractModel
from domain.query_models import QueryScene


class QueryReviewStatus(StrEnum):
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class QueryReviewRequest(ContractModel):
    query_id: str = Field(min_length=1, max_length=100)
    query_checksum: str = Field(min_length=16, max_length=128)
    status: QueryReviewStatus
    reviewer: str = Field(default="现场审核员", min_length=1, max_length=100)
    comment: str = Field(default="", max_length=4000)
    original_risk_labels: list[str] = Field(default_factory=list, max_length=27)
    corrected_risk_labels: list[str] = Field(default_factory=list, max_length=27)
    corrected_scene: QueryScene | None = None
    analysis_schema_version: str = Field(default="4.0", max_length=20)


class QueryReviewRecord(QueryReviewRequest):
    review_id: str
    reviewed_at: datetime
