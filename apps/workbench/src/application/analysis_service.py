"""Query 准备与风险判断的统一应用接口。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

from application.query_service import QueryProcessingService
from domain.query_models import QueryScene
from domain.risk_models import QueryAnalysisResult
from query.classification import ClassifierStatus
from risk.pipeline import RiskAssessmentPipeline


class QueryAnalysisService:
    def __init__(
        self,
        *,
        query_service: QueryProcessingService,
        risk_pipeline: RiskAssessmentPipeline,
    ) -> None:
        self.query_service = query_service
        self.risk_pipeline = risk_pipeline

    @property
    def classifier_status(self) -> ClassifierStatus:
        return self.query_service.classifier_status

    async def process_text(
        self,
        text: str,
        *,
        metadata: Mapping[str, object] | None = None,
        scene: QueryScene | None = None,
    ) -> QueryAnalysisResult:
        started = perf_counter()
        prepared = await self.query_service.process_text(text, metadata=metadata, scene=scene)
        preparation_ms = (perf_counter() - started) * 1000
        risk_started = perf_counter()
        result = await self.risk_pipeline.analyze(prepared)
        risk_ms = (perf_counter() - risk_started) * 1000
        return result.model_copy(
            update={
                "stage_timings_ms": {
                    "query_preparation": round(preparation_ms, 3),
                    "risk_pipeline": round(risk_ms, 3),
                    "total": round((perf_counter() - started) * 1000, 3),
                }
            }
        )

    async def process_file(
        self,
        file_path: str | Path,
        *,
        source_uri: str | None = None,
        metadata: Mapping[str, object] | None = None,
        scene: QueryScene | None = None,
    ) -> QueryAnalysisResult:
        started = perf_counter()
        prepared = await self.query_service.process_file(
            file_path,
            source_uri=source_uri,
            metadata=metadata,
            scene=scene,
        )
        preparation_ms = (perf_counter() - started) * 1000
        risk_started = perf_counter()
        result = await self.risk_pipeline.analyze(prepared)
        risk_ms = (perf_counter() - risk_started) * 1000
        return result.model_copy(
            update={
                "stage_timings_ms": {
                    "file_extraction_and_preparation": round(preparation_ms, 3),
                    "risk_pipeline": round(risk_ms, 3),
                    "total": round((perf_counter() - started) * 1000, 3),
                }
            }
        )
