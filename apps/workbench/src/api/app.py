"""FinRisk FastAPI 应用和本地 Web 工作台入口。"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import copy
from datetime import date
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from application.analysis_service import QueryAnalysisService
from application.case_service import CaseManagementService
from application.collection_service import CollectionService
from application.evaluation_service import load_evaluation
from application.evidence_service import KnowledgeEvidenceService
from application.query_service import QueryProcessingService
from application.retrieval_service import CaseRetrievalService, CaseSearchFilters
from application.review_service import QueryReviewService
from domain.common import ContractModel
from domain.knowledge_models import (
    CaseDetail,
    CaseIngestionResult,
    CaseListPage,
    CaseStatistics,
    MatchedCase,
)
from domain.query_models import QueryScene
from domain.review_models import QueryReviewRecord, QueryReviewRequest
from domain.risk_models import QueryAnalysisResult
from extraction import ExtractionError
from llm import (
    LLMProviderError,
    PiAgentLLMClient,
    create_llm_client_from_env,
)
from risk.detection import RuleRiskDetector
from risk.pipeline import RiskAssessmentPipeline
from risk.taxonomy import load_taxonomy_descriptions
from settings import AppSettings
from storage.case_repository import CaseNotFoundError, CaseRepository, DuplicateCaseError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "src" / "web" / "static"
_ANALYSIS_OPTIONS: dict[tuple[str, str], QueryAnalysisService] = {}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.agent_start_error = None
    application.state.retrieval_start_error = None
    # Load the prebuilt local index before the first query, never encode the corpus here.
    try:
        await asyncio.to_thread(get_retrieval_service().warmup)
    except (ValueError, RuntimeError, OSError) as error:
        application.state.retrieval_start_error = str(error)
    client = get_llm_client()
    if client is not None:
        try:
            await client.start()
        except LLMProviderError as error:
            application.state.agent_start_error = str(error)
    collector = get_collection_service()
    collector.start()
    try:
        yield
    finally:
        await collector.stop()
    for variant in _ANALYSIS_OPTIONS.values():
        variant.risk_pipeline.retrieval_service.close()
    _ANALYSIS_OPTIONS.clear()
    if get_llm_client.cache_info().currsize:
        client = get_llm_client()
        if client is not None:
            await client.aclose()
    if get_retrieval_service.cache_info().currsize:
        get_retrieval_service().close()
    for factory in (
        get_analysis_service,
        get_query_service,
        get_case_service,
        get_collection_service,
        get_llm_client,
        get_retrieval_service,
        get_evidence_service,
        get_repository,
        get_rule_detector,
        get_review_service,
        get_settings,
    ):
        factory.cache_clear()


app = FastAPI(
    title="FinRisk 保险合规风险工作台",
    description="案例增量入库、案例管理、Query 风险判断与相似案例匹配",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TextQueryRequest(ContractModel):
    text: str = Field(min_length=1, max_length=1_000_000, description="待分析完整文本")
    scene: QueryScene | None = Field(default=None, description="用户指定场景；为空时自动识别")
    rerank_provider: Literal["none", "cloud", "deepseek"] | None = None
    retrieval_timing: Literal["post", "pre", "adaptive"] | None = None


class ClassifierStatusResponse(ContractModel):
    taxonomy_version: str
    classifier_version: str
    rule_version: str
    llm_configured: bool
    llm_model: str | None


class CaseStatusRequest(ContractModel):
    active: bool


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings.from_env()


@lru_cache(maxsize=1)
def get_repository() -> CaseRepository:
    return CaseRepository(get_settings())


@lru_cache(maxsize=1)
def get_llm_client() -> PiAgentLLMClient | None:
    return create_llm_client_from_env()


@lru_cache(maxsize=1)
def get_rule_detector() -> RuleRiskDetector:
    return RuleRiskDetector(get_settings().risk_keyword_path)


@lru_cache(maxsize=1)
def get_retrieval_service() -> CaseRetrievalService:
    return CaseRetrievalService(get_repository(), get_settings())


@lru_cache(maxsize=1)
def get_evidence_service() -> KnowledgeEvidenceService | None:
    root = get_settings().case_kb_root
    return (
        KnowledgeEvidenceService(root, get_settings().additional_evidence_databases)
        if (root / "evidence_registry.sqlite3").is_file()
        else None
    )


@lru_cache(maxsize=1)
def get_query_service() -> QueryProcessingService:
    return QueryProcessingService(llm_client=get_llm_client())


@lru_cache(maxsize=1)
def get_analysis_service() -> QueryAnalysisService:
    repository = get_repository()
    settings = get_settings()
    pipeline = RiskAssessmentPipeline(
        rule_detector=get_rule_detector(),
        retrieval_service=get_retrieval_service(),
        taxonomy=repository.risk_dictionary(),
        llm_client=get_llm_client(),
        taxonomy_descriptions=load_taxonomy_descriptions(settings.risk_dictionary_path),
        risk_display_threshold=settings.risk_display_threshold,
        retrieval_threshold=settings.retrieval_risk_threshold,
        retrieval_timing=settings.retrieval_timing,
        semantic_context_gate=settings.semantic_context_gate,
        retrieval_top_k=settings.retrieval_top_k,
        llm_concurrency=settings.risk_llm_concurrency,
        evidence_service=get_evidence_service(),
        agent_retrieval_enabled=settings.agent_retrieval_enabled,
        longtext_screening=settings.longtext_screening,
    )
    return QueryAnalysisService(query_service=get_query_service(), risk_pipeline=pipeline)


@lru_cache(maxsize=1)
def get_review_service() -> QueryReviewService:
    settings = get_settings()
    return QueryReviewService(settings.query_reviews_path, get_repository().risk_dictionary())


def analysis_with_options(
    service: QueryAnalysisService, rerank: str | None, timing: str | None
) -> QueryAnalysisService:
    if rerank is None and timing is None:
        return service
    settings = get_settings()
    key = (rerank or settings.rerank_provider, timing or settings.retrieval_timing)
    if key not in _ANALYSIS_OPTIONS:
        pipeline = copy(service.risk_pipeline)
        pipeline.retrieval_service = service.risk_pipeline.retrieval_service.with_reranker(key[0])
        pipeline.adaptive = key[1] == "adaptive"
        pipeline.retrieval_timing = "hybrid" if pipeline.adaptive else key[1]
        _ANALYSIS_OPTIONS[key] = QueryAnalysisService(
            query_service=service.query_service, risk_pipeline=pipeline
        )
    return _ANALYSIS_OPTIONS[key]


@app.get("/api/evaluation")
async def evaluation_summary() -> JSONResponse:
    return JSONResponse(
        load_evaluation(PROJECT_ROOT, get_settings().case_kb_root),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/evaluation/report")
async def evaluation_report() -> JSONResponse:
    payload = load_evaluation(PROJECT_ROOT, get_settings().case_kb_root)
    if not payload["available"]:
        raise HTTPException(status_code=404, detail="评测尚未完成")
    return JSONResponse(payload, headers={
        "Cache-Control": "no-store",
        "Content-Disposition": 'attachment; filename="FinRisk-evaluation-20261024.json"',
    })


@lru_cache(maxsize=1)
def get_case_service() -> CaseManagementService:
    return CaseManagementService(
        settings=get_settings(),
        repository=get_repository(),
        rule_detector=get_rule_detector(),
        llm_client=get_llm_client(),
    )


@lru_cache(maxsize=1)
def get_collection_service() -> CollectionService:
    return CollectionService(get_settings(), get_case_service(), get_retrieval_service())


@app.get("/api/collection/status")
async def collection_status() -> dict[str, object]:
    return await asyncio.to_thread(get_collection_service().status)


AnalysisDependency = Annotated[QueryAnalysisService, Depends(get_analysis_service)]
CaseDependency = Annotated[CaseManagementService, Depends(get_case_service)]
RetrievalDependency = Annotated[CaseRetrievalService, Depends(get_retrieval_service)]
ReviewDependency = Annotated[QueryReviewService, Depends(get_review_service)]


@app.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"}
    )


@app.get("/api/health")
async def health(case_service: CaseDependency) -> dict[str, object]:
    statistics = case_service.statistics()
    release_manifest = json.loads(get_settings().manifest_path.read_text("utf-8"))
    llm_client = get_llm_client()
    evidence_service = get_evidence_service()
    return {
        "status": "ok",
        "case_count": statistics.total_cases,
        "active_case_count": statistics.active_cases,
        "subject_case_count": statistics.subject_case_count,
        "case_listing_unit": statistics.listing_unit,
        "data_version": statistics.data_version,
        "latest_crawl_at": statistics.latest_crawl_at,
        "latest_penalty_date": statistics.latest_penalty_date,
        "llm_model": llm_client.model_name if llm_client else None,
        "agent_runtime": {
            "name": "pi-agent-core",
            "version": "1.2.0",
            "available": llm_client.runtime_available if llm_client else False,
            "configured": llm_client is not None,
            "worker_running": llm_client.worker_running if llm_client else False,
            "startup_warning": (
                getattr(app.state, "agent_start_error", None)
                if llm_client and not llm_client.worker_running
                else None
            ),
        },
        "context_version": "sentence-context-1.0",
        "retrieval": {
            "configured_embedding": get_settings().embedding_provider,
            "actual_vector_channel": get_retrieval_service().last_diagnostics.vector_channel,
            "timing": get_settings().retrieval_timing,
            "rerank_provider": get_settings().rerank_provider,
            "index_ready": get_retrieval_service().ready,
            "startup_warning": getattr(app.state, "retrieval_start_error", None),
        },
        "knowledge_versions": evidence_service.versions if evidence_service else {},
        "knowledge_release": {
            "upstream_revision": release_manifest.get("upstream_revision"),
            "raw_pdf_count": release_manifest.get("raw_pdf_count"),
            "partitions": release_manifest.get("partitions", {}),
            "article_count": len(evidence_service.articles),
            "mechanism_count": len(evidence_service.p0.cards) if evidence_service.p0 else 0,
            "full_projection": get_settings().full_corpus,
        }
        if evidence_service
        else {},
        "knowledge_review_status": "model_candidate_pending_human"
        if get_evidence_service()
        else "legacy",
    }


@app.get("/api/knowledge/evidence/{identifier:path}")
async def read_knowledge_evidence(
    identifier: str, layer: str = Query(default="hit", pattern="^(hit|supplement|extended)$")
) -> dict[str, object]:
    service = get_evidence_service()
    if service is None:
        raise HTTPException(status_code=404, detail="当前库未安装证据登记表")
    try:
        return await asyncio.to_thread(service.read, identifier, layer)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/query/classifier", response_model=ClassifierStatusResponse)
async def classifier_status(service: AnalysisDependency) -> ClassifierStatusResponse:
    current = service.classifier_status
    return ClassifierStatusResponse(
        taxonomy_version=current.taxonomy_version,
        classifier_version=current.classifier_version,
        rule_version=current.rule_version,
        llm_configured=current.llm_configured,
        llm_model=current.llm_model,
    )


@app.post("/api/query/reviews", response_model=QueryReviewRecord, status_code=201)
async def save_query_review(
    payload: QueryReviewRequest,
    service: ReviewDependency,
) -> QueryReviewRecord:
    try:
        return service.save(payload)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/query/analyze/text", response_model=QueryAnalysisResult)
async def analyze_text_query(
    payload: TextQueryRequest,
    service: AnalysisDependency,
) -> QueryAnalysisResult:
    try:
        selected = analysis_with_options(service, payload.rerank_provider, payload.retrieval_timing)
        return await selected.process_text(
            payload.text,
            metadata={"channel": "web"},
            scene=payload.scene,
        )
    except Exception as error:
        _raise_query_error(error)
        raise


@app.post("/api/query/analyze/file", response_model=QueryAnalysisResult)
async def analyze_file_query(
    file: Annotated[UploadFile, File(description="PDF、图片、TXT、Markdown 或 DOCX")],
    service: AnalysisDependency,
    scene: Annotated[QueryScene | None, Form()] = None,
    rerank_provider: Annotated[Literal["none", "cloud", "deepseek"] | None, Form()] = None,
    retrieval_timing: Annotated[Literal["post", "pre", "adaptive"] | None, Form()] = None,
) -> QueryAnalysisResult:
    safe_filename = _safe_filename(file.filename)
    try:
        with TemporaryDirectory(prefix="finrisk-query-") as temporary_directory:
            temporary_path = Path(temporary_directory) / safe_filename
            await _save_upload(file, temporary_path, get_settings().max_upload_size_bytes)
            selected = analysis_with_options(service, rerank_provider, retrieval_timing)
            return await selected.process_file(
                temporary_path,
                source_uri=f"upload://{safe_filename}",
                metadata={"upload_content_type": file.content_type or "application/octet-stream"},
                scene=scene,
            )
    except HTTPException:
        raise
    except Exception as error:
        _raise_query_error(error)
        raise
    finally:
        await file.close()


@app.get("/api/cases/statistics", response_model=CaseStatistics)
async def case_statistics(service: CaseDependency) -> CaseStatistics:
    return service.statistics()


@app.get("/api/cases", response_model=CaseListPage)
async def list_cases(
    service: CaseDependency,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    q: Annotated[str | None, Query(max_length=200)] = None,
    risk_tag: Annotated[str | None, Query(max_length=100)] = None,
    retrieval_status: Annotated[str | None, Query(pattern="^(active|inactive)$")] = None,
    origin: Annotated[str | None, Query(max_length=100)] = None,
    regulator: Annotated[str | None, Query(max_length=200)] = None,
    party_name: Annotated[str | None, Query(max_length=200)] = None,
    institution_type: Annotated[str | None, Query(max_length=100)] = None,
    penalty_date_from: date | None = None,
    penalty_date_to: date | None = None,
) -> CaseListPage:
    return service.list_cases(
        page=page,
        page_size=page_size,
        query=q,
        risk_tag=risk_tag,
        status=retrieval_status,
        origin=origin,
        regulator=regulator,
        party_name=party_name,
        institution_type=institution_type,
        penalty_date_from=penalty_date_from,
        penalty_date_to=penalty_date_to,
    )


@app.get("/api/cases/{case_id}", response_model=CaseDetail)
async def get_case(case_id: str, service: CaseDependency) -> CaseDetail:
    try:
        return service.get_case(case_id)
    except CaseNotFoundError as error:
        raise HTTPException(status_code=404, detail="案例不存在") from error


@app.get("/api/cases/{case_id}/source")
async def case_source(case_id: str, service: CaseDependency) -> FileResponse:
    try:
        item = service.get_case(case_id)
    except CaseNotFoundError as error:
        raise HTTPException(status_code=404, detail="案例不存在") from error
    settings = get_settings()
    roots = (
        (settings.project_root / "raw_pdfs").resolve(),
        settings.raw_case_upload_dir.resolve(),
        settings.collection_root.resolve(),
    )
    source = settings.project_root / (item.raw_source_file or "").replace("\\", "/")
    # Linux limits each filename to 255 bytes, including multibyte Chinese names.
    # Deployment may retain such originals under a deterministic short alias.
    alias = source.with_name(sha256(source.name.encode("utf-8")).hexdigest() + source.suffix)
    path = (alias if alias.is_file() else source).resolve()
    if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
        raise HTTPException(status_code=404, detail="暂无已绑定的来源原件")
    is_pdf = path.suffix.lower() == ".pdf"
    return FileResponse(
        path,
        media_type="application/pdf" if is_pdf else "application/octet-stream",
        filename=source.name,
        content_disposition_type="inline" if is_pdf else "attachment",
    )


@app.post("/api/cases/upload", response_model=CaseIngestionResult, status_code=201)
async def upload_case(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="待增量入库的案例 PDF 或文档")],
    service: CaseDependency,
) -> CaseIngestionResult:
    safe_filename = _safe_filename(file.filename)
    try:
        with TemporaryDirectory(prefix="finrisk-case-") as temporary_directory:
            temporary_path = Path(temporary_directory) / f"document{Path(safe_filename).suffix}"
            await _save_upload(file, temporary_path, get_settings().max_upload_size_bytes)
            result = await service.ingest_file(temporary_path, original_filename=safe_filename)
            background_tasks.add_task(get_collection_service().refresh_indexes)
            return result
    except HTTPException:
        raise
    except DuplicateCaseError as error:
        raise HTTPException(status_code=409, detail="该文件对应的案例已经入库") from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except (ExtractionError, OSError, UnicodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()


@app.patch("/api/cases/{case_id}/status", response_model=CaseDetail)
async def set_case_status(
    case_id: str,
    payload: CaseStatusRequest,
    service: CaseDependency,
) -> CaseDetail:
    try:
        return service.set_active(case_id, active=payload.active)
    except CaseNotFoundError as error:
        raise HTTPException(status_code=404, detail="案例不存在") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.delete("/api/cases/{case_id}", response_model=CaseDetail)
async def deactivate_case(case_id: str, service: CaseDependency) -> CaseDetail:
    """兼容管理 UI 的删除动作，实际执行可恢复停用。"""

    try:
        return service.set_active(case_id, active=False)
    except CaseNotFoundError as error:
        raise HTTPException(status_code=404, detail="案例不存在") from error


@app.get("/api/knowledge/search", response_model=list[MatchedCase])
async def search_cases(
    service: RetrievalDependency,
    q: Annotated[str, Query(min_length=1, max_length=5000)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    risk_tag: Annotated[str | None, Query(max_length=100)] = None,
    regulator: Annotated[str | None, Query(max_length=200)] = None,
    party_name: Annotated[str | None, Query(max_length=200)] = None,
    institution_type: Annotated[str | None, Query(max_length=100)] = None,
    penalty_date_from: date | None = None,
    penalty_date_to: date | None = None,
) -> list[MatchedCase]:
    filters = CaseSearchFilters(
        risk_tags=frozenset({risk_tag}) if risk_tag else frozenset(),
        regulator=regulator,
        party_name=party_name,
        institution_type=institution_type,
        penalty_date_from=penalty_date_from,
        penalty_date_to=penalty_date_to,
    )
    return await asyncio.to_thread(service.search, q, top_k=top_k, filters=filters)


def _raise_query_error(error: Exception) -> None:
    if isinstance(error, LLMProviderError):
        raise HTTPException(status_code=502, detail=str(error)) from error
    if isinstance(error, (ExtractionError, OSError, UnicodeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(error)) from error


def _safe_filename(filename: str | None) -> str:
    normalized = (filename or "").replace("\\", "/")
    safe_name = Path(normalized).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="上传文件缺少有效文件名")
    return safe_name


async def _save_upload(upload: UploadFile, destination: Path, maximum_size: int) -> int:
    total_size = 0
    with destination.open("wb") as output_file:
        while chunk := await upload.read(1024 * 1024):
            total_size += len(chunk)
            if total_size > maximum_size:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"文件不能超过 {maximum_size // 1024 // 1024} MB",
                )
            output_file.write(chunk)
    if total_size == 0:
        raise HTTPException(status_code=400, detail="上传文件不能为空")
    return total_size
