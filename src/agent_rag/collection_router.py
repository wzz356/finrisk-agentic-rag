"""Local management API, available independently of model/GPU readiness."""

import os
import secrets
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_rag.collection_service import CollectionService
from agent_rag.contracts import PROJECT_ROOT
from agent_rag.knowledge_registry import retrieval_snapshot
from agent_rag.maintenance_router import create_router as maintenance_router


def collection_root():
    root = Path(
        os.environ.get("FINRISK_AGENT_COLLECTION_ROOT", PROJECT_ROOT / "runtime/knowledge_updates")
    ).resolve()
    if not root.is_relative_to((PROJECT_ROOT / "runtime").resolve()):
        raise ValueError("COLLECTION_ROOT_MUST_BE_IN_PROJECT_RUNTIME")
    return root


@lru_cache(maxsize=1)
def get_collection_service():
    roots = [
        Path(p) for p in os.environ.get("FINRISK_AGENT_LEGACY_CRAWL", "").split(os.pathsep) if p
    ]
    return CollectionService(collection_root(), history_roots=roots)


_csrf = secrets.token_urlsafe(32)


def local_access(request: Request):
    if not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "管理接口仅允许本机访问。")
    if request.url.hostname not in {"127.0.0.1", "::1", "localhost", "testserver"}:
        raise HTTPException(403, "管理接口需要本机地址。")
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme != request.url.scheme or parsed.netloc != request.url.netloc:
            raise HTTPException(403, "请从本机同源管理页面操作。")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "管理接口拒绝跨站请求。")


def write_access(request: Request):
    local_access(request)
    if not secrets.compare_digest(request.headers.get("x-knowledge-csrf", ""), _csrf):
        raise HTTPException(403, "管理令牌失效，请刷新页面。")


router = APIRouter()
api = APIRouter(prefix="/api/agent-rag/knowledge", dependencies=[Depends(local_access)])


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Settings(Payload):
    crawl_enabled: bool = Field(strict=True)


class Crawl(Payload):
    since: str
    until: str
    max_pages: int = Field(default=2, ge=1, le=100, strict=True)


class Process(Payload):
    batch_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    cap_cny: float = Field(gt=0, le=100, allow_inf_nan=False, strict=True)
    max_documents: int = Field(ge=1, le=100, strict=True)
    max_requests: int = Field(ge=1, le=200, strict=True)


class Activate(Payload):
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Usage(Payload):
    enabled: bool = Field(strict=True)


def error_response(error):
    code = str(error)
    return HTTPException(409 if code == "KNOWLEDGE_JOB_BUSY" else 422, code)


@api.get("/status")
def status(service: CollectionService = Depends(get_collection_service)):
    result = service.status()
    # Reading status must not initialize a GPU/model just to show the manager.
    from agent_rag.router import _initialization, _RegistryTaskService

    actual = _initialization.peek_ready()
    if isinstance(actual, _RegistryTaskService):
        actual.request_refresh()
    root = getattr(getattr(actual, "snapshot", None), "root", None)
    target = retrieval_snapshot(service.registry) if result["configuration"]["ready"] else None
    result["selection"].update(
        actual_version=root.parent.name if root else None,
        target_version=target.parent.name if target else None,
        pending_switch=bool(target and (not root or target.resolve() != root.resolve())),
        restart_required=bool(
            actual
            and (
                not isinstance(actual, _RegistryTaskService)
                or actual.registry.resolve() != service.registry.resolve()
            )
        ),
    )
    return JSONResponse({"csrf_token": _csrf, **result}, headers={"Cache-Control": "no-store"})


@api.patch("/settings", dependencies=[Depends(write_access)])
def settings(payload: Settings, service: CollectionService = Depends(get_collection_service)):
    return service.set_crawl_enabled(payload.crawl_enabled)


def submit(service, kind, payload):
    try:
        job = service.submit(kind, payload.model_dump())
    except ValueError as error:
        raise error_response(error) from None
    return JSONResponse({"job_id": job.job_id, "status": "queued"}, status_code=202)


@api.post("/import-history", dependencies=[Depends(write_access)])
def history(payload: Payload, service: CollectionService = Depends(get_collection_service)):
    return submit(service, "import_history", payload)


@api.post("/crawl", dependencies=[Depends(write_access)])
def crawl(payload: Crawl, service: CollectionService = Depends(get_collection_service)):
    return submit(service, "crawl", payload)


@api.post("/process", dependencies=[Depends(write_access)])
def process(payload: Process, service: CollectionService = Depends(get_collection_service)):
    return submit(service, "process", payload)


@api.post("/activate", dependencies=[Depends(write_access)])
def activate(payload: Activate, service: CollectionService = Depends(get_collection_service)):
    return submit(service, "activate", payload)


@api.post("/usage", dependencies=[Depends(write_access)])
def usage(payload: Usage, service: CollectionService = Depends(get_collection_service)):
    return submit(service, "usage", payload)


@api.get("/documents")
def documents(
    batch_id: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    service: CollectionService = Depends(get_collection_service),
):
    return service.store.documents(batch_id or None, offset, limit)


@api.get("/documents/{digest}")
def document(
    digest: str,
    cursor: int = Query(default=0, ge=0),
    service: CollectionService = Depends(get_collection_service),
):
    try:
        value = service.store.read_document(digest)
    except ValueError as error:
        raise HTTPException(404 if str(error) == "UNKNOWN_DOCUMENT" else 422, str(error)) from None
    text = value.pop("text")
    end = min(cursor + 8000, len(text))
    return {
        **value,
        "text": text[cursor:end],
        "has_more": end < len(text),
        "next_cursor": end if end < len(text) else None,
        "total_chars": len(text),
    }


@router.get("/agent-rag/knowledge", include_in_schema=False, dependencies=[Depends(local_access)])
def page():
    return FileResponse(PROJECT_ROOT / "src/web/static/knowledge-updates.html")


@router.get("/agent-rag/knowledge-updates.js", include_in_schema=False)
def script():
    return FileResponse(
        PROJECT_ROOT / "src/web/static/knowledge-updates.js", media_type="text/javascript"
    )


def start_collection():
    get_collection_service().start()


def stop_collection():
    if get_collection_service.cache_info().currsize:
        get_collection_service().close()


router.include_router(api)
router.include_router(maintenance_router(local_access, write_access, get_collection_service, _csrf))
router.add_event_handler("startup", start_collection)
router.add_event_handler("shutdown", stop_collection)
