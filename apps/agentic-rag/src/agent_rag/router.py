"""与旧审查API分开的入口，内部接口只对本机认证服务开放。"""

import json
import os
import secrets
import threading
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_rag.collection_router import collection_root
from agent_rag.collection_router import router as collection_router
from agent_rag.contracts import PROJECT_ROOT
from agent_rag.knowledge_registry import retrieval_snapshot
from agent_rag.knowledge_snapshot import KnowledgeSnapshot
from agent_rag.service_initialization import ServiceInitialization
from agent_rag.task_service import TaskService
from agent_rag.token_count import official_counter

router = APIRouter()
router.include_router(collection_router)


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=10000)
    user_context: dict[str, str | None] = Field(default_factory=dict)
    engine: Literal["controlled_agent", "fixed_rag"] = "controlled_agent"
    experiment_profile_id: str | None = None
    mock_scenario: Literal["generic", "gift-example"] = "generic"


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(pattern="^[a-f0-9]{32}$")
    params: dict[str, Any]
    query_limit: int = Field(default=6, ge=0, le=6, strict=True)
    timeout_ms: int | None = Field(default=None, gt=0, strict=True)


class CancelRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(pattern="^[a-f0-9]{32}$")


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(pattern="^[a-f0-9]{32}$")
    evidence_ids: list[str] = Field(max_length=80)


class TokenCountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(pattern="^[a-f0-9]{32}$")
    payload: dict[str, Any]


def load_v2_snapshot(path: Path) -> Any:
    from agent_rag.v2_runtime import load_v2_snapshot as load

    return load(path)


def _registry_path() -> Path | None:
    registry = os.environ.get("FINRISK_AGENT_KB_REGISTRY")
    if registry:
        return Path(registry)
    if os.environ.get("FINRISK_AGENT_SNAPSHOT"):
        return None
    managed = collection_root() / "registry"
    return managed if (managed / "usage.json").exists() else None


def _snapshot_path() -> Path:
    registry = _registry_path()
    if registry:
        return retrieval_snapshot(registry)
    return Path(os.environ.get(
        "FINRISK_AGENT_SNAPSHOT", PROJECT_ROOT / "runtime/agent_rag/snapshot-v1"))


def _close_snapshot(snapshot: Any) -> None:
    dense = getattr(getattr(snapshot, "backend", None), "dense", None)
    if dense is not None:
        dense.close()


class _RegistryTaskService:
    """Keep task ownership stable while preparing a replacement snapshot in the background."""

    def __init__(self, registry: Path, service: TaskService) -> None:
        self.registry = registry.resolve()
        self._current = service
        self._services = [service]
        self._owners: dict[str, TaskService] = {}
        self._lock = threading.RLock()
        self._refresh_thread: threading.Thread | None = None
        self._candidate: Any = None
        self._candidate_path: Path | None = None
        self._loading_path: Path | None = None
        self._failed_path: Path | None = None
        self._error: str | None = None
        self._closed = False

    @property
    def snapshot(self) -> Any:
        with self._lock:
            return self._current.snapshot

    @property
    def config(self) -> dict[str, Any]:
        with self._lock:
            return self._current.config

    def registry_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "registry_snapshot_id": self._current.snapshot.manifest["snapshot_id"],
                "registry_refresh_state": (
                    "loading" if self._loading_path else
                    "waiting_for_idle" if self._candidate is not None else
                    "failed" if self._error else "idle"),
                "registry_refresh_error": self._error,
            }

    def _install_candidate_locked(self) -> None:
        if self._candidate is None or self._current.active_task_id is not None:
            return
        candidate, path = self._candidate, self._candidate_path
        self._candidate, self._candidate_path = None, None
        try:
            if retrieval_snapshot(self.registry) != path:
                _close_snapshot(candidate)
                return
            # Construct only at an idle boundary: its restart scan must not interrupt a job.
            replacement = TaskService(self._current.root, candidate)
        except Exception:
            self._error, self._failed_path = "REGISTRY_REFRESH_FAILED", path
            _close_snapshot(candidate)
            return
        previous = self._current
        for task_id in previous.tools:
            self._owners.setdefault(task_id, previous)
            previous.tools[task_id].search_disabled = True
        self._current = replacement
        self._services.append(replacement)
        self._error, self._failed_path = None, None
        # Historical sources and reports need no dense worker. No running task uses it.
        _close_snapshot(previous.snapshot)

    def request_refresh(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._install_candidate_locked()
            if (self._current.active_task_id is not None or self._loading_path is not None
                    or self._candidate is not None):
                return
            try:
                path = retrieval_snapshot(self.registry)
            except (OSError, ValueError, KeyError):
                self._error = "REGISTRY_REFRESH_FAILED"
                return
            if path == self._current.snapshot.root or path == self._failed_path:
                return
            self._loading_path = path
            self._error = None
            self._refresh_thread = threading.Thread(
                target=self._load_candidate, args=(path,), daemon=True,
                name="agent-registry-refresh")
            self._refresh_thread.start()

    def _load_candidate(self, path: Path) -> None:
        try:
            candidate = load_v2_snapshot(path)
        except Exception:
            with self._lock:
                self._loading_path = None
                self._failed_path, self._error = path, "REGISTRY_REFRESH_FAILED"
            return
        with self._lock:
            self._loading_path = None
            if self._closed:
                _close_snapshot(candidate)
                return
            self._candidate, self._candidate_path = candidate, path
            self._install_candidate_locked()

    def admit(self, request: dict[str, Any], key: str | None) -> tuple[dict[str, Any], bool]:
        with self._lock:
            if self._closed:
                raise ValueError("PROFILE_NOT_READY")
            self._install_candidate_locked()
            record, reused = self._current.admit(request, key)
            self._owners[record["task_id"]] = self._current
            return record, reused

    def _owner(self, task_id: str) -> TaskService:
        with self._lock:
            if task_id in self._owners:
                return self._owners[task_id]
            if task_id not in self._current.records:
                raise KeyError("TASK_NOT_FOUND")
            manifest = json.loads(
                (self._current.root / task_id / "manifest.json").read_text(encoding="utf-8"))
            snapshot_id = manifest["snapshot_id"]
            for service in self._services:
                if service.snapshot.manifest["snapshot_id"] == snapshot_id:
                    self._owners[task_id] = service
                    return service
            # After restart, restore historical source reads without spawning GPU/API work.
            from copy import copy

            from agent_rag.v2_snapshot import V2KnowledgeSnapshot

            versions = self.registry / "versions"
            for directory in sorted(versions.iterdir()):
                if not directory.is_dir():
                    continue
                path = (directory / "snapshot").resolve()
                if not path.is_relative_to(versions.resolve()):
                    continue
                try:
                    saved = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if saved.get("snapshot_id") != snapshot_id:
                    continue
                snapshot = V2KnowledgeSnapshot(path)
                frozen = manifest.get("snapshot", {})
                if snapshot.assets != frozen.get("assets"):
                    raise ValueError("SOURCE_SNAPSHOT_MISMATCH")
                snapshot.manifest.update(
                    rerank_mode=frozen.get("rerank_mode"),
                    rerank_protocol=frozen.get("rerank_protocol"))
                historical = copy(self._current)
                historical.snapshot = snapshot
                historical.snapshot_manifest = snapshot.manifest
                historical.profile = manifest["retrieval_profile"]
                historical.rerank_mode = manifest["rerank_mode"]
                historical.config = manifest["config"]
                historical.lock = threading.Lock()
                historical.records = {
                    key: value for key, value in self._current.records.items()
                    if json.loads((self._current.root / key / "manifest.json").read_text(
                        encoding="utf-8"))["snapshot_id"] == snapshot_id}
                historical.tools, historical.controls = {}, {}
                historical.jobs, historical.active_task_id = set(), None
                self._services.append(historical)
                self._owners[task_id] = historical
                return historical
            raise ValueError("SOURCE_SNAPSHOT_MISMATCH")

    def get(self, task_id: str) -> dict[str, Any]:
        return self._owner(task_id).get(task_id)

    def bundle(self, task_id: str) -> dict[str, Any]:
        return self._owner(task_id).bundle(task_id)

    def tool(self, task_id: str) -> Any:
        return self._owner(task_id).tool(task_id)

    def verify(self, task_id: str, evidence_ids: list[str]) -> dict[str, Any]:
        return self._owner(task_id).verify(task_id, evidence_ids)

    def cancel_retrieval(self, task_id: str) -> dict[str, Any]:
        return self._owner(task_id).cancel_retrieval(task_id)

    def launch(self, task_id: str, node_url: str, token: str) -> None:
        self._owner(task_id).launch(task_id, node_url, token)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            snapshots = [service.snapshot for service in self._services]
            if self._candidate is not None:
                snapshots.append(self._candidate)
                self._candidate = None
        seen = set()
        for snapshot in snapshots:
            if id(snapshot) not in seen:
                _close_snapshot(snapshot)
                seen.add(id(snapshot))


def _close_service(service: Any) -> None:
    if isinstance(service, _RegistryTaskService):
        service.close()
    else:
        _close_snapshot(service.snapshot)


def _build_service(stage: Any) -> TaskService | _RegistryTaskService:
    stage("asset_validation")
    snapshot_path = _snapshot_path()
    if not (snapshot_path / "manifest.json").exists():
        raise ValueError("SNAPSHOT_NOT_READY")
    manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    stage("snapshot_load")
    snapshot = (load_v2_snapshot(snapshot_path)
                if manifest.get("schema_version") == "finrisk-v2-snapshot/1.0"
                else KnowledgeSnapshot(snapshot_path))
    try:
        stage("service_build")
        task_root = Path(os.environ.get(
            "FINRISK_AGENT_TASK_ROOT", PROJECT_ROOT / "runtime/agent_rag/tasks"))
        service = TaskService(task_root, snapshot)
        registry = _registry_path()
        return _RegistryTaskService(registry, service) if registry else service
    except BaseException:
        _close_snapshot(snapshot)
        raise


_initialization = ServiceInitialization(
    _build_service, closer=_close_service)


def _requires_warmup() -> bool:
    if os.environ.get("FINRISK_AGENT_WARMUP_V2") == "1":
        return True
    try:
        manifest = json.loads((_snapshot_path() / "manifest.json").read_text(encoding="utf-8"))
        return manifest.get("schema_version") == "finrisk-v2-snapshot/1.0"
    except (OSError, ValueError):
        return False


def get_service() -> TaskService:
    if not _requires_warmup():
        _initialization.request_warmup()
        return _initialization.wait_ready()
    service = _initialization.get_ready()
    if isinstance(service, _RegistryTaskService):
        service.request_refresh()
    return service


Service = Annotated[TaskService, Depends(get_service)]


@router.get("/api/agent-rag/status")
def agent_status() -> dict[str, Any]:
    service = _initialization.peek_ready()
    config = getattr(service, "config", {})
    snapshot = getattr(service, "snapshot", None)
    manifest = getattr(snapshot, "manifest", {})
    return {
        **_initialization.status(),
        **(service.registry_status() if isinstance(service, _RegistryTaskService) else {}),
        "default_execution_mode": config.get("execution_mode", "mock"),
        "live_enabled": os.environ.get("FINRISK_AGENT_ALLOW_LIVE") == "1",
        "retrieval_live_enabled": os.environ.get("FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE") == "1",
        "live_model": "deepseek-flash",
        "retrieval_profile": config.get("retrieval_profile", "bm25-source-v1"),
        "score_mode": config.get("rerank_mode", manifest.get("rerank_mode", "not_enabled")),
        "default_profile_id": config.get("profile_id", "mock-bm25-baseline-v1"),
    }


@router.get("/api/agent-rag/readiness")
def readiness() -> JSONResponse:
    state = _initialization.status()
    return JSONResponse(state, status_code=200 if state["initialization_state"] == "ready" else 503)


def close_service() -> None:
    _initialization.close()


router.add_event_handler("shutdown", close_service)


def internal_auth(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    if request.client and request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "内部接口仅允许本机调用。")
    token = os.environ.get("FINRISK_AGENT_SERVICE_TOKEN", "")
    if not token:
        raise HTTPException(503, "内部服务认证未配置。")
    if not secrets.compare_digest(authorization or "", f"Bearer {token}"):
        raise HTTPException(401, "内部服务认证失败。")


@router.post("/internal/agent-rag/warmup", status_code=202,
             dependencies=[Depends(internal_auth)])
def warmup() -> dict[str, Any]:
    return _initialization.request_warmup()


def map_error(error: ValueError) -> HTTPException:
    code = str(error).split(":", 1)[0]
    return HTTPException(
        {"IDEMPOTENCY_CONFLICT": 409, "CAPACITY_EXCEEDED": 429, "PROFILE_NOT_READY": 503}.get(
            code, 422
        ),
        detail=code,
    )


@router.get("/agent-rag", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "src/web/static/agent-rag.html")


@router.get("/agent-rag/progress-view.js", include_in_schema=False)
def progress_view_script() -> FileResponse:
    return FileResponse(
        PROJECT_ROOT / "src/web/static/agent-progress.js", media_type="text/javascript"
    )


@router.post("/api/agent-rag/tasks", status_code=202)
async def create_task(
    payload: TaskRequest, service: Service, idempotency_key: Annotated[str | None, Header()] = None
) -> dict[str, Any]:
    token = os.environ.get("FINRISK_AGENT_SERVICE_TOKEN", "")
    if not token:
        raise HTTPException(503, "内部服务认证未配置，请使用启动脚本。")
    try:
        record, reused = service.admit(payload.model_dump(), idempotency_key)
        if not reused:
            service.launch(
                record["task_id"],
                os.environ.get("FINRISK_AGENT_NODE_URL", "http://127.0.0.1:8766"),
                token,
            )
        return {
            "task_id": record["task_id"],
            "task_status": record["task_status"],
            "reused": reused,
            "execution_mode": record["execution_mode"],
            "poll_url": f"/api/agent-rag/tasks/{record['task_id']}",
        }
    except ValueError as error:
        raise map_error(error) from error


@router.get("/api/agent-rag/tasks/{task_id}")
def get_task(task_id: str, service: Service) -> dict[str, Any]:
    try:
        value = service.get(task_id)
        return {
            key: value[key]
            for key in ("task_id", "task_status", "report", "error", "message", "total_e2e_ms",
                        "progress", "progress_history")
            if key in value
        }
    except KeyError as error:
        raise HTTPException(404, "任务不存在。") from error


@router.get("/api/agent-rag/tasks/{task_id}/source/{source_id}")
def read_review_source(
    task_id: str, source_id: str, service: Service, cursor: int = 0
) -> dict[str, Any]:
    try:
        return service.tool(task_id).call(
            "read_source", {"source_id": source_id, "cursor": cursor, "max_chars": 3000}
        )
    except KeyError as error:
        raise HTTPException(404, "当前进程没有该任务的来源注册表。") from error
    except ValueError as error:
        raise HTTPException(409, "该任务的来源或输入版本不一致，不能标为已核验。") from error


@router.get("/internal/agent-rag/tasks/{task_id}/bundle", dependencies=[Depends(internal_auth)])
def get_bundle(task_id: str, service: Service) -> dict[str, Any]:
    try:
        return service.bundle(task_id)
    except KeyError as error:
        raise HTTPException(404, "任务不存在。") from error


def tool_result(name: str, payload: ToolRequest, service: TaskService) -> dict[str, Any]:
    try:
        tool = service.tool(payload.task_id)
        timer = None
        if name == "search_cases" and tool.control is not None:
            config = service.bundle(payload.task_id)["config"]
            timeout = config["tool_request_timeout_seconds"]
            if payload.timeout_ms is not None:
                timeout = min(timeout, payload.timeout_ms / 1000)
            timeout = min(timeout, tool.control.remaining_seconds())
            timer = threading.Timer(timeout, tool.control.cancel)
            timer.daemon = True
            timer.start()
        try:
            return tool.call(name, payload.params, payload.query_limit)
        finally:
            if timer is not None:
                timer.cancel()
    except KeyError as error:
        raise HTTPException(404, "任务工具未注册。") from error


@router.post("/internal/agent-rag/retrieval/cancel", dependencies=[Depends(internal_auth)])
def cancel_retrieval(payload: CancelRetrievalRequest, service: Service) -> dict[str, Any]:
    try:
        return service.cancel_retrieval(payload.task_id)
    except KeyError as error:
        raise HTTPException(404, "任务不存在。") from error


@router.post("/internal/agent-rag/search", dependencies=[Depends(internal_auth)])
def search(payload: ToolRequest, service: Service) -> dict[str, Any]:
    return tool_result("search_cases", payload, service)


@router.post("/internal/agent-rag/source/read", dependencies=[Depends(internal_auth)])
def read_source(payload: ToolRequest, service: Service) -> dict[str, Any]:
    return tool_result("read_source", payload, service)


@router.post("/internal/agent-rag/source/preview", dependencies=[Depends(internal_auth)])
def preview_source(payload: ToolRequest, service: Service) -> dict[str, Any]:
    try:
        return service.tool(payload.task_id).preview_read(payload.params)
    except KeyError as error:
        raise HTTPException(404, "任务工具未注册。") from error


@router.post("/internal/agent-rag/token-count", dependencies=[Depends(internal_auth)])
def token_count(payload: TokenCountRequest, service: Service) -> dict[str, Any]:
    try:
        service.tool(payload.task_id)  # 同一任务/冻结快照来源限制，计数不登记证据或动作。
    except KeyError as error:
        raise HTTPException(404, "任务工具未注册。") from error
    try:
        if len(str(payload.payload)) > 250000:
            raise ValueError("TOKEN_COUNTER_PAYLOAD_TOO_LARGE")
        return official_counter.count(payload.payload)
    except (ValueError, ImportError) as error:
        raise HTTPException(503, str(error).split(":", 1)[0]) from error


@router.post("/internal/agent-rag/risk-definitions", dependencies=[Depends(internal_auth)])
def definitions(payload: ToolRequest, service: Service) -> dict[str, Any]:
    return tool_result("get_risk_definition", payload, service)


@router.post("/internal/agent-rag/verify", dependencies=[Depends(internal_auth)])
def verify(payload: VerifyRequest, service: Service) -> dict[str, Any]:
    try:
        return service.verify(payload.task_id, payload.evidence_ids)
    except KeyError as error:
        raise HTTPException(404, "任务工具未注册。") from error
