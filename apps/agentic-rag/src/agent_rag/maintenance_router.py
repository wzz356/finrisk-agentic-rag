"""Local-only maintenance admission and safe, public-scope management views."""

import re
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_rag.maintenance_service import MaintenanceService

Identifier = Annotated[
    str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
RunId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
_HIDDEN = re.compile(
    r"key|secret|credential|password|authorization|token|trusted_runs|approval_context|(^|_)path$|(^|_)root$",
    re.I,
)
_LOCAL_PATH = re.compile(
    r'[A-Za-z]:[\\/][^\s"<>]*|\\\\[^\s"<>]+|(?:^|\s)/(?:Users|home|tmp|var|etc)/[^\s"<>]*'
)


@lru_cache(maxsize=1)
def get_maintenance_service():
    from agent_rag.collection_router import collection_root

    return MaintenanceService(collection_root() / "maintenance")


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Event(Payload):
    event_type: Literal[
        "new_source",
        "source_revision",
        "extraction_correction",
        "duplicate_relation",
        "law_revision",
        "deactivation",
        "conflict_report",
        "periodic_consolidation",
    ]
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=3)
    target_record_ids: list[Identifier] = Field(default_factory=list, max_length=40)
    reason: str = Field(min_length=1, max_length=2000, strict=True)
    idempotency_key: Identifier | None = None


class ImportBatch(Payload):
    batch_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class Run(Payload):
    event_ids: list[Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]] = Field(
        min_length=1, max_length=20
    )
    execution_mode: Literal["fixture", "live"] = "fixture"


def _public(value):
    if isinstance(value, dict):
        return {
            key: _public(item)
            for key, item in value.items()
            if key == "csrf_token" or not _HIDDEN.search(key)
        }
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, str):
        return _LOCAL_PATH.sub("[本地路径已隐藏]", value)
    return value


def _error(error):
    code = getattr(error, "code", str(error))
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,100}", code):
        code = "MAINTENANCE_REQUEST_FAILED"
    status = (
        404
        if code.endswith("_NOT_FOUND")
        or code in {"UNKNOWN_BATCH", "UNKNOWN_DOCUMENT", "UNKNOWN_RUN"}
        else 422
    )
    if code in {"LIVE_ADAPTERS_NOT_READY", "FIXTURE_RUNNER_NOT_READY"}:
        status = 503
    elif any(part in code for part in ("CONFLICT", "BUSY", "LEASE_HELD")):
        status = 409
    return HTTPException(status, code)


def _public_event(service, event_id):
    try:
        event = service.store.get_event(event_id)
    except ValueError as error:
        raise _error(error) from None
    if event.get("scope_id") != "public":
        raise HTTPException(403, "PRIVATE_MAINTENANCE_NOT_ADMITTED")
    return event


def _public_run(service, run_id):
    try:
        run = service.get_run(run_id)
    except ValueError as error:
        raise _error(error) from None
    for event_id in run.get("event_ids", []):
        _public_event(service, event_id)
    return run


def create_router(local_access, write_access, get_collection_service, csrf_token):
    router = APIRouter(
        prefix="/api/agent-rag/knowledge/maintenance", dependencies=[Depends(local_access)]
    )

    @router.get("/status")
    def status(service=Depends(get_maintenance_service)):
        value = service.status()
        events = [event for event in value.get("events", []) if event.get("scope_id") == "public"]
        event_ids = {event["id"] for event in events}
        jobs = [job for job in value.get("jobs", []) if job.get("event_id") in event_ids]
        runs = [run for run in value.get("runs", []) if set(run.get("event_ids", [])) <= event_ids]
        result = {
            key: value[key]
            for key in ("live_ready", "execution_mode", "policy_id", "limits", "knowledge_commit")
            if key in value
        }
        result.update(
            csrf_token=csrf_token,
            events=_public(events),
            jobs=_public(jobs),
            runs=_public(runs),
            publication_ready=False,
            fixture_ready=False,
            live_reason="缺少经验证的第二个独立模型适配器；当前仅支持离线工程回放。",
            fixture_reason="管理页面未配置工程回放脚本；请通过离线测试入口回放。",
            publication_reason="生产发布与版本选择适配尚未配置。",
            stages={
                "candidate": len(events),
                "model_review": sum(job["status"] == "approved" for job in jobs),
                "engineering_replay": sum(job["status"] == "engineering_complete" for job in jobs),
                "build": 0,
            },
        )
        return JSONResponse(_public(result), headers={"Cache-Control": "no-store"})

    @router.get("/events")
    def events(
        offset: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),
        service=Depends(get_maintenance_service),
    ):
        values = [item for item in service.store.list_events() if item.get("scope_id") == "public"]
        return {
            "items": _public(values[offset : offset + limit]),
            "total": len(values),
            "offset": offset,
            "limit": limit,
        }

    @router.post("/events", dependencies=[Depends(write_access)], status_code=201)
    def create_event(payload: Event, service=Depends(get_maintenance_service)):
        try:
            return _public(service.create_event(**payload.model_dump()))
        except ValueError as error:
            raise _error(error) from None

    @router.post("/import-batch", dependencies=[Depends(write_access)])
    def import_batch(
        payload: ImportBatch,
        service=Depends(get_maintenance_service),
        collection=Depends(get_collection_service),
    ):
        try:
            # An unknown batch must not be reported as a successful empty import.
            if not any(
                batch["id"] == payload.batch_id for batch in collection.store.list_batches()
            ):
                raise ValueError("UNKNOWN_BATCH")
            if (collection.root / "build-config.json").is_file():
                config = collection._config()
                service.bind_snapshot(config["base_snapshot"], config["base_manifest_sha256"])
            return _public(service.import_inventory(collection.store, payload.batch_id))
        except ValueError as error:
            raise _error(error) from None

    @router.get("/changesets/{event_id}")
    def changeset(
        event_id: RunId,
        revision: int | None = Query(None, ge=1),
        service=Depends(get_maintenance_service),
    ):
        _public_event(service, event_id)
        try:
            value = service.store.get_changeset(event_id, revision)
            scope = value["payload"].get("scope", {})
            if scope.get("scope_id") != "public" or scope.get("classification") != "public":
                raise HTTPException(403, "PRIVATE_MAINTENANCE_NOT_ADMITTED")
            return _public(
                {
                    "event_id": event_id,
                    "revision": value["revision"],
                    "changeset": value["payload"],
                    "review_runs": service.store.list_review_runs(event_id, value["revision"]),
                    "publication_allowed": False,
                }
            )
        except ValueError as error:
            raise _error(error) from None

    @router.post("/runs", dependencies=[Depends(write_access)], status_code=202)
    def submit_run(payload: Run, service=Depends(get_maintenance_service)):
        if payload.execution_mode == "live" and not service.status().get("live_ready"):
            raise HTTPException(503, "LIVE_ADAPTERS_NOT_READY")
        for event_id in payload.event_ids:
            _public_event(service, event_id)
        try:
            result = service.submit_events(payload.event_ids, execution_mode=payload.execution_mode)
            return _public(result)
        except ValueError as error:
            raise _error(error) from None

    @router.get("/runs/{run_id}")
    def get_run(run_id: RunId, service=Depends(get_maintenance_service)):
        return _public(_public_run(service, run_id))

    @router.post("/runs/{run_id}/cancel", dependencies=[Depends(write_access)])
    def cancel_run(run_id: RunId, payload: Payload, service=Depends(get_maintenance_service)):
        _public_run(service, run_id)
        try:
            return _public(service.cancel_run(run_id))
        except ValueError as error:
            raise _error(error) from None

    @router.post("/runs/{run_id}/resume", dependencies=[Depends(write_access)])
    def resume_run(run_id: RunId, payload: Payload):
        raise HTTPException(409, "RESUME_NOT_CONFIGURED_BUDGET_PRESERVED")

    return router
