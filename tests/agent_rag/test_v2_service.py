import asyncio
import importlib
import json
import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from test_v2_snapshot import create_snapshot, fixture_assets
from test_v2_tools import Backend, payload

from agent_rag.task_service import TaskService


def make_service(tmp_path, mode="replay"):
    sources, units, dictionary, _, _ = fixture_assets(tmp_path)
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    snapshot.manifest["rerank_mode"] = mode
    snapshot.backend = Backend(snapshot)
    return TaskService(tmp_path / "tasks", snapshot)


@pytest.fixture(autouse=True)
def isolated_live_guards(monkeypatch):
    for key in (
        "FINRISK_AGENT_ALLOW_LIVE",
        "FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE",
        "FINRISK_AGENT_RERANK_CAP_CNY",
        "FINRISK_AGENT_RERANK_MAX_REQUESTS",
    ):
        monkeypatch.delenv(key, raising=False)


def enable_live(monkeypatch):
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE", "1")
    monkeypatch.setenv("FINRISK_AGENT_RERANK_CAP_CNY", "0.15")
    monkeypatch.setenv("FINRISK_AGENT_RERANK_MAX_REQUESTS", "240")


def test_v2_service_admission_binds_profile_and_read_only_restore(tmp_path):
    sources, units, dictionary, _, _ = fixture_assets(tmp_path)
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    snapshot.manifest["rerank_mode"] = "replay"
    snapshot.backend = Backend(snapshot)
    service = TaskService(tmp_path / "tasks", snapshot)
    assert service.config["retrieval_profile"] == "v2-hybrid-flash-v1"
    record, _ = service.admit(
        {"text": "赠送购物卡。", "experiment_profile_id": "mock-v2-flash-replay-v1"}, "v2-test"
    )
    bundle = service.bundle(record["task_id"])
    assert bundle["config"]["max_queries_per_search"] == 1
    assert bundle["config"]["max_search_calls"] == 2
    assert bundle["snapshot"]["snapshot_id"] == snapshot.manifest["snapshot_id"]
    result = service.cancel_retrieval(record["task_id"])
    assert result["cancelled"] is True
    assert result["external_retrieval_requests"] == 0


def test_v2_config_preserves_context_limits_and_replay_has_no_paid_control(tmp_path):
    service = make_service(tmp_path)
    record, _ = service.admit({"text": "赠送购物卡。"}, None)
    config = service.bundle(record["task_id"])["config"]
    assert config["profile_id"] == "mock-v2-flash-replay-v1"
    assert config["rerank_mode"] == "replay"
    assert config["max_search_query_attempts"] == config["max_search_calls"] == 2
    assert config["max_queries_per_search"] == 1
    assert config["max_tool_actions"] == 8 and config["max_model_requests"] == 6
    assert config["max_external_retrieval_requests"] == 0
    assert config["tool_request_timeout_seconds"] == 180
    assert config["task_deadline_seconds"] == 270
    assert config["model_request_timeout_seconds"] == 30
    assert config["service_max_input_tokens"] == 16000
    assert config["task_token_budget"] == 60000
    assert config["context_safety_margin_tokens"] == 1024
    assert service.tool(record["task_id"]).control is None
    assert not (service.root / record["task_id"] / "rerank-ledger.json").exists()


@pytest.mark.parametrize(
    "mode,profile",
    [
        ("replay", "mock-v2-flash-live-v1"),
        ("replay", "live-deepseek-flash-v2-v1"),
        ("live", "mock-v2-flash-replay-v1"),
        ("replay", "mock-bm25-baseline-v1"),
        ("live", "live-deepseek-flash-bm25-v1"),
    ],
)
def test_profile_cannot_silently_change_snapshot_or_score_mode(
    tmp_path, monkeypatch, mode, profile
):
    enable_live(monkeypatch)
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_LIVE", "1")
    service = make_service(tmp_path, mode)
    with pytest.raises(ValueError, match="PROFILE_NOT_READY|RERANK_MODE_MISMATCH"):
        service.admit({"text": "材料", "experiment_profile_id": profile}, None)
    assert not list(service.root.glob("*/admission.json"))


def test_v2_live_retrieval_has_separate_guard_and_agent_live_guard(tmp_path, monkeypatch):
    service = make_service(tmp_path, "live")
    with pytest.raises(ValueError, match="RETRIEVAL_LIVE_MODE_NOT_ENABLED"):
        service.admit({"text": "材料"}, None)
    enable_live(monkeypatch)
    with pytest.raises(ValueError, match="LIVE_MODE_NOT_ENABLED"):
        service.admit({"text": "材料", "experiment_profile_id": "live-deepseek-flash-v2-v1"}, None)
    record, _ = service.admit({"text": "材料", "experiment_profile_id": None}, None)
    assert record["execution_mode"] == "mock"
    assert service.bundle(record["task_id"])["config"]["rerank_mode"] == "live"


def test_v2_live_agent_requires_separate_model_cost_ledger(tmp_path, monkeypatch):
    service = make_service(tmp_path, "live")
    enable_live(monkeypatch)
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_LIVE", "1")
    monkeypatch.delenv("FINRISK_AGENT_TRIAL_LEDGER", raising=False)
    request = {"text": "材料", "experiment_profile_id": "live-deepseek-flash-v2-v1"}
    with pytest.raises(ValueError, match="MODEL_TRIAL_LEDGER_REQUIRED"):
        service.admit(request, None)
    monkeypatch.setenv("FINRISK_AGENT_TRIAL_LEDGER", str(tmp_path / "new-model-ledger.json"))
    record, _ = service.admit(request, None)
    config = service.bundle(record["task_id"])["config"]
    assert config["model_trial_cap_cny"] == 1
    assert config["task_total_cost_cap_cny"] == pytest.approx(1.15)


@pytest.mark.parametrize(
    "cap,requests",
    [
        (None, "240"),
        ("0.15", None),
        ("0", "240"),
        ("nan", "240"),
        ("inf", "240"),
        ("-1", "240"),
        ("0.15", "241"),
        ("0.15", "0"),
        ("0.15", "1.5"),
    ],
)
def test_live_requires_explicit_finite_budget_and_request_cap(tmp_path, monkeypatch, cap, requests):
    service = make_service(tmp_path, "live")
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE", "1")
    if cap is not None:
        monkeypatch.setenv("FINRISK_AGENT_RERANK_CAP_CNY", cap)
    if requests is not None:
        monkeypatch.setenv("FINRISK_AGENT_RERANK_MAX_REQUESTS", requests)
    with pytest.raises(ValueError, match="INVALID_RERANK_LIMITS|RERANK_LIMITS_REQUIRED"):
        service.admit({"text": "材料"}, None)
    assert not list(service.root.glob("*/rerank-ledger.json"))


def test_live_control_is_task_local_deadline_and_cancel_are_durable(tmp_path, monkeypatch):
    enable_live(monkeypatch)
    service = make_service(tmp_path, "live")
    before = time.time()
    first, _ = service.admit({"text": "材料"}, "first")
    task_id = first["task_id"]
    tool = service.tool(task_id)
    ledger_path = service.root / task_id / "rerank-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert 269 < ledger["deadline_epoch_seconds"] - before <= 270.1
    assert tool.control.ledger_path == ledger_path
    assert tool.audit_dir == service.root / task_id / "retrieval-audits"
    assert tool.control.check()
    # Physical transport accounting comes from the durable task ledger.
    request = {"model": "deepseek-flash", "max_tokens": 512}
    ordinal, _ = tool.control._reserve(request, "pair")
    tool.control._commit_dispatch(ordinal)
    tool.control._start_transport(ordinal)
    result = service.cancel_retrieval(task_id)
    assert result["cancelled"] and result["external_retrieval_requests"] == 1
    assert result["cap_cny"] == 0.15 and result["max_requests"] == 240
    assert not tool.control.check()
    assert service.cancel_retrieval(task_id)["external_retrieval_requests"] == 1
    service.active_task_id = None
    second, _ = service.admit({"text": "下一份"}, "second")
    second_control = service.tool(second["task_id"]).control
    assert second_control is not tool.control and second_control.check()
    assert second_control.meta()["external_retrieval_requests"] == 0


def test_v2_restart_read_and_cancel_do_not_reopen_paid_control_or_search(tmp_path, monkeypatch):
    enable_live(monkeypatch)
    service = make_service(tmp_path, "live")
    record, _ = service.admit({"text": "赠送购物卡。"}, None)
    directory = service.root / record["task_id"]
    before = (directory / "rerank-ledger.json").read_bytes()
    restarted = TaskService(service.root, service.snapshot)
    restored = restarted.tool(record["task_id"])
    assert restored.control is None
    assert restored.search_disabled is True
    assert restored.call("search_cases", payload())["status"] == "error"
    assert restored.snapshot.backend.calls == 0
    source_id = next(iter(service.snapshot.sources))
    assert (
        restored.call("read_source", {"source_id": source_id, "max_chars": 500})["status"] == "ok"
    )
    result = restarted.cancel_retrieval(record["task_id"])
    assert result["cancelled"] and result["external_retrieval_requests"] == 0
    assert (directory / "rerank-ledger.json").read_bytes() == before


def test_dispatch_uses_task_manifest_deadline_and_always_cancels(tmp_path, monkeypatch):
    enable_live(monkeypatch)
    service = make_service(tmp_path, "live")
    record, _ = service.admit({"text": "材料"}, None)
    service.config["task_deadline_seconds"] = 1
    observed = {}

    class Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise RuntimeError("offline failure")

    monkeypatch.setattr("agent_rag.task_service.httpx.AsyncClient", Client)
    asyncio.run(service._dispatch(record["task_id"], "http://127.0.0.1:8766", "token"))
    assert observed["timeout"] == 285
    assert service.get(record["task_id"])["task_status"] == "failed"
    assert not service.tool(record["task_id"]).control.check()
    assert service.active_task_id is None


def test_router_v2_loader_and_actual_status(tmp_path, monkeypatch):
    module = importlib.import_module("agent_rag.router")
    service = make_service(tmp_path)
    snapshot_path = tmp_path / "snapshot"
    loaded = []
    monkeypatch.setenv("FINRISK_AGENT_SNAPSHOT", str(snapshot_path))
    monkeypatch.setenv("FINRISK_AGENT_TASK_ROOT", str(tmp_path / "router-tasks"))
    monkeypatch.setattr(module, "_initialization", module.ServiceInitialization(module._build_service))
    monkeypatch.setattr(
        module, "load_v2_snapshot", lambda path: loaded.append(path) or service.snapshot
    )
    assert module.agent_status()["initialization_state"] == "not_started"
    with pytest.raises(HTTPException) as error:
        module.get_service()
    assert error.value.status_code == 503
    assert loaded == []
    module._initialization.request_warmup()
    result = module._initialization.wait_ready(2)
    assert module.get_service() is result
    assert loaded == [snapshot_path]
    status = module.agent_status()
    assert status["retrieval_profile"] == "v2-hybrid-flash-v1"
    assert status["score_mode"] == "replay"
    assert status["default_profile_id"] == "mock-v2-flash-replay-v1"


def test_router_cancel_is_authenticated_and_task_request_defaults_to_snapshot(
    tmp_path, monkeypatch
):
    module = importlib.import_module("agent_rag.router")
    service = make_service(tmp_path)
    monkeypatch.setenv("FINRISK_AGENT_SERVICE_TOKEN", "unit-secret")
    monkeypatch.setattr(service, "launch", lambda *args: None)
    app = FastAPI()
    app.include_router(module.router)
    app.dependency_overrides[module.get_service] = lambda: service
    with TestClient(app) as client:
        result = client.post("/api/agent-rag/tasks", json={"text": "材料"})
        assert result.status_code == 202, result.text
        body = {"task_id": result.json()["task_id"]}
        endpoint = "/internal/agent-rag/retrieval/cancel"
        assert client.post(endpoint, json=body).status_code == 401
        headers = {"Authorization": "Bearer unit-secret"}
        cancelled = client.post(endpoint, json=body, headers=headers)
        assert cancelled.status_code == 200 and cancelled.json()["cancelled"] is True
        assert client.post(endpoint, json={"task_id": "f" * 32}, headers=headers).status_code == 404
        assert (
            client.post(endpoint, json={**body, "cap_cny": 100}, headers=headers).status_code == 422
        )


def test_search_timeout_cancels_new_transport_and_disarms_timer(tmp_path, monkeypatch):
    enable_live(monkeypatch)
    service = make_service(tmp_path, "live")
    record, _ = service.admit({"text": "赠送购物卡。"}, None)
    module = importlib.import_module("agent_rag.router")
    timers = []

    class Timer:
        def __init__(self, seconds, callback):
            self.seconds, self.callback, self.disarmed = seconds, callback, False
            timers.append(self)

        def start(self):
            self.callback()

        def cancel(self):
            self.disarmed = True

    monkeypatch.setattr(module.threading, "Timer", Timer)
    request = module.ToolRequest(task_id=record["task_id"], params=payload(), timeout_ms=20)
    module.tool_result("search_cases", request, service)
    assert timers[0].seconds == 0.02 and timers[0].disarmed
    assert not service.tool(record["task_id"]).control.check()
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            module.ToolRequest(task_id=record["task_id"], params=payload(), timeout_ms=invalid)
