from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_rag.knowledge_snapshot import KnowledgeSnapshot
from agent_rag.router import get_service, router
from agent_rag.task_service import TaskService


@pytest.fixture
def client(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "case.txt").write_text("购买保险给予合同外利益。", encoding="utf-8")
    dictionary = (
        Path(__file__).resolve().parents[2] / "contracts/agent_rag/risk_type_dictionary.csv"
    )
    snapshot = KnowledgeSnapshot.build(raw, dictionary, tmp_path / "snapshot")
    service = TaskService(tmp_path / "tasks", snapshot)
    monkeypatch.setenv("FINRISK_AGENT_SERVICE_TOKEN", "unit-secret")
    monkeypatch.setattr(service, "launch", lambda *args: None)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as value:
        yield value


def test_task_post_poll_idempotency_and_capacity(client):
    body = {"text": "购买这份保险就送购物卡。"}
    response = client.post("/api/agent-rag/tasks", json=body, headers={"Idempotency-Key": "unit"})
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    same = client.post("/api/agent-rag/tasks", json=body, headers={"Idempotency-Key": "unit"})
    assert same.json()["task_id"] == task_id
    assert client.get(f"/api/agent-rag/tasks/{task_id}").status_code == 200
    assert client.post("/api/agent-rag/tasks", json={"text": "另一份"}).status_code == 429
    assert (
        client.post(
            "/api/agent-rag/tasks", json={"text": "不同"}, headers={"Idempotency-Key": "unit"}
        ).status_code
        == 409
    )


def test_internal_tools_require_auth_and_accept_only_registered_input(client):
    task_id = client.post("/api/agent-rag/tasks", json={"text": "购买保险"}).json()["task_id"]
    request = {"task_id": task_id, "params": {"tags": ["赠送利益"]}, "query_limit": 6}
    assert client.post("/internal/agent-rag/risk-definitions", json=request).status_code == 401
    good = client.post(
        "/internal/agent-rag/risk-definitions",
        json=request,
        headers={"Authorization": "Bearer unit-secret"},
    )
    assert good.status_code == 200 and good.json()["status"] == "ok"
    request["params"] = {"source_id": "INPUT:other", "max_chars": 500}
    bad = client.post(
        "/internal/agent-rag/source/read",
        json=request,
        headers={"Authorization": "Bearer unit-secret"},
    )
    assert bad.json()["error"]["code"] == "SOURCE_NOT_ALLOWED"


def test_invalid_request_fields_and_unknown_task(client):
    assert (
        client.post(
            "/api/agent-rag/tasks", json={"text": "资料", "max_model_requests": 100}
        ).status_code
        == 422
    )
    assert client.get("/api/agent-rag/tasks/unknown").status_code == 404
    assert client.get("/agent-rag").status_code == 200


def test_internal_count_and_source_preview_are_authenticated_and_task_bound(client):
    task_id = client.post("/api/agent-rag/tasks", json={"text": "购买保险"}).json()["task_id"]
    headers = {"Authorization": "Bearer unit-secret"}
    payload = {"task_id": task_id, "payload": {"messages": [{"role": "system", "content": "资料"}]}}
    path = "/internal/agent-rag/token-count"
    assert client.post(path, json=payload).status_code == 401
    result = client.post(path, json=payload, headers=headers)
    assert result.status_code == 200
    assert result.json()["counter_kind"] == "deepseek_v41_official"
    assert result.json()["prompt_tokens"] > 0
    payload["task_id"] = "f" * 32
    assert client.post(path, json=payload, headers=headers).status_code == 404
    request = {"task_id": task_id, "params": {"source_id": f"INPUT:{task_id}", "max_chars": 500}}
    preview_path = "/internal/agent-rag/source/preview"
    assert client.post(preview_path, json=request).status_code == 401
    preview = client.post(preview_path, json=request, headers=headers)
    assert preview.status_code == 200
    assert preview.json()["data"]["evidence"]["quote"] == "购买保险"


def test_progress_view_script_is_served_by_lightweight_router(client):
    response = client.get("/agent-rag/progress-view.js")
    assert response.status_code == 200
    assert "FinRiskProgress" in response.text


def test_cold_service_initialization_is_singleton_across_threads(tmp_path, monkeypatch):
    import concurrent.futures
    import time

    import agent_rag.router as module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FINRISK_AGENT_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(module, "_initialization", module.ServiceInitialization(module._build_service))
    calls = []
    singleton = object()

    def load_snapshot(path):
        calls.append(path)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(module, "KnowledgeSnapshot", load_snapshot)
    monkeypatch.setattr(module, "TaskService", lambda *args: singleton)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: module.get_service(), range(2)))
    assert values == [singleton, singleton]
    assert len(calls) == 1
