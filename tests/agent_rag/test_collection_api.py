import importlib
import importlib.util
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient


def setup(tmp_path):
    assert importlib.util.find_spec("agent_rag.collection_router"), "管理API未实现"
    m = importlib.import_module("agent_rag.collection_router")
    from agent_rag.collection_service import CollectionService

    service = CollectionService(tmp_path / "managed")
    app = FastAPI()
    app.include_router(m.router)
    app.dependency_overrides[m.get_collection_service] = lambda: service
    return TestClient(app), service


def test_management_works_without_rag_snapshot_and_writes_require_csrf_and_origin(tmp_path):
    client, s = setup(tmp_path)
    status = client.get("/api/agent-rag/knowledge/status")
    assert status.status_code == 200
    token = status.json()["csrf_token"]
    assert (
        client.patch("/api/agent-rag/knowledge/settings", json={"crawl_enabled": True}).status_code
        == 403
    )
    headers = {"X-Knowledge-CSRF": token, "Origin": "https://evil.example"}
    assert (
        client.patch(
            "/api/agent-rag/knowledge/settings", json={"crawl_enabled": True}, headers=headers
        ).status_code
        == 403
    )
    headers["Origin"] = "http://testserver"
    good = client.patch(
        "/api/agent-rag/knowledge/settings", json={"crawl_enabled": True}, headers=headers
    )
    assert good.status_code == 200
    assert s.status()["settings"]["crawl_enabled"] is True
    assert s.status()["settings"]["usage_enabled"] is False
    assert (
        client.patch(
            "/api/agent-rag/knowledge/settings", json={"usage_enabled": True}, headers=headers
        ).status_code
        == 422
    )
    assert client.get("/agent-rag/knowledge").status_code == 200
    assert client.get("/agent-rag/knowledge-updates.js").status_code == 200


def test_localhost_hostname_and_remote_client_restrictions(tmp_path):
    client, s = setup(tmp_path)
    assert (
        client.get(
            "/api/agent-rag/knowledge/status", headers={"Host": "attacker.example"}
        ).status_code
        == 403
    )
    client._transport.client = ("192.0.2.4", 1234)
    assert client.get("/api/agent-rag/knowledge/status").status_code == 403


def test_raw_read_is_paginated_and_restricted_to_inventory(tmp_path):
    client, s = setup(tmp_path)
    batch = s.store.create_batch("crawl")
    item = s.store.ingest(
        batch["id"], ("保险🙂" * 4000).encode(), {"source_url": "https://www.nfra.gov.cn/a"}
    )
    result = client.get("/api/agent-rag/knowledge/documents/" + item["digest"]).json()
    assert result["has_more"] and result["next_cursor"] > 0
    tail = client.get(
        "/api/agent-rag/knowledge/documents/" + item["digest"],
        params={"cursor": result["next_cursor"]},
    ).json()
    assert result["text"] + tail["text"] == "保险🙂" * 4000
    assert client.get("/api/agent-rag/knowledge/documents/" + ("a" * 64)).status_code == 404
    assert (
        client.get("/api/agent-rag/knowledge/documents", params={"limit": 10000}).status_code == 422
    )


def test_background_acceptance_is_immediate_and_duplicate_job_rejected(tmp_path):
    client, s = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def crawl(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return {"complete": False, "documents": 0, "errors": [], "next_pages": {"4113": 2}}

    s.crawler = crawl
    headers = {
        "X-Knowledge-CSRF": client.get("/api/agent-rag/knowledge/status").json()["csrf_token"]
    }
    payload = {"since": "2026-08-28", "until": "2026-10-18", "max_pages": 1}
    try:
        result = client.post("/api/agent-rag/knowledge/crawl", json=payload, headers=headers)
        assert result.status_code == 202 and result.json()["job_id"]
        assert entered.wait(1)
        assert (
            client.post("/api/agent-rag/knowledge/crawl", json=payload, headers=headers).status_code
            == 409
        )
        assert (
            client.post(
                "/api/agent-rag/knowledge/process",
                json={"batch_id": "x", "cap_cny": 1, "max_documents": 1, "max_requests": 1},
                headers=headers,
            ).status_code
            == 422
        )
    finally:
        release.set()
        for thread in s._threads:
            thread.join(5)
    assert client.get("/api/agent-rag/knowledge/status").json()["jobs"][0]["status"] == "incomplete"


def test_different_registry_requires_restart_instead_of_waiting_forever(tmp_path, monkeypatch):
    client, s = setup(tmp_path)
    from test_knowledge_registry import version_snapshot
    from test_registry_runtime import loaded

    from agent_rag import router as runtime
    from agent_rag.task_service import TaskService

    other = tmp_path / "other-registry"
    snapshot = version_snapshot(other, "old")
    facade = runtime._RegistryTaskService(other, TaskService(tmp_path / "tasks", loaded(snapshot)))
    monkeypatch.setattr(facade, "request_refresh", lambda: None)
    monkeypatch.setattr(runtime._initialization, "peek_ready", lambda: facade)
    try:
        assert (
            client.get("/api/agent-rag/knowledge/status").json()["selection"]["restart_required"]
            is True
        )
    finally:
        facade.close()
