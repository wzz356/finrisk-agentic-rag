"""Local maintenance admission, durable execution and public-scope projections."""

import importlib
import importlib.util
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PREFIX = "/api/agent-rag/knowledge/maintenance"


def setup(tmp_path):
    assert importlib.util.find_spec("agent_rag.maintenance_router"), "维护管理 API 尚未实现"
    module = importlib.import_module("agent_rag.maintenance_router")
    from agent_rag.collection_router import get_collection_service, router
    from agent_rag.collection_service import CollectionService
    from agent_rag.maintenance_service import MaintenanceService

    collection = CollectionService(tmp_path / "collection")
    service = MaintenanceService(tmp_path / "maintenance")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[module.get_maintenance_service] = lambda: service
    app.dependency_overrides[get_collection_service] = lambda: collection
    client = TestClient(app)
    token = client.get(PREFIX + "/status").json()["csrf_token"]
    return client, service, collection, {"X-Knowledge-CSRF": token}


def event(service):
    source = service.corpus.register_evidence(
        "甲保险公司未承诺保本。".encode(), origin_id="fixture:api"
    )
    return service.create_event("new_source", [source["evidence_id"]], [], "需独立模型核查")


def test_status_is_available_without_rag_and_separates_maintenance_from_usage(tmp_path):
    client, service, _, _ = setup(tmp_path)
    response = client.get(PREFIX + "/status")
    assert response.status_code == 200
    value = response.json()
    assert value["live_ready"] is False and value["publication_ready"] is False
    assert "第二" in value["live_reason"]
    assert "usage_enabled" not in value
    assert set(value["stages"]) == {"candidate", "model_review", "engineering_replay", "build"}
    assert response.headers["cache-control"] == "no-store"


def test_all_maintenance_endpoints_keep_local_origin_and_csrf_guards(tmp_path):
    client, service, _, headers = setup(tmp_path)
    item = event(service)
    assert client.post(PREFIX + "/runs", json={"event_ids": [item["id"]]}).status_code == 403
    assert (
        client.post(
            PREFIX + "/import-batch",
            json={"batch_id": "a" * 32},
            headers={**headers, "Origin": "https://evil.example"},
        ).status_code
        == 403
    )
    assert client.get(PREFIX + "/events", headers={"Host": "attacker.example"}).status_code == 403
    assert (
        client.get(PREFIX + "/events", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    )
    client._transport.client = ("192.0.2.4", 1234)
    assert client.get(PREFIX + "/status").status_code == 403


def test_event_creation_is_idempotent_and_only_accepts_registered_public_references(tmp_path):
    client, service, _, headers = setup(tmp_path)
    source = service.corpus.register_evidence("保险材料".encode(), origin_id="fixture:event")
    body = {
        "event_type": "new_source",
        "evidence_ids": [source["evidence_id"]],
        "target_record_ids": [],
        "reason": "核查来源",
        "idempotency_key": "api-event-1",
    }
    first = client.post(PREFIX + "/events", json=body, headers=headers)
    second = client.post(PREFIX + "/events", json=body, headers=headers)
    assert first.status_code == 201 and first.json()["id"] == second.json()["id"]
    listed = client.get(PREFIX + "/events").json()
    assert listed["total"] == 1 and listed["items"][0]["status"] == "pending"
    assert client.get(PREFIX + "/events", params={"limit": 1000}).status_code == 422
    assert (
        client.post(
            PREFIX + "/events", json={**body, "evidence_ids": ["missing"]}, headers=headers
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"trusted_runs": {}},
        {"approval_context": {}},
        {"key": "secret"},
        {"path": "C:/private"},
        {"model_id": "claimed-model"},
        {"scripts": {}},
        {"scope_id": "private"},
    ],
)
def test_client_cannot_supply_trust_paths_models_or_private_scope(tmp_path, extra):
    client, service, _, headers = setup(tmp_path)
    item = event(service)
    response = client.post(
        PREFIX + "/runs", json={"event_ids": [item["id"]], **extra}, headers=headers
    )
    assert response.status_code == 422
    assert not service.store.list_jobs()


def test_import_batch_creates_pending_events_without_build_selection_or_usage(tmp_path):
    client, service, collection, headers = setup(tmp_path)
    batch = collection.store.create_batch("fixture")
    collection.store.ingest(
        batch["id"],
        "甲保险公司未承诺保本。".encode(),
        {"source_url": "https://www.nfra.gov.cn/example"},
    )
    first = client.post(PREFIX + "/import-batch", json={"batch_id": batch["id"]}, headers=headers)
    second = client.post(PREFIX + "/import-batch", json={"batch_id": batch["id"]}, headers=headers)
    assert first.status_code == 200
    assert first.json()["event_ids"] == second.json()["event_ids"] and first.json()["count"] == 1
    assert service.store.list_events()[0]["status"] == "pending"
    assert not service.store.list_jobs()
    assert collection.status()["settings"]["usage_enabled"] is False
    assert not collection.status()["versions"]
    assert (
        client.post(
            PREFIX + "/import-batch", json={"batch_id": "f" * 32}, headers=headers
        ).status_code
        == 404
    )


def test_unconfigured_live_run_is_503_without_jobs_or_budget(tmp_path):
    client, service, _, headers = setup(tmp_path)
    item = event(service)
    response = client.post(
        PREFIX + "/runs",
        json={"event_ids": [item["id"]], "execution_mode": "live"},
        headers=headers,
    )
    assert response.status_code == 503 and response.json()["detail"] == "LIVE_ADAPTERS_NOT_READY"
    assert not service.store.list_jobs() and not (service.root / "budgets").exists()


def test_direct_import_binds_existing_trusted_baseline_before_creating_events(
    tmp_path, monkeypatch
):
    client, service, collection, headers = setup(tmp_path)
    batch = collection.store.create_batch("fixture")
    collection.store.ingest(
        batch["id"], "保险文书".encode(), {"source_url": "https://www.nfra.gov.cn/example"}
    )
    (collection.root / "build-config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        collection,
        "_config",
        lambda: {"base_snapshot": "host-pinned-base", "base_manifest_sha256": "a" * 64},
    )
    bound = []
    monkeypatch.setattr(service, "bind_snapshot", lambda root, pin: bound.append((root, pin)))
    response = client.post(
        PREFIX + "/import-batch", json={"batch_id": batch["id"]}, headers=headers
    )
    assert response.status_code == 200 and bound == [("host-pinned-base", "a" * 64)]


def test_run_is_async_persisted_and_scriptless_fixture_never_claims_success(tmp_path):
    client, service, _, headers = setup(tmp_path)
    item = event(service)
    start = time.monotonic()
    response = client.post(
        PREFIX + "/runs",
        json={"event_ids": [item["id"]], "execution_mode": "fixture"},
        headers=headers,
    )
    assert response.status_code == 202 and time.monotonic() - start < 2
    run_id = response.json()["run_id"]
    for thread in service._threads:
        thread.join(5)
    response = client.get(PREFIX + "/runs/" + run_id)
    assert response.status_code == 200
    result = response.json()
    assert result["publication_allowed"] is False
    assert result["status"] in {"failed", "engineering_failed"}
    assert "FIXTURE_SCRIPT_REQUIRED" in str(result)
    from agent_rag.maintenance_service import MaintenanceService

    assert any(row["run_id"] == run_id for row in client.get(PREFIX + "/status").json()["runs"])
    reopened = MaintenanceService(service.root)
    assert reopened.get_run(run_id)["status"] == result["status"]
    module = importlib.import_module("agent_rag.maintenance_router")
    client.app.dependency_overrides[module.get_maintenance_service] = lambda: reopened
    assert any(row["run_id"] == run_id for row in client.get(PREFIX + "/status").json()["runs"])
    assert client.get(PREFIX + "/runs/" + "f" * 32).status_code == 404


def test_cancel_unknown_and_resume_are_explicit_and_do_not_reset_budget(tmp_path):
    client, service, _, headers = setup(tmp_path)
    assert (
        client.post(PREFIX + "/runs/" + "f" * 32 + "/cancel", json={}, headers=headers).status_code
        == 404
    )
    assert (
        client.post(PREFIX + "/runs/" + "f" * 32 + "/resume", json={}, headers=headers).status_code
        == 409
    )
    assert not service.store.list_jobs()


def test_run_accepts_before_worker_finishes_and_cancel_preserves_accounting(tmp_path, monkeypatch):
    client, service, _, headers = setup(tmp_path)
    item = event(service)
    entered, release = threading.Event(), threading.Event()

    def blocked_run(event_ids, **kwargs):
        entered.set()
        assert release.wait(5)
        return {
            "items": [{"event_id": item["id"], "status": "cancelled"}],
            "budget": {"requests": 2, "settled_micro": 7, "reserved_micro": 3},
            "publication_allowed": False,
        }

    monkeypatch.setattr(service, "run_events", blocked_run)
    try:
        response = client.post(PREFIX + "/runs", json={"event_ids": [item["id"]]}, headers=headers)
        assert response.status_code == 202 and entered.wait(1)
        run_id = response.json()["run_id"]
        assert client.get(PREFIX + "/runs/" + run_id).json()["status"] == "running"
        cancelled = client.post(PREFIX + "/runs/" + run_id + "/cancel", json={}, headers=headers)
        assert cancelled.status_code == 200 and cancelled.json()["cancel_requested"] is True
        assert (
            client.post(
                PREFIX + "/runs/" + run_id + "/cancel",
                json={"approval_context": {}},
                headers=headers,
            ).status_code
            == 422
        )
    finally:
        release.set()
        for thread in service._threads:
            thread.join(5)
    value = client.get(PREFIX + "/runs/" + run_id).json()
    assert value["budget"]["settled_micro"] == 7 and value["budget"]["reserved_micro"] == 3
    assert value["publication_allowed"] is False


def test_private_details_rejected_and_public_diff_review_are_safely_projected(tmp_path):
    client, service, _, _ = setup(tmp_path)
    public = event(service)
    private = service.store.create_event(
        "new_source", ["private-ref"], [], "private", "private-test", "secret"
    )
    service.store.save_changeset(
        private["id"],
        {"scope": {"scope_id": "private", "classification": "private"}, "operations": []},
    )
    assert client.get(PREFIX + "/changesets/" + private["id"]).status_code == 403
    assert private["id"] not in str(client.get(PREFIX + "/events").json())
    service.store.save_changeset(
        public["id"],
        {
            "scope": {"scope_id": "public", "classification": "public"},
            "operations": [
                {
                    "kind": "replace",
                    "before": {"text": "旧条目"},
                    "after": {"text": "新条目"},
                    "path": "C:/private/example",
                }
            ],
            "api_key": "never-public",
        },
    )
    service.store.save_review_run(
        public["id"],
        1,
        {
            "role": "reviewer",
            "decision": "request_changes",
            "feedback": "需补充证据",
            "provider": "fixture",
            "trace_path": "C:/private/trace",
        },
    )
    response = client.get(PREFIX + "/changesets/" + public["id"])
    assert response.status_code == 200
    value = response.json()
    assert value["changeset"]["operations"][0]["before"]["text"] == "旧条目"
    assert value["review_runs"][0]["payload"]["decision"] == "request_changes"
    assert "C:/private" not in str(value) and "never-public" not in str(value)
    assert client.get(PREFIX + "/changesets/" + "f" * 32).status_code == 404
