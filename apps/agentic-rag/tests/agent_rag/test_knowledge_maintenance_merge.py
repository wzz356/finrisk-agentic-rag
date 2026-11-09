"""Main-agent/maintenance seams: explicit sources win and status stays model-free."""

import importlib
import importlib.util
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_knowledge_registry import version_snapshot


def test_process_help_describes_pending_maintenance_registration():
    path = Path(__file__).resolve().parents[2] / "scripts/manage_knowledge.py"
    spec = importlib.util.spec_from_file_location("maintenance_merge_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert "登记待审" in cli._parser().format_help()
    assert "收费处理批次" not in cli._parser().format_help()


@pytest.mark.parametrize(
    "registry_explicit,snapshot_explicit",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_explicit_registry_then_snapshot_then_bound_managed_fallback(
    tmp_path,
    monkeypatch,
    registry_explicit,
    snapshot_explicit,
):
    router = importlib.import_module("agent_rag.router")
    managed = tmp_path / "managed"
    (managed / "registry").mkdir(parents=True)
    (managed / "registry/usage.json").write_text("{}", encoding="utf-8")
    explicit_registry = tmp_path / "explicit-registry"
    explicit_snapshot = tmp_path / "explicit-snapshot"
    monkeypatch.delenv("FINRISK_AGENT_KB_REGISTRY", raising=False)
    monkeypatch.delenv("FINRISK_AGENT_SNAPSHOT", raising=False)
    monkeypatch.setattr(router, "collection_root", lambda: managed)
    monkeypatch.setattr(router, "retrieval_snapshot", lambda root: root / "resolved")
    if registry_explicit:
        monkeypatch.setenv("FINRISK_AGENT_KB_REGISTRY", str(explicit_registry))
    if snapshot_explicit:
        monkeypatch.setenv("FINRISK_AGENT_SNAPSHOT", str(explicit_snapshot))
    expected = (
        explicit_registry / "resolved"
        if registry_explicit
        else explicit_snapshot
        if snapshot_explicit
        else managed / "registry/resolved"
    )
    assert router._snapshot_path() == expected


def test_unbound_managed_inventory_does_not_replace_default_snapshot(tmp_path, monkeypatch):
    router = importlib.import_module("agent_rag.router")
    monkeypatch.delenv("FINRISK_AGENT_KB_REGISTRY", raising=False)
    monkeypatch.delenv("FINRISK_AGENT_SNAPSHOT", raising=False)
    monkeypatch.setattr(router, "collection_root", lambda: tmp_path)
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry/current.json").write_text("{}", encoding="utf-8")
    assert router._registry_path() is None


def test_registry_without_usage_retains_old_current_pointer_api(tmp_path):
    from agent_rag import knowledge_registry as registry

    root = tmp_path / "registry"
    snapshot = version_snapshot(root, "base")
    registry.publish_snapshot(root, snapshot)
    assert not (root / "usage.json").exists()
    assert registry.current_snapshot(root) == registry.retrieval_snapshot(root) == snapshot


def test_main_router_management_status_never_initializes_model(tmp_path, monkeypatch):
    router = importlib.import_module("agent_rag.router")
    collection = importlib.import_module("agent_rag.collection_router")
    maintenance = importlib.import_module("agent_rag.maintenance_router")
    from agent_rag.collection_service import CollectionService
    from agent_rag.maintenance_service import MaintenanceService

    def forbidden(*args, **kwargs):
        pytest.fail("管理状态不得初始化模型或 GPU")

    monkeypatch.setattr(router._initialization, "request_warmup", forbidden)
    monkeypatch.setattr(router._initialization, "wait_ready", forbidden)
    monkeypatch.setattr(router._initialization, "get_ready", forbidden)
    monkeypatch.setattr(router._initialization, "peek_ready", lambda: None)
    monkeypatch.setattr(router, "load_v2_snapshot", forbidden)
    app = FastAPI()
    app.include_router(router.router)
    app.dependency_overrides[collection.get_collection_service] = lambda: CollectionService(
        tmp_path / "collection"
    )
    app.dependency_overrides[maintenance.get_maintenance_service] = lambda: MaintenanceService(
        tmp_path / "maintenance"
    )
    client = TestClient(app)
    status = client.get("/api/agent-rag/knowledge/status")
    assert status.status_code == 200
    assert status.json()["settings"]["crawl_enabled"] is False
    assert status.json()["settings"]["usage_enabled"] is False
    review = client.get("/api/agent-rag/knowledge/maintenance/status")
    assert review.status_code == 200 and review.json()["live_ready"] is False
