import importlib
import json
import threading
import time

import pytest
from test_knowledge_registry import version_snapshot, write_json

from agent_rag.task_service import TaskService, atomic_json
from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash


def point(registry, snapshot):
    write_json(
        registry / "current.json",
        {
            "schema_version": "finrisk-knowledge-registry/1.0",
            "version": snapshot.parent.name,
            "snapshot_id": "snapshot-" + snapshot.parent.name,
            "snapshot_path": snapshot.relative_to(registry).as_posix(),
            "manifest_sha256": file_hash(snapshot / "manifest.json"),
            "history": [],
        },
    )


def loaded(path):
    snapshot = V2KnowledgeSnapshot(path)
    snapshot.manifest["rerank_mode"] = "replay"
    return snapshot


def facade(registry, task_root, first):
    module = importlib.import_module("agent_rag.router")
    assert hasattr(module, "_RegistryTaskService"), "尚未实现registry空闲刷新"
    return module, module._RegistryTaskService(registry, TaskService(task_root, loaded(first)))


def finish(service, record):
    task_id = record["task_id"]
    service.records[task_id]["task_status"] = "completed"
    atomic_json(service.root / task_id / "admission.json", service.records[task_id])
    atomic_json(
        service.root / task_id / "report.json",
        {"task_id": task_id, "task_status": "completed", "retrieved_evidence": []},
    )
    with service.lock:
        service.active_task_id = None


def test_registry_environment_overrides_legacy_snapshot_and_builds_runtime(tmp_path, monkeypatch):
    module = importlib.import_module("agent_rag.router")
    registry = tmp_path / "registry"
    first = version_snapshot(registry, "v1")
    point(registry, first)
    monkeypatch.setenv("FINRISK_AGENT_KB_REGISTRY", str(registry))
    monkeypatch.setenv("FINRISK_AGENT_SNAPSHOT", str(tmp_path / "ignored"))
    monkeypatch.setenv("FINRISK_AGENT_TASK_ROOT", str(tmp_path / "tasks"))
    monkeypatch.setattr(module, "load_v2_snapshot", loaded)
    assert module._snapshot_path() == first
    service = module._build_service(lambda *args: None)
    assert service.snapshot.manifest["snapshot_id"] == "snapshot-v1"
    assert hasattr(service, "request_refresh")
    service.close()


def test_refresh_only_starts_idle_then_waits_for_idle_before_install(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    first = version_snapshot(registry, "v1")
    second = version_snapshot(registry, "v2")
    point(registry, first)
    module, service = facade(registry, tmp_path / "tasks", first)
    old = service._current
    record, _ = service.admit({"text": "材料"}, None)
    old_tool = service.tool(record["task_id"])
    point(registry, second)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def load(path):
        calls.append(path)
        entered.set()
        assert release.wait(2)
        return loaded(path)

    monkeypatch.setattr(module, "load_v2_snapshot", load)
    try:
        service.request_refresh()
        assert calls == []
        assert service.snapshot is old.snapshot
        finish(old, record)
        started = time.monotonic()
        service.request_refresh()
        assert time.monotonic() - started < 0.3
        assert entered.wait(1)
        # A new task can use the still-healthy current version while loading.
        running, _ = service.admit({"text": "下一份"}, None)
        release.set()
        service._refresh_thread.join(2)
        assert service.snapshot is old.snapshot
        assert service.get(record["task_id"])["report"]["task_status"] == "completed"
        finish(old, running)
        service.request_refresh()
        assert service.snapshot.manifest["snapshot_id"] == "snapshot-v2"
        assert service.tool(record["task_id"]) is old_tool
        assert service.bundle(record["task_id"])["snapshot"]["snapshot_id"] == "snapshot-v1"
        source_id = next(iter(old.snapshot.sources))
        assert (
            service.tool(record["task_id"]).call(
                "read_source", {"source_id": source_id, "max_chars": 500}
            )["status"]
            == "ok"
        )
        assert service.get(record["task_id"])["task_status"] == "completed"
    finally:
        release.set()
        service.close()


def test_bad_refresh_keeps_current_service_and_records_readable(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    first = version_snapshot(registry, "v1")
    second = version_snapshot(registry, "v2")
    point(registry, first)
    module, service = facade(registry, tmp_path / "tasks", first)
    before = service.snapshot
    record, _ = service.admit({"text": "材料"}, None)
    finish(service._current, record)
    point(registry, second)

    def fail(path):
        raise ValueError("DENSE_STARTUP_FAILED: secret diagnostic")

    monkeypatch.setattr(module, "load_v2_snapshot", fail)
    try:
        service.request_refresh()
        service._refresh_thread.join(2)
        assert service.snapshot is before
        assert service.get(record["task_id"])["task_status"] == "completed"
        assert service.registry_status()["registry_refresh_error"] == "REGISTRY_REFRESH_FAILED"
        assert "secret" not in json.dumps(service.registry_status())
    finally:
        service.close()


def test_no_registry_preserves_existing_snapshot_environment(tmp_path, monkeypatch):
    module = importlib.import_module("agent_rag.router")
    monkeypatch.delenv("FINRISK_AGENT_KB_REGISTRY", raising=False)
    monkeypatch.setenv("FINRISK_AGENT_SNAPSHOT", str(tmp_path / "legacy"))
    assert module._snapshot_path() == tmp_path / "legacy"


def test_restarted_registry_can_read_old_retained_task_source_without_gpu(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    first = version_snapshot(registry, "v1")
    second = version_snapshot(registry, "v2")
    point(registry, first)
    module, service = facade(registry, tmp_path / "tasks", first)
    old, _ = service.admit({"text": "材料"}, None)
    source_id = next(iter(service.snapshot.sources))
    finish(service._current, old)
    service.close()
    point(registry, second)
    _, restarted = facade(registry, tmp_path / "tasks", second)
    monkeypatch.setattr(
        module, "load_v2_snapshot", lambda *args: pytest.fail("旧任务只读不得开GPU")
    )
    try:
        assert restarted.get(old["task_id"])["task_status"] == "completed"
        assert restarted.bundle(old["task_id"])["snapshot"]["snapshot_id"] == "snapshot-v1"
        tool = restarted.tool(old["task_id"])
        assert tool.search_disabled
        assert (
            tool.call("read_source", {"source_id": source_id, "max_chars": 500})["status"] == "ok"
        )
    finally:
        restarted.close()


def test_external_base_task_source_restores_after_first_registry_publish(tmp_path, monkeypatch):
    registry_module = importlib.import_module("agent_rag.knowledge_registry")
    assert hasattr(registry_module, "register_base_snapshot"), "外部基础快照尚未注册"
    registry = tmp_path / "registry"
    external_base = version_snapshot(tmp_path / "old-original-library", "base")
    tasks = tmp_path / "tasks"
    original_service = TaskService(tasks, loaded(external_base))
    old, _ = original_service.admit({"text": "材料"}, None)
    source_id = next(iter(original_service.snapshot.sources))
    finish(original_service, old)
    bootstrap = registry_module.register_base_snapshot(registry, external_base)
    registry_module.publish_snapshot(registry, bootstrap)
    newer = version_snapshot(registry, "new")
    registry_module.publish_snapshot(registry, newer)
    module, restarted = facade(registry, tasks, newer)
    monkeypatch.setattr(module, "load_v2_snapshot", lambda *args: pytest.fail("旧任务不得开GPU"))
    try:
        assert restarted.get(old["task_id"])["task_status"] == "completed"
        assert restarted.bundle(old["task_id"])["snapshot"]["snapshot_id"] == "snapshot-base"
        tool = restarted.tool(old["task_id"])
        assert tool.snapshot.root == bootstrap
        assert tool.search_disabled
        assert (
            tool.call("read_source", {"source_id": source_id, "max_chars": 500})["status"] == "ok"
        )
    finally:
        restarted.close()
