"""Durable maintenance run summaries and internal-tool route isolation."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_dev30_experiment import module as experiment_module
from test_maintenance_api import PREFIX, event, setup

from agent_rag.knowledge_updates import write_json


def saved_run(service, run_id, event_ids, **extra):
    value = {
        "run_id": run_id,
        "event_ids": event_ids,
        "status": "failed",
        "execution_mode": "fixture",
        "publication_allowed": False,
        **extra,
    }
    path = service.root / "api_runs" / f"{run_id}.json"
    write_json(path, value)
    return path


def test_status_returns_only_fifty_recent_persisted_runs_in_time_order(tmp_path):
    _, service, _, _ = setup(tmp_path)
    item = event(service)
    ids = [f"{index:032x}" for index in range(55)]
    for index, run_id in enumerate(ids):
        path = saved_run(service, run_id, [item["id"]])
        os.utime(path, (1000 + index, 1000 + index))
    assert [row["run_id"] for row in service.status()["runs"]] == list(reversed(ids[-50:]))


def test_run_status_filters_private_and_mixed_scope_and_redacts_nested_details(tmp_path):
    client, service, _, _ = setup(tmp_path)
    public = event(service)
    private = service.store.create_event(
        "new_source", ["private-ref"], [], "private", "private-run", "private"
    )
    public_id, private_id, mixed_id = "a" * 32, "b" * 32, "c" * 32
    saved_run(
        service,
        public_id,
        [public["id"]],
        result={
            "trace_path": "C:/private/trace.json",
            "api_key": "never-public",
            "message": "trace C:/private/trace.json",
            "nested": {"authorization": "never-public", "knowledge_commit": "host-commit"},
        },
    )
    saved_run(service, private_id, [private["id"]])
    saved_run(service, mixed_id, [public["id"], private["id"]])
    response = client.get(PREFIX + "/status")
    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [row["run_id"] for row in runs] == [public_id]
    assert runs[0]["result"]["nested"]["knowledge_commit"] == "host-commit"
    assert "C:/private" not in str(runs) and "never-public" not in str(runs)
    assert private_id not in str(runs) and mixed_id not in str(runs)


def test_status_skips_corrupt_invalid_and_mismatched_run_files(tmp_path):
    _, service, _, _ = setup(tmp_path)
    item = event(service)
    valid = "a" * 32
    saved_run(service, valid, [item["id"]])
    directory = service.root / "api_runs"
    for name, raw in (
        ("not-a-run", "{}"),
        ("b" * 32, "{broken"),
        ("c" * 32, "[]"),
        ("d" * 32, "{}"),
    ):
        (directory / f"{name}.json").write_text(raw, encoding="utf-8")
    saved_run(service, "e" * 32, [item["id"]]).write_text(
        json.dumps(
            {
                "run_id": "f" * 32,
                "event_ids": [item["id"]],
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )
    assert [row["run_id"] for row in service.status()["runs"]] == [valid]


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        {
            "run_id": "b" * 32,
            "event_ids": ["c" * 32],
            "status": "failed",
        },
    ],
)
def test_get_run_rejects_invalid_records_instead_of_exposing_arbitrary_json(tmp_path, value):
    _, service, _, _ = setup(tmp_path)
    path = service.root / "api_runs" / ("a" * 32 + ".json")
    write_json(path, value)
    with pytest.raises(ValueError, match="INVALID_RUN_RECORD"):
        service.get_run("a" * 32)


def test_get_run_rejects_resolved_escape_before_reading_json(tmp_path, monkeypatch):
    _, service, _, _ = setup(tmp_path)
    item = event(service)
    path = saved_run(service, "a" * 32, [item["id"]])
    outside = tmp_path / "outside.json"
    resolve, read = Path.resolve, Path.read_text

    def redirected(candidate, *args, **kwargs):
        return outside if candidate == path else resolve(candidate, *args, **kwargs)

    def forbidden(candidate, *args, **kwargs):
        if candidate == path:
            pytest.fail("越界作业不得读取原文")
        return read(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected)
    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(ValueError, match="INVALID_RUN_PATH"):
        service.get_run("a" * 32)


def test_status_does_not_read_outside_symlink(tmp_path, monkeypatch):
    _, service, _, _ = setup(tmp_path)
    directory = service.root / "api_runs"
    directory.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{private", encoding="utf-8")
    link = directory / ("a" * 32 + ".json")
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("系统不允许测试进程创建symlink；resolve越界哨兵另有覆盖")
    read = Path.read_text

    def forbidden(candidate, *args, **kwargs):
        if candidate == link or candidate == outside:
            pytest.fail("越界symlink不得读取JSON")
        return read(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbidden)
    assert service.status()["runs"] == []
    with pytest.raises(ValueError, match="INVALID_RUN_PATH"):
        service.get_run("a" * 32)


def test_experiment_internal_tool_app_never_includes_manager_or_public_routes():
    client = TestClient(experiment_module().local_tool_app(object()))
    for path in (
        "/api/agent-rag/knowledge/status",
        "/api/agent-rag/knowledge/maintenance/status",
        "/agent-rag/knowledge",
    ):
        assert client.get(path).status_code == 404
    assert client.post("/api/agent-rag/tasks", json={"text": "fixture"}).status_code == 404


def test_run_reader_preserves_persisted_terminal_status_label(tmp_path):
    _, service, _, _ = setup(tmp_path)
    item = event(service)
    run_id = "a" * 32
    saved_run(service, run_id, [item["id"]], status="engineering_failed")
    assert service.get_run(run_id)["status"] == "engineering_failed"
