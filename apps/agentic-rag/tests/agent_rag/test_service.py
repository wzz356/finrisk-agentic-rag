import json
from pathlib import Path

import pytest

from agent_rag.knowledge_snapshot import KnowledgeSnapshot
from agent_rag.task_service import TaskService


@pytest.fixture
def service(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "test.txt").write_text("给予合同外利益，罚款。", encoding="utf-8")
    dictionary = (
        Path(__file__).resolve().parents[2] / "contracts/agent_rag/risk_type_dictionary.csv"
    )
    snapshot = KnowledgeSnapshot.build(raw, dictionary, tmp_path / "snapshot")
    return TaskService(tmp_path / "tasks", snapshot)


def request(text="测试材料", context=None):
    return {
        "text": text,
        "user_context": context or {},
        "engine": "controlled_agent",
        "experiment_profile_id": "mock-bm25-baseline-v1",
        "mock_scenario": "generic",
    }


def test_idempotency_same_input_and_changed_background(service):
    task, reused = service.admit(request(), "same-key")
    assert not reused
    again, reused = service.admit(request(), "same-key")
    assert reused and again["task_id"] == task["task_id"]
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        service.admit(request(context={"purpose_hint": "广告"}), "same-key")


def test_atomic_capacity_no_unlimited_queue(service):
    service.admit(request(), None)
    with pytest.raises(ValueError, match="CAPACITY_EXCEEDED"):
        service.admit(request("另一份材料"), None)


def test_read_is_side_effect_free_and_unknown_rejected(service):
    record, _ = service.admit(request(), None)
    before = (service.root / record["task_id"] / "manifest.json").read_bytes()
    assert service.get(record["task_id"])["task_status"] == "prepared"
    assert before == (service.root / record["task_id"] / "manifest.json").read_bytes()
    with pytest.raises(KeyError):
        service.get("../secret")


def progress_sample(task_id):
    from datetime import UTC, datetime

    return {
        "schema_version": "finrisk-progress/1.0", "task_id": task_id,
        "sequence": 3, "updated_at": datetime.now(UTC).isoformat(),
        "phase": "reviewing", "action": "read_source", "stage_label": "回查处罚原文",
        "task_status": "running", "execution_mode": "mock", "stop_reason": None,
        "message": "正在回查处罚原文。",
        "counts": {"segments_total": 1, "segments_analyzed": 1, "segments_finalized": 0,
                   "risk_points": 1, "read_sources": 0, "pending_input_conditions": 1,
                   "pending_case_queries": 1},
        "limits": {"model_requests": {"used": 3, "limit": 6},
                   "tool_actions": {"used": 2, "limit": 8},
                   "search_calls": {"used": 1, "limit": 3}},
        "time": {"remaining_seconds": 60, "elapsed_ms": 30000},
    }


def test_poll_returns_valid_progress_and_bounded_dedicated_history(service):
    record, _ = service.admit(request(), None)
    directory = service.root / record["task_id"]
    snapshot = progress_sample(record["task_id"])
    (directory / "progress.json").write_text(json.dumps(snapshot), encoding="utf-8")
    rows = [{"sequence": i, "type": "progress.snapshot", "data": {**snapshot, "sequence": i}}
            for i in range(1, 16)]
    (directory / "progress_events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    result = service.get(record["task_id"])
    assert result.get("progress") is not None
    assert result["progress"]["stage_label"] == "回查处罚原文"
    assert len(result["progress_history"]) == 10
    assert result["progress_history"][0]["sequence"] == 6


def test_restart_progress_overrides_stale_running_without_writing_or_resume(service):
    record, _ = service.admit(request(), None)
    path = service.root / record["task_id"] / "progress.json"
    path.write_text(json.dumps(progress_sample(record["task_id"])), encoding="utf-8")
    before = path.read_bytes()
    restarted = TaskService(service.root, service.snapshot)
    result = restarted.get(record["task_id"])
    assert result.get("progress") is not None
    assert result["progress"]["task_status"] == "interrupted"
    assert result["progress"]["time"]["remaining_seconds"] == 0
    assert path.read_bytes() == before


def test_optional_case_point_count_and_old_snapshot_survive_readonly_restart(service):
    record, _ = service.admit(request(), None)
    directory = service.root / record["task_id"]
    old = progress_sample(record["task_id"])
    current = json.loads(json.dumps(old))
    current["counts"].update(pending_case_queries=0, pending_case_points=1)
    latest = directory / "progress.json"
    history = directory / "progress_events.jsonl"
    latest.write_text(json.dumps(current), encoding="utf-8")
    history.write_text("\n".join(json.dumps({"type": "progress.snapshot", "data": value})
                                 for value in [old, current]), encoding="utf-8")
    before = (latest.read_bytes(), history.read_bytes())
    restarted = TaskService(service.root, service.snapshot)
    for _ in range(2):
        result = restarted.get(record["task_id"])
        assert result["progress"]["counts"]["pending_case_points"] == 1
        assert result["progress"]["counts"]["pending_case_queries"] == 0
        assert "pending_case_points" not in result["progress_history"][0]["counts"]
        assert result["progress_history"][1]["counts"]["pending_case_points"] == 1
        assert len(result["progress_history"]) <= 10
    assert (latest.read_bytes(), history.read_bytes()) == before


def test_progress_missing_corrupt_or_foreign_task_does_not_leak_or_break_poll(service):
    record, _ = service.admit(request(), None)
    path = service.root / record["task_id"] / "progress.json"
    assert service.get(record["task_id"]).get("progress") is None
    for content in ["{invalid", json.dumps(progress_sample("another-task"))]:
        path.write_text(content, encoding="utf-8")
        assert service.get(record["task_id"]).get("progress") is None


def test_malformed_history_events_are_skipped_without_breaking_poll(service):
    record, _ = service.admit(request(), None)
    directory = service.root / record["task_id"]
    snapshot = progress_sample(record["task_id"])
    rows = ["[]", "null", "1", '"text"', "{invalid", json.dumps({
        "type": "progress.snapshot", "data": snapshot,
    })]
    (directory / "progress_events.jsonl").write_text("\n".join(rows), encoding="utf-8")
    before = (directory / "progress_events.jsonl").read_bytes()
    result = service.get(record["task_id"])
    assert result["progress_history"] == [snapshot]
    assert (directory / "progress_events.jsonl").read_bytes() == before


def test_manifest_freezes_wire_and_internal_criteria_bytes(service):
    import hashlib

    record, _ = service.admit(request(), None)
    manifest = json.loads(
        (service.root / record["task_id"] / "manifest.json").read_text(encoding="utf-8")
    )
    project = Path(__file__).resolve().parents[2]
    assert manifest["model_wire_version"] == "finrisk-model-wire/1.2"
    assert manifest["model_wire_sha256"] == hashlib.sha256(
        (project / "contracts/agent_rag/model-wire.schema.json").read_bytes()
    ).hexdigest()
    assert manifest["criteria_sha256"] == hashlib.sha256(
        (project / "contracts/agent_rag/risk-criteria.json").read_bytes()
    ).hexdigest()
    execution_contract = service.bundle(record["task_id"])["execution_contract"]
    assert execution_contract["model_wire_sha256"] == manifest["model_wire_sha256"]


def test_restart_marks_running_interrupted_and_does_not_resume(service):
    record, _ = service.admit(request(), "crash-key")
    restarted = TaskService(service.root, service.snapshot)
    assert restarted.get(record["task_id"])["task_status"] == "interrupted"
    same, reused = restarted.admit(request(), "crash-key")
    assert reused and same["task_id"] == record["task_id"]
    assert restarted.active_task_id is None


def test_final_artifacts_preserved_and_late_interrupted_result_not_reopened(service):
    record, _ = service.admit(request(), None)
    task_dir = service.root / record["task_id"]
    restarted = TaskService(service.root, service.snapshot)
    (task_dir / "report.json").write_text(
        json.dumps({"task_status": "completed", "task_id": record["task_id"]}), encoding="utf-8"
    )
    assert restarted.get(record["task_id"])["task_status"] == "interrupted"


def test_trusted_bundle_no_secret_or_arbitrary_path_from_model(service):
    record, _ = service.admit(request(), None)
    bundle = service.bundle(record["task_id"])
    assert bundle["input"]["task_id"] == record["task_id"]
    assert len(bundle["definitions"]) == 27
    assert bundle["config"]["execution_mode"] == "mock"
    assert "api_key" not in json.dumps(bundle)


def test_real_mode_and_hybrid_not_silently_enabled(service):
    payload = request()
    payload["experiment_profile_id"] = "live-hybrid-v1"
    with pytest.raises(ValueError, match="PROFILE_NOT_READY"):
        service.admit(payload, None)


def test_completed_source_can_be_read_after_restart_without_model_calls(service):
    record, _ = service.admit(request("购买保险"), None)
    original = service.tool(record["task_id"])
    source_id = next(iter(service.snapshot.sources))
    read = original.call("read_source", {"source_id": source_id, "max_chars": 500})
    report = {
        "task_status": "completed",
        "task_id": record["task_id"],
        "retrieved_evidence": [read["data"]["evidence"]],
    }
    (service.root / record["task_id"] / "report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    restarted = TaskService(service.root, service.snapshot)
    restored = restarted.tool(record["task_id"])
    assert (
        restored.call("read_source", {"source_id": source_id, "max_chars": 500})["status"] == "ok"
    )
    assert restarted.active_task_id is None and not restarted.jobs
