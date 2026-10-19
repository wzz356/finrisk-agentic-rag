import importlib
import importlib.util

import pytest


def service(tmp_path):
    assert importlib.util.find_spec("agent_rag.maintenance_service"), "维护编排服务尚未实现"
    return importlib.import_module("agent_rag.maintenance_service").MaintenanceService(tmp_path)


def test_live_without_verified_pair_is_refused_before_work_or_budget(tmp_path):
    s = service(tmp_path)
    with pytest.raises(ValueError, match="LIVE_ADAPTERS_NOT_READY"):
        s.run_events([], execution_mode="live")
    assert s.status()["live_ready"] is False
    assert s.status()["usage_enabled"] is False


def test_inventory_registration_isolated_and_idempotent(tmp_path):
    from agent_rag.collection_store import CollectionStore

    inventory = CollectionStore(tmp_path / "inventory")
    b = inventory.create_batch("fixture")
    inventory.ingest(
        b["id"], "甲保险公司未承诺保本。".encode(), {"source_url": "https://www.nfra.gov.cn/a"}
    )
    s = service(tmp_path / "maintenance")
    first = s.import_inventory(inventory, b["id"])
    second = s.import_inventory(inventory, b["id"])
    assert first["event_ids"] == second["event_ids"] and len(first["event_ids"]) == 1
    assert len(s.store.list_events()) == 1
    assert s.status()["usage_enabled"] is False


def test_actual_node_loop_accounts_every_round_and_cannot_publish_fixture(tmp_path):
    s = service(tmp_path)
    source = s.corpus.register_evidence("甲保险公司未承诺保本。".encode(), origin_id="fixture:1")
    event = s.create_event("new_source", [source["evidence_id"]], [], "fixture example")
    script = [
        [
            {
                "type": "toolCall",
                "id": "read",
                "name": "read_evidence",
                "arguments": {
                    "evidence_id": source["evidence_id"],
                    "text_sha256": source["text_sha256"],
                    "start": 0,
                    "end": len("甲保险公司未承诺保本。"),
                },
            }
        ],
        [
            {
                "type": "toolCall",
                "id": "search",
                "name": "search_evidence",
                "arguments": {"query": "保险"},
            }
        ],
        {"operations": [], "claims": [], "source_coverage": []},
    ]
    result = s.run_events(
        [event["id"]], execution_mode="fixture", scripts={event["id"]: {"propose": script}}
    )
    assert result["publication_allowed"] is False
    assert result["items"][0]["status"] == "failed"
    assert result["items"][0]["error"] == "PROTOCOL_SCHEMA"
    assert result["budget"]["requests"] == 3
    import json

    trace = json.loads(next((s.root / "traces").glob("*.json")).read_text(encoding="utf-8"))
    contract = trace["task"]["input"]["maintenance_contract"]
    assert len(contract["official_risk_tags"]) == 27
    assert contract["coordinate_system"] == "unicode_codepoints_half_open"
    assert contract["$defs"]["Operation"]["required"]
    assert result["budget"]["settled_micro"] == 0
    assert s.store.list_jobs()[0]["status"] == "failed"
    assert s.status()["usage_enabled"] is False
    replay = s.run_events(
        [event["id"]], execution_mode="fixture", scripts={event["id"]: {"propose": script}}
    )
    assert replay["budget"]["requests"] == 3


def test_async_runs_are_durable_and_no_script_is_not_success(tmp_path):
    import time

    s = service(tmp_path)
    evidence = s.corpus.register_evidence("保险原文".encode(), origin_id="fixture:async")
    event = s.create_event("new_source", [evidence["evidence_id"]], [], "fixture example")
    run = s.submit_events([event["id"]], execution_mode="fixture")
    deadline = time.monotonic() + 5
    while (
        s.get_run(run["run_id"])["status"] in {"queued", "running"} and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    result = s.get_run(run["run_id"])
    assert result["status"] == "failed"
    assert result["result"]["items"][0]["error"] == "FIXTURE_SCRIPT_REQUIRED"
    reopened = service(tmp_path)
    assert reopened.get_run(run["run_id"])["publication_allowed"] is False
    with pytest.raises(ValueError, match="INVALID_RUN_ID"):
        s.get_run("../outside")


def test_completed_or_failed_event_replay_does_not_reset_budget(tmp_path):
    s = service(tmp_path)
    evidence = s.corpus.register_evidence("保险原文".encode(), origin_id="fixture:replay")
    event = s.create_event("new_source", [evidence["evidence_id"]], [], "fixture example")
    first = s.run_events([event["id"]], scripts={})
    again = s.run_events([event["id"]], scripts={})
    assert first["batch_id"] == again["batch_id"]
    assert first["items"] == again["items"]


def test_snapshot_binding_imports_existing_knowledge_as_legacy_candidates(tmp_path):
    from test_knowledge_registry import version_snapshot

    from agent_rag.v2_snapshot import file_hash

    base = version_snapshot(tmp_path / "old", "base")
    s = service(tmp_path / "maintenance")
    result = s.bind_snapshot(base, file_hash(base / "manifest.json"))
    assert result["source_count"] == 1 and result["record_count"] >= 1
    records = s.corpus.records(s.base_commit)
    assert records and all(r["admission_status"] == "legacy_candidate" for r in records)
    head = s.base_commit
    assert s.bind_snapshot(base, file_hash(base / "manifest.json"))["knowledge_commit"] == head
    assert service(tmp_path / "maintenance").base_commit == head


def test_external_idempotency_key_is_bound_to_protocol_hash_before_model_work(tmp_path):
    s = service(tmp_path)
    evidence = s.corpus.register_evidence("保险原文".encode(), origin_id="fixture:idempotency")
    first = s.create_event(
        "new_source",
        [evidence["evidence_id"]],
        [],
        "fixture example",
        idempotency_key="api-event-1",
    )
    again = s.create_event(
        "new_source",
        [evidence["evidence_id"]],
        [],
        "fixture example",
        idempotency_key="api-event-1",
    )
    assert first["id"] == again["id"]
    assert len(first["idempotency_key"]) == 64 and all(
        c in "0123456789abcdef" for c in first["idempotency_key"]
    )


def test_existing_baseline_pins_original_host_seed_instead_of_current_head(tmp_path, monkeypatch):
    from test_knowledge_registry import version_snapshot

    from agent_rag.v2_snapshot import file_hash

    base = version_snapshot(tmp_path / "old", "base")
    s = service(tmp_path / "maintenance")
    pin = file_hash(base / "manifest.json")
    original = s.bind_snapshot(base, pin)["knowledge_commit"]
    calls = []
    monkeypatch.setattr(
        s.publisher,
        "pin_legacy_seed",
        lambda commit, **kwargs: calls.append((commit, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(s.publisher, "head", lambda: "f" * 40)
    assert s.bind_snapshot(base, pin)["knowledge_commit"] == "f" * 40
    assert calls == [
        (original, {"base_manifest_path": base / "manifest.json", "expected_sha256": pin})
    ]


def test_cancel_between_queue_and_worker_start_is_never_cleared(tmp_path, monkeypatch):
    import contextlib
    import threading
    import time

    s = service(tmp_path)
    module = importlib.import_module("agent_rag.maintenance_service")
    entered, release = threading.Event(), threading.Event()
    original = module.update_lock

    @contextlib.contextmanager
    def gated(path):
        with original(path):
            if path.name == "operations":
                entered.set()
                assert release.wait(5)
            yield

    monkeypatch.setattr(module, "update_lock", gated)

    class ForbiddenRunner:
        def run(self, *args, **kwargs):
            raise AssertionError("取消后不得再启动角色进程")

    s.runner = ForbiddenRunner()
    evidence = s.corpus.register_evidence("保险原文".encode(), origin_id="fixture:cancel")
    event = s.create_event("new_source", [evidence["evidence_id"]], [], "fixture example")
    run = s.submit_events([event["id"]], scripts={event["id"]: {"propose": [{}]}})
    assert entered.wait(3)
    s.cancel_run(run["run_id"])
    release.set()
    deadline = time.monotonic() + 5
    while (
        s.get_run(run["run_id"])["status"] in {"queued", "running"} and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert s.get_run(run["run_id"])["status"] == "cancelled"
    assert s.get_run(run["run_id"])["cancel_requested"] is True


def test_existing_batch_cannot_capture_new_events_and_shrink_their_budget(tmp_path):
    s = service(tmp_path)
    events = []
    for i in range(2):
        source = s.corpus.register_evidence(
            ("保险原文" + str(i)).encode(), origin_id="fixture:batch" + str(i)
        )
        events.append(
            s.create_event("new_source", [source["evidence_id"]], [], "fixture example")["id"]
        )
    s.run_events(events[:1], scripts={})
    with pytest.raises(ValueError, match="BATCH_REPLAY_SCOPE_MISMATCH"):
        s.run_events(events, scripts={})
    assert s.store.get_artifact("budget:" + events[1]) is None
