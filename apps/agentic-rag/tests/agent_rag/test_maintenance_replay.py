"""Real fixture-only Node/Pi replay; no provider calls or production publication."""

import importlib.util
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/evaluate_knowledge_maintenance.py"
MANIFEST = ROOT / "tests/fixtures/knowledge_maintenance/events.json"


def replay_module():
    assert SCRIPT.is_file(), "缺少可重复的维护工程回放入口"
    spec = importlib.util.spec_from_file_location("maintenance_replay", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return replay_module().run_replay(tmp_path_factory.mktemp("kms"), suite="smoke")


def test_three_node_pi_replays_persist_real_reports_receipts_and_ledger(smoke):
    assert smoke["suite"] == "smoke"
    assert smoke["expectations_met"] is True
    assert len(smoke["items"]) == 3
    assert {item["status"] for item in smoke["items"]} == {"engineering_complete"}
    assert smoke["requests"] == 16
    assert smoke["ledger_states"] == {"settled": 16}
    assert smoke["all_requests_settled"] is True
    assert smoke["physical_provider_requests"] == 0
    assert smoke["settled_micro"] == 0
    assert smoke["remaining_requests"] == 68
    report = json.loads(Path(smoke["service_report"]).read_text(encoding="utf-8"))
    assert report["budget"]["requests"] == smoke["requests"]
    assert report["budget"]["unknown_micro"] == 0
    with sqlite3.connect(smoke["ledger_path"]) as db:
        rows = db.execute("SELECT state,actual_micro FROM requests").fetchall()
    assert len(rows) == 16 and all(row == ("settled", 0) for row in rows)
    for item in smoke["items"]:
        assert item["publication_allowed"] is False
        receipt = json.loads(Path(item["receipt_path"]).read_text(encoding="utf-8"))
        assert receipt["decision"] == "approve"
        assert receipt["reviewer"]["execution_mode"] == "fixture"
        assert receipt["source_reads"] and all(read["complete"] for read in receipt["source_reads"])
        assert receipt["independent_search_trace_ids"]
        assert item["production_gate"] == "FIXTURE_FORBIDDEN"
        for trace_path in item["trace_paths"]:
            trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            assert trace["result"]["stop_reason"] == "completed"
            assert trace["result"]["resources"]["physical_provider_requests"] == 0
            assert any(tool["name"] == "read_evidence" for tool in trace["tools"])
            if trace["task"]["role"] == "reviewer":
                assert any(tool["name"] == "search_evidence" for tool in trace["tools"])
                assert any(tool["name"] == "inspect_changeset" for tool in trace["tools"])


def test_negation_loss_is_logged_then_fixed_once_and_deactivation_stays_candidate(smoke):
    repair = next(item for item in smoke["items"] if item["fixture_id"] == "new02")
    assert repair["revisions"] == 2 and repair["repair_rounds"] == 1
    proposals = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in repair["changeset_paths"]
    ]
    assert proposals[0]["operations"][0]["after"]["case_summary"] == "承诺保本。"
    assert proposals[1]["operations"][0]["after"]["case_summary"] == "未承诺保本。"
    assert repair["review_decisions"] == ["request_changes", "approve"]
    stopped = next(item for item in smoke["items"] if item["fixture_id"] == "deactivate01")
    changeset = json.loads(Path(stopped["changeset_paths"][0]).read_text(encoding="utf-8"))
    assert changeset["operations"][0]["kind"] == "deactivate"
    assert changeset["operations"][0]["after"]["active"] is False
    assert stopped["publication_allowed"] is False


def test_twenty_event_manifest_has_requested_distribution_and_synthetic_boundary():
    replay_module()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["example_only"] is True
    events = manifest["events"]
    assert len(events) == 20
    assert Counter(event["category"] for event in events) == {
        "new_source": 8,
        "source_revision": 3,
        "extraction_correction": 2,
        "duplicate_relation": 2,
        "law_revision": 2,
        "deactivation": 1,
        "insufficient_evidence": 1,
        "permission": 1,
    }
    assert all(event["example_only"] is True for event in events)
    assert all(
        event["expected_status"]
        in {"engineering_complete", "needs_evidence", "authorization_rejected"}
        for event in events
    )


def test_full_replay_uses_real_outcomes_and_rejects_cross_scope_reads(tmp_path):
    full = replay_module().run_replay(tmp_path / "full", suite="full")
    assert full["expectations_met"] is True
    assert len(full["items"]) == 20
    assert Counter(item["status"] for item in full["items"]) == {
        "engineering_complete": 18,
        "needs_evidence": 1,
        "authorization_rejected": 1,
    }
    assert full["requests"] == 80 and full["remaining_requests"] == 452
    assert full["ledger_states"] == {"settled": 80}
    assert full["all_requests_settled"] is True
    assert full["canonical_records_unchanged"] is True
    assert full["physical_provider_requests"] == 0 and full["settled_micro"] == 0
    permissions = next(item for item in full["items"] if item["fixture_id"] == "permission01")
    assert permissions["checks"] == {
        "search_evidence_hidden": True,
        "read_evidence": "NOT_AUTHORIZED",
        "read_record": "NOT_AUTHORIZED",
        "service_admission": "PRIVATE_MAINTENANCE_NOT_ADMITTED",
    }
    assert permissions["requests"] == 0 and permissions["trace_path"]
    missing = next(item for item in full["items"] if item["fixture_id"] == "insufficient01")
    assert missing["receipt_path"] is None and missing["review_decisions"] == ["needs_evidence"]
    assert all(item["repair_rounds"] <= 1 for item in full["items"])
    summary = json.loads(Path(full["summary_path"]).read_text(encoding="utf-8"))
    assert summary["limitations"]["real_semantics_evaluated"] is False
    assert summary["limitations"]["retrieval_regression_executed"] is False
    assert summary["expectations_are"] == "scripted_engineering_expectations_not_model_quality"


def test_permission_report_counts_observed_tool_trace(tmp_path):
    module = replay_module()
    fixture = next(
        item
        for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["events"]
        if item["id"] == "permission01"
    )
    service = module.MaintenanceService(tmp_path / "maintenance")
    item = module._permission_check(service, fixture, tmp_path)
    trace = json.loads(Path(item["trace_path"]).read_text(encoding="utf-8"))
    assert item["tool_actions"] == len(trace["tools"]) == 3
    assert item["status"] == "authorization_rejected"
    assert [tool["success"] for tool in trace["tools"]] == [True, False, False]
