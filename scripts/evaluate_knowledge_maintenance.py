"""Offline scripted engineering replay through the actual Python -> Node/Pi bridge.

Fixtures are synthetic protocol examples, never model-quality labels or live facts.
The script admits no provider configuration and never invokes a publisher.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent_rag.maintenance_contracts import canonical_bytes, require_approved  # noqa: E402
from agent_rag.maintenance_service import MaintenanceService, business_validator  # noqa: E402

MANIFEST = ROOT / "tests/fixtures/knowledge_maintenance/events.json"
PUBLIC = {"scope_id": "public", "classification": "public"}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return str(path.resolve())


def _statement(value):
    return value if type(value) is str else canonical_bytes(value).decode("utf-8")


def _tool(name, arguments, suffix):
    return {
        "type": "toolCall",
        "id": name + "-" + str(suffix),
        "name": name,
        "arguments": arguments,
    }


def _citation(source, statement):
    start = source["text"].find(statement)
    if start < 0:
        raise ValueError("FIXTURE_QUOTE_MISSING")
    return {
        **{key: source[key] for key in ("evidence_id", "raw_sha256", "text_sha256")},
        "start": start,
        "end": start + len(statement),
        "quote": statement,
    }


def _proposal(fixture, sources, previous, *, repaired=False):
    body = copy.deepcopy(fixture.get("repair_body") if repaired else fixture["body"])
    operation = {
        "operation_id": "OP-" + fixture["id"],
        "kind": fixture["operation"],
        "record_kind": fixture["record_kind"],
        "target_id": fixture["target_id"],
        "expected_record_version": previous["record_version"] if previous else None,
        "before_sha256": previous["sha256"] if previous else None,
        "after": body,
        "reason": fixture["reason"],
    }
    claims = []
    for field, value in body.items():
        statement = _statement(value)
        claims.append(
            {
                "claim_id": fixture["id"] + "-" + field,
                "operation_id": operation["operation_id"],
                "pointer": "/" + field,
                "claim_type": "legal_rule" if fixture["record_kind"] == "law_article" else "fact",
                "statement": statement,
                "citations": [_citation(sources[-1], statement)],
                "uncertainty": None,
            }
        )
    coverage = []
    for source in sources:
        latest = source["evidence_id"] == sources[-1]["evidence_id"]
        unresolved = fixture["category"] == "insufficient_evidence"
        coverage.append(
            {
                "evidence_id": source["evidence_id"],
                "text_sha256": source["text_sha256"],
                "start": 0,
                "end": len(source["text"]),
                "disposition": "needs_clarification"
                if unresolved
                else "mapped"
                if latest
                else "not_material",
                "claim_ids": [claim["claim_id"] for claim in claims]
                if latest and not unresolved
                else [],
                "reason": "合成未核实材料，缺少原件。"
                if unresolved
                else "合成当前字段映射；全文由实际工具读取。"
                if latest
                else "历史版本仅用于比较，当前主张引用更正版本。",
            }
        )
    return {"operations": [operation], "claims": claims, "source_coverage": coverage}


def _review(proposal, decision):
    checks = [
        {
            "claim_id": claim["claim_id"],
            "verdict": "supported",
            "citations": claim["citations"],
            "comment": "预定工程审核输出；不是实际模型判断。",
        }
        for claim in proposal["claims"]
    ]
    issues = []
    if decision != "approve":
        affected = next(
            (claim for claim in proposal["claims"] if claim["pointer"] == "/case_summary"),
            proposal["claims"][0],
        )
        for check in checks:
            if check["claim_id"] == affected["claim_id"]:
                check["verdict"] = (
                    "unsupported" if decision == "request_changes" else "needs_evidence"
                )
        issues = [
            {
                "issue_id": "ISSUE-" + affected["claim_id"],
                "severity": "blocking",
                "category": "qualifier" if decision == "request_changes" else "source",
                "operation_id": affected["operation_id"],
                "claim_id": affected["claim_id"],
                "pointer": affected["pointer"],
                "diff_line": None,
                "citations": affected["citations"],
                "change_request": "恢复原文否定词“未”。"
                if decision == "request_changes"
                else "补充可核实原件后再审。",
            }
        ]
    return {
        "decision": decision,
        "claim_checks": checks,
        "omission_check_completed": True,
        "issues": issues,
    }


def _script(fixture, event, sources, proposal, *, role, revision, decision="approve"):
    calls = [
        _tool(
            "read_evidence",
            {
                "evidence_id": source["evidence_id"],
                "text_sha256": source["text_sha256"],
                "start": 0,
                "end": len(source["text"]),
            },
            i,
        )
        for i, source in enumerate(sources)
    ]
    if role == "reviewer":
        calls.extend(
            [
                _tool("search_evidence", {"query": fixture["search_query"]}, "independent"),
                _tool(
                    "inspect_changeset",
                    {"changeset_id": event["id"], "revision": revision},
                    revision,
                ),
                _tool("get_dependencies", {"record_ids": event["target_record_ids"]}, revision),
            ]
        )
        final = _review(proposal, decision)
    else:
        calls.append(_tool("search_knowledge", {"query": fixture["search_query"]}, "existing"))
        if len(sources) > 1:
            calls.append(
                _tool(
                    "compare_versions",
                    {"old_ref": sources[0]["evidence_id"], "new_ref": sources[-1]["evidence_id"]},
                    "versions",
                )
            )
        final = proposal
    # One real Pi tool-use turn, followed by a final JSON turn. Both are ledgered.
    return [calls, final]


def _register(service, fixture):
    sources = []
    for index, text in enumerate(fixture["source_texts"]):
        # Extraction correction intentionally keeps raw original bytes fixed.
        raw_text = (
            fixture["source_texts"][-1] if fixture["category"] == "extraction_correction" else text
        )
        metadata = service.corpus.register_evidence(
            raw_text.encode("utf-8"),
            origin_id="fixture:maintenance:" + fixture["id"],
            text=text,
            extractor_version="fixture-extraction/" + str(index + 1),
        )
        sources.append(service.corpus.resolve_evidence(metadata["evidence_id"]))
    previous = None
    if fixture.get("prior_body"):
        service.corpus.register_record(
            fixture["target_id"],
            fixture["record_kind"],
            fixture["prior_body"],
            commit=service.base_commit,
            evidence_ids=[sources[0]["evidence_id"]],
        )
        previous = service.corpus.read_record(fixture["target_id"], service.base_commit, PUBLIC)
    for seed in fixture.get("seed_records", []):
        service.corpus.register_record(
            seed["id"],
            seed["kind"],
            seed["body"],
            commit=service.base_commit,
            evidence_ids=[sources[-1]["evidence_id"]],
        )
    targets = (
        [previous["record_id"]]
        if previous
        else [record["id"] for record in fixture.get("seed_records", [])]
    )
    event = service.create_event(
        fixture["event_type"],
        [source["evidence_id"] for source in sources],
        targets,
        fixture["reason"],
    )
    proposal = _proposal(fixture, sources, previous)
    decision = (
        "request_changes"
        if fixture.get("repair_body")
        else "needs_evidence"
        if fixture["category"] == "insufficient_evidence"
        else "approve"
    )
    scripts = {
        "propose": _script(fixture, event, sources, proposal, role="proposer", revision=1),
        "review": _script(
            fixture, event, sources, proposal, role="reviewer", revision=1, decision=decision
        ),
    }
    if fixture.get("repair_body"):
        repaired = _proposal(fixture, sources, previous, repaired=True)
        scripts.update(
            repair=_script(fixture, event, sources, repaired, role="proposer", revision=2),
            rereview=_script(fixture, event, sources, repaired, role="reviewer", revision=2),
        )
    return event, scripts


def _permission_check(service, fixture, output):
    text = fixture["source_texts"][0]
    private = service.corpus.register_evidence(
        text.encode("utf-8"),
        origin_id="fixture:private",
        scope_id="team-secret",
        classification="team_private",
    )
    service.corpus.register_record(
        fixture["target_id"],
        "case",
        fixture["body"],
        commit=service.base_commit,
        scope_id="team-secret",
        classification="team_private",
        evidence_ids=[private["evidence_id"]],
    )
    tools = service.corpus.tools("reviewer", service.base_commit, PUBLIC)
    checks = {
        "search_evidence_hidden": tools.call("search_evidence", {"query": fixture["search_query"]})[
            "data"
        ]["total"]
        == 0
    }
    for name, args in [
        (
            "read_evidence",
            {
                "evidence_id": private["evidence_id"],
                "text_sha256": private["text_sha256"],
                "start": 0,
                "end": len(text),
            },
        ),
        ("read_record", {"record_id": fixture["target_id"], "record_version": 1}),
    ]:
        try:
            tools.call(name, args)
            checks[name] = "UNEXPECTED_ALLOWED"
        except ValueError as error:
            checks[name] = str(error)
    try:
        service.create_event("new_source", [private["evidence_id"]], [], fixture["reason"])
        checks["service_admission"] = "UNEXPECTED_ALLOWED"
    except ValueError as error:
        checks["service_admission"] = str(error)
    expected = {
        "search_evidence_hidden": True,
        "read_evidence": "NOT_AUTHORIZED",
        "read_record": "NOT_AUTHORIZED",
        "service_admission": "PRIVATE_MAINTENANCE_NOT_ADMITTED",
    }
    path = _write(
        output / "artifacts" / (fixture["id"] + "-permission.json"),
        {"checks": checks, "tools": tools.trace},
    )
    return {
        "fixture_id": fixture["id"],
        "category": fixture["category"],
        "expected_status": fixture["expected_status"],
        "status": "authorization_rejected" if checks == expected else "failed",
        "checks": checks,
        "trace_path": path,
        "requests": 0,
        "repair_rounds": 0,
        "tool_actions": len(tools.trace),
        "publication_allowed": False,
        "execution_mode": "direct_scoped_tool_check",
        "event_id": None,
    }


def _event_report(service, fixture, event, outcome, traces, output, ledger_rows):
    runs = [row["payload"] for row in service.store.list_review_runs(event["id"])]
    trusted = {run["run_id"]: run for run in runs}
    event_traces = [
        trace for trace in traces if trace["value"]["task"]["input"]["event"]["id"] == event["id"]
    ]
    event_traces.sort(
        key=lambda trace: (
            trace["value"]["task"]["revision"],
            trace["value"]["task"]["role"] == "reviewer",
        )
    )
    revisions = sorted({run["revision"] for run in runs})
    changesets = [
        service.store.get_changeset(event["id"], revision)["payload"] for revision in revisions
    ]
    paths = [
        _write(
            output
            / "artifacts"
            / (fixture["id"] + "-revision-" + str(value["revision"]) + ".json"),
            value,
        )
        for value in changesets
    ]
    receipt = service.store.get_receipt(event["id"], revisions[-1]) if revisions else None
    receipt_path, gate = None, None
    if receipt:
        receipt = receipt["payload"]
        receipt_path = _write(output / "artifacts" / (fixture["id"] + "-receipt.json"), receipt)
        try:
            require_approved(
                changesets[-1],
                receipt,
                trusted_runs=trusted,
                key_resolver={"maintenance-local-v1": (service.root / "receipt.key").read_bytes()},
                policy=service.policy,
                expected_base=service.base_commit,
                expected_bindings=changesets[-1]["bindings"],
                evidence_resolver=service.corpus.resolve_evidence,
                business_validator=business_validator,
            )
            gate = "UNEXPECTED_ALLOWED"
        except ValueError as error:
            gate = getattr(error, "code", str(error))
    decisions = []
    for trace in event_traces:
        if trace["value"]["task"]["role"] == "reviewer":
            actual = trace["value"]["result"]["output"]
            decisions.append(actual.get("decision") if isinstance(actual, dict) else None)
    requests = [row for row in ledger_rows if row["event_id"] == event["id"]]
    return {
        **outcome,
        "fixture_id": fixture["id"],
        "category": fixture["category"],
        "expected_status": fixture["expected_status"],
        "revisions": len(changesets),
        "repair_rounds": max(0, len(changesets) - 1),
        "review_decisions": decisions,
        "requests": len(requests),
        "requests_by_stage": dict(Counter(row["stage"].split(":")[-1] for row in requests)),
        "trace_paths": [trace["path"] for trace in event_traces],
        "changeset_paths": paths,
        "receipt_path": receipt_path,
        "production_gate": gate,
        "tool_actions": sum(len(trace["value"]["tools"]) for trace in event_traces),
        "base_knowledge_commit": service.base_commit,
    }


def run_replay(output, *, suite="smoke", manifest_path=MANIFEST):
    """Run real host/Node fixture calls and return only observed engineering results."""
    if suite not in {"smoke", "full"}:
        raise ValueError("INVALID_REPLAY_SUITE")
    output = Path(output).resolve()
    # A fresh output avoids presenting cached jobs as new replay evidence.
    if output.exists() and any(output.iterdir()):
        raise ValueError("REPLAY_OUTPUT_MUST_BE_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    if manifest.get("example_only") is not True or any(
        item.get("example_only") is not True for item in manifest["events"]
    ):
        raise ValueError("SYNTHETIC_FIXTURES_REQUIRED")
    fixtures = [
        item
        for item in manifest["events"]
        if suite == "full" or item["id"] in manifest["smoke_ids"]
    ]
    service = MaintenanceService(output / "maintenance")
    registered, scripts, permissions = [], {}, []
    for fixture in fixtures:
        if fixture["category"] == "permission":
            permissions.append((fixture, _permission_check(service, fixture, output)))
        else:
            event, event_scripts = _register(service, fixture)
            registered.append((fixture, event))
            scripts[event["id"]] = event_scripts
    baseline = canonical_bytes(service.corpus.records(service.base_commit))
    report = service.run_events(
        [event["id"] for _, event in registered], execution_mode="fixture", scripts=scripts
    )
    report_path = service.root / "reports" / (report["batch_id"] + ".json")
    ledger_path = service.root / "budgets" / (report["batch_id"] + ".sqlite3")
    with sqlite3.connect(ledger_path) as db:
        db.row_factory = sqlite3.Row
        ledger_rows = [
            dict(row)
            for row in db.execute(
                "SELECT request_key,event_id,stage,role,state,actual_micro "
                "FROM requests ORDER BY created_at,ticket_id"
            )
        ]
    traces = [
        {"path": str(path), "value": json.loads(path.read_text(encoding="utf-8"))}
        for path in sorted((service.root / "traces").glob("*.json"))
    ]
    outcomes = {item["event_id"]: item for item in report["items"]}
    items = [
        _event_report(service, fixture, event, outcomes[event["id"]], traces, output, ledger_rows)
        for fixture, event in registered
    ] + [item for _, item in permissions]
    order = {fixture["id"]: index for index, fixture in enumerate(fixtures)}
    items.sort(key=lambda item: order[item["fixture_id"]])
    unchanged = baseline == canonical_bytes(service.corpus.records(service.base_commit))
    expectations = all(item["status"] == item["expected_status"] for item in items)
    expectations &= all(
        item.get("production_gate") == "FIXTURE_FORBIDDEN"
        for item in items
        if item["status"] == "engineering_complete"
    )
    expectations &= unchanged and len(ledger_rows) == report["budget"]["requests"]
    physical = sum(
        trace["value"]["result"]["resources"]["physical_provider_requests"] for trace in traces
    )
    settled = all(row["state"] == "settled" and row["actual_micro"] == 0 for row in ledger_rows)
    expectations &= (
        settled
        and physical == 0
        and all(trace["value"]["result"]["stop_reason"] == "completed" for trace in traces)
    )
    summary = {
        "schema": "finrisk-maintenance-engineering-replay/1.0",
        "suite": suite,
        "example_only": True,
        "expectations_are": "scripted_engineering_expectations_not_model_quality",
        "execution_mode": "fixture",
        "publication_allowed": False,
        "expectations_met": bool(expectations),
        "items": items,
        "status_counts": dict(Counter(item["status"] for item in items)),
        "requests": report["budget"]["requests"],
        "physical_provider_requests": physical,
        "ledger_states": dict(Counter(row["state"] for row in ledger_rows)),
        "all_requests_settled": settled,
        "settled_micro": report["budget"]["settled_micro"],
        "currency": "CNY",
        "remaining_requests": report["budget"]["max_requests"] - report["budget"]["requests"],
        "remaining_micro": report["budget"]["available_micro"],
        "budget": report["budget"],
        "ledger_path": str(ledger_path),
        "service_report": str(report_path),
        "canonical_records_unchanged": unchanged,
        "knowledge_commit": service.base_commit,
        "trace_count": len(traces),
        "tool_actions": sum(len(trace["value"]["tools"]) for trace in traces),
        "direct_permission_tool_actions": sum(item["tool_actions"] for _, item in permissions),
        "max_repair_rounds": service.policy["max_repair_rounds"],
        "limitations": {
            "real_semantics_evaluated": False,
            "retrieval_regression_executed": False,
            "live_provider_executed": False,
            "publication_executed": False,
            "scripted_review_is_model_effect": False,
        },
        "summary_path": str(output / "summary.json"),
    }
    _write(output / "ledger-requests.json", ledger_rows)
    _write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="零收费合成夹具工程回放；不评估真实模型语义或发布。"
    )
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, required=True, help="必须为空的新回放目录")
    args = parser.parse_args()
    report = run_replay(args.output, suite=args.suite)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "suite",
                    "status_counts",
                    "expectations_met",
                    "requests",
                    "physical_provider_requests",
                    "settled_micro",
                    "remaining_requests",
                    "remaining_micro",
                    "summary_path",
                )
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0 if report["expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
