import importlib.util
import json
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[2]


def api(name="experiment"):
    path = P / "scripts" / "agentic_v3" / (name + ".py")
    assert path.exists(), "new loop experiment implementation is required"
    spec = importlib.util.spec_from_file_location("loop_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schedule_joins_by_id_and_preserves_all_planned_arms():
    m = api()
    qs = [{"sample_id": str(i), "text": "材料", "user_context": {}} for i in range(8)]
    ids = ["S02", "C15", "S01", "C07", "C18", "C19", "C06", "C01"]
    refs = [
        {
            "sample_id": str(i),
            "review_id": rid,
            "human_approved": True,
            "reference": {"supported_tags": []},
            "difficulty": "simple" if i < 2 else "complex",
        }
        for i, rid in enumerate(ids)
    ]
    tasks = m.schedule("development", qs[::-1], refs)
    assert len(tasks) == 24
    assert tasks[0]["input"]["sample_id"] == "0"
    assert [t["arm"] for t in tasks[:6]] == ["R1", "R2", "A", "R2", "A", "R1"]
    rows = m.aggregate(tasks, {})
    assert all(row["denominator"] == 8 for row in rows["arms"].values())
    assert rows["arms"]["A"]["correct_valid"] == 0


def test_invalid_empty_gold_cannot_score_as_correct():
    m = api()
    task = {
        "task_key": "N:R1",
        "arm": "R1",
        "review_id": "N",
        "difficulty": "simple",
        "reference": {"supported_tags": [], "overall_judgment": "no_clear_risk"},
    }
    invalid = {
        "business_valid": False,
        "report": {"overall_judgment": "no_clear_risk", "risk_points": []},
    }
    assert m.aggregate([task], {"N:R1": invalid})["arms"]["R1"]["correct_valid"] == 0


def test_query_unicode_limit_refuses_instead_of_truncating():
    m = api()
    assert "😀" in m.evaluation_query({"text": "😀", "user_context": {}})
    with pytest.raises(ValueError, match="QUERY_TOO_LONG"):
        m.evaluation_query({"text": "字" * 501, "user_context": {}})


def test_review_real_approval_binding_and_new_ids(tmp_path):
    m = api("review")
    samples = [
        {
            "review_id": f"N{i:02}",
            "difficulty": "simple" if i <= 4 else "complex",
            "family_id": str(i),
            "text": "话术",
            "user_context": {},
            "proposed_reference": {"supported_tags": [], "overall_judgment": "normal"},
        }
        for i in range(1, 11)
    ]
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"samples": samples}), encoding="utf8")
    approval = tmp_path / "approval.json"
    approved = {
        "authority": "human_user_in_current_conversation",
        "decision": "accept_all_10_proposals",
        "statement": "认可N01–N10全部参照",
        "review_ids": [s["review_id"] for s in samples],
        "source_proposal_sha256": m.file_sha(proposal),
    }
    approval.write_text(json.dumps(approved), encoding="utf8")
    b = m.make_review_bundle(proposal, approval, expected_sha=m.file_sha(proposal))
    assert len({r["sample_id"] for r in b["references"]}) == 10
    assert b["source_approval_sha256"] == m.file_sha(approval)
    assert b["bundle_sha256"] == m.obj_sha({k: v for k, v in b.items() if k != "bundle_sha256"})
    approved["source_proposal_sha256"] = "0" * 64
    approval.write_text(json.dumps(approved), encoding="utf8")
    with pytest.raises(ValueError, match="APPROVAL"):
        m.make_review_bundle(proposal, approval, expected_sha=m.file_sha(proposal))


def register():
    return {
        "schema_version": "finrisk-loop-stage-budget-register/1.0",
        "hard_cap_micro_cny": 40000000,
        "stage_limits_micro_cny": {
            "smoke": 3000000,
            "development": 15000000,
            "independent": 20000000,
        },
        "known_micro_cny": 0,
        "unknown_reserved_micro_cny": 0,
        "executions": [],
    }


def test_stage_diagnostic_rounds_share_cap_and_unknown_is_not_refunded(tmp_path):
    m = api()
    path = tmp_path / "stage.json"
    m.save(path, register())
    budget = m.StageBudget(path, "smoke", "plan1", 3000000)
    budget.reserve("one")
    budget.finish("one", 0, 1000000)
    with pytest.raises(ValueError, match="BUDGET"):
        m.StageBudget(path, "smoke", "plan2", 3000000).reserve("two")
    with pytest.raises(ValueError, match="SETTLED"):
        budget.finish("one", 1, 0)
    assert m.load(path)["unknown_reserved_micro_cny"] == 1000000


def test_execution_is_exclusive_and_freeze_checks_are_read_only(tmp_path):
    m = api()
    assert m.create_execution(tmp_path / "execution").is_dir()
    with pytest.raises(ValueError, match="ALREADY_STARTED"):
        m.create_execution(tmp_path / "execution")
    f = tmp_path / "frozen.txt"
    f.write_text("old")
    rows = [{"path": str(f), "sha256": m.file_sha(f)}]
    m.verify_files(rows)
    f.write_text("new")
    with pytest.raises(ValueError, match="FROZEN"):
        m.verify_files(rows)


def test_physical_requests_do_not_count_reservations_or_dispatch():
    m = api()
    assert m.physical_counts(
        [{"type": "model.reserved", "data": {}}],
        {"request_reservations": [{"dispatch_committed_at": "now"}]},
    ) == (0, 0)
    assert m.physical_counts(
        [
            {"type": "model.request", "data": {"physical_request": True}},
            {"type": "model.failure", "data": {"physical_requests": 1}},
        ],
        {"request_reservations": [{"transport_started_at": "now"}]},
    ) == (1, 1)


def test_missing_and_malformed_ledger_remain_full_unknown(tmp_path):
    m = api()
    assert m.read_costs(tmp_path / "model.json", tmp_path / "rerank.json", {})[:2] == (0, 2500000)


def test_local_partial_can_be_valid_but_global_failure_and_stale_acceptance_cannot():
    m = api()
    report = {
        "task_status": "partial",
        "stop": {"reason": "evidence_sufficient"},
        "coverage": {"unprocessed_segment_ids": []},
        "validation": {
            "schema_valid": True,
            "references_valid": True,
            "semantic_review_status": "model_assessed",
            "errors": [],
        },
        "risk_points": [{"risk_point_id": "RP0001", "supported_tags": [], "tag_checks": []}],
    }
    a = {
        "active_blockers": [],
        "points": [
            {
                "risk_point_id": "RP0001",
                "processing_state": "assessed",
                "accepted_assessment_id": "AS1",
                "state_revision": 1,
                "last_assessed_revision": 1,
                "additional_mechanisms_uncovered": False,
                "retrieval_state": "local_partial",
            }
        ],
    }
    r = {
        "retrieval_status": "partial",
        "points": [
            {"risk_point_id": "RP0001", "assessment_status": "model_assessed_with_retrieval_limits"}
        ],
    }
    assert m.business_valid(report, a, r)[0]
    a["active_blockers"] = [{"code": "STOPPED"}]
    assert not m.business_valid(report, a, r)[0]
    a["active_blockers"] = []
    a["points"][0]["last_assessed_revision"] = 0
    assert not m.business_valid(report, a, r)[0]


def test_experiment_cache_off_and_default_cache_unchanged(tmp_path):
    from test_v2_backend import Dense, Scorer, Snapshot

    from agent_rag.v2_backend import HybridV2Backend

    scorer = Scorer()
    backend = HybridV2Backend(Snapshot(), Dense(), scorer, cache_size=0)
    backend.search("购物卡", 3)
    backend.search("购物卡", 3)
    assert scorer.calls == 2
    assert not backend.cache
    scorer = Scorer()
    backend = HybridV2Backend(Snapshot(), Dense(), scorer)
    backend.search("购物卡", 3)
    backend.search("购物卡", 3)
    assert scorer.calls == 1


def test_pair_cache_off_really_rescores_across_controls_and_cancel(tmp_path):
    from test_rerank_backend import response, unit

    from agent_rag.rerank_backend import DeepSeekReranker, RerankTaskControl

    physical = []

    def send(payload, timeout_seconds):
        physical.append(1)
        return response(payload)

    scorer = DeepSeekReranker("规则", send, cache_size=0)
    c1 = RerankTaskControl(tmp_path / "a.json", cap_cny=1, max_requests=10)
    c2 = RerankTaskControl(tmp_path / "b.json", cap_cny=1, max_requests=10)
    assert scorer.score_query("同查询", [unit()], c1)["complete"]
    c1.cancel()
    assert scorer.score_query("同查询", [unit()], c2)["complete"]
    assert len(physical) == 2
    assert not scorer._cache


def test_physical_failure_fallback_is_deduplicated_per_request():
    m = api()
    events = [
        {"type": "model.request", "data": {"request_id": "m1", "physical_request": True}},
        {"type": "model.failure", "data": {"request_id": "m1", "physical_requests": 1}},
        {"type": "model.failure", "data": {"request_id": "m2", "physical_requests": 1}},
    ]
    assert m.physical_counts(events, {"request_reservations": []}) == (2, 0)


@pytest.mark.parametrize("stop", ["STOPPED", "transport_circuit_open", "circuit_open"])
def test_global_stop_is_invalid_even_with_no_points(stop):
    m = api()
    report = {
        "task_status": "partial",
        "stop": {"reason": stop},
        "coverage": {"all_submitted_segments_reviewed": True},
        "validation": {
            "schema_valid": True,
            "references_valid": True,
            "semantic_review_status": "model_assessed",
            "errors": [],
        },
        "risk_points": [],
    }
    assert not m.business_valid(report, {"active_blockers": [], "points": []}, {"points": []})[0]


def test_query_alone_is_not_a_gain_candidate():
    m = api()
    events = [
        {"event_id": "T:1", "type": "progress.observed", "data": {}},
        {"event_id": "T:2", "type": "gap.derived", "data": {"point_id": "RP1"}},
        {
            "event_id": "T:3",
            "type": "action.proposed",
            "data": {
                "origin": "adaptive_model",
                "action_type": "search_cases",
                "action_id": "A1",
                "query_hash": "new",
                "point_ids": ["RP1"],
                "preceding_observation_event_id": "T:1",
            },
        },
    ]
    assert m.gain_candidates(events) == []


def test_late_cancelled_task_cannot_populate_disabled_cache():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from test_v2_backend import Dense, Scorer, Snapshot

    from agent_rag.v2_backend import HybridV2Backend

    entered, release = threading.Event(), threading.Event()

    class Late(Scorer):
        def score_query(self, query, units, control):
            entered.set()
            assert release.wait(5)
            return super().score_query(query, units, control)

    scorer = Late()
    backend = HybridV2Backend(Snapshot(), Dense(), scorer, cache_size=0)
    with ThreadPoolExecutor(1) as pool:
        late = pool.submit(backend.search, "同查询", 3)
        assert entered.wait(5)
        release.set()
        late.result()
    backend.search("同查询", 3)
    assert scorer.calls == 2 and not backend.cache


def test_python_service_base_configs_match_real_node_description(tmp_path, monkeypatch):
    import shutil

    from test_budget_profiles import live_environment, make_service

    m = api()
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is supplied explicitly by the offline acceptance command")
    live_environment(monkeypatch, tmp_path)
    d = m.describe(P, node)
    for arm in ("R1", "R2", "A"):
        (tmp_path / arm).mkdir()
        service = make_service(tmp_path / arm, "live")
        record, _ = service.admit(
            {
                "text": "材料",
                "experiment_profile_id": "live-deepseek-flash-planned-v1",
                "engine": "fixed_rag" if arm == "R2" else "controlled_agent",
            },
            None,
        )
        base = m.load(service.root / record["task_id"] / "manifest.json")
        assert base["config"] == d["stages"]["development"][arm]["base_config"]
        assert (
            base["config_sha256"]
            == m.service_admission_bindings(d, "development")[arm]["config_sha256"]
        )


def test_corrupt_rerank_attempt_binding_is_conservative_unknown(tmp_path):
    from agent_rag.budget_profiles import budget_profile_sha256, load_budget_profile
    from agent_rag.rerank_backend import RerankTaskControl

    m = api()
    profile = load_budget_profile("planned-evidence-loop-v1")
    digest = budget_profile_sha256(profile)
    path = tmp_path / "rerank.json"
    control = RerankTaskControl(
        path,
        1.5,
        800,
        expected_budget_profile_id=profile["profile_id"],
        expected_budget_profile_sha256=digest,
    )
    ordinal, _ = control._reserve({"model": "deepseek-flash", "max_tokens": 512}, "PAIR")
    control._commit_dispatch(ordinal)
    control._start_transport(ordinal)
    audit = m.load(control.attempt_audit_path)
    audit["attempts"][0]["pair_id"] = "FOREIGN"
    m.save(control.attempt_audit_path, audit)
    known, unknown, detail = m.read_costs(
        tmp_path / "missing-model.json",
        path,
        {"project_root": str(P), "description": {"budget_profile_sha256": digest}},
    )
    assert (known, unknown) == (0, 2500000)
    assert not detail["rerank"]["valid"]


def test_gain_requires_new_necessary_read_accepted_use_and_reference_change():
    m = api()

    def event(n, kind, **data):
        return {"event_id": "T:" + str(n), "type": kind, "data": data}

    events = [
        event(1, "assessment.accepted", assessment_id="AS1", point_id="RP1"),
        event(2, "progress.observed", observation_kind="assessment"),
        event(3, "gap.derived", gap_id="G1", point_id="RP1"),
        event(
            4,
            "action.proposed",
            origin="adaptive_model",
            action_type="search_cases",
            action_id="A1",
            query_hash="NEW",
            point_ids=["RP1"],
            target_gap_ids=["G1"],
            preceding_observation_event_id="T:2",
        ),
        event(5, "action.completed", action_id="A1", status="completed"),
        event(
            6,
            "progress.observed",
            observation_kind="source_read",
            evidence_gain=True,
            affected_point_ids=["RP1"],
            evidence_ids=["NEW-E"],
            windows=[{"anchor_evidence_id": "ANCHOR"}],
        ),
        event(
            7,
            "assessment.accepted",
            assessment_id="AS2",
            point_id="RP1",
            used_evidence_ids=["NEW-E"],
        ),
    ]
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {"kind": "assess", "assessment_id": "AS1", "tag_checks": []}
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "kind": "assess",
                            "assessment_id": "AS2",
                            "tag_checks": [{"tag": "赠送利益", "status": "supported"}],
                        }
                    ),
                }
            ],
        },
    ]
    found = m.gain_candidates(events, messages, ["赠送利益"])
    assert len(found) == 1 and found[0]["confirmed_benefit"] is False
    assert found[0]["case_applicability"] == "pending_human_review"
    events[5]["data"]["evidence_gain"] = False
    assert m.gain_candidates(events, messages, ["赠送利益"]) == []


def test_execute_key_isolation_and_restore_without_saving_environment(tmp_path, monkeypatch):
    import agent_rag.v2_runtime as runtime

    m = api()
    plan_path = tmp_path / "plan.json"
    m.save(plan_path, {})
    register_path = tmp_path / "register.json"
    m.save(register_path, register())
    key_path = tmp_path / "synthetic-key.txt"
    key_path.write_text("offline-only-synthetic")
    plan = {
        "project_root": str(P),
        "stage": "smoke",
        "node": "offline-node",
        "round_cap_micro_cny": 3000000,
        "tasks": [],
        "stage_budget_register": str(register_path),
        "snapshot_dir": str(tmp_path / "frozen-snapshot"),
    }
    monkeypatch.setattr(m, "preflight", lambda p: plan)
    plan["node"] = str(Path("offline-node").resolve())
    authorization = tmp_path / "authorization.json"
    m.save(authorization, m.authorization_record(plan_path, plan))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-inherited-sentinel")
    before = dict(__import__("os").environ)

    def forbidden_load(root, **kwargs):
        import os

        assert "DEEPSEEK_API_KEY" not in os.environ
        assert os.environ["DEEPSEEK_API_KEY_FILE"] == str(key_path.resolve())
        assert os.environ["FINRISK_AGENT_RERANK_CAP_CNY"] == "1.5"
        assert os.environ["FINRISK_AGENT_RERANK_MAX_REQUESTS"] == "800"
        assert kwargs["result_cache_size"] == kwargs["rerank_cache_size"] == 0
        assert kwargs["audit_root"] == tmp_path / "execution" / "retrieval-audits"
        raise ValueError("OFFLINE_SENTINEL_NO_GPU_NO_API")

    monkeypatch.setattr(runtime, "load_v2_snapshot", forbidden_load)
    with pytest.raises(ValueError, match="OFFLINE_SENTINEL"):
        m.execute(plan_path, authorization, plan["node"], key_path)
    assert dict(__import__("os").environ) == before
    assert "synthetic-inherited-sentinel" not in "".join(
        p.read_text() for p in (tmp_path / "execution").rglob("*.json")
    )


def test_cancelled_late_pair_cannot_contaminate_next_task(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from test_rerank_backend import response, unit

    from agent_rag.rerank_backend import DeepSeekReranker, RerankTaskControl

    entered, release = threading.Event(), threading.Event()
    physical = []

    def transport(payload, timeout):
        physical.append(1)
        if len(physical) == 1:
            entered.set()
            assert release.wait(5)
        return response(payload)

    scorer = DeepSeekReranker("原规则", transport, cache_size=0)
    controls = [RerankTaskControl(tmp_path / (str(i) + ".json"), 1, 10) for i in range(3)]
    with ThreadPoolExecutor(1) as pool:
        late = pool.submit(scorer.score_query, "同query", [unit()], controls[0])
        assert entered.wait(5)
        controls[0].cancel()
        assert scorer.score_query("同query", [unit()], controls[1])["complete"]
        release.set()
        late.result()
    assert scorer.score_query("同query", [unit()], controls[2])["complete"]
    assert len(physical) == 3 and not scorer._cache


def test_score_of_unstarted_plan_keeps_denominator_without_creating_execution(
    tmp_path, monkeypatch
):
    import agent_rag.v2_snapshot as snapshots

    m = api()
    qs = [{"sample_id": str(i), "text": "材料", "user_context": {}} for i in range(3)]
    refs = [
        {
            "sample_id": str(i),
            "review_id": rid,
            "human_approved": True,
            "difficulty": "simple" if i == 0 else "complex",
            "reference": {"supported_tags": []},
        }
        for i, rid in enumerate(("S01", "C15", "C01"))
    ]
    plan_path = tmp_path / "plan.json"
    m.save(plan_path, {})
    register_path = tmp_path / "register.json"
    m.save(register_path, register())
    plan = {
        "project_root": str(P),
        "stage": "smoke",
        "tasks": m.schedule("smoke", qs, refs),
        "stage_budget_register": str(register_path),
        "round_cap_micro_cny": 3000000,
        "snapshot_dir": str(tmp_path),
        "evaluation_protocol_sha256": "0" * 64,
    }
    monkeypatch.setattr(m, "preflight", lambda p: plan)
    monkeypatch.setattr(snapshots, "V2KnowledgeSnapshot", lambda p: None)
    result = m.score(plan_path)
    assert result["arms"]["A"]["denominator"] == 3
    assert result["arms"]["A"]["correct_valid"] == 0
    assert not (tmp_path / "execution").exists()


def test_invalid_row_keeps_unverified_original_artifacts_in_review(tmp_path, monkeypatch):
    import agent_rag.v2_snapshot as snapshots

    m = api()
    qs = [{"sample_id": str(i), "text": "offline material", "user_context": {}} for i in range(3)]
    refs = [
        {
            "sample_id": str(i),
            "review_id": rid,
            "human_approved": True,
            "difficulty": "simple" if i == 0 else "complex",
            "reference": {"supported_tags": []},
        }
        for i, rid in enumerate(("S01", "C15", "C01"))
    ]
    register_path, plan_path = tmp_path / "register.json", tmp_path / "plan.json"
    m.save(register_path, register())
    plan = {
        "project_root": str(P),
        "stage": "smoke",
        "tasks": m.schedule("smoke", qs, refs),
        "stage_budget_register": str(register_path),
        "round_cap_micro_cny": 3000000,
        "snapshot_dir": str(tmp_path),
        "evaluation_protocol_sha256": "0" * 64,
    }
    m.save(plan_path, plan)
    monkeypatch.setattr(m, "preflight", lambda p: plan)
    monkeypatch.setattr(snapshots, "V2KnowledgeSnapshot", lambda p: None)
    output = tmp_path / "execution"
    directory = output / "tasks" / ("a" * 32)
    directory.mkdir(parents=True)
    m.save(output / "authorization.json", m.authorization_record(plan_path, plan))
    task = plan["tasks"][0]
    report = {"marker": "RAW_REPORT_EXISTS", "stop": {"reason": "RAW_STOP_EXISTS"}}
    m.save(directory / "report.json", report)
    m.save(output / "reports" / "S01_A.json", {"different": True})
    m.save(directory / "model-response-M1.json", {"reply": "RAW_REPLY_EXISTS"})
    (directory / "model_messages.jsonl").write_text("RAW_UNPARSABLE_ASSISTANT", encoding="utf8")
    (directory / "events.jsonl").write_text(
        json.dumps({"type": "RAW_TRACE_EXISTS"}) + "\n", encoding="utf8"
    )
    m.save(
        output / "S01_A.json",
        {
            "plan_sha256": m.file_sha(plan_path),
            "task_key": task["task_key"],
            "task_id": "a" * 32,
            "report": report,
            "known_micro_cny": 123,
            "unknown_micro_cny": 456,
            "cost_detail": {"marker": "RAW_COST_EXISTS"},
        },
    )
    result = m.score(plan_path)
    row = m.load(output / "review-data.json")["records"][task["task_key"]]
    assert row["audit_status"] == "invalid" and not row["business_valid"]
    assert row["error"] == "REPORT_COPY_OR_RESULT_MISMATCH"
    assert "report" not in row  # Raw evidence must never become accepted scoring input.
    assert result["arms"]["A"]["denominator"] == 3
    assert result["arms"]["A"]["correct_valid"] == 0
    raw = row["raw_evidence"]
    assert raw["status"] == "raw_unverified"
    assert raw["files"]["report"]["path"] == str(directory / "report.json")
    assert raw["files"]["report"]["status"] == "available"
    assert raw["files"]["model_messages"]["status"] == "parse_error"
    assert raw["files"]["rerank_ledger"]["status"] == "missing"
    assert raw["ledger_cost_review"]["known_micro_cny"] == 0
    assert raw["ledger_cost_review"]["unknown_micro_cny"] == 2500000
    assert m.load(register_path) == register()
    rendered = (output / "review.html").read_text(encoding="utf8")
    assert "原始/未验收" in rendered
    for marker in (
        "RAW_REPORT_EXISTS",
        "RAW_STOP_EXISTS",
        "RAW_REPLY_EXISTS",
        "RAW_TRACE_EXISTS",
        "RAW_COST_EXISTS",
        "RAW_UNPARSABLE_ASSISTANT",
    ):
        assert marker in rendered


@pytest.mark.parametrize("stage", ["development", "dev"])
def test_prepare_cli_accepts_development_and_dev_alias(stage, monkeypatch, capsys):
    m = api()
    calls = []
    monkeypatch.setattr(m, "prepare", lambda **kwargs: calls.append(kwargs) or {"offline": True})
    monkeypatch.setattr(
        m.sys,
        "argv",
        [
            "experiment.py",
            "prepare",
            "--project-root",
            str(P),
            "--stage",
            stage,
            "--output",
            "not-created.json",
            "--round-cap-cny",
            "15",
            "--fast-path",
            "on",
            "--node",
            "offline-node",
            "--stage-budget-register",
            "stage.json",
        ],
    )
    m.main()
    assert calls[0]["stage"] == stage
    assert calls[0]["round_cap_cny"] == "15"
    assert json.loads(capsys.readouterr().out) == {"offline": True}


def test_execute_timing_includes_admission_and_excludes_report_cleanup(tmp_path):
    import os
    from types import SimpleNamespace
    from unittest.mock import patch

    import uvicorn

    import agent_rag.task_service as ts
    import agent_rag.v2_runtime as vr

    m = api()
    root = tmp_path / "timing"
    root.mkdir(exist_ok=False)
    plan_path, auth_path = root / "plan.json", root / "authorization.json"
    key = root / "synthetic-placeholder.txt"
    key.write_text("offline synthetic placeholder; never read", encoding="utf8")
    task_id = "a" * 32
    clock = {"now": 0.0}
    times = {}
    base = {"config": {}, "config_sha256": "base"}
    binding = {
        "config": {},
        "planned_execution_contract": {},
        "base_config_node_sha256": "node",
        "config_sha256": "c" * 64,
    }
    plan = {
        "project_root": str(P),
        "node": str((root / "offline-node").resolve()),
        "stage": "smoke",
        "round_cap_micro_cny": 3000000,
        "stage_budget_register": str(root / "register.json"),
        "snapshot_dir": str(root / "snapshot"),
        "runtime_snapshot_projection": {},
        "evaluation_protocol_sha256": "protocol",
        "service_admission_bindings": {"A": base},
        "description": {"stages": {"smoke": {"A": binding}}},
        "tasks": [
            {
                "task_key": "S01:A",
                "arm": "A",
                "input": {"text": "离线", "user_context": {}},
                "input_sha256": "input",
            }
        ],
    }
    m.save(plan_path, plan)
    m.save(auth_path, m.authorization_record(plan_path, plan))
    m.save(
        root / "register.json",
        {
            "schema_version": "finrisk-loop-stage-budget-register/1.0",
            "hard_cap_micro_cny": 40000000,
            "stage_limits_micro_cny": m.STAGE_LIMITS,
            "known_micro_cny": 0,
            "unknown_reserved_micro_cny": 0,
            "executions": [],
        },
    )

    class Service:
        def __init__(self, root, snapshot):
            self.root, self.records, self.controls = Path(root), {}, {}
            self.active_task_id = None

        def admit(self, request, key):
            times["submission"] = clock["now"]
            clock["now"] += 7
            directory = self.root / task_id
            m.save(
                directory / "manifest.json",
                {"config": {}, "config_sha256": "base", "input_sha256": "input", "snapshot": {}},
            )
            self.records[task_id] = {"task_id": task_id, "binding_sha256": "admission"}
            return self.records[task_id], False

        def bundle(self, unused):
            clock["now"] += 3
            return {"input": {"task_id": task_id}}

        def cancel_retrieval(self, unused):
            clock["now"] += 5
            times["cancel_return"] = clock["now"]

    class Socket:
        def bind(self, address):
            pass

        def listen(self, count):
            pass

        def getsockname(self):
            return ("127.0.0.1", 12345)

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

    class Circuit:
        def __init__(self, path):
            pass

        def before_task(self):
            pass

        def observe(self, *args):
            pass

    def load_snapshot(*args, **kwargs):
        assert kwargs["result_cache_size"] == kwargs["rerank_cache_size"] == 0
        clock["now"] += 100  # shared warmup must remain outside per-task timing
        return SimpleNamespace(manifest={})

    def run_cli(*args, **kwargs):
        times["cli_launch"] = clock["now"]
        clock["now"] += 11
        directory = root / "execution" / "tasks" / task_id
        m.save(directory / "report.json", {"task_status": "completed"})
        m.save(directory / "rerank-ledger.json", {"request_reservations": []})
        times["report_persisted"] = clock["now"]
        receipt = {
            "schema_version": "finrisk-loop-report-persistence/1.0",
            "task_id": task_id,
            "input_sha256": "input",
            "profile_sha256": "c" * 64,
            "report_sha256": m.file_sha(directory / "report.json"),
            "persisted_at_epoch_ms": int(clock["now"] * 1000),
        }
        (directory / "events.jsonl").write_text(
            json.dumps({"type": "report.persisted", "event_id": task_id + ":1", "data": receipt})
            + "\n",
            encoding="utf8",
        )
        return SimpleNamespace(returncode=0)

    def cleanup(service, snapshot, server, thread, sock, old_env):
        os.environ.clear()
        os.environ.update(old_env)
        return []

    infra = SimpleNamespace(
        RoundCircuit=Circuit, local_tool_app=lambda service: None, cleanup_resources=cleanup
    )
    with (
        patch.object(m, "preflight", return_value=plan),
        patch.object(m, "module_file", return_value=infra),
        patch.object(m.time, "monotonic", side_effect=lambda: clock["now"]),
        patch.object(m.time, "time", side_effect=lambda: clock["now"]),
        patch.object(vr, "load_v2_snapshot", side_effect=load_snapshot),
        patch.object(ts, "TaskService", Service),
        patch.object(uvicorn, "Config", return_value=None),
        patch.object(
            uvicorn, "Server", return_value=SimpleNamespace(started=True, run=lambda **kwargs: None)
        ),
        patch.object(m.threading, "Thread", Thread),
        patch.object(m.socket, "socket", Socket),
        patch.object(m.subprocess, "run", side_effect=run_cli),
        patch.object(m, "read_costs", return_value=(0, 0, {})),
    ):
        m.execute(plan_path, auth_path, plan["node"], key)

    record = m.load(root / "execution" / "S01_A.json")
    assert record["wall_seconds"] == 16
    assert record["task_e2e_seconds"] == 21
    assert record["task_timing"]["status"] == "verified"
    assert record["task_timing"]["cli_to_report_seconds"] == 11
    assert record["task_timing"]["post_report_cleanup_seconds"] == 5
    assert record["task_timing"]["cli_wall_seconds"] == 11
    assert m.load(root / "execution" / "warmup.json")["model_gpu_warmup_seconds"] == 100


def test_report_timing_uses_last_receipt_and_rejects_bad_clock_or_identity():
    m = api()
    anchors = {
        name: {"epoch_ms": value * 1000, "monotonic_ms": value * 1000}
        for name, value in (
            ("submitted", 100),
            ("cli_started", 110),
            ("cli_returned", 125),
            ("cleanup_completed", 130),
        )
    }

    def receipt(n, at, digest):
        return {
            "type": "report.persisted",
            "event_id": "a" * 32 + ":" + str(n),
            "data": {
                "schema_version": "finrisk-loop-report-persistence/1.0",
                "task_id": "a" * 32,
                "input_sha256": "input",
                "profile_sha256": "profile",
                "report_sha256": digest,
                "persisted_at_epoch_ms": at * 1000,
            },
        }

    events = [receipt(1, 121, "a" * 64), receipt(2, 123, "b" * 64)]
    result = m.report_timing("a" * 32, "input", "profile", "b" * 64, events, anchors)
    assert result["status"] == "verified" and result["task_e2e_seconds"] == 23
    assert result["receipt_event_id"].endswith(":2")
    for bad_events in ([], events[:1], events + [receipt(3, 124, "a" * 64)]):
        invalid = m.report_timing("a" * 32, "input", "profile", "b" * 64, bad_events, anchors)
        assert invalid["status"] == "unverifiable" and invalid["task_e2e_seconds"] is None
    anchors["cli_returned"]["epoch_ms"] = 99000
    invalid = m.report_timing("a" * 32, "input", "profile", "b" * 64, events, anchors)
    assert invalid["status"] == "unverifiable" and invalid["task_e2e_seconds"] is None


def test_score_summary_uses_only_verified_submission_to_report_time():
    m = api()
    task = {
        "task_key": "N:A",
        "arm": "A",
        "review_id": "N",
        "difficulty": "simple",
        "reference": {"supported_tags": [], "overall_judgment": "normal"},
    }
    record = {
        "business_valid": True,
        "wall_seconds": 1,
        "task_e2e_seconds": 21,
        "task_timing": {"status": "verified", "task_e2e_seconds": 21},
        "report": {
            "task_status": "completed",
            "overall_judgment": "no_clear_risk",
            "risk_points": [],
        },
    }
    assert m.aggregate([task], {"N:A": record})["arms"]["A"]["median_seconds"] == 21
    record.pop("task_timing")
    summary = m.aggregate([task], {"N:A": record})
    assert summary["arms"]["A"]["median_seconds"] is None
    assert summary["rows"][0]["wall_seconds"] == 1



@pytest.mark.parametrize(("a_seconds", "missing_receipt", "expected"), [
    ([1, 10, 20, 20], None, False),
    ([1, 10, 12, 14], "A", False),
    ([1, 10, 12, 14], "R1", False),
    ([1, 10, 12, 14], None, True),
])
def test_simple_speed_uses_only_correct_valid_timings(
    tmp_path, monkeypatch, a_seconds, missing_receipt, expected
):
    import statistics

    import agent_rag.v2_snapshot as snapshots

    m = api()
    tasks, records = [], {}
    for arm in ("A", "R1"):
        for index, seconds in enumerate(a_seconds if arm == "A" else [12] * 4):
            key = f"N{index}:{arm}"
            tasks.append({
                "task_key": key, "review_id": f"N{index}", "arm": arm,
                "difficulty": "simple", "reference": {"supported_tags": ["赠送利益"]},
            })
            records[key] = {
                "audit_status": "valid", "business_valid": True,
                "model_physical_requests": 1, "query_count": 1,
                "task_e2e_seconds": seconds,
                "task_timing": {"status": "verified", "task_e2e_seconds": seconds},
                "report": {
                    "task_status": "completed",
                    "risk_points": [{"supported_tags": ["赠送利益"] if index else []}],
                },
            }
    if missing_receipt:
        records[f"N3:{missing_receipt}"].pop("task_timing")
    plan_path, register_path = tmp_path / "plan.json", tmp_path / "register.json"
    m.save(plan_path, {})
    m.save(register_path, register())
    plan = {
        "project_root": str(P), "snapshot_dir": str(tmp_path), "stage": "independent",
        "tasks": tasks, "stage_budget_register": str(register_path),
        "round_cap_micro_cny": 20000000, "evaluation_protocol_sha256": "0" * 64,
    }
    m.save(tmp_path / "execution/authorization.json", m.authorization_record(plan_path, plan))
    monkeypatch.setattr(m, "preflight", lambda p: plan)
    monkeypatch.setattr(snapshots, "V2KnowledgeSnapshot", lambda p: None)
    monkeypatch.setattr(m, "audit_task", lambda p, pp, task, s, e: records[task["task_key"]])
    result = m.score(plan_path, write=False)
    assert result["independent_targets"]["simple_pass"] is expected
    assert result["independent_targets"]["simple_correct_valid_one_model_one_query"] == 3
    for arm in ("A", "R1"):
        assert result["arms"][arm]["denominator"] == 4
        assert result["arms"][arm]["correct_valid"] == 3
        rows = [r for r in result["rows"] if r["arm"] == arm]
        assert [r["correct_valid"] for r in rows] == [False, True, True, True]
        measured = [r["task_e2e_seconds"] for r in rows if r["task_e2e_seconds"] is not None]
        assert result["arms"][arm]["median_seconds"] == statistics.median(measured)
        assert result["arms"][arm]["mean_seconds"] == statistics.mean(measured)
        assert result["arms"][arm]["slowest_seconds"] == max(measured)


def test_simple_speed_population_rule_is_frozen():
    m = api()
    assert m.STATISTICS_RULES["timing"].get("simple_speed_population") == (
        "per-arm correct_valid simple rows; every included row requires verified e2e receipt"
    )
    assert "correct-valid A/R1 only" in m.STATISTICS_RULES["simple_target"]


@pytest.fixture
def nonempty_audit_case(tmp_path, monkeypatch):
    import shutil
    import subprocess

    from test_budget_profiles import live_environment
    from test_v2_service import make_service

    from agent_rag.task_service import TaskService

    m = api()
    node = shutil.which("node")
    if node is None:
        pytest.skip("offline contract integration requires Node on PATH")
    live_environment(monkeypatch, tmp_path)
    source = make_service(tmp_path, "live")
    source.snapshot.manifest["rerank_protocol"] = "span-ref-v2"
    service = TaskService(tmp_path / "execution/tasks", source.snapshot)
    text = "购买这份保险就送购物卡。"
    context = {"purpose_hint": "当前保险营销宣传"}
    record, _ = service.admit(
        {"text": text, "user_context": context, "experiment_profile_id": m.PROFILE}, None
    )
    task_id = record["task_id"]
    description = m.describe(P, node)
    base = m.service_admission_bindings(description, "smoke")
    bundle = service.bundle(task_id)
    task = {"task_key": "SYN:A", "review_id": "SYN", "arm": "A", "difficulty": "simple",
            "input": {"text": text, "user_context": context},
            "input_sha256": bundle["input"]["input_sha256"],
            "reference": {"supported_tags": ["赠送利益"]}}
    plan = {"project_root": str(P), "node": node, "stage": "smoke", "tasks": [task],
            "round_cap_micro_cny": 3000000, "snapshot_dir": str(tmp_path),
            "stage_budget_register": str(tmp_path / "register.json"),
            "description": description, "service_admission_bindings": base,
            "snapshot_id": bundle["snapshot"]["snapshot_id"],
            "runtime_snapshot_projection": bundle["snapshot"],
            "evaluation_protocol_sha256": description["evaluation_protocol_sha256"]}
    plan_path = tmp_path / "plan.json"
    m.save(plan_path, plan)
    binding = description["stages"]["smoke"]["A"]
    bundle["config"] = binding["config"]
    bundle["execution_contract"] = binding["planned_execution_contract"]
    bundle["loop_evaluation"] = {
        "protocol_sha256": plan["evaluation_protocol_sha256"], "stage": "smoke", "arm": "A",
        "base_config_node_sha256": binding["base_config_node_sha256"],
        "base_admission_config_sha256": base["A"]["config_sha256"],
        "plan_sha256": m.file_sha(plan_path), "admission_binding_sha256": record["binding_sha256"],
    }
    directory = service.root / task_id
    m.save(directory / "loop-bundle.json", bundle)
    model_path = tmp_path / "execution/model-ledgers/SYN_A.json"
    model_path.parent.mkdir()
    # Real Node registry/report/sidecars, with direct scripted assessment only.
    # No runTask, gateway, key loader, network, tokenizer or GPU is constructed.
    code = """
import { readFileSync, writeFileSync } from 'node:fs';
const root = process.argv[1], directory = process.argv[2], ledger = process.argv[3];
const load = name => import(new URL('agent_harness/dist/' + name + '.js', root));
const { fixture } = await load('tests/agent_loop_fixture');
const { LoopRegistry } = await load('src/agent_loop_state');
const { TaskState } = await load('src/state');
const { Budget } = await load('src/budget');
const { assembleLoopReport, agentAssessmentSidecar } = await load('src/agent_loop_report');
const { retrievalAssessment } = await load('src/retrieval_quality');
const { plannedSystemPrompt } = await load('src/agent_loop_runner');
const { TrialBudget } = await load('src/trial_budget');
const { hash } = await load('src/contracts');
const bundle = JSON.parse(readFileSync(directory + '/loop-bundle.json', 'utf8'));
const f = fixture(); Object.assign(f.task, bundle);
const state = new TaskState(f.task.input), budget = new Budget(f.task.config);
state.evaluationCitations = true; state.citationSnapshotId = bundle.snapshot.snapshot_id;
const registry = new LoopRegistry(f.task, state, budget);
registry.registerPlan(f.plan(1, ["赠送利益", "合同外利益"]));
const request = registry.beginAssessment('RP0001');
await registry.acceptPointAssessment(f.assess(request), request, f.bridge);
const report = await assembleLoopReport(registry, f.bridge);
report.versions.prompt_sha256 = hash(plannedSystemPrompt);
const bytes = JSON.stringify(report), digest = hash(bytes);
writeFileSync(directory + '/report.json', bytes);
writeFileSync(directory + '/agent-assessment.json', JSON.stringify(
  agentAssessmentSidecar(registry, report, digest)));
writeFileSync(directory + '/retrieval-assessment.json', JSON.stringify(
  retrievalAssessment(state, report, digest)));
writeFileSync(directory + '/events.jsonl', '');
new TrialBudget(ledger, 1, 'v2-v1', {
  budget_profile_id: budget.config.budget_profile_id,
  budget_profile_sha256: budget.config.budget_profile_sha256,
  model_trial_cap_micro_cny: budget.config.model_trial_cap_micro_cny,
}).initializeEmpty();
"""
    subprocess.run(
        [node, "--input-type=module", "-e", code, P.as_uri() + "/", str(directory),
         str(model_path)], check=True, capture_output=True, timeout=30,
    )
    report = m.load(directory / "report.json")
    assert report["risk_points"][0]["tag_checks"][0]["conditions"]
    assert m.load(directory / "agent-assessment.json")["points"][0]["facts"]
    known, unknown, detail = m.read_costs(model_path, directory / "rerank-ledger.json", plan)
    assert known == unknown == 0
    execution = {"plan_sha256": m.file_sha(plan_path), "stage": "smoke",
                 "round_cap_micro_cny": 3000000,
                 "tasks": [{"task_key": task["task_key"], "status": "settled",
                            "known_micro_cny": 0, "unknown_micro_cny": 0}]}
    local_register = register()
    local_register["executions"] = [execution]
    m.save(plan["stage_budget_register"], local_register)
    result_path = tmp_path / "execution/SYN_A.json"
    result = {"task_id": task_id, "task_key": task["task_key"],
              "plan_sha256": m.file_sha(plan_path), "report": report,
              "known_micro_cny": 0, "unknown_micro_cny": 0, "cost_detail": detail,
              "model_physical_requests": 0, "rerank_physical_requests": 0}
    copy = tmp_path / "execution/reports/SYN_A.json"
    copy.parent.mkdir()
    copy.write_bytes((directory / "report.json").read_bytes())
    m.save(result_path, result)
    m.save(tmp_path / "execution/authorization.json", m.authorization_record(plan_path, plan))

    def rewrite_report(report):
        m.save(directory / "report.json", report)
        copy.write_bytes((directory / "report.json").read_bytes())
        agent = m.load(directory / "agent-assessment.json")
        retrieval = m.load(directory / "retrieval-assessment.json")
        agent["identity"]["report_sha256"] = m.file_sha(copy)
        retrieval["report_sha256"] = m.file_sha(copy)
        m.save(directory / "agent-assessment.json", agent)
        m.save(directory / "retrieval-assessment.json", retrieval)
        result["report"] = report
        m.save(result_path, result)

    return m, plan, plan_path, task, source.snapshot, execution, directory, rewrite_report


def test_real_nonempty_sidecars_and_public_conditions_complete_full_audit(nonempty_audit_case):
    m, plan, path, task, snapshot, execution, _, _ = nonempty_audit_case
    audited = m.audit_task(plan, path, task, snapshot, execution)
    assert audited["audit_status"] == "valid" and audited["business_valid"], audited["limitations"]
    assert not audited["hard_engineering_error"]


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_public_condition_id_set_must_be_complete_and_unique(nonempty_audit_case, change):
    m, plan, path, task, snapshot, execution, directory, rewrite = nonempty_audit_case
    report = m.load(directory / "report.json")
    conditions = report["risk_points"][0]["tag_checks"][0]["conditions"]
    assert all("condition_id" in c and "criterion_id" not in c for c in conditions)
    if change == "missing":
        conditions.pop()
    else:
        conditions.append(dict(conditions[0]))
    rewrite(report)
    with pytest.raises(ValueError, match="REPORT_CRITERIA_INCOMPLETE"):
        m.audit_task(plan, path, task, snapshot, execution)


@pytest.mark.parametrize("uri", ["urn:finrisk:unregistered:1", "https://invalid.test/schema"])
def test_unknown_sidecar_reference_stays_closed_invalid_row(
    nonempty_audit_case, monkeypatch, uri
):
    import copy
    import socket

    import agent_rag.contracts as contracts
    import agent_rag.v2_snapshot as snapshots

    m, plan, path, _, snapshot, _, _, _ = nonempty_audit_case
    original = contracts.load_contract

    def changed(name):
        schema = original(name)
        if name == "agent-assessment.schema.json":
            schema = copy.deepcopy(schema)
            schema["$defs"]["facts"]["$ref"] = uri
        return schema

    monkeypatch.setattr(contracts, "load_contract", changed)
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("remote ref accessed"))
    monkeypatch.setattr(m, "preflight", lambda p: plan)
    monkeypatch.setattr(snapshots, "V2KnowledgeSnapshot", lambda p: snapshot)
    result = m.score(path, write=False)
    assert result["arms"]["A"]["denominator"] == 1
    assert result["arms"]["A"]["correct_valid"] == 0
    assert result["rows"][0]["status"] == "invalid"
    assert result["development_gate"]["hard_engineering_errors"] == 1
