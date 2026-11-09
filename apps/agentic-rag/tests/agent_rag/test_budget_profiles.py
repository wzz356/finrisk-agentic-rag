import copy
import importlib
import json

import pytest
from test_v2_service import make_service


def api():
    return importlib.import_module("agent_rag.budget_profiles")


def live_environment(monkeypatch, tmp_path, cap="1.5", requests="800"):
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE", "1")
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_LIVE", "1")
    monkeypatch.setenv("FINRISK_AGENT_TRIAL_LEDGER", str(tmp_path / "models.json"))
    monkeypatch.setenv("FINRISK_AGENT_RERANK_CAP_CNY", cap)
    monkeypatch.setenv("FINRISK_AGENT_RERANK_MAX_REQUESTS", requests)


def test_registered_planned_profile_admits_frozen_limits_and_hash(tmp_path, monkeypatch):
    live_environment(monkeypatch, tmp_path)
    service = make_service(tmp_path, "live")
    record, _ = service.admit(
        {"text": "材料", "experiment_profile_id": "live-deepseek-flash-planned-v1"}, None
    )
    bundle = service.bundle(record["task_id"])
    config = bundle["config"]
    expected = {
        "max_model_requests": 12,
        "max_search_calls": 4,
        "max_search_query_attempts": 4,
        "max_tool_actions": 14,
        "task_deadline_seconds": 600,
        "rerank_max_requests": 800,
        "max_external_retrieval_requests": 800,
        "rerank_cap_cny": 1.5,
        "model_trial_cap_cny": 1,
        "task_total_cost_cap_cny": 2.5,
        "initial_query_limit": 2,
        "feedback_query_reserve": 2,
        "max_feedback_rounds": 2,
        "max_read_source_actions": 8,
        "max_definition_actions": 2,
        "max_json_repairs_per_task": 1,
        "max_output_tokens_per_request": 4096,
        "service_max_input_tokens": 16000,
        "task_token_budget": 250000,
        "max_candidate_risk_points": 4,
        "orchestration_mode": "planned_adaptive",
        "fast_path_enabled": False,
        "retrieval_profile": "v2-hybrid-flash-v1",
    }
    for key, value in expected.items():
        assert config[key] == value, key
    profile = api().load_budget_profile("planned-evidence-loop-v1")
    digest = api().budget_profile_sha256(profile)
    assert config["budget_profile_sha256"] == digest
    assert bundle["execution_contract"]["budget_profile_sha256"] == digest
    manifest = json.loads((service.root / record["task_id"] / "manifest.json").read_text())
    assert manifest["budget_profile_sha256"] == digest
    ledger = json.loads((service.root / record["task_id"] / "rerank-ledger.json").read_text())
    assert ledger["cap_micro_cny"] == 1500000 and ledger["max_requests"] == 800
    assert ledger["budget_profile_sha256"] == digest


@pytest.mark.parametrize("profile", ["live-deepseek-flash-v2-v1", "mock-v2-flash-live-v1"])
def test_legacy_registered_profile_still_rejects_800(tmp_path, monkeypatch, profile):
    live_environment(monkeypatch, tmp_path, cap="0.5")
    service = make_service(tmp_path, "live")
    with pytest.raises(ValueError, match="INVALID_RERANK_LIMITS"):
        service.admit({"text": "材料", "experiment_profile_id": profile}, None)


@pytest.mark.parametrize(
    "cap,requests",
    [
        ("1.500001", "800"),
        ("1.5", "801"),
        ("0.500001", "240"),
        ("0.5", "9007199254740992"),
        ("0.0000001", "1"),
    ],
)
def test_environment_cannot_raise_registered_limits(tmp_path, monkeypatch, cap, requests):
    live_environment(monkeypatch, tmp_path, cap=cap, requests=requests)
    service = make_service(tmp_path, "live")
    profile = "live-deepseek-flash-v2-v1" if cap == "0.500001" else "live-deepseek-flash-planned-v1"
    with pytest.raises(ValueError, match="INVALID_RERANK_LIMITS"):
        service.admit({"text": "材料", "experiment_profile_id": profile}, None)
    assert not list(service.root.glob("*/admission.json"))


def test_environment_can_tighten_without_requiring_theoretical_full_capacity(tmp_path, monkeypatch):
    live_environment(monkeypatch, tmp_path, cap="0.15", requests="100")
    service = make_service(tmp_path, "live")
    record, _ = service.admit(
        {"text": "材料", "experiment_profile_id": "live-deepseek-flash-planned-v1"}, None
    )
    config = service.bundle(record["task_id"])["config"]
    assert config["rerank_max_requests"] == 100 and config["rerank_cap_cny"] == 0.15
    assert config["task_total_cost_cap_cny"] == 1.15
    diagnostic = api().budget_capacity_diagnostic(
        api().load_budget_profile("planned-evidence-loop-v1"), 100
    )
    assert diagnostic["may_stop_before_all_queries"] is True
    assert diagnostic["theoretical_rerank_requests"] == 800


def test_legacy_defaults_remain_unchanged(tmp_path, monkeypatch):
    live_environment(monkeypatch, tmp_path, cap="0.5", requests="240")
    service = make_service(tmp_path, "live")
    record, _ = service.admit(
        {"text": "材料", "experiment_profile_id": "live-deepseek-flash-v2-v1"}, None
    )
    config = service.bundle(record["task_id"])["config"]
    assert [
        config[k]
        for k in (
            "max_model_requests",
            "max_search_calls",
            "max_search_query_attempts",
            "max_tool_actions",
            "task_deadline_seconds",
        )
    ] == [6, 2, 2, 8, 270]
    assert config["task_total_cost_cap_cny"] == 1.5
    assert "orchestration_mode" not in config


def test_canonical_hash_fixed_vector_and_integral_float_normalization():
    # SHA256 of UTF-8 {"a":[1,{"x":2}],"z":3}, independently frozen in both languages.
    expected = "a773a4bd6210a6665c9776d4213b13ffbca00ffb8e0eb1ce7fb74dc89b7fe57b"
    value = {"z": 3.0, "a": [1.0, {"x": 2}]}
    assert api().canonical_budget_json(value) == '{"a":[1,{"x":2}],"z":3}'
    assert api().budget_profile_sha256(value) == expected


@pytest.mark.parametrize(
    "bad", [True, False, float("nan"), float("inf"), -1, 1.5, 9007199254740992, "1", None]
)
@pytest.mark.parametrize("field", ["max_model_requests", "rerank_cap_micro_cny"])
def test_budget_rejects_non_safe_integer_counts_and_money(field, bad):
    profile = api().load_budget_profile("planned-evidence-loop-v1")
    profile[field] = bad
    with pytest.raises(ValueError, match="INVALID_BUDGET_PROFILE"):
        api().validate_budget_profile(profile)


@pytest.mark.parametrize(
    "mutation",
    [
        {"surprise": 1},
        {"initial_query_limit": 5},
        {"feedback_query_reserve": 5},
        {"task_total_cost_cap_micro_cny": 2500001},
        {"max_search_calls": 0},
    ],
)
def test_budget_rejects_unknown_fields_and_inconsistent_limits(mutation):
    profile = api().load_budget_profile("planned-evidence-loop-v1")
    profile.update(mutation)
    with pytest.raises(ValueError, match="INVALID_BUDGET_PROFILE"):
        api().validate_budget_profile(profile)


def test_validated_profiles_are_independent_and_unknown_id_fails():
    original = api().load_budget_profile("planned-evidence-loop-v1")
    changed = copy.deepcopy(original)
    changed["max_model_requests"] = 11.0
    assert type(api().validate_budget_profile(changed)["max_model_requests"]) is int
    changed["max_model_requests"] = 1
    assert api().load_budget_profile("planned-evidence-loop-v1") == original
    with pytest.raises(ValueError, match="UNKNOWN_BUDGET_PROFILE"):
        api().load_budget_profile("unknown")


@pytest.mark.parametrize(
    "update",
    [
        {"orchestration_mode": "unknown"},
        {"budget_profile_id": "unknown"},
        {"orchestration_mode": "planned_adaptive"},
        {"fast_path_enabled": True},
        {"budget_profile_id": "planned-evidence-loop-v1", "orchestration_mode": "planned_fixed"},
    ],
)
def test_service_rejects_mismatched_orchestration_config(tmp_path, update):
    service = make_service(tmp_path)
    service.baseline_config.update(update)
    with pytest.raises(ValueError, match="BUDGET_CONFIG|UNKNOWN_BUDGET_PROFILE"):
        service._profile_config("mock-v2-flash-replay-v1")


def test_extremely_large_integer_rejected_with_validation_error():
    profile = api().load_budget_profile("planned-evidence-loop-v1")
    profile["rerank_cap_micro_cny"] = 10**1000
    with pytest.raises(ValueError, match="INVALID_BUDGET_PROFILE"):
        api().validate_budget_profile(profile)


def test_service_fixed_rag_binds_planned_fixed_without_changing_budget_hash(tmp_path, monkeypatch):
    live_environment(monkeypatch, tmp_path)
    service = make_service(tmp_path, "live")
    record, _ = service.admit(
        {
            "text": "材料",
            "experiment_profile_id": "live-deepseek-flash-planned-v1",
            "engine": "fixed_rag",
        },
        None,
    )
    config = service.bundle(record["task_id"])["config"]
    assert config["orchestration_mode"] == "planned_fixed"
    assert config["budget_profile_sha256"] == api().budget_profile_sha256(
        api().load_budget_profile("planned-evidence-loop-v1")
    )


@pytest.mark.parametrize(
    "update",
    [
        {"budget_profile_id": None},
        {"orchestration_mode": None},
        {"fast_path_enabled": None},
        {"budget_profile_id": True},
    ],
)
def test_explicit_null_or_boolean_config_is_not_an_absent_option(update):
    with pytest.raises(ValueError, match="BUDGET_CONFIG|UNKNOWN_BUDGET_PROFILE"):
        api().validate_budget_config(update)


def test_floating_currency_aliases_cannot_diverge_from_integer_caps():
    profile = api().load_budget_profile("planned-evidence-loop-v1")
    config = {
        **profile,
        "profile_id": "live-deepseek-flash-planned-v1",
        "budget_profile_id": profile["profile_id"],
        "budget_profile_sha256": api().budget_profile_sha256(profile),
        "retrieval_profile": "v2-hybrid-flash-v1",
        "engine": "controlled_agent",
        "orchestration_mode": "planned_adaptive",
    }
    for key, value in [
        ("rerank_cap_cny", 1.500001),
        ("model_trial_cap_cny", True),
        ("task_total_cost_cap_cny", 3),
    ]:
        with pytest.raises(ValueError, match="INVALID_BUDGET_CONFIG"):
            api().validate_budget_config({**config, key: value})


def test_environment_extreme_exponent_is_a_validation_error(tmp_path, monkeypatch):
    live_environment(monkeypatch, tmp_path, cap="1e1000000")
    service = make_service(tmp_path, "live")
    with pytest.raises(ValueError, match="INVALID_RERANK_LIMITS"):
        service.admit(
            {"text": "材料", "experiment_profile_id": "live-deepseek-flash-planned-v1"}, None
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget_profile_id", "legacy-live-v2-v1"),
        ("budget_profile_sha256", "0" * 64),
        ("budget_profile_sha256", None),
    ],
)
@pytest.mark.parametrize("boundary", ["check", "reserve", "commit", "start_transport"])
def test_service_control_rejects_tampered_profile_at_every_dispatch_boundary(
    tmp_path, monkeypatch, field, value, boundary
):
    from test_rerank_backend import response, unit

    from agent_rag.rerank_backend import DeepSeekReranker

    live_environment(monkeypatch, tmp_path)
    service = make_service(tmp_path, "live")
    record, _ = service.admit(
        {"text": "材料", "experiment_profile_id": "live-deepseek-flash-planned-v1"}, None
    )
    control = service.tool(record["task_id"]).control
    request = {"model": "deepseek-flash", "max_tokens": 512}
    ordinal = None
    if boundary in {"commit", "start_transport"}:
        ordinal, _ = control._reserve(request, "test-pair")
    if boundary == "start_transport":
        control._commit_dispatch(ordinal)
    ledger = json.loads(control.ledger_path.read_text(encoding="utf-8"))
    if value is None:
        ledger.pop(field)
    else:
        ledger[field] = value
    control.ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    before = control.ledger_path.read_bytes()
    actions = {
        "check": control.check,
        "reserve": lambda: control._reserve(request, "new-pair"),
        "commit": lambda: control._commit_dispatch(ordinal),
        "start_transport": lambda: control._start_transport(ordinal),
    }
    with pytest.raises(ValueError, match="INVALID_EXISTING_RERANK_LEDGER"):
        actions[boundary]()
    calls = []

    def send(payload, timeout):
        calls.append(payload)
        return response(payload)

    reranker = DeepSeekReranker("offline rules", send)
    with pytest.raises(ValueError, match="INVALID_EXISTING_RERANK_LEDGER"):
        reranker.score_query("材料", [unit()], control)
    assert calls == []
    assert control.ledger_path.read_bytes() == before
    assert all("transport_started_at" not in row for row in ledger["request_reservations"])


def test_bound_control_cannot_reopen_with_missing_or_different_expected_profile(tmp_path):
    from agent_rag.rerank_backend import RerankTaskControl

    profile = api().load_budget_profile("planned-evidence-loop-v1")
    path = tmp_path / "bound-ledger.json"
    expected = {
        "expected_budget_profile_id": profile["profile_id"],
        "expected_budget_profile_sha256": api().budget_profile_sha256(profile),
    }
    control = RerankTaskControl(path, 1.5, 800, **expected)
    assert control.check()
    assert RerankTaskControl(path, 1.5, 800, **expected).check()
    for binding in ({}, {**expected, "expected_budget_profile_sha256": "0" * 64}):
        with pytest.raises(ValueError, match="INVALID_EXISTING_RERANK_LEDGER"):
            RerankTaskControl(path, 1.5, 800, **binding)
    old = RerankTaskControl(tmp_path / "old-ledger.json", 0.5, 240)
    assert old.check()
    assert "budget_profile_sha256" not in json.loads(old.ledger_path.read_text())
