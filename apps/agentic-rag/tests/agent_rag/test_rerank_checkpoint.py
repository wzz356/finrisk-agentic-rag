"""Offline regressions for durable per-task pair admission; no real API requests."""

import json
import subprocess
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_rerank_backend import backend, control, response, unit


def test_reranker_explicitly_advertises_stable_query_binding():
    assert getattr(backend().DeepSeekReranker("规则"), "supports_query_binding", False) is True


def test_repeated_bad_quote_has_two_total_attempts_and_structured_failure(tmp_path):
    calls = []

    def send(request, timeout):
        calls.append(request)
        return response(request, source_quote="原文不存在的引文")

    budget = control(tmp_path)
    reranker = backend().DeepSeekReranker("规则", send)
    first = reranker.score_query("查询", [unit()], budget)
    second = reranker.score_query("查询", [unit()], budget)
    assert len(calls) == 2
    assert not first["complete"] and not second["complete"]
    assert second["meta"]["external_retrieval_requests"] == 0
    assert second["meta"]["duplicate_paid_dispatches_prevented"] == 1
    assert second["errors"][0] == {
        "unit_id": "U1", "file_id": "F1", "reason": "QUOTE_MISMATCH",
        "code": "QUOTE_MISMATCH", "attempts_used": 2, "remaining_attempts": 0,
        "terminal": True, "audit_ordinals": [1, 2],
    }
    checkpoint = json.loads((tmp_path / "ledger.json.pairs.json").read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == "finrisk-rerank-checkpoint/1.0"
    pair = next(iter(checkpoint["pairs"].values()))
    assert pair["attempts_used"] == 2 and pair["remaining_attempts"] == 0
    assert pair["terminal"] is True and pair["audit_ordinals"] == [1, 2]
    assert [attempt["code"] for attempt in pair["attempts"]] == ["QUOTE_MISMATCH"] * 2


def test_reopening_control_and_evicted_success_reuses_durable_pair_scores(tmp_path):
    calls = []

    def send(request, timeout):
        calls.append(request)
        return response(request)

    budget = control(tmp_path)
    reranker = backend().DeepSeekReranker("规则", send, cache_size=1)
    assert reranker.score_query("查询", [unit(), unit(2)], budget)["complete"]
    reopened = control(tmp_path)
    second = backend().DeepSeekReranker("规则", send, cache_size=0).score_query(
        "查询", [unit(), unit(2)], reopened
    )
    assert second["complete"] and len(calls) == 2
    assert second["meta"]["cache_hits"] == 2


def test_reopening_control_does_not_restore_exhausted_pair_attempts(tmp_path):
    calls = []

    def send(request, timeout):
        calls.append(request)
        return response(request, source_quote="错位")

    backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], control(tmp_path))
    result = backend().DeepSeekReranker("规则", send).score_query(
        "查询", [unit()], control(tmp_path)
    )
    assert len(calls) == 2 and result["meta"]["external_retrieval_requests"] == 0


def test_binding_ignores_query_action_candidate_ids_but_binds_snapshot_index_profile(tmp_path):
    calls = []

    def send(request, timeout):
        calls.append(request)
        return response(request, source_quote="错位")

    reranker = backend().DeepSeekReranker("规则", send)
    budget = control(tmp_path)
    binding = {"snapshot_id": "S1", "index_sha256": "I1", "retrieval_profile": "P1"}
    reranker.score_query("查询", [unit()], budget, query_binding={**binding, "query_id": "Q1"})
    repeat = reranker.score_query("查询", [unit()], budget, query_binding={
        **binding, "query_id": "Q2", "action": "retry", "candidate_pool": ["U1", "U2"],
    })
    assert len(calls) == 2 and repeat["meta"]["external_retrieval_requests"] == 0
    for field in ("snapshot_id", "index_sha256", "retrieval_profile"):
        reranker.score_query("查询", [unit()], budget, query_binding={**binding, field: "changed"})
    assert len(calls) == 8


def test_action_id_and_all_nonstable_binding_fields_cannot_reset_pair_allowance(tmp_path):
    calls = []

    def send(request, timeout):
        calls.append(request)
        return response(request, source_quote="错位")

    reranker = backend().DeepSeekReranker("规则", send)
    budget = control(tmp_path)
    stable = {"snapshot_id": "S1", "index_sha256": "I1", "retrieval_profile": "P1"}
    reranker.score_query("查询", [unit()], budget, query_binding={
        **stable, "query_id": "Q1", "action_id": "A1", "unexpected_context": "one",
    })
    repeated = reranker.score_query("查询", [unit()], budget, query_binding={
        **stable, "query_id": "Q2", "action_id": "A2", "unexpected_context": "two",
        "profile": "not-the-stable-profile", "candidate_pool": ["U1", "U2"],
    })
    assert len(calls) == 2
    assert not repeated["complete"] and repeated["meta"]["external_retrieval_requests"] == 0


def test_file_id_is_part_of_pair_binding_and_failures_do_not_cross_tasks(tmp_path):
    calls = []

    def send(request, timeout):
        calls.append(request)
        return response(request, source_quote="错位")

    reranker = backend().DeepSeekReranker("规则", send)
    budget = control(tmp_path / "one")
    reranker.score_query("查询", [unit()], budget)
    reranker.score_query("查询", [unit(file_id="other-file")], budget)
    reranker.score_query("查询", [unit()], control(tmp_path / "two"))
    assert len(calls) == 6


def test_concurrent_same_pair_across_controls_executes_once_without_holding_task_lock(tmp_path):
    calls = []
    entered = threading.Event()
    release = threading.Event()
    first_control = control(tmp_path)
    second_control = control(tmp_path)

    def send(request, timeout):
        calls.append(request)
        entered.set()
        assert release.wait(5)
        return response(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend().DeepSeekReranker("规则", send).score_query,
                            "查询", [unit()], first_control)
        assert entered.wait(5)
        second = pool.submit(backend().DeepSeekReranker("规则", send).score_query,
                             "查询", [unit()], second_control)
        assert second_control.meta()["in_flight_requests"] == 1
        release.set()
        assert first.result(timeout=5)["complete"]
        assert second.result(timeout=5)["complete"]
    assert len(calls) == 1


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 409, 422])
def test_all_permanent_4xx_are_terminal_without_retry(tmp_path, status):
    calls = []

    def send(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError("https://example.invalid", status, "private-body", {}, None)

    reranker = backend().DeepSeekReranker("规则", send)
    budget = control(tmp_path)
    result = reranker.score_query("查询", [unit()], budget)
    reranker.score_query("查询", [unit()], budget)
    assert len(calls) == 1
    assert result["errors"][0]["terminal"] is True
    assert result["errors"][0]["attempts_used"] == 1
    assert "private-body" not in json.dumps(result)


@pytest.mark.parametrize("damage", ["missing", "bad_json", "binding", "attempts"])
def test_missing_or_corrupt_checkpoint_with_sent_ledger_never_grants_new_attempts(tmp_path, damage):
    calls = []

    def send(request, timeout):
        calls.append(request)
        return response(request, source_quote="错位")

    budget = control(tmp_path)
    backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    path = tmp_path / "ledger.json.pairs.json"
    if damage == "missing":
        path.unlink()
    elif damage == "bad_json":
        path.write_text("{", encoding="utf-8")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if damage == "binding":
            data["task_binding"] = "changed"
        else:
            pair = next(iter(data["pairs"].values()))
            pair["attempts"] = []
            pair.update(attempts_used=0, remaining_attempts=2, terminal=False, audit_ordinals=[])
        path.write_text(json.dumps(data), encoding="utf-8")
    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], control(tmp_path))
    assert not result["complete"] and len(calls) == 2
    assert result["errors"][0]["code"] == "CHECKPOINT_UNAVAILABLE"
    assert result["errors"][0]["attempts_used"] is None
    assert result["errors"][0]["remaining_attempts"] is None
    assert result["errors"][0]["audit_ordinals"] is None


def test_foreign_service_generation_cannot_take_over_checkpoint(tmp_path, monkeypatch):
    module = backend()
    calls = []

    def send(request, timeout):
        calls.append(request)
        return response(request)

    module.DeepSeekReranker("规则", send).score_query("查询", [unit()], control(tmp_path))
    checkpoint_module = __import__("agent_rag.rerank_checkpoint", fromlist=["SERVICE_GENERATION"])
    monkeypatch.setattr(checkpoint_module, "SERVICE_GENERATION", "foreign-generation")
    result = module.DeepSeekReranker("规则", send).score_query("新查询", [unit()], control(tmp_path))
    assert not result["complete"] and len(calls) == 1
    assert result["errors"][0]["code"] == "CHECKPOINT_UNAVAILABLE"


@pytest.mark.parametrize("when", ["claim", "finish"])
def test_checkpoint_write_failure_never_creates_extra_paid_retry(tmp_path, monkeypatch, when):
    checkpoint_module = __import__("agent_rag.rerank_checkpoint", fromlist=["PairCheckpoint"])
    original = checkpoint_module.PairCheckpoint._persist
    failed = False
    calls = []

    def persist(self, data, **kwargs):
        nonlocal failed
        pairs = list(data["pairs"].values())
        matching = bool(pairs) and (when == "claim" or pairs[0]["attempts"][0]["code"] is not None)
        if not failed and matching:
            failed = True
            raise OSError("private-checkpoint-error")
        return original(self, data, **kwargs)

    monkeypatch.setattr(checkpoint_module.PairCheckpoint, "_persist", persist)

    def send(request, timeout):
        calls.append(request)
        return response(request, source_quote="错位")

    budget = control(tmp_path)
    reranker = backend().DeepSeekReranker("规则", send)
    first = reranker.score_query("查询", [unit()], budget)
    second = reranker.score_query("查询", [unit()], control(tmp_path))
    assert not first["complete"] and not second["complete"]
    assert len(calls) == (0 if when == "claim" else 1)
    assert "private-checkpoint-error" not in json.dumps(second)
    if when == "claim":
        assert first["errors"][0]["attempts_used"] is None
        assert first["errors"][0]["remaining_attempts"] is None
        assert first["errors"][0]["audit_ordinals"] == []


@pytest.mark.parametrize("when", ["bind", "finish", "claim_read", "abort_bind"])
def test_checkpoint_failure_preserves_known_attempt_and_ordinal_without_refunding_unknown_usage(
    tmp_path, monkeypatch, when
):
    checkpoint_module = __import__("agent_rag.rerank_checkpoint", fromlist=["PairCheckpoint"])
    original = checkpoint_module.PairCheckpoint._persist
    original_get = checkpoint_module.PairCheckpoint.get
    original_claim = checkpoint_module.PairCheckpoint.claim
    failed = False
    calls = []
    claimed = False
    budget = control(tmp_path)

    def persist(self, data, **kwargs):
        nonlocal failed
        pairs = list(data["pairs"].values())
        matching = bool(pairs) and (
            (when in {"bind", "abort_bind"} and bool(pairs[0]["audit_ordinals"]))
            or (when == "finish" and pairs[0]["attempts"][0]["code"] is not None)
        )
        if not failed and matching:
            failed = True
            raise OSError("private-checkpoint-error")
        return original(self, data, **kwargs)

    def claim(self, key):
        nonlocal claimed
        result = original_claim(self, key)
        claimed = True
        return result

    def get(self, key):
        nonlocal failed
        if when == "claim_read" and claimed and not failed:
            failed = True
            raise checkpoint_module.RerankCheckpointError()
        return original_get(self, key)

    monkeypatch.setattr(checkpoint_module.PairCheckpoint, "_persist", persist)
    monkeypatch.setattr(checkpoint_module.PairCheckpoint, "claim", claim)
    monkeypatch.setattr(checkpoint_module.PairCheckpoint, "get", get)
    if when == "abort_bind":
        # Cancel during the reservation write so an unsent audit ordinal exists.
        persist_ledger = budget._persist

        def cancel_after_reservation(ledger):
            persist_ledger(ledger)
            if ledger["request_reservations"]:
                budget.cancel_event.set()

        monkeypatch.setattr(budget, "_persist", cancel_after_reservation)

    def send(request, timeout):
        calls.append(request)
        result = response(request, source_quote="错位")
        result.pop("usage")
        return result

    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    error = result["errors"][0]
    assert error["code"] == "CHECKPOINT_UNAVAILABLE"
    assert error["attempts_used"] == 1 and error["remaining_attempts"] == 1
    assert error["terminal"] is True
    assert error["audit_ordinals"] == ([] if when == "claim_read" else [1])
    assert result["meta"]["external_retrieval_requests"] == (1 if when == "finish" else 0)
    assert len(calls) == (1 if when == "finish" else 0)
    assert (result["meta"]["reserved_cost_micro_cny"] > 0) is (when == "finish")
    assert "private-checkpoint-error" not in json.dumps(result)


def test_cancel_unknown_usage_keeps_existing_fees_and_blocks_new_attempts(tmp_path):
    calls = []
    budget = control(tmp_path)

    def send(request, timeout):
        calls.append(request)
        budget.cancel_event.set()
        result = response(request, source_quote="错位")
        result.pop("usage")
        return result

    reranker = backend().DeepSeekReranker("规则", send)
    reranker.score_query("查询", [unit()], budget)
    result = reranker.score_query("查询", [unit()], budget)
    assert not result["complete"] and len(calls) == 1
    assert result["meta"]["reserved_cost_micro_cny"] > 0
    assert result["meta"]["finished_usage_unknown_requests"] == 1


def test_98_successful_pairs_are_reused_when_two_pairs_exhaust_validation_attempts(tmp_path):
    calls = {}

    def send(request, timeout):
        packet = json.loads(request["messages"][1]["content"])
        text = packet["unit_text"]
        calls[text] = calls.get(text, 0) + 1
        return response(request, **({"source_quote": "错位"} if text in {"text98", "text99"} else {}))

    units = [unit(index, text=f"text{index}") for index in range(100)]
    budget = control(tmp_path, max_requests=200)
    reranker = backend().DeepSeekReranker("规则", send, cache_size=1)
    first = reranker.score_query("查询", units, budget)
    assert not first["complete"] and len(first["scores"]) == 98
    assert calls == {f"text{index}": (2 if index >= 98 else 1) for index in range(100)}
    second = reranker.score_query("查询", units, control(tmp_path, max_requests=200))
    assert not second["complete"] and len(second["scores"]) == 98
    assert second["meta"]["external_retrieval_requests"] == 0
    assert second["meta"]["cache_hits"] == 98
    assert all(error["remaining_attempts"] == 0 for error in second["errors"])
    assert sum(calls.values()) == 102


def test_pair_claim_and_ordinal_are_durable_before_reservation_and_transport(tmp_path, monkeypatch):
    budget = control(tmp_path)
    reserve = budget._reserve

    def checkpoint_pair():
        data = json.loads((tmp_path / "ledger.json.pairs.json").read_text(encoding="utf-8"))
        return next(iter(data["pairs"].values()))

    def reserve_after_claim(*args):
        pair = checkpoint_pair()
        assert pair["attempts_used"] == 1 and pair["attempts"][0]["state"] == "claimed"
        assert pair["audit_ordinals"] == []
        return reserve(*args)

    monkeypatch.setattr(budget, "_reserve", reserve_after_claim)

    def send(request, timeout):
        assert checkpoint_pair()["audit_ordinals"] == [1]
        return response(request)

    assert backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)["complete"]


@pytest.mark.parametrize("state", ["transport_started", "settled"])
def test_audit_write_failure_is_durable_terminal_across_repeated_queries(tmp_path, monkeypatch, state):
    budget = control(tmp_path)
    original = budget._persist_attempts
    calls = []
    failed = False

    def persist(audit):
        nonlocal failed
        if not failed and any(row["state"] == state for row in audit["attempts"]):
            failed = True
            raise OSError("private-audit-error")
        return original(audit)

    monkeypatch.setattr(budget, "_persist_attempts", persist)

    def send(request, timeout):
        calls.append(request)
        return response(request)

    reranker = backend().DeepSeekReranker("规则", send)
    first = reranker.score_query("查询", [unit()], budget)
    second = reranker.score_query("查询", [unit()], control(tmp_path))
    assert not first["complete"] and not second["complete"]
    assert len(calls) == (0 if state == "transport_started" else 1)
    assert second["errors"][0]["code"] == "AUDIT_WRITE_FAILED"
    assert second["errors"][0]["terminal"] is True


@pytest.mark.parametrize("stop", ["request_limit", "cancelled"])
def test_same_pair_waiter_obeys_cancellation_and_can_reuse_inflight_success_at_cap(tmp_path, stop):
    budget = control(tmp_path, max_requests=1)
    waiter_budget = control(tmp_path, max_requests=1)
    entered = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    calls = []

    def send(request, timeout):
        calls.append(request)
        entered.set()
        assert release.wait(5)
        return response(request)

    checkpoint_module = __import__("agent_rag.rerank_checkpoint", fromlist=["PairCheckpoint"])
    checkpoint = checkpoint_module.PairCheckpoint(budget)
    original_wait = checkpoint._condition.wait

    def wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    # Observe the condition boundary instead of depending on a scheduling sleep.
    checkpoint._condition.wait = wait
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(backend().DeepSeekReranker("规则", send).score_query,
                                "查询", [unit()], budget)
            assert entered.wait(5)
            second = pool.submit(backend().DeepSeekReranker("规则", send).score_query,
                                 "查询", [unit()], waiter_budget)
            if stop == "cancelled":
                assert waiting.wait(1)
                waiter_budget.cancel()
                result = second.result(timeout=2)
                assert result["errors"][0]["code"] == "CANCELLED"
            release.set()
            assert first.result(timeout=5)["complete"]
            if stop == "request_limit":
                assert second.result(timeout=5)["complete"]
    finally:
        release.set()
        checkpoint._condition.wait = original_wait
    assert len(calls) == 1


def test_competing_first_ledger_creation_cannot_overwrite_dispatched_unknown_fees(tmp_path):
    """Pause a separate generation after its exists check, before initial write."""
    module = backend()
    child = r'''
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from agent_rag.rerank_backend import RerankTaskControl
p = Path(sys.argv[1])
method = '_create_initial_ledger' if hasattr(RerankTaskControl, '_create_initial_ledger') else '_persist'
original = getattr(RerankTaskControl, method)
def pause(self, ledger):
    if not ledger['request_reservations'] and not self.ledger_path.exists():
        (p / 'child-ready').write_text('ready', encoding='utf-8')
        deadline = time.monotonic() + 10
        while not (p / 'child-release').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('test coordination timed out')
            time.sleep(.01)
    return original(self, ledger)
setattr(RerankTaskControl, method, pause)
c = RerankTaskControl(p / 'ledger.json', cap_cny=1, max_requests=100)
meta = c.meta()
print(json.dumps({key: meta[key] for key in (
    'total_external_retrieval_requests', 'reserved_cost_micro_cny',
)}))
'''
    from pathlib import Path
    source_path = str(Path(module.__file__).resolve().parents[1])
    process = subprocess.Popen(
        [sys.executable, "-c", child, str(tmp_path), source_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "child-ready").exists() and process.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert (tmp_path / "child-ready").exists()
        budget = control(tmp_path)

        def send(request, timeout):
            result = response(request)
            result.pop("usage")
            return result

        result = module.DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
        assert result["complete"]
        before = budget.meta()
        before_bytes = (tmp_path / "ledger.json").read_bytes()
        assert before["total_external_retrieval_requests"] == 1
        assert before["reserved_cost_micro_cny"] > 0
        (tmp_path / "child-release").write_text("release", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        after = budget.meta()
        assert after["total_external_retrieval_requests"] == 1
        assert after["reserved_cost_micro_cny"] == before["reserved_cost_micro_cny"]
        assert (tmp_path / "ledger.json").read_bytes() == before_bytes
        assert json.loads(stdout) == {
            "total_external_retrieval_requests": 1,
            "reserved_cost_micro_cny": before["reserved_cost_micro_cny"],
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
