import hashlib
import importlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

HASH = "a" * 64
CONFIG = {"provider": "fixture", "model": "test", "max_retries": 0}


def module():
    try:
        return importlib.import_module("agent_rag.maintenance_budget")
    except ModuleNotFoundError:
        pytest.fail("BudgetLedger has not been implemented")


def ledger(path, **kwargs):
    return module().BudgetLedger(
        path,
        "p1",
        kwargs.pop("cap_micro", 100),
        kwargs.pop("max_requests", 8),
        provider_config=CONFIG,
        **kwargs,
    )


def reserve(book, key="request-1", **kwargs):
    return book.reserve(
        key,
        kwargs.pop("payload_sha256", HASH),
        kwargs.pop("role", "proposer"),
        kwargs.pop("event_id", "event-1"),
        kwargs.pop("stage", "draft"),
        kwargs.pop("reserved_micro", 20),
        **kwargs,
    )


def assert_code(code, call):
    with pytest.raises(ValueError) as caught:
        call()
    assert caught.value.code == code


def test_reservation_send_settlement_cache_replay_and_restart(tmp_path):
    path = tmp_path / "budget.db"
    book = ledger(path)
    ticket = reserve(book)
    assert book.summary()["reserved_micro"] == 20
    book.mark_sent(ticket)
    result = book.settle(ticket, 12, response={"text": "result"})
    assert result["state"] == "settled"
    reopened = ledger(path)
    cached = reserve(reopened)
    assert cached["cache_hit"] is True
    assert cached["response"] == {"text": "result"}
    assert_code("REQUEST_ALREADY_SENT", lambda: reopened.mark_sent(cached))
    summary = reopened.summary()
    assert summary["request_count"] == 1
    assert summary["spent_micro"] == 12
    assert summary["reserved_micro"] == 0
    assert summary["available_micro"] == 88


def test_unknown_and_crash_after_send_keep_reservation_and_cannot_resend(tmp_path):
    path = tmp_path / "budget.db"
    book = ledger(path)
    ticket = reserve(book)
    book.mark_sent(ticket)
    assert_code("REQUEST_IN_FLIGHT", lambda: reserve(ledger(path)))
    book.settle(ticket, None)
    assert_code("REQUEST_OUTCOME_UNKNOWN", lambda: reserve(ledger(path)))
    assert ledger(path).summary()["unknown_micro"] == 20
    assert ledger(path).summary()["available_micro"] == 80
    book.settle(ticket, 15, response={"reconciled": True})
    assert reserve(ledger(path))["cache_hit"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("payload_sha256", "b" * 64),
        ("role", "reviewer"),
        ("event_id", "other"),
        ("stage", "repair"),
        ("reserved_micro", 21),
        ("provider_config", {**CONFIG, "model": "different"}),
    ],
)
def test_request_idempotency_binds_all_request_details(tmp_path, field, value):
    book = ledger(tmp_path / "budget.db")
    reserve(book)
    assert_code("REQUEST_BINDING_CONFLICT", lambda: reserve(book, **{field: value}))


def test_whole_batch_shared_across_roles_and_process_restart(tmp_path):
    path = tmp_path / "budget.db"
    book = ledger(path, cap_micro=35)
    reserve(book, role="proposer")
    assert_code(
        "BUDGET_EXCEEDED", lambda: reserve(ledger(path, cap_micro=35), "review", role="reviewer")
    )
    assert_code("BATCH_CONFIG_CONFLICT", lambda: ledger(path, cap_micro=999))
    assert book.summary()["request_count"] == 1


def test_concurrent_reservations_cannot_oversubscribe_money_or_requests(tmp_path):
    path = tmp_path / "budget.db"
    ledger(path, cap_micro=50, max_requests=3)

    def attempt(i):
        try:
            return reserve(ledger(path, cap_micro=50, max_requests=3), str(i), reserved_micro=20)
        except ValueError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(attempt, range(16)))
    assert sum(isinstance(value, dict) for value in values) == 2
    assert values.count("BUDGET_EXCEEDED") == 14
    assert ledger(path, cap_micro=50, max_requests=3).summary()["reserved_micro"] == 40


def test_same_key_concurrent_send_has_one_winner(tmp_path):
    book = ledger(tmp_path / "budget.db")
    ticket = reserve(book)

    def send(_):
        try:
            return book.mark_sent(ticket)
        except ValueError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(send, range(12)))
    assert sum(isinstance(value, dict) for value in values) == 1
    assert values.count("REQUEST_ALREADY_SENT") == 11


@pytest.mark.parametrize(
    "dimension,value,code",
    [
        ("role", "proposer", "ROLE_LIMIT_EXCEEDED"),
        ("stage", "draft", "STAGE_LIMIT_EXCEEDED"),
        ("event", "event-1", "EVENT_LIMIT_EXCEEDED"),
    ],
)
def test_dimension_request_and_money_limits_are_atomic(tmp_path, dimension, value, code):
    limits = {dimension: {value: {"max_requests": 1, "cap_micro": 30}}}
    book = ledger(tmp_path / "budget.db", limits=limits)
    reserve(book)
    assert_code(code, lambda: reserve(book, "repair"))
    assert book.summary()["request_count"] == 1


def test_format_repair_and_fixture_zero_cost_count_as_requests(tmp_path):
    book = ledger(tmp_path / "budget.db", cap_micro=0, max_requests=2)
    for key, role, stage in [("draft", "proposer", "draft"), ("repair", "reviewer", "repair")]:
        ticket = reserve(book, key, role=role, stage=stage, reserved_micro=0)
        book.mark_sent(ticket)
        book.settle(ticket, 0, response={})
    assert_code("REQUEST_LIMIT_EXCEEDED", lambda: reserve(book, "third", reserved_micro=0))
    assert book.summary()["request_count"] == 2


def test_price_overrun_is_persisted_and_trips_entire_batch(tmp_path):
    path = tmp_path / "budget.db"
    book = ledger(path)
    ticket = reserve(book)
    book.mark_sent(ticket)
    assert_code(
        "PRICE_RESERVATION_EXCEEDED", lambda: book.settle(ticket, 21, response={"ok": True})
    )
    assert ledger(path).summary()["circuit_broken"] is True
    assert ledger(path).summary()["spent_micro"] == 21
    assert_code("BATCH_CIRCUIT_OPEN", lambda: reserve(ledger(path), "other", role="reviewer"))


def test_unsent_cancellation_never_resets_count_and_sent_cannot_release(tmp_path):
    path = tmp_path / "budget.db"
    book = ledger(path, max_requests=1)
    ticket = reserve(book)
    book.release(ticket)
    assert ledger(path, max_requests=1).summary()["request_count"] == 1
    assert_code("REQUEST_LIMIT_EXCEEDED", lambda: reserve(book, "new"))
    assert_code("REQUEST_RELEASED", lambda: reserve(book))


def test_live_requires_positive_explicit_cap_verifiable_price_and_no_sdk_retries(tmp_path):
    assert_code("LIVE_CAP_REQUIRED", lambda: ledger(tmp_path / "a.db", live=True, cap_micro=0))
    assert_code("PRICE_POLICY_REQUIRED", lambda: ledger(tmp_path / "b.db", live=True))
    price = {
        "policy_id": "p1",
        "verified": True,
        "currency": "CNY",
        "unit": "micro_cny",
        "source": "local-reviewed-price-sheet",
        "rates": {"input": 1, "output": 2},
    }
    source = tmp_path / "reviewed-price.json"
    source.write_text(json.dumps(price), encoding="utf-8")
    price["source_path"] = str(source)
    price["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    book = ledger(tmp_path / "c.db", live=True, price_policy=price)
    assert book.summary()["live"] is True
    assert_code(
        "SDK_RETRIES_FORBIDDEN",
        lambda: module().BudgetLedger(
            tmp_path / "d.db", "p1", 100, 8, provider_config={**CONFIG, "max_retries": 2}
        ),
    )
    assert module().SDK_REQUIRED_MAX_RETRIES == 0


@pytest.mark.parametrize("cost", [-1, 1.5, True])
def test_costs_are_nonnegative_integer_micro_cny(tmp_path, cost):
    book = ledger(tmp_path / "budget.db")
    assert_code("INVALID_MONEY", lambda: reserve(book, reserved_micro=cost))


def test_process_exit_keeps_sent_request_and_forbids_resend(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    code = """import os, sys
sys.path.insert(0, sys.argv[2])
from agent_rag.maintenance_budget import BudgetLedger
book = BudgetLedger(sys.argv[1], 'p1', 100, 8,
    provider_config={'provider':'fixture','model':'test','max_retries':0})
ticket = book.reserve('request-1', 'a'*64, 'proposer', 'event-1', 'draft', 20)
book.mark_sent(ticket)
os._exit(0)
"""
    path = tmp_path / "budget.db"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(Path(__file__).resolve().parents[2] / "src")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    reopened = ledger(path)
    assert reopened.get_request("request-1")["state"] == "sent"
    assert_code("REQUEST_IN_FLIGHT", lambda: reserve(reopened))
    assert reopened.summary()["available_micro"] == 80


def test_configuration_is_snapshotted_and_cannot_mutate_budget_after_creation(tmp_path):
    limits = {"role": {"proposer": {"max_requests": 1}}}
    book = ledger(tmp_path / "budget.db", limits=limits)
    reserve(book)
    limits["role"]["proposer"]["max_requests"] = 999
    book.config["cap_micro"] = 999999
    book.config["limits"]["role"]["proposer"]["max_requests"] = 999
    assert_code("ROLE_LIMIT_EXCEEDED", lambda: reserve(book, "other"))
    assert book.summary()["cap_micro"] == 100


@pytest.mark.parametrize(
    "dimension,value,code",
    [
        ("role", "proposer", "ROLE_LIMIT_EXCEEDED"),
        ("stage", "draft", "STAGE_LIMIT_EXCEEDED"),
        ("event", "event-1", "EVENT_LIMIT_EXCEEDED"),
    ],
)
def test_dimension_amount_limits_apply_even_with_request_capacity(tmp_path, dimension, value, code):
    book = ledger(
        tmp_path / "budget.db", limits={dimension: {value: {"max_requests": 10, "cap_micro": 30}}}
    )
    reserve(book)
    assert_code(code, lambda: reserve(book, "second"))


def test_only_one_zero_cost_request_can_reserve_under_concurrency(tmp_path):
    path = tmp_path / "budget.db"
    ledger(path, cap_micro=0, max_requests=1)

    def attempt(i):
        try:
            return reserve(ledger(path, cap_micro=0, max_requests=1), str(i), reserved_micro=0)
        except ValueError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(12)))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert results.count("REQUEST_LIMIT_EXCEEDED") == 11


def test_sent_request_cannot_be_cancelled_to_release_money(tmp_path):
    book = ledger(tmp_path / "budget.db")
    ticket = reserve(book)
    book.mark_sent(ticket)
    assert_code("REQUEST_ALREADY_SENT", lambda: book.release(ticket))
    assert book.summary()["available_micro"] == 80


def test_price_circuit_also_blocks_already_reserved_unsent_requests(tmp_path):
    book = ledger(tmp_path / "budget.db")
    one, two = reserve(book), reserve(book, "second")
    book.mark_sent(one)
    assert_code("PRICE_RESERVATION_EXCEEDED", lambda: book.settle(one, 21))
    assert_code("BATCH_CIRCUIT_OPEN", lambda: book.mark_sent(two))


def test_live_price_source_is_hash_checked_and_policy_bound(tmp_path):
    price = {
        "policy_id": "p1",
        "verified": True,
        "currency": "CNY",
        "unit": "micro_cny",
        "rates": {"input": 1},
    }
    assert_code(
        "PRICE_POLICY_UNVERIFIED", lambda: ledger(tmp_path / "a.db", live=True, price_policy=price)
    )
    source = tmp_path / "price.json"
    source.write_text(json.dumps(price), encoding="utf-8")
    price.update(source_path=str(source), source_sha256="b" * 64)
    assert_code(
        "PRICE_POLICY_UNVERIFIED", lambda: ledger(tmp_path / "b.db", live=True, price_policy=price)
    )
    price["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    price["rates"]["input"] = 999
    assert_code(
        "PRICE_POLICY_UNVERIFIED", lambda: ledger(tmp_path / "c.db", live=True, price_policy=price)
    )


def test_live_request_requires_positive_money_reservation(tmp_path):
    price = {
        "policy_id": "p1",
        "verified": True,
        "currency": "CNY",
        "unit": "micro_cny",
        "rates": {"input": 1},
    }
    source = tmp_path / "price.json"
    source.write_text(json.dumps(price), encoding="utf-8")
    price.update(
        source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    book = ledger(tmp_path / "budget.db", live=True, price_policy=price)
    assert_code("LIVE_RESERVATION_REQUIRED", lambda: reserve(book, reserved_micro=0))
    assert book.summary()["request_count"] == 0
