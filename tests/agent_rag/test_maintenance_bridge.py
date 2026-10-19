"""Host-side cumulative token admission survives role and process boundaries."""

import pytest

from agent_rag.maintenance_bridge import NodeMaintenanceRunner
from agent_rag.maintenance_budget import BudgetLedger


def packet(request, units):
    return {
        "request_id": request,
        "payload_sha256": "a" * 64,
        "prompt_token_upper_bound": units - 10,
        "max_tokens": 10,
        "reserved_token_units": units,
        "token_counter_kind": "conservative_byte_upper_bound",
    }


def test_token_reservations_share_event_limit_and_survive_new_runner(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", "fixture", 0, 28)
    NodeMaintenanceRunner._reserve_tokens(ledger, packet("proposer-1", 60), "event", 100)
    NodeMaintenanceRunner._reserve_tokens(ledger, packet("proposer-1", 60), "event", 100)
    reopened = BudgetLedger(tmp_path / "budget.sqlite3", "fixture", 0, 28)
    with pytest.raises(ValueError, match="EVENT_TOKEN_BUDGET_EXHAUSTED"):
        NodeMaintenanceRunner._reserve_tokens(reopened, packet("reviewer-1", 41), "event", 100)
    NodeMaintenanceRunner._reserve_tokens(reopened, packet("reviewer-1", 40), "event", 100)
    assert NodeMaintenanceRunner.token_summary(reopened) == {
        "counter_kind": "conservative_byte_upper_bound",
        "reservation_rule": "no_refund",
        "reserved_units": 100,
        "by_event": {"event": 100},
    }


def test_token_reservations_reject_unknown_or_changed_counter(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", "fixture", 0, 28)
    with pytest.raises(ValueError, match="TOKEN_COUNTER_REQUIRED"):
        NodeMaintenanceRunner._reserve_tokens(ledger, {}, "event", 100)
    NodeMaintenanceRunner._reserve_tokens(ledger, packet("one", 20), "event", 100)
    with pytest.raises(ValueError, match="TOKEN_REQUEST_BINDING"):
        NodeMaintenanceRunner._reserve_tokens(ledger, packet("one", 21), "event", 100)
