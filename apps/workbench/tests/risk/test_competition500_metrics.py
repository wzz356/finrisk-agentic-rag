import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark_competition500 import ranking_metrics, risk_metrics  # noqa: E402


def test_closed_pool_recall_uses_every_relevant_case_as_denominator():
    grades = {f"c{i}": 0 for i in range(500)}
    grades["c1"] = 3
    grades["c2"] = 2
    result = ranking_metrics(
        [{"query_id": "q", "case_ids": ["c1", "c3"], "seconds": 1}], {"q": grades}
    )
    assert result["recall_at_5"] == 0.5
    assert result["hit1"] == 1
    assert result["mrr"] == 1


def test_failed_query_remains_in_classification_denominator():
    rows = [
        {
            "query_id": "q1",
            "scene": "s",
            "decision_state": "risk",
            "risk_labels": ["A"],
            "seconds": 1,
            "warnings": [],
            "chars": 80,
        },
        {
            "query_id": "q2",
            "scene": "error",
            "decision_state": "error",
            "risk_labels": [],
            "seconds": 3,
            "warnings": [],
            "chars": 80,
            "error": "timeout",
        },
    ]
    gold = {
        "q1": {"scene": "s", "decision_state": "risk", "risk_labels": ["A"]},
        "q2": {"scene": "s", "decision_state": "risk", "risk_labels": ["A"]},
    }
    result = risk_metrics(rows, gold, ["A"])
    assert result["state_accuracy"] == 0.5
    assert result["risk_recall"] == 0.5
    assert result["errors"] == 1
    assert result["label_micro_f1"] == pytest.approx(2 / 3)
