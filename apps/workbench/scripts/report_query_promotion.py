"""Compare frozen 200-query runs against the promotion gate declared before optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/query_optimization_20261024"
RUNS = ROOT / "outputs/competition_upgrade_20261019/runs"
REPORT = ROOT / "reports/query_optimization_20261024"


def read(path: Path):
    return json.loads(path.read_text("utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="query_20261024_baseline")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--development", required=True)
    args = parser.parse_args()
    protocol = read(WORK / "protocol.json")
    for name, key in [
        ("test.jsonl", "test_inputs_sha256"),
        ("reference/labels.jsonl", "test_reference_sha256"),
    ]:
        actual = hashlib.sha256((ROOT / "data/eval/competition500_v1" / name).read_bytes())
        assert actual.hexdigest() == protocol[key], "Frozen test data changed"
    values = {}
    prediction_keys = (
        "query_id",
        "scene",
        "decision_state",
        "risk_labels",
        "seconds",
        "error",
        "missing_model_targets",
        "unchecked_mechanisms",
    )
    for label, run in [("baseline", args.baseline), ("candidate", args.candidate)]:
        folder = RUNS / run
        risk = read(folder / "test_summary.json")["deepseek_adaptive"]
        retrieval = read(folder / "retrieval_test_summary.json")["deepseek"]
        assert risk["queries"] == retrieval["judged_queries"] == 200
        values[label] = {"run": run, "risk": risk, "retrieval": retrieval}
        rows = read(folder / "test_deepseek_adaptive.json")
        assert len({r["query_id"] for r in rows}) == 200
        write(
            REPORT / f"{label}_predictions.json",
            [{key: row[key] for key in prediction_keys if key in row} for row in rows],
        )
        write(REPORT / f"{label}_source_manifest.json", read(folder / "run_manifest.json"))
    baseline, candidate = values["baseline"]["risk"], values["candidate"]["risk"]
    gate = protocol["promotion_gate"]
    core = ("state_accuracy", "risk_binary_accuracy", "label_micro_f1")
    delta = {
        key: candidate[key] - baseline[key]
        for key in [*core, "risk_recall", "scene_accuracy", "label_macro_f1"]
    }
    epsilon = 1e-10
    checks = {
        "meaningful_gain": max(delta[k] for k in core) + epsilon
        >= gate["minimum_gain_any_of_state_accuracy_binary_accuracy_label_micro_f1"],
        "core_no_regression": min(delta[k] for k in core) + epsilon
        >= -gate["maximum_regression_state_accuracy_binary_accuracy_label_micro_f1"],
        "risk_recall": delta["risk_recall"] + epsilon >= -gate["maximum_risk_recall_regression"],
        "scene_accuracy": delta["scene_accuracy"] + epsilon
        >= -gate["maximum_scene_accuracy_regression"],
        "query_errors": candidate["errors"] == gate["query_errors"],
        "missing_model_targets": candidate["missing_model_targets"]
        == gate["missing_model_targets"],
    }
    retrieval_delta = {}
    for metric in ("hit5", "ndcg_at_10"):
        difference = (
            values["candidate"]["retrieval"][metric] - values["baseline"]["retrieval"][metric]
        )
        retrieval_delta[metric] = difference
        checks[f"retrieval_{metric}"] = (
            difference + epsilon >= -gate["maximum_retrieval_hit5_ndcg_regression"]
        )
    report = {
        "protocol": protocol,
        "runs": values,
        "risk_delta": delta,
        "retrieval_delta": retrieval_delta,
        "gate_checks": checks,
        "promote": all(checks.values()),
        "development": {
            "baseline": read(WORK / "baseline_development/development_summary.json"),
            "candidate": read(WORK / args.development / "development_summary.json"),
        },
        "interpretation": (
            "Repeated engineering regression, not new independent or expert validation; "
            "single real-API run per version with model nondeterminism. "
            "No test-error-guided candidate selection or reference relabeling."
        ),
    }
    write(REPORT / "comparison.json", report)
    print(
        json.dumps(
            {
                "promote": report["promote"],
                "risk_delta": delta,
                "retrieval_delta": retrieval_delta,
                "gate_checks": checks,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
