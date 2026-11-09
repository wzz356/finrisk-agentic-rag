"""Select defaults exclusively from development predictions and source-grounded relevance."""

import argparse
import hashlib
import json
import os

from benchmark_competition500 import ranking_metrics, risk_metrics
from build_competition_dataset import DATA, ROOT, read_rows, save

from risk.taxonomy import load_taxonomy_descriptions
from settings import AppSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=os.getenv("FINRISK_BENCH_RUN", "reproduction"))
    args = parser.parse_args()
    base = ROOT / "outputs/competition_upgrade_20261019"
    run = base / "runs" / args.run
    protocol = {
        "risk_validation_queries": 100,
        "retrieval_validation_queries": 60,
        "selection_uses": "development_only",
        "weights": {
            "state_accuracy": 0.45,
            "label_micro_f1": 0.30,
            "retrieval_ndcg_at_10": 0.20,
            "retrieval_hit5": 0.05,
        },
        "tie_break": "median_seconds",
        "test_used_for_selection": False,
    }
    dev_ids = {r["query_id"] for r in read_rows(DATA / "development.jsonl")[:100]}
    gold = {
        r["query_id"]: r
        for r in read_rows(DATA / "reference/labels.jsonl")
        if r["query_id"] in dev_ids
    }
    taxonomy = list(load_taxonomy_descriptions(AppSettings.from_env().risk_dictionary_path))
    qrels = {
        r["query_id"]: r["grades"] for r in read_rows(DATA / "reference/qrels_development.jsonl")
    }
    assert len(qrels) == 60
    ranking = {}
    for provider in ["none", "cloud", "deepseek"]:
        path = run / f"retrieval_development_{provider}.json"
        rows = json.loads(path.read_text("utf-8"))
        assert len(rows) == 60 and all(r["query_id"].startswith("DEV") for r in rows)
        ranking[provider] = ranking_metrics(rows, qrels)
    candidates = {}
    for mode in ["none_adaptive", "cloud_adaptive", "deepseek_adaptive"]:
        rows = json.loads((run / f"development_{mode}.json").read_text("utf-8"))
        rows = [r for r in rows if r["query_id"] in dev_ids]
        assert len(rows) == 100
        metrics = risk_metrics(rows, gold, taxonomy)
        retrieval = ranking[mode.split("_")[0]]
        weights = protocol["weights"]
        score = (
            weights["state_accuracy"] * metrics["state_accuracy"]
            + weights["label_micro_f1"] * metrics["label_micro_f1"]
            + weights["retrieval_ndcg_at_10"] * retrieval["ndcg_at_10"]
            + weights["retrieval_hit5"] * retrieval["hit5"]
        )
        score -= 0.05 * (metrics["errors"] + metrics["missing_model_targets"]) / 100
        candidates[mode] = {"score": score, "risk": metrics, "retrieval": retrieval}
    winner = max(
        candidates, key=lambda k: (candidates[k]["score"], -candidates[k]["risk"]["median_seconds"])
    )
    result = {
        "mode": winner,
        "calibration_enabled": True,
        "selection_set": "development100_and_relevance60",
        "test_used_for_selection": False,
        "protocol": protocol,
        "candidates": candidates,
        "development_qrels_sha256": hashlib.sha256(
            (DATA / "reference/qrels_development.jsonl").read_bytes()
        ).hexdigest(),
    }
    save(base / "selection_frozen.json", result)
    old = run / "selection.json"
    if old.is_file() and not (run / "selection_coarse.json").exists():
        save(run / "selection_coarse.json", json.loads(old.read_text("utf-8")))
    save(old, result)
    save(run / "retrieval_development_summary.json", ranking)
    print(
        json.dumps(
            {
                "selected": winner,
                "development_scores": {k: round(v["score"], 5) for k, v in candidates.items()},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
