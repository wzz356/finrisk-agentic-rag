"""Fit the local candidate classifier on 400 development queries, without reading test labels."""

import argparse
import hashlib
from pathlib import Path

from build_competition_dataset import DATA, ROOT, read_rows, save

from risk.local_calibration import train_model
from risk.taxonomy import load_taxonomy_descriptions
from settings import AppSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-training", type=Path)
    args = parser.parse_args()
    rows = read_rows(DATA / "development.jsonl")
    ids = {r["query_id"] for r in rows}
    labels = {
        r["query_id"]: r for r in read_rows(DATA / "reference/labels.jsonl") if r["query_id"] in ids
    }
    settings = AppSettings.from_env()
    taxonomy = list(load_taxonomy_descriptions(settings.risk_dictionary_path))
    path = ROOT / "data/risk/calibration/competition500.npz"
    extra = read_rows(args.boundary_training) if args.boundary_training else []
    if any(r.get("split") != "train" for r in extra):
        raise ValueError("只能使用声明为train的边界样本")
    all_rows = [*rows, *extra]
    references = {**labels, **{r["query_id"]: r for r in extra}}
    metadata = [
        {
            "candidate_labels": references[r["query_id"]].get("candidate_labels", []),
            "missing_facts": references[r["query_id"]].get("missing_facts", []),
        }
        for r in all_rows
    ]
    train_model(
        [r["text"] for r in all_rows],
        [references[r["query_id"]]["decision_state"] for r in all_rows],
        [references[r["query_id"]]["risk_labels"] for r in all_rows],
        taxonomy,
        path,
        example_metadata=metadata,
    )
    save(
        path.with_suffix(".json"),
        {
            "training_queries": len(all_rows),
            "training_ids": sorted(r["query_id"] for r in all_rows),
            "boundary_training_queries": len(extra),
            "boundary_training_sha256": hashlib.sha256(
                args.boundary_training.read_bytes()
            ).hexdigest()
            if args.boundary_training
            else None,
            "validation_queries_used": 0,
            "taxonomy_size": len(taxonomy),
            "test_queries_used": 0,
            "training_reference_sha256": hashlib.sha256(
                (DATA / "reference/labels.jsonl").read_bytes()
            ).hexdigest(),
            "training_input_sha256": hashlib.sha256(
                (DATA / "development.jsonl").read_bytes()
            ).hexdigest(),
            "reference_version": "competition500-codex-reviewed-v2-20261019",
            "training_api_calls": 0,
            "method": "hashed character 2/3-gram, softmax states and 27 sigmoid labels",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    print(f"Local candidate model trained on {len(all_rows)} declared training queries.")


if __name__ == "__main__":
    main()
