"""Evaluate only development data, keeping original test labels outside candidate selection."""

import argparse
import asyncio
import json
from dataclasses import replace

import benchmark_competition500 as bench
from dotenv import load_dotenv

from application.retrieval_service import CaseRetrievalService
from settings import AppSettings


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--boundary-split", choices=["validation", "train"], default="validation")
    args = parser.parse_args()
    load_dotenv(bench.ROOT / ".env")
    bench.OUT = bench.ROOT / "outputs/query_optimization_20261024" / args.run
    bench.freeze_runtime()
    settings = replace(
        AppSettings.from_env(), type_bridge_enabled=False, query_rewrite_original_anchors=0
    )
    repository = bench.PoolRepository(
        settings, [r["case_id"] for r in bench.read_rows(bench.DATA / "cases.jsonl")]
    )
    retrieval = CaseRetrievalService(repository, settings)
    retrieval.rebuild()
    retrieval._ensure_dense_index(allow_build=True)
    development = bench.read_rows(bench.DATA / "development.jsonl")[: args.dev_limit]
    dev_ids = {r["query_id"] for r in development}
    reference = {
        r["query_id"]: r
        for r in bench.read_rows(bench.DATA / "reference/labels.jsonl")
        if r["query_id"] in dev_ids
    }
    boundaries = bench.read_rows(
        bench.ROOT / f"data/eval/query_boundaries_20261024/{args.boundary_split}.jsonl"
    )
    reference.update({r["query_id"]: r for r in boundaries})
    samples = [*development, *[{**r, "split": "development"} for r in boundaries]]
    try:
        rows = await bench.run_risk("deepseek_adaptive", samples, settings, repository, retrieval)
        report = {}
        for name, selected in [
            ("existing_development", dev_ids),
            ("new_boundaries", {r["query_id"] for r in boundaries}),
        ]:
            values = [r for r in rows if r["query_id"] in selected]
            if values:
                report[name] = bench.risk_metrics(values, reference, repository.risk_dictionary())
        bench.save(bench.OUT / "development_summary.json", report)
        print(
            json.dumps(
                {
                    k: {
                        key: v[key]
                        for key in [
                            "queries",
                            "state_accuracy",
                            "risk_binary_accuracy",
                            "risk_recall",
                            "label_micro_f1",
                            "errors",
                            "missing_model_targets",
                        ]
                    }
                    for k, v in report.items()
                },
                ensure_ascii=False,
            )
        )
    finally:
        retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
