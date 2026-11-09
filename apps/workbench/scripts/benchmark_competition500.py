"""Real-API closed-500 evaluation. Answers are scorer-only, never model or retrieval input."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from statistics import mean, median
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

from build_competition_dataset import DATA, ROOT, read_rows, save
from dotenv import load_dotenv

sys.path.insert(0, str(ROOT / "src"))
from benchmark_support import RecordingLLM  # noqa: E402

from application.evidence_service import KnowledgeEvidenceService  # noqa: E402
from application.query_service import QueryProcessingService  # noqa: E402
from application.retrieval_service import CaseRetrievalService  # noqa: E402
from llm import create_llm_client_from_env  # noqa: E402
from query.classification import QuerySceneClassifier  # noqa: E402
from risk.detection import RuleRiskDetector  # noqa: E402
from risk.pipeline import RiskAssessmentPipeline  # noqa: E402
from risk.taxonomy import load_taxonomy_descriptions  # noqa: E402
from settings import AppSettings  # noqa: E402
from storage.case_repository import CaseRepository  # noqa: E402

OUT = (
    ROOT / "outputs/competition_upgrade_20261019/runs" / os.getenv("FINRISK_BENCH_RUN", "baseline")
)


def freeze_runtime():
    import llm.client as bridge

    folder = OUT / "frozen_runtime"
    if not (OUT / "run_manifest.json").is_file():
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["runtime.mjs", "worker.mjs"]:
            shutil.copy2(ROOT / "agent_runtime" / name, folder / name)
        modules = folder / "node_modules"
        if os.name == "nt" and not modules.exists():
            command = (
                "New-Item -ItemType Junction -Path '"
                + str(modules).replace("'", "''")
                + "' -Value '"
                + str(ROOT / "agent_runtime/node_modules").replace("'", "''")
                + "' | Out-Null"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command], capture_output=True
            )
            if result.returncode:
                shutil.copytree(ROOT / "agent_runtime/node_modules", modules, dirs_exist_ok=True)
        elif not modules.exists():
            modules.symlink_to(ROOT / "agent_runtime/node_modules", target_is_directory=True)
        with ZipFile(OUT / "source_snapshot.zip", "w", ZIP_DEFLATED) as archive:
            for path in (ROOT / "src").rglob("*.py"):
                archive.write(path, path.relative_to(ROOT))
        save(
            OUT / "run_manifest.json",
            {
                "source_hashes": source_hashes(),
                "calibration_enabled": os.getenv("FINRISK_LOCAL_CALIBRATION", "true"),
                "test_used_for_selection": False,
            },
        )
    else:
        manifest = json.loads((OUT / "run_manifest.json").read_text("utf-8"))
        if manifest["source_hashes"] != source_hashes():
            raise ValueError("Source changed since this run; use a new FINRISK_BENCH_RUN")
    bridge.PI_RUNTIME_DIR = folder


class PoolRepository(CaseRepository):
    def __init__(self, settings, selected):
        self.selected = set(selected)
        super().__init__(settings)

    def summaries(self):
        return tuple(self.get_summary(cid) for cid in sorted(self.selected))

    def retrieval_summaries(self):
        return self.summaries()

    def iter_retrieval_documents(self):
        for document in super().iter_retrieval_documents():
            ids = tuple(cid for cid in document.case_ids if cid in self.selected)
            if ids:
                yield replace(document, case_ids=ids)


def source_hashes():
    paths = [
        *list((ROOT / "src").rglob("*.py")),
        ROOT / "agent_runtime/runtime.mjs",
        ROOT / "data/risk/calibration/competition500.npz",
        DATA / "manifest.json",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def risk_metrics(rows, gold, labels):
    n = len(rows)
    matched = scene = tp = fp = fn = tn = 0
    per = {label: [0, 0, 0] for label in labels}
    for r in rows:
        g = gold[r["query_id"]]
        p = set(r["risk_labels"])
        actual = set(g["risk_labels"])
        matched += r["decision_state"] == g["decision_state"]
        scene += r["scene"] == g["scene"]
        a = g["decision_state"] == "risk"
        b = r["decision_state"] == "risk"
        if r["decision_state"] not in {"risk", "no_clear_risk", "insufficient_information"}:
            fn += a
            fp += not a
            for label in actual:
                if label in per:
                    per[label][2] += 1
            continue
        tp += a and b
        fp += not a and b
        fn += a and not b
        tn += not a and not b
        for label in labels:
            per[label][0] += label in p and label in actual
            per[label][1] += label in p and label not in actual
            per[label][2] += label not in p and label in actual

    def div(a, b):
        return a / b if b else 0.0

    a = sum(x[0] for x in per.values())
    b = sum(x[1] for x in per.values())
    c = sum(x[2] for x in per.values())
    times = sorted(r["seconds"] for r in rows)
    return {
        "queries": n,
        "state_accuracy": div(matched, n),
        "risk_binary_accuracy": div(tp + tn, n),
        "risk_precision": div(tp, tp + fp),
        "risk_recall": div(tp, tp + fn),
        "risk_f1": div(2 * tp, 2 * tp + fp + fn),
        "scene_accuracy": div(scene, n),
        "label_micro_f1": div(2 * a, 2 * a + b + c),
        "label_macro_f1": mean(div(2 * t, 2 * t + f + m) for t, f, m in per.values()),
        "label_supported_macro_f1": mean(
            div(2 * t, 2 * t + f + m) for t, f, m in per.values() if t + m
        )
        if any(t + m for t, f, m in per.values())
        else 0,
        "per_label": {
            label: {"tp": t, "fp": f, "fn": m, "f1": div(2 * t, 2 * t + f + m), "support": t + m}
            for label, (t, f, m) in per.items()
        },
        "median_seconds": median(times),
        "p95_seconds": times[min(n - 1, math.ceil(n * 0.95) - 1)],
        "warnings": sum(bool(r["warnings"]) for r in rows),
        "errors": sum(bool(r.get("error")) for r in rows),
        "missing_model_targets": sum(r.get("missing_model_targets", 0) for r in rows),
        "long_queries": sum(r["chars"] >= 900 for r in rows),
        "long_median_seconds": median([r["seconds"] for r in rows if r["chars"] >= 900])
        if any(r["chars"] >= 900 for r in rows)
        else None,
    }


def ranking_metrics(rankings, qrels):
    scored = []
    for row in rankings:
        grades = qrels[row["query_id"]]
        assert len(grades) == 500
        relevant = {cid for cid, g in grades.items() if g >= 2}
        if not relevant:
            continue
        ids = row["case_ids"]
        positions = [i for i, cid in enumerate(ids, 1) if cid in relevant]
        gains = sorted((2**g - 1 for g in grades.values()), reverse=True)
        ideal = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:10]))
        scored.append(
            {
                "hit1": float(bool(positions and positions[0] == 1)),
                "hit5": float(bool(positions and positions[0] <= 5)),
                "mrr": 1 / positions[0] if positions else 0,
                **{
                    f"recall_at_{k}": len(set(ids[:k]) & relevant) / len(relevant)
                    for k in [3, 5, 10, 20]
                },
                "ndcg_at_10": sum(
                    (2 ** grades[cid] - 1) / math.log2(i + 2) for i, cid in enumerate(ids[:10])
                )
                / ideal
                if ideal
                else 0,
            }
        )
    return {
        "judged_queries": len(rankings),
        "positive_queries": len(scored),
        "no_positive_queries": len(rankings) - len(scored),
        **{
            k: mean(r[k] for r in scored) if scored else 0
            for k in [
                "hit1",
                "hit5",
                "mrr",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "ndcg_at_10",
            ]
        },
        "median_seconds": median(r["seconds"] for r in rankings),
    }


async def run_risk(mode, samples, settings, repository, retrieval, *, local=False):
    name = "rules_only" if local else mode
    path = OUT / f"{samples[0]['split']}_{name}.json"
    completed = json.loads(path.read_text("utf-8")) if path.exists() else []
    have = {r["query_id"] for r in completed}
    pending = [r for r in samples if r["query_id"] not in have]
    if not pending:
        return completed
    provider, timing = mode.split("_")
    retrieval._rerank_provider = provider
    retrieval._reranker = None
    inner = None if local else create_llm_client_from_env()
    client = RecordingLLM(inner) if inner else None
    service = QueryProcessingService(classifier=QuerySceneClassifier(client))
    pipeline = RiskAssessmentPipeline(
        rule_detector=RuleRiskDetector(settings.risk_keyword_path),
        retrieval_service=retrieval,
        taxonomy=repository.risk_dictionary(),
        llm_client=client,
        taxonomy_descriptions=load_taxonomy_descriptions(settings.risk_dictionary_path),
        retrieval_timing=timing,
        retrieval_top_k=5,
        llm_concurrency=6,
        evidence_service=KnowledgeEvidenceService(settings.case_kb_root),
        agent_retrieval_enabled=False,
        longtext_screening=settings.longtext_screening,
    )
    semaphore = asyncio.Semaphore(3)

    async def one(sample):
        async with semaphore:
            start = perf_counter()
            try:
                prepared = await service.process_text(sample["text"])
                result = await pipeline.analyze(prepared)
                summary = result.risk_summary
                row = {
                    "query_id": sample["query_id"],
                    "scene": result.classification.scene.value,
                    "decision_state": summary.decision_state,
                    "risk_labels": summary.risk_labels,
                    "seconds": perf_counter() - start,
                    "chars": len(sample["text"]),
                    "warnings": summary.warnings,
                    "missing_model_targets": sum(
                        not a.llm_reviewed and a.review_status == "needs_review"
                        for a in result.risk_assessments
                    )
                    if inner
                    else 0,
                    "unchecked_mechanisms": sum(
                        len(v) for v in summary.mechanism_coverage.get("unchecked", {}).values()
                    ),
                    "matched_case_ids": [r.case_id for r in summary.top_matched_cases],
                    "stage_timings_ms": result.stage_timings_ms,
                    "result": result.model_dump(mode="json"),
                }
            except Exception as error:
                row = {
                    "query_id": sample["query_id"],
                    "scene": "error",
                    "decision_state": "error",
                    "risk_labels": [],
                    "seconds": perf_counter() - start,
                    "chars": len(sample["text"]),
                    "warnings": [],
                    "error": str(error)[:200],
                }
            completed.append(row)
            save(path, completed)
            print(
                f"risk {name} {sample['query_id']} {row['decision_state']} {row['seconds']:.2f}s",
                flush=True,
            )

    try:
        await asyncio.gather(*(one(r) for r in pending))
    finally:
        if inner:
            await inner.aclose()
        if client:
            api_path = OUT / f"{samples[0]['split']}_{name}_api.json"
            previous = json.loads(api_path.read_text("utf-8")) if api_path.exists() else []
            save(api_path, [*previous, *client.calls])
    return completed


async def run_retrieval(split, settings, repository, retrieval, *, providers=None, limit=0):
    reference_path = DATA / f"reference/qrels_{split}.jsonl"
    qrels = (
        {r["query_id"]: r["grades"] for r in read_rows(reference_path)}
        if reference_path.exists()
        else {}
    )
    samples = [r for r in read_rows(DATA / f"{split}.jsonl") if not qrels or r["query_id"] in qrels]
    if limit:
        samples = samples[:limit]
    reports = {}
    for provider in providers or ["deepseek"]:
        path = OUT / f"retrieval_{split}_{provider}.json"
        rows = json.loads(path.read_text("utf-8")) if path.exists() else []
        have = {r["query_id"] for r in rows}
        retrieval._rerank_provider = provider
        retrieval._reranker = None
        for sample in samples:
            if sample["query_id"] in have:
                continue
            start = perf_counter()
            matches = await asyncio.to_thread(retrieval.search, sample["text"], top_k=20)
            rows.append(
                {
                    "query_id": sample["query_id"],
                    "case_ids": [m.case_id for m in matches],
                    "seconds": perf_counter() - start,
                    "literal_verified": all(m.literal_verified for m in matches),
                    "warnings": list(retrieval.last_diagnostics.warnings),
                }
            )
            save(path, rows)
        reports[provider] = (
            ranking_metrics(rows, qrels)
            if qrels
            else {"reference_pending": True, "predicted_queries": len(rows)}
        )
        print("ranking " + provider + " " + json.dumps(reports[provider]), flush=True)
    save(OUT / f"retrieval_{split}_summary.json", reports)
    return reports


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["index", "dev", "dev-full", "test", "retrieval-dev", "retrieval-test"]
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", default="")
    parser.add_argument(
        "--rerank-providers",
        default="deepseek",
        help="Comma-separated selected rerankers; avoids unrequested API comparisons",
    )
    parser.add_argument(
        "--query-ids", default="", help="Comma-separated IDs for an engineering replay"
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env()
    settings = replace(settings, type_bridge_enabled=False, query_rewrite_original_anchors=0)
    freeze_runtime()
    repository = PoolRepository(settings, [r["case_id"] for r in read_rows(DATA / "cases.jsonl")])
    retrieval = CaseRetrievalService(repository, settings)
    retrieval.rebuild()
    retrieval._ensure_dense_index(allow_build=True)
    try:
        if args.phase == "index":
            save(
                OUT / "index500.json",
                {
                    "cases": 500,
                    "units": len(retrieval._index.documents),
                    "source_hashes": source_hashes(),
                },
            )
            return
        if args.phase.startswith("retrieval-"):
            split = "development" if args.phase == "retrieval-dev" else "test"
            providers = args.rerank_providers.split(",")
            if not set(providers) <= {"none", "cloud", "deepseek"}:
                raise ValueError("Unknown reranker")
            await run_retrieval(
                split, settings, repository, retrieval, providers=providers, limit=args.limit
            )
            return
        gold = {r["query_id"]: r for r in read_rows(DATA / "reference/labels.jsonl")}
        split = "test" if args.phase == "test" else "development"
        samples = read_rows(DATA / f"{split}.jsonl")
        if args.query_ids:
            requested_ids = set(args.query_ids.split(","))
            samples = [r for r in samples if r["query_id"] in requested_ids]
            if {r["query_id"] for r in samples} != requested_ids:
                raise ValueError("Unknown query ID in selected split")
        if args.limit:
            samples = samples[: args.limit]
        if args.phase == "dev":
            samples = samples[:36]
            summary = {}
            for provider in ["none", "cloud", "deepseek"]:
                for timing in ["post", "pre", "adaptive"]:
                    mode = provider + "_" + timing
                    rows = await run_risk(mode, samples, settings, repository, retrieval)
                    summary[mode] = risk_metrics(rows, gold, repository.risk_dictionary())
                    save(OUT / "development_matrix.json", summary)
            ranking = (
                json.loads((OUT / "retrieval_development_summary.json").read_text("utf-8"))
                if (OUT / "retrieval_development_summary.json").exists()
                else {}
            )
            winner = max(
                summary,
                key=lambda k: (
                    summary[k]["state_accuracy"],
                    summary[k]["label_micro_f1"],
                    ranking.get(k.split("_")[0], {}).get("ndcg_at_10", 0),
                    -summary[k]["median_seconds"],
                ),
            )
            save(
                OUT / "selection.json",
                {
                    "mode": winner,
                    "selection_set": "development_first36",
                    "source_hashes": source_hashes(),
                    "test_used_for_selection": False,
                    "metrics": summary[winner],
                },
            )
        else:
            selection = (
                json.loads((OUT / "selection.json").read_text("utf-8"))
                if (OUT / "selection.json").exists()
                else {"mode": args.mode}
            )
            modes = (
                [args.mode]
                if args.mode
                else (["none_post", selection["mode"]] if split == "test" else [selection["mode"]])
            )
            summary = {}
            for mode in dict.fromkeys(modes):
                rows = await run_risk(mode, samples, settings, repository, retrieval)
                summary[mode] = risk_metrics(rows, gold, repository.risk_dictionary())
            rows = await run_risk("none_post", samples, settings, repository, retrieval, local=True)
            summary["rules_only"] = risk_metrics(rows, gold, repository.risk_dictionary())
            save(OUT / f"{split}_summary.json", summary)
    finally:
        retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
