"""准备只读V2工具运行清单，不出网、不读取密钥、不重算文档向量。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_rag.contracts import canonical_json  # noqa: E402
from agent_rag.input_registry import text_hash  # noqa: E402
from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash, rows  # noqa: E402


def prepare(
    experiment_root: Path, destination: Path, scope: str, mode: str, gpu_python: Path | None = None
):
    experiment_root, destination = experiment_root.resolve(), destination.resolve()
    dataset = experiment_root / "outputs/FinRisk_可用测试集_v1_20261015"
    baseline = experiment_root / "outputs/FinRisk_知识库V2_20261012"
    full_index = baseline / "indexes/all9639_final/structured.jsonl"
    source = baseline / "normalized_final/data/documents.jsonl"
    index = full_index
    subset = None
    if (destination / "manifest.json").exists():
        raise ValueError("SNAPSHOT_ALREADY_EXISTS")
    destination.mkdir(parents=True, exist_ok=True)
    if scope == "closed100":
        subset = sorted({r["file_id"] for r in rows(dataset / "corpus.jsonl")})
        source_copy = destination / "sources.jsonl"
        with source_copy.open("x", encoding="utf-8") as handle:
            for row in rows(source):
                if row["official_file_id"] in subset:
                    handle.write(canonical_json(row) + "\n")
        source = source_copy
        index = dataset / "closed_index_units.jsonl"
    cache = experiment_root / "work/finrisk_evalset_20261015/vector_cache"
    doc_meta = cache / "5439ca546467f0674ef4a3de046941defb3c2268072fdf4927d583702ae76c07.json"
    query_meta = cache / "77bc6cccfdc5c3e6881dacd52610cbab46e11da80380d73212693c12d4e07ed3.json"
    metadata = json.loads(doc_meta.read_text(encoding="utf-8"))
    gpu_python = (
        gpu_python or experiment_root / "work/finrisk_retrieval_20261012/.venv/Scripts/python.exe"
    )
    if not gpu_python.is_file():
        raise ValueError("GPU_PYTHON_NOT_FOUND")
    dense_config = {
        "python_executable": str(gpu_python),
        "index_path": str(full_index),
        "index_sha256": file_hash(full_index),
        "vector_path": str(doc_meta.with_suffix(".npy")),
        "vector_meta_path": str(doc_meta),
        "vector_meta_sha256": file_hash(doc_meta),
        "model_path": metadata["binding"]["model"]["path"],
        "model_revision": metadata["binding"]["model"]["revision"],
        "device": "cuda",
        "precision": "float16",
        "startup_timeout_seconds": 120,
        "request_timeout_seconds": 20,
        "subset_file_ids": subset,
    }
    if mode == "replay":
        dense_config.update(
            frozen_queries_path=str(dataset / "queries.jsonl"),
            query_vectors_path=str(query_meta.with_suffix(".npy")),
            query_meta_path=str(query_meta),
            query_meta_sha256=file_hash(query_meta),
        )
    dense_path = destination / "dense-config.json"
    dense_path.write_text(canonical_json(dense_config), encoding="utf-8")
    snapshot = V2KnowledgeSnapshot.create(
        source, index, ROOT / "contracts/agent_rag/risk_type_dictionary.csv", destination
    )
    assets = {
        "dense_config": dense_path,
        "rules": ROOT / "resources/agent_rag/deepseek-rerank-rules.md",
    }
    if mode == "replay":
        assets.update(
            score_inputs=dataset / "score_inputs.jsonl",
            scores=dataset / "scores.jsonl",
            candidate_map=dataset / "candidate_map.jsonl",
        )
    snapshot.manifest.update(
        rerank_mode=mode,
        corpus_scope=scope,
        backend={
            "mode": mode,
            "assets": {
                k: {"path": str(v.resolve()), "sha256": file_hash(v)} for k, v in assets.items()
            },
        },
    )
    identity = {
        "core": snapshot.manifest["snapshot_id"],
        "scope": scope,
        "mode": mode,
        "backend_assets": {
            k: v["sha256"] for k, v in snapshot.manifest["backend"]["assets"].items()
        },
    }
    snapshot.manifest["snapshot_id"] = "v2-" + text_hash(canonical_json(identity))[:24]
    (destination / "manifest.json").write_text(canonical_json(snapshot.manifest), encoding="utf-8")
    return {
        "snapshot_id": snapshot.manifest["snapshot_id"],
        "scope": scope,
        "score_mode": mode,
        "sources": len(snapshot.sources),
        "units": len(snapshot.units),
        "model_requests_sent": 0,
        "path": str(destination),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--scope", choices=["closed100", "full9639"], default="full9639")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--gpu-python", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.experiment_root, args.destination, args.scope, args.mode, args.gpu_python),
            ensure_ascii=False,
        )
    )
