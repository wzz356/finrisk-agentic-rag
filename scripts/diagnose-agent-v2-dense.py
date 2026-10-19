"""Read-only, local-GPU BGE shape experiment; never imports a retrieval service."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_LENGTH = 512
CONDITIONS = [
    {"name": "batch16_dynamic", "batch_size": 16, "padding": True},
    {"name": "batch1_dynamic", "batch_size": 1, "padding": True},
    {"name": "batch16_fixed512", "batch_size": 16, "padding": "max_length"},
    {"name": "batch1_fixed512", "batch_size": 1, "padding": "max_length"},
]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_sha(path, expected):
    actual = sha_file(path)
    if actual != expected:
        raise ValueError(f"ASSET_SHA_MISMATCH: {path}")
    return actual


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


@contextmanager
def offline_only():
    def denied(*args, **kwargs):
        raise RuntimeError("NETWORK_FORBIDDEN")

    original = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = original


def validate_profile(config):
    if (config.get("model_revision") != MODEL_REVISION
            or Path(config.get("model_path", "")).name != MODEL_REVISION):
        raise ValueError("MODEL_REVISION_MISMATCH")
    if config.get("device") != "cuda" or config.get("precision") != "float16":
        raise ValueError("ENCODER_PROFILE_MISMATCH")


def ordered_text_sha(rows):
    digest = hashlib.sha256()
    for row in rows:
        line = json.dumps([row["unit_id"], row["text"]], ensure_ascii=False,
                          separators=(",", ":"))
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def validate_cache(meta, rows, kind):
    binding = meta["binding"]
    expected = {"tokenizer_class": "BertTokenizerFast", "token_audit_schema_version": 2,
                "max_length": MAX_LENGTH, "query_instruction": QUERY_INSTRUCTION
                if kind == "query" else "", "pooling": "CLS", "normalization": "L2",
                "torch": "2.6.0+cu124", "transformers": "4.57.1", "precision": "float16",
                "kind": kind}
    if (any(binding.get(k) != v for k, v in expected.items())
            or binding.get("model", {}).get("revision") != MODEL_REVISION
            or meta.get("shape") != [len(rows), 512]):
        raise ValueError("CACHE_PROFILE_MISMATCH")
    text_sha = ordered_text_sha(rows)
    key = hashlib.sha256(json.dumps({"text_sha": text_sha, "config": binding},
                                   sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if meta.get("ordered_unit_text_sha256") != text_sha or meta.get("key") != key:
        raise ValueError("CACHE_INPUT_BINDING_MISMATCH")
    audit = meta.get("token_audit", [])
    if len(audit) != len(rows) or any(
        token.get("id") != row["unit_id"] or token.get("truncated") is not False
        or not 0 < token.get("input_tokens", 0) <= MAX_LENGTH
        or token.get("used_tokens") != token.get("input_tokens")
        for row, token in zip(rows, audit)
    ):
        raise ValueError("CACHE_TOKEN_AUDIT_MISMATCH")


def load_assets(experiment_root, trial_root):
    """Validate manifests, original code, text, cache and model before model loading."""
    root, trial = Path(experiment_root).resolve(), Path(trial_root).resolve()
    dataset = root / "outputs/FinRisk_可用测试集_v1_20261015"
    work = root / "work/finrisk_evalset_20261015"
    config_path = trial / "snapshot-live/dense-config.json"
    manifest_path = trial / "snapshot-live/manifest.json"
    manifest, config = read_json(manifest_path), read_json(config_path)
    checked_sha(config_path, manifest["backend"]["assets"]["dense_config"]["sha256"])
    validate_profile(config)
    expected_python = root / "work/finrisk_retrieval_20261012/.venv/Scripts/python.exe"
    if (Path(config["python_executable"]).resolve() != expected_python.resolve()
            or Path(sys.executable).resolve() != expected_python.resolve()):
        raise ValueError("PYTHON_EXECUTABLE_BINDING_MISMATCH")
    tracked = [config_path, manifest_path, expected_python]
    # Both freezes are originals. Every referenced input/code is checked, never rewritten.
    for name in ("input_freeze.json", "candidate_freeze.json"):
        path = dataset / name
        freeze = read_json(path)
        tracked.append(path)
        for section in ("sha256", "code_sha256"):
            for source, expected in freeze.get(section, {}).items():
                checked_sha(Path(source), expected)
                tracked.append(Path(source))
    dataset_manifest = dataset / "dataset_manifest.json"
    query_path = dataset / "queries.jsonl"
    frozen = read_json(dataset_manifest)
    expected_queries = next(row["sha256"] for row in frozen["files"]
                            if row["path"] == "queries.jsonl")
    checked_sha(query_path, expected_queries)
    tracked.extend([dataset_manifest, query_path])
    queries = read_rows(query_path)
    if (len(queries) != 20 or [q.get("query_id") for q in queries]
            != [f"KBT{i:03}" for i in range(1, 21)] or any(
                q.get("text_sha256") != hashlib.sha256(q["text"].encode("utf-8")).hexdigest()
                for q in queries)):
        raise ValueError("FROZEN_QUERY_INPUT_MISMATCH")
    log = read_json(dataset / "local_model_log.json")
    query_event = next(event for event in log["cache_events"] if event["kind"] == "query")
    query_meta_path = work / "vector_cache" / (query_event["key"] + ".json")
    query_vector_path = query_meta_path.with_suffix(".npy")
    index_path = Path(config["index_path"])
    checked_sha(index_path, config["index_sha256"])
    checked_sha(index_path, manifest["index_sha256"])
    document_meta_path = Path(config["vector_meta_path"])
    checked_sha(document_meta_path, config["vector_meta_sha256"])
    doc_meta, query_meta = read_json(document_meta_path), read_json(query_meta_path)
    rows = read_rows(index_path)
    if len(rows) != 65475 or len({r["unit_id"] for r in rows}) != len(rows):
        raise ValueError("DOCUMENT_INDEX_BINDING_MISMATCH")
    validate_cache(doc_meta, rows, "document")
    validate_cache(query_meta, [{"unit_id": q["query_id"], "text": QUERY_INSTRUCTION + q["text"]}
                                for q in queries], "query")
    if doc_meta["binding"]["model"] != query_meta["binding"]["model"]:
        raise ValueError("QUERY_DOCUMENT_MODEL_MISMATCH")
    model = Path(config["model_path"])
    fingerprint = doc_meta["binding"]["model"]
    files = []
    for path in sorted(model.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(model).as_posix(), "size": path.stat().st_size,
                          "sha256": sha_file(path)})
            tracked.append(path)
    model_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    if (files != fingerprint["files"] or model_sha != fingerprint["cache_sha256"]
            or Path(fingerprint["path"]).resolve() != model.resolve()):
        raise ValueError("MODEL_FILE_BINDING_MISMATCH")
    document_vector_path = Path(config["vector_path"])
    checked_sha(document_vector_path, doc_meta["npy_sha256"])
    checked_sha(query_vector_path, query_meta["npy_sha256"])
    tracked.extend([index_path, document_meta_path, document_vector_path,
                    query_meta_path, query_vector_path])
    tracked.extend(path for path in trial.rglob("*") if path.is_file())
    routes = {r["query_id"]: r["BGE"] for r in read_rows(dataset / "routes.jsonl")
              if r["scope"] == "full9639"}
    if set(routes) != {q["query_id"] for q in queries}:
        raise ValueError("FROZEN_ROUTE_BINDING_MISMATCH")
    return {"config": config, "queries": queries, "unit_ids": [r["unit_id"] for r in rows],
            "document_vector_path": document_vector_path, "query_vector_path": query_vector_path,
            "tracked": sorted(set(tracked)), "historical_bge": routes, "original_runtime": log,
            "query_token_audit": query_meta["token_audit"]}


class LocalEncoder:
    def __init__(self, config, queries):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1")
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        if (torch.__version__ != "2.6.0+cu124" or transformers.__version__ != "4.57.1"
                or not torch.cuda.is_available()):
            raise ValueError("GPU_RUNTIME_BINDING_MISMATCH")
        self.torch = torch
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True, trust_remote_code=False)
        if type(self.tokenizer).__name__ != "BertTokenizerFast":
            raise ValueError("TOKENIZER_BINDING_MISMATCH")
        self.model = AutoModel.from_pretrained(config["model_path"], local_files_only=True,
                                               trust_remote_code=False, torch_dtype=torch.float16)
        self.model.to("cuda").eval()
        torch.cuda.synchronize()
        self.info = {"python_executable": sys.executable, "python": sys.version,
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
                     "torch_cpu_threads": torch.get_num_threads(),
                     "torch_interop_threads": torch.get_num_interop_threads(),
                     "model_load_seconds": time.perf_counter() - started,
                     "attention_implementation": self.model.config._attn_implementation,
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                     "model_dtype": str(next(self.model.parameters()).dtype)}
        started = time.perf_counter()
        self.encode_batch(queries[:16], CONDITIONS[0])
        self.info["warmup_seconds"] = time.perf_counter() - started

    def encode_batch(self, rows, condition):
        texts = [QUERY_INSTRUCTION + row["text"] for row in rows]
        unlimited = self.tokenizer(texts, truncation=False, padding=False)
        if any(len(ids) > MAX_LENGTH for ids in unlimited["input_ids"]):
            raise ValueError("QUERY_TRUNCATION_FORBIDDEN")
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        tokens = self.tokenizer(texts, padding=condition["padding"], truncation=True,
                                max_length=MAX_LENGTH, return_tensors="pt")
        audit = [{"query_id": row["query_id"], "query_instruction": QUERY_INSTRUCTION,
                  "input_tokens": len(unlimited["input_ids"][i]),
                  **{key: value[i].tolist() for key, value in tokens.items()}}
                 for i, row in enumerate(rows)]
        tokens = {key: value.to("cuda") for key, value in tokens.items()}
        with self.torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state[:, 0]
            vectors = self.torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
        result = vectors.cpu().numpy().astype("float32").tolist()
        self.torch.cuda.synchronize()
        return result, audit, time.perf_counter() - started


def ranking(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:100]


def compare(left, right, ids):
    rows = []
    for query_id in ids:
        a, b = left[query_id], right[query_id]
        vector_diff = [abs(x - y) for x, y in zip(a["vector"], b["vector"])]
        score_diff = [abs(x - y) for x, y in zip(a["scores"], b["scores"])]
        a_order, b_order = a["order"], b["order"]
        rows.append({"query_id": query_id, "max_abs_vector_difference": max(vector_diff),
                     "l2_vector_difference": math.sqrt(sum(d * d for d in vector_diff)),
                     "max_abs_score_difference": max(score_diff),
                     "scores_with_difference_gt_1e_6": sum(d > 1e-6 for d in score_diff),
                     "top100_set_identical": set(a_order) == set(b_order),
                     "top100_order_identical": a_order == b_order,
                     "top100_overlap": len(set(a_order) & set(b_order)),
                     "top100_rank_positions_changed": sum(x != y for x, y in zip(a_order, b_order)),
                     "same_unit_top100_max_score_difference": max(score_diff[i]
                         for i in set(a_order) | set(b_order))})
    return {"max_abs_vector_difference": max(r["max_abs_vector_difference"] for r in rows),
            "max_abs_score_difference": max(r["max_abs_score_difference"] for r in rows),
            "top100_set_identical_queries": sum(r["top100_set_identical"] for r in rows),
            "top100_order_identical_queries": sum(r["top100_order_identical"] for r in rows),
            "per_query": rows}


def diagnose(queries, frozen_vectors, score_fn, unit_ids, encoder, output, source_paths,
             historical_bge=None, source_before=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("DIAGNOSTIC_OUTPUT_EXISTS")
    before = source_before or {str(p): sha_file(p) for p in source_paths}
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "finrisk-dense-diagnostic/1.0", "repetitions": 3,
              "model_revision": MODEL_REVISION, "query_instruction": QUERY_INSTRUCTION,
              "precision": "float16", "normalization": "float32 L2", "pooling": "CLS",
              "max_length": MAX_LENGTH, "score_tolerance": 1e-6,
              "query_order": [q["query_id"] for q in queries], "runtime": encoder.info,
              "external_requests": 0, "network_policy": "socket connect forbidden",
              "source_assets_before": before, "phases": []}
    try:
        with offline_only(), (output / "tokens.jsonl").open("w", encoding="utf-8") as tokens_out, \
                (output / "vectors.jsonl").open("w", encoding="utf-8") as vectors_out, \
                (output / "top100.jsonl").open("w", encoding="utf-8") as hits_out:
            frozen = {}
            for row, vector in zip(queries, frozen_vectors):
                scores = score_fn(vector)
                frozen[row["query_id"]] = {"vector": vector, "scores": scores,
                                           "order": ranking(scores)}
            if historical_bge is not None:
                replay = []
                for query_id, data in frozen.items():
                    actual = [{"unit_id": unit_ids[i], "score": data["scores"][i], "rank": rank}
                              for rank, i in enumerate(data["order"], 1)]
                    replay.append({"query_id": query_id,
                                   "exact_ids_order_scores": actual == historical_bge[query_id]})
                report["frozen_replay"] = {"all_exact": all(r["exact_ids_order_scores"]
                                                           for r in replay), "per_query": replay}
            for phase_name, wanted in [("KBT005_first", {"KBT005"}),
                                       ("frozen20", set(report["query_order"]))]:
                ids = [q["query_id"] for q in queries if q["query_id"] in wanted]
                phase = {"name": phase_name, "query_ids": ids, "conditions": [], "comparisons": {}}
                results = {}
                for condition in CONDITIONS:
                    repeats, timings = [], []
                    for repeat in range(1, 4):
                        result, seconds = {}, 0.0
                        for start in range(0, len(queries), condition["batch_size"]):
                            batch = queries[start:start + condition["batch_size"]]
                            if not any(row["query_id"] in wanted for row in batch):
                                continue
                            vectors, token_rows, elapsed = encoder.encode_batch(batch, condition)
                            if len(vectors) != len(batch):
                                raise ValueError("ENCODER_ROW_COUNT_MISMATCH")
                            seconds += elapsed
                            label = {"phase": phase_name, "condition": condition["name"],
                                     "repeat": repeat,
                                     "batch_query_ids": [r["query_id"] for r in batch]}
                            tokens_out.write(json.dumps({**label, "tokens": token_rows},
                                                        ensure_ascii=False, allow_nan=False) + "\n")
                            for row, vector in zip(batch, vectors):
                                query_id = row["query_id"]
                                if query_id not in wanted:
                                    continue
                                scores = score_fn(vector)
                                if (len(scores) != len(unit_ids)
                                        or not all(math.isfinite(v) for v in vector + scores)):
                                    raise ValueError("INVALID_ENCODING_OR_SCORES")
                                order = ranking(scores)
                                result[query_id] = {"vector": vector, "scores": scores,
                                                    "order": order}
                                vectors_out.write(json.dumps({**label, "query_id": query_id,
                                                              "vector": vector},
                                                             allow_nan=False) + "\n")
                                hits_out.write(json.dumps({**label, "query_id": query_id, "hits": [
                                    {"unit_id": unit_ids[i], "score": scores[i], "rank": rank}
                                    for rank, i in enumerate(order, 1)]}, allow_nan=False) + "\n")
                        repeats.append(result)
                        timings.append(seconds)
                    stability = [compare(repeats[a], repeats[b], ids)
                                 for a, b in itertools.combinations(range(3), 2)]
                    phase["conditions"].append({**condition, "encoding_seconds": timings,
                        "repeat_bitwise_stable": all(a == b for a, b in zip(repeats, repeats[1:])),
                        "repeat_pair_comparisons": stability,
                        "versus_frozen": [compare(frozen, result, ids) for result in repeats]})
                    results[condition["name"]] = repeats
                    print(json.dumps({"phase": phase_name, "condition": condition["name"],
                                      "repeats_completed": 3}), flush=True)
                for left, right in [(0, 1), (0, 2), (1, 3), (2, 3)]:
                    a, b = CONDITIONS[left]["name"], CONDITIONS[right]["name"]
                    values = [compare(results[a][i], results[b][i], ids) for i in range(3)]
                    phase["comparisons"][a + "__" + b] = values[0] | {"all_repeats": values}
                report["phases"].append(phase)
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        after = {str(p): sha_file(p) if Path(p).is_file() else None for p in source_paths}
        report["source_assets_after"] = after
        report["source_assets_unchanged"] = before == after
        write_json(output / "report.json", report)
        write_markdown(output / "report.md", report)
    if not report["source_assets_unchanged"]:
        raise ValueError("SOURCE_ASSET_MUTATED")
    return report


def write_markdown(path, report):
    lines = ["# BGE 本地输入形状诊断", "", f"模型 revision：`{MODEL_REVISION}`。",
             "FP16 模型、CLS、float32 L2、原文档向量保持不变；未取整，门槛仍为 1e-6。",
             "", f"原资产 SHA 前后相同：{report['source_assets_unchanged']}。外部请求：0。", "",
             "| 阶段 | 条件 | 相对冻结最大向量差 | 相对冻结最大分差 | "
             "Top100 集合一致 | 顺序一致 | 三次完全稳定 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for phase in report["phases"]:
        for row in phase["conditions"]:
            compared = row["versus_frozen"][0]
            lines.append(f"| {phase['name']} | {row['name']} | "
                         f"{compared['max_abs_vector_difference']:.12g} | "
                         f"{compared['max_abs_score_difference']:.12g} | "
                         f"{compared['top100_set_identical_queries']}/{len(phase['query_ids'])} | "
                         f"{compared['top100_order_identical_queries']}/"
                         f"{len(phase['query_ids'])} | "
                         f"{row['repeat_bitwise_stable']} |")
    lines.extend(["", "完整逐查询、三次复现和单变量比较见 report.json；原始向量、tokens/masks、",
                  "Top100 原始分数分别见 vectors.jsonl、tokens.jsonl、top100.jsonl。",
                  "该表只报告观测；需要结合单变量比较判断输入形状影响，"
                  "不能据此认定具体 CUDA 算子。"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--trial-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("DIAGNOSTIC_OUTPUT_EXISTS")
    with offline_only():
        started = time.perf_counter()
        assets = load_assets(args.experiment_root, args.trial_root)
        before = {str(p): sha_file(p) for p in assets["tracked"]}
        validation_seconds = time.perf_counter() - started
        import numpy as np

        documents = np.load(assets["document_vector_path"], allow_pickle=False, mmap_mode="r")
        frozen = np.load(assets["query_vector_path"], allow_pickle=False, mmap_mode="r")
        if (documents.dtype != np.float32 or documents.shape != (65475, 512)
                or frozen.dtype != np.float32 or frozen.shape != (20, 512)):
            raise ValueError("VECTOR_SHAPE_DTYPE_MISMATCH")
        encoder = LocalEncoder(assets["config"], assets["queries"])
        encoder.info.update(numpy=np.__version__, source_validation_seconds=validation_seconds,
                            original_runtime={k: assets["original_runtime"].get(k) for k in
                                ("python", "torch", "transformers", "numpy", "torch_cpu_threads",
                                 "torch_interop_threads", "gpu", "cuda_runtime")})

        def scores(vector):
            return (documents @ np.asarray(vector, dtype="float32")).tolist()

        report = diagnose(assets["queries"], frozen.tolist(), scores, assets["unit_ids"],
                          encoder, args.output, assets["tracked"], assets["historical_bge"], before)
    print(json.dumps({"output": str(args.output.resolve()),
                      "frozen_replay_exact": report["frozen_replay"]["all_exact"],
                      "source_assets_unchanged": report["source_assets_unchanged"]}), flush=True)


if __name__ == "__main__":
    main()
