"""加载受信运行清单；收费评分与已冻结评分回放明确分开。"""

import json
import os
from pathlib import Path
from typing import Any

from agent_rag.dense_backend import ResidentDenseIndex, validate_startup_timeout_seconds
from agent_rag.input_registry import text_hash
from agent_rag.rerank_backend import DeepSeekReranker, validate_completion
from agent_rag.v2_backend import HybridV2Backend
from agent_rag.v2_context import validate_context_expansion
from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash, rows


class FrozenFlashReranker:
    def __init__(self, inputs_path: Path, scores_path: Path, mapping_path: Path):
        packets = {row["pair_id"]: row for row in rows(inputs_path)}
        scores = {row["pair_id"]: row for row in rows(scores_path)}
        if set(packets) != set(scores):
            raise ValueError("REPLAY_PAIR_SET_MISMATCH")
        self.scores = {}
        for mapping in rows(mapping_path):
            pid = mapping["pair_id"]
            packet, score = packets[pid], scores[pid]
            if (
                packet["query_text_sha256"] != text_hash(packet["query_text"])
                or packet["unit_text_sha256"] != text_hash(packet["unit_text"])
                or score.get("query_text_sha256") != packet["query_text_sha256"]
                or score.get("unit_text_sha256") != packet["unit_text_sha256"]
                or score.get("origin") != "actual_deepseek_flash_API"
            ):
                raise ValueError("REPLAY_TEXT_BINDING_MISMATCH")
            opinion = validate_completion(
                packet,
                {
                    "model": score.get("returned_model"),
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(score, ensure_ascii=False)},
                        }
                    ],
                },
            )
            key = (packet["query_text_sha256"], mapping["unit_id"], packet["unit_text_sha256"])
            value = {
                **opinion,
                "unit_id": mapping["unit_id"],
                "file_id": mapping["file_id"],
                "origin": "frozen_actual_API_score_replay",
                "response_id": score.get("response_id"),
            }
            if key in self.scores and self.scores[key] != value:
                raise ValueError("REPLAY_DUPLICATE_BINDING")
            self.scores[key] = value

    def score_query(self, query_text: str, units: list[dict], control: Any = None) -> dict:
        found, errors = [], []
        for unit in units:
            key = (text_hash(query_text), unit["unit_id"], text_hash(unit["text"]))
            score = self.scores.get(key)
            if score is None or score["file_id"] != unit["file_id"]:
                errors.append({"unit_id": unit["unit_id"], "code": "REPLAY_SCORE_NOT_AVAILABLE"})
            else:
                found.append(dict(score))
        return {
            "scores": found,
            "complete": not errors,
            "errors": errors,
            "meta": {
                "external_retrieval_requests": 0,
                "cache_hits": len(found),
                "rerank_mode": "replay",
                "model": "deepseek-flash",
                "scores_not_new_API_calls": True,
            },
        }


def load_v2_snapshot(
    root: Path, *, startup_timeout_seconds: float | None = None,
    result_cache_size: int = 128, rerank_cache_size: int = 4096, audit_root: Path | None = None
) -> V2KnowledgeSnapshot:
    validate_startup_timeout_seconds(startup_timeout_seconds)
    snapshot = V2KnowledgeSnapshot(root)
    backend = snapshot.manifest.get("backend")
    if not backend or backend.get("mode") not in {"live", "replay"}:
        raise ValueError("INVALID_V2_BACKEND_CONFIG")
    context_expansion = validate_context_expansion(backend.get("context_expansion"))
    protocol = backend.get("rerank_protocol", "quote-copy-v1")
    if protocol not in {"quote-copy-v1", "span-ref-v2"} or backend["mode"] == "replay" and protocol != "quote-copy-v1":
        raise ValueError("RERANK_PROTOCOL_NOT_READY")
    paths = {}
    for name, asset in backend["assets"].items():
        path = Path(asset["path"])
        if file_hash(path) != asset["sha256"]:
            raise ValueError("BACKEND_ASSET_HASH_MISMATCH")
        paths[name] = path
    rules = paths["rules"].read_text(encoding="utf-8")
    if backend["mode"] == "live":
        if os.environ.get("FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE") != "1":
            raise ValueError("LIVE_RERANK_NOT_ENABLED")
        key_file = os.environ.get("FINRISK_AGENT_DEEPSEEK_KEY_FILE")
        rerank_kwargs = ({} if type(rerank_cache_size) is int and rerank_cache_size == 4096
                         else {"cache_size": rerank_cache_size})
        scorer = DeepSeekReranker(rules, key_file=Path(key_file) if key_file else None, protocol=protocol,
                                  **rerank_kwargs)
    else:
        scorer = FrozenFlashReranker(paths["score_inputs"], paths["scores"], paths["candidate_map"])
    dense_kwargs = ({} if startup_timeout_seconds is None
        else {"startup_timeout_seconds": startup_timeout_seconds})
    dense = ResidentDenseIndex(paths["dense_config"], **dense_kwargs)
    context_kwargs = ({} if context_expansion is None
                      else {"context_expansion": context_expansion})
    cache_kwargs = ({} if type(result_cache_size) is int and result_cache_size == 128
                    else {"cache_size": result_cache_size})
    try:
        snapshot.backend = HybridV2Backend(
            snapshot, dense, scorer, audit_root=audit_root if audit_root is not None else root / "retrieval-audits",
            **cache_kwargs, **context_kwargs
        )
    except BaseException:
        dense.close()
        raise
    snapshot.manifest["rerank_mode"] = backend["mode"]
    snapshot.manifest["rerank_protocol"] = protocol
    return snapshot
