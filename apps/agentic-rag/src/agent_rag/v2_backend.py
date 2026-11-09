"""常驻BM25/BGE候选融合及Flash重排，保持冻结实验的排序规则。"""

import copy
import json
import math
import time
import uuid
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from agent_rag.contracts import canonical_json
from agent_rag.input_registry import text_hash
from agent_rag.v2_context import ParentContextExpander, validate_context_expansion
from agent_rag.v2_snapshot import PROFILE
from retrieval.sparse import RetrievalDocument, SparseRetrievalIndex


def rrf(routes: list[list[dict[str, Any]]], k: int = 60, top_k: int = 100) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for route in routes:
        seen: set[str] = set()
        for row in route:
            uid = row["unit_id"]
            if uid in seen:
                continue
            seen.add(uid)
            scores[uid] = scores.get(uid, 0) + 1 / (k + len(seen))
    return [
        {"unit_id": uid, "score": score, "rank": rank}
        for rank, (uid, score) in enumerate(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k], 1
        )
    ]


def aggregate_files(ranked: list[dict[str, Any]], units: dict[str, Any]) -> list[dict[str, Any]]:
    best = {}
    for row in sorted(ranked, key=lambda hit: (-hit["score"], hit["unit_id"])):
        fid = units[row["unit_id"]]["file_id"]
        best.setdefault(fid, {**row, "file_id": fid})
    return [
        {**row, "rank": rank}
        for rank, row in enumerate(
            sorted(best.values(), key=lambda hit: (-hit["score"], hit["file_id"])), 1
        )
    ]


class HybridV2Backend:
    def __init__(
        self, snapshot: Any, dense_index: Any, reranker: Any, *,
        audit_root: Path | None = None, context_expansion: dict[str, Any] | None = None,
        cache_size: int = 128
    ) -> None:
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("INVALID_BACKEND_CACHE_SIZE")
        self.cache_size = cache_size
        self.snapshot, self.dense, self.reranker = snapshot, dense_index, reranker
        self.context_expansion = validate_context_expansion(context_expansion)
        self.context_expander = (
            ParentContextExpander(snapshot, self.context_expansion)
            if self.context_expansion is not None else None
        )
        self.units = {unit["unit_id"]: unit for unit in snapshot.units}
        self.sparse = SparseRetrievalIndex(
            (
                RetrievalDocument(
                    unit_id=unit["unit_id"],
                    case_ids=tuple(
                        unit.get("case_ids") or [unit.get("case_id") or unit["file_id"] + ":SOURCE"]
                    ),
                    text=unit["text"],
                    source="structured",
                )
                for unit in snapshot.units
            ),
            k1=1.5,
            b=0.75,
        )
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.audit_root = audit_root
        if audit_root is not None:
            audit_root.mkdir(parents=True, exist_ok=True)

    def cache_key(self, text: str, top_k: int) -> str:
        binding = {
            "text": text,
            "top_k": top_k,
            "snapshot": self.snapshot.manifest["snapshot_id"],
            "profile": PROFILE,
            "rerank_protocol": getattr(self.reranker, "protocol", "quote-copy-v1"),
            "rules_sha256": getattr(self.reranker, "rules_sha256", None),
            "protocol_fingerprint": getattr(self.reranker, "protocol_fingerprint", None),
        }
        if self.context_expander is not None:
            binding["context_expansion_sha256"] = self.context_expander.policy_sha256
        return text_hash(canonical_json(binding))

    def is_cached(self, text: str, top_k: int) -> bool:
        return self.cache_key(text, top_k) in self.cache

    def search(self, text: str, top_k: int, control: Any = None) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip() or not 1 <= top_k <= 100:
            raise ValueError("INVALID_SEARCH_ARGUMENTS")
        if control is not None and not control.check():
            return {
                "candidates": [],
                "complete": False,
                "audit": {},
                "meta": {
                    "backend_query_attempts": 0,
                    "external_retrieval_requests": 0,
                    "cache_hits": 0,
                    "stage_status": {
                        stage: "not_started" for stage in ["bm25", "bge", "rrf", "rerank"]
                    },
                    "stop_reason": control.meta().get("stop_reason"),
                },
            }
        key = self.cache_key(text, top_k)
        if key in self.cache:
            self.cache.move_to_end(key)
            result = copy.deepcopy(self.cache[key])
            if control is not None:
                # 排名可跨任务复用，费用、截止时间和账本必须来自当前任务。
                result["meta"].update(control.meta())
            result["meta"].update(
                backend_query_attempts=0,
                external_retrieval_requests=0,
                cache_hits=1,
                cache_reused=True,
                retrieval_elapsed_ms=0,
            )
            return result
        started = time.monotonic()
        sparse = [
            {"unit_id": hit.document.unit_id, "score": hit.score, "rank": hit.rank}
            for hit in self.sparse.search(text, top_k=100)
        ]
        dense_result = self.dense.search(text, top_k=100)
        dense = dense_result["hits"]
        if len({row["unit_id"] for row in dense}) != len(dense) or any(
            row["unit_id"] not in self.units
            or not isinstance(row["score"], (int, float))
            or not math.isfinite(row["score"])
            for row in dense
        ):
            raise ValueError("DENSE_UNIT_MISMATCH")
        fused = rrf([sparse, dense])
        packets = [self.units[row["unit_id"]] for row in fused]
        binding = {"snapshot_id": self.snapshot.manifest["snapshot_id"],
                   "index_sha256": self.snapshot.manifest["index_sha256"],
                   "retrieval_profile": PROFILE}
        if getattr(self.reranker, "supports_query_binding", False):
            result = self.reranker.score_query(text, packets, control, query_binding=binding)
        else:
            result = self.reranker.score_query(text, packets, control)
        scores = result["scores"]
        expected = {unit["unit_id"] for unit in packets}
        scored = {row["unit_id"] for row in scores}
        if (
            len(scored) != len(scores)
            or not scored <= expected
            or any(
                row.get("file_id") != self.units[row["unit_id"]]["file_id"]
                or type(row.get("score")) is not int
                or not 0 <= row["score"] <= 100
                for row in scores
            )
            or (result["complete"] and scored != expected)
        ):
            raise ValueError("RERANK_BINDING_MISMATCH")
        missing = expected - scored
        error_rows = result.get("errors", [])
        if (not isinstance(error_rows, list)
                or any(not isinstance(e, dict) or e.get("unit_id") not in missing for e in error_rows)
                or len({e["unit_id"] for e in error_rows}) != len(error_rows)):
            raise ValueError("RERANK_BINDING_MISMATCH")
        error_by_unit = {e["unit_id"]: e for e in error_rows}
        failure_counts = Counter()
        for uid in missing:
            code = error_by_unit.get(uid, {}).get("code", "UNCLASSIFIED_RERANK_FAILURE")
            if not isinstance(code, str) or not code or len(code) > 64 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in code):
                raise ValueError("RERANK_BINDING_MISMATCH")
            failure_counts[code] += 1
        coverage = {
            "schema_version": "finrisk-ranking-coverage/1.0",
            "expected_units": len(expected), "scored_units": len(scored), "failed_units": len(missing),
            "ranking_complete": bool(result["complete"]),
            "topk_scope": "full_pool" if result["complete"] else "scored_subset",
            "attempts_exhausted": bool(missing) and all(
                error_by_unit.get(uid, {}).get("terminal") is True
                and type(error_by_unit[uid].get("remaining_attempts")) is int
                and error_by_unit[uid]["remaining_attempts"] == 0 for uid in missing
            ),
            "failure_counts": dict(sorted(failure_counts.items())),
            "failed_unit_ids": sorted(missing)[:3],
            "expected_set_sha256": text_hash(canonical_json(sorted(expected))),
            "scored_set_sha256": text_hash(canonical_json(sorted(scored))),
            "failed_set_sha256": text_hash(canonical_json(sorted(missing))),
            "rerank_protocol": getattr(self.reranker, "protocol", "quote-copy-v1"),
        }
        ranked = [
            {**row, "rank": rank}
            for rank, row in enumerate(
                sorted(scores, key=lambda hit: (-hit["score"], hit["unit_id"])), 1
            )
        ]
        files = aggregate_files(ranked, self.units)
        candidates = []
        for row in files[:top_k]:
            candidate = self.snapshot.candidate(
                self.units[row["unit_id"]], row["rank"], row["score"]
            )
            if self.context_expander is not None:
                candidate = self.context_expander.expand(self.units[row["unit_id"]], candidate)
            candidate["rerank_assessment"] = {
                key: row.get(key) for key in ["level", "reason", "limitation"]
            }
            candidate["rerank_assessment"]["origin"] = "model_relevance_score_not_risk_decision"
            candidates.append(candidate)
        audit = {
            "query_text": text,
            "query_text_sha256": text_hash(text),
            "snapshot_id": self.snapshot.manifest["snapshot_id"],
            "profile": PROFILE,
            "stages": {"BM25": sparse, "BGE": dense, "RRF": fused, "ND": ranked},
            "file_rankings": {
                name: aggregate_files(route, self.units)
                for name, route in [
                    ("BM25", sparse),
                    ("BGE", dense),
                    ("RRF", fused),
                    ("ND", ranked),
                ]
            },
            "top20": files[:20],
            "complete": result["complete"],
            "expected_unit_ids": sorted(expected),
            "scored_unit_ids": sorted(scored),
            "failed_unit_ids": sorted(missing),
            "ranking_coverage": coverage,
        }
        meta = {
            **result.get("meta", {}),
            "backend_query_attempts": 1,
            "index_sha256": self.snapshot.manifest["index_sha256"],
            "retrieval_profile": PROFILE,
            "snapshot_id": self.snapshot.manifest["snapshot_id"],
            "candidate_units": len(fused),
            "candidate_documents": len(files),
            "dense": dense_result.get("meta", {}),
            "stage_status": {
                "bm25": "completed",
                "bge": "completed",
                "rrf": "completed",
                "rerank": "completed" if result["complete"] else "partial",
            },
            "retrieval_elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
        if self.context_expander is not None:
            diagnoses = [candidate["context_expansion"] for candidate in candidates]
            stats = {
                **self.context_expansion,
                "policy_sha256": self.context_expander.policy_sha256,
                "expanded_candidates": sum(item["status"] == "expanded" for item in diagnoses),
                "skipped_candidates": sum(item["status"] == "skipped" for item in diagnoses),
                "added_chars": sum(item["added_chars"] for item in diagnoses),
                "skip_reasons": dict(sorted(Counter(
                    item["reason"] for item in diagnoses if item["status"] == "skipped"
                ).items())),
            }
            audit["context_expansion"] = copy.deepcopy(stats)
            meta["context_expansion"] = stats
        if self.audit_root is not None:
            audit_id = uuid.uuid4().hex
            with (self.audit_root / f"{audit_id}.json").open("x", encoding="utf-8") as handle:
                json.dump(audit, handle, ensure_ascii=False, allow_nan=False)
            meta["audit_id"] = audit_id
        value = {
            "candidates": candidates,
            "complete": bool(result["complete"]),
            "meta": meta,
            "audit": audit,
            "errors": result.get("errors", []),
            "ranking_coverage": coverage,
        }
        if result["complete"] and self.cache_size:
            self.cache[key] = copy.deepcopy(value)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return value
