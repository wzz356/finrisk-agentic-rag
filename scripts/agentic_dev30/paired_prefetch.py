"""Optional, frozen first-query sharing; no decision model and no automatic replay."""

from __future__ import annotations

import copy
import threading
import time
from pathlib import Path

try:
    from scripts.agentic_dev30.experiment import canonical, file_sha, load, obj_sha, save
except ModuleNotFoundError as error:
    if error.name not in {"scripts", "scripts.agentic_dev30"}:
        raise
    from experiment import canonical, file_sha, load, obj_sha, save


PREFETCH_RESERVE_MICRO = 500_000
FULL_SOURCE_COUNT = 9639
BOUND_LIMITS = {
    "task_rerank_requests": 240,
    "task_rerank_cap_cny": 0.5,
    "task_model_ledger_cap_cny": 1,
    "round_task_reserve_cny": 1.5,
    "prefetch_rerank_requests": 240,
    "prefetch_rerank_cap_cny": 0.5,
    "prefetch_deadline_seconds": 180,
    "round_prefetch_reserve_cny": 0.5,
}
CACHE_POLICY = (
    "paired-prefetch/1.1: track actual backend.search activity until the whole call returns; "
    "under the same admission mutex every ensure fails closed while any search is active; "
    "before each new material clear whole-query and successful "
    "rerank-pair caches under reranker cache lock at a quiescent task boundary; "
    "one complete canonical({text,user_context}) top3 query before all three arms; "
    "R1/A reuse the same ranking; R2 retains fixed multi-step diagnostic behavior; "
    "task checkpoints, deadlines and budgets are never cleared"
)
SEARCH_BOUNDARY = {
    "version": "finrisk-backend-search-boundary/1.0",
    "activity": "actual backend.search entry through final return, including cache writes",
    "mutex": "search admission, ensure activity check and cache clear share one mutex",
    "failure_policy": "stop round without waiting, replay or a new prefetch reservation",
    "checked_before": "every arm including same-material cached prefetch reuse",
}
TIMING_POLICY = {
    "wall_seconds": "actual task subprocess duration; historical meaning unchanged",
    "actual_task_seconds": "same measured duration as wall_seconds",
    "shared_prefetch_seconds": "actual shared query duration, recorded once per material",
    "comparable_e2e_seconds": (
        "shared_prefetch_seconds + actual_task_seconds; R1/A only; "
        "composed estimate, not measured cold start"
    ),
    "actual_cost": "round ledger includes each PREFETCH once plus each task once",
    "method_cost_estimate": (
        "R1/A task cost + shared cost as hypothetical per-method cost; "
        "never sum into actual round cost"
    ),
    "R2": (
        "fixed multi-step diagnostic; resident caches may be shared; no paired comparable estimate"
    ),
}


def _materials(tasks):
    """Only contiguous, identically bound three-arm materials can share evidence."""
    if not isinstance(tasks, list) or not tasks or len(tasks) % 3:
        raise ValueError("INVALID_PAIRED_PREFETCH_TASKS")
    inputs = []
    seen = set()
    for i in range(0, len(tasks), 3):
        group = tasks[i : i + 3]
        q = group[0].get("input")
        if (
            not isinstance(q, dict)
            or set(q) != {"sample_id", "text", "user_context"}
            or not isinstance(q["sample_id"], str)
            or not q["sample_id"]
            or not isinstance(q["text"], str)
            or not isinstance(q["user_context"], dict)
            or q["sample_id"] in seen
            or {t.get("arm") for t in group} != {"R1", "R2", "A"}
            or any(
                canonical(t.get("input")) != canonical(q)
                or t.get("task_key") != q["sample_id"] + ":" + t.get("arm", "")
                for t in group
            )
        ):
            raise ValueError("INVALID_PAIRED_PREFETCH_TASKS")
        seen.add(q["sample_id"])
        inputs.append(q)
    return inputs


def configuration(tasks, snapshot_id):
    inputs = _materials(tasks)
    return {
        "schema_version": "finrisk-paired-prefetch/1.1",
        "snapshot_id": snapshot_id,
        "top_k": 3,
        "query": "canonical({text,user_context})",
        "cap_cny": 0.5,
        "max_requests": 240,
        "deadline_seconds": 180,
        "reserve_micro_cny": PREFETCH_RESERVE_MICRO,
        "task_count": len(inputs),
        "inputs_sha256": obj_sha(inputs),
        "search_boundary": SEARCH_BOUNDARY,
    }


def validate_plan(plan, project):
    """Called before runtime/key access; an absent setting retains historical behavior."""
    if "paired_prefetch" not in plan:
        if plan.get("cache_policy") == CACHE_POLICY or plan.get("timing_policy") == TIMING_POLICY:
            raise ValueError("PAIRED_PREFETCH_BINDING_MISMATCH")
        return False
    expected = configuration(plan.get("tasks"), plan.get("snapshot_id"))
    module_path = Path(__file__).resolve()
    records = plan.get("frozen_files", [])
    if (
        canonical(plan["paired_prefetch"]) != canonical(expected)
        or plan.get("cache_policy") != CACHE_POLICY
        or plan.get("timing_policy") != TIMING_POLICY
        or any(
            canonical((plan.get("limits") or {}).get(k)) != canonical(v)
            for k, v in BOUND_LIMITS.items()
        )
        or plan.get("task_count") != len(plan["tasks"])
        or plan.get("max_rerank_requests") != (len(plan["tasks"]) + expected["task_count"]) * 240
        or plan.get("max_judgment_requests")
        != sum(1 if t["arm"] == "R1" else 6 for t in plan["tasks"])
        or not any(
            Path(r.get("path", "")).resolve() == module_path
            and r.get("sha256") == file_sha(module_path)
            for r in records
        )
        or not module_path.is_relative_to(Path(project).resolve())
    ):
        raise ValueError("PAIRED_PREFETCH_BINDING_MISMATCH")
    return True


def validate_snapshot(snapshot_dir, mode, project):
    """Validation only reads assets; it starts neither GPU nor paid scorer."""
    if mode not in {"full9639", "incremental-v1"}:
        raise ValueError("INVALID_EXPERIMENT_SNAPSHOT_MODE")
    manifest = load(Path(snapshot_dir) / "manifest.json")
    if manifest.get("rerank_mode") != "live" or manifest.get("corpus_scope") != mode:
        raise ValueError(
            "FULL_LIVE_V2_SNAPSHOT_REQUIRED"
            if mode == "full9639"
            else "INCREMENTAL_LIVE_V2_SNAPSHOT_REQUIRED"
        )
    import sys

    source_path = str(Path(project) / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    if (manifest.get("backend") or {}).get("mode") != "live":
        raise ValueError("LIVE_V2_BACKEND_REQUIRED")
    from agent_rag.dense_backend import DOCUMENT_COUNT
    from agent_rag.knowledge_registry import _validate_snapshot

    checked = _validate_snapshot(Path(snapshot_dir))
    dense = load(manifest["backend"]["assets"]["dense_config"]["path"])
    if (
        dense.get("subset_file_ids") is not None
        or manifest["index_sha256"] != dense["index_sha256"]
        or type(manifest.get("source_count")) is not int
        or type(manifest.get("chunk_count")) is not int
        or len(checked.sources) != manifest["source_count"]
        or len(checked.units) != manifest["chunk_count"]
    ):
        raise ValueError("EXPERIMENT_SNAPSHOT_BINDING_MISMATCH")
    if mode == "full9639":
        if (
            "corpus_mode" in dense
            or "document_count" in dense
            or manifest["source_count"] != FULL_SOURCE_COUNT
            or manifest["chunk_count"] != DOCUMENT_COUNT
        ):
            raise ValueError("FULL_LIVE_V2_SNAPSHOT_REQUIRED")
    elif (
        dense.get("corpus_mode") != "incremental-v1"
        or type(dense.get("document_count")) is not int
        or dense["document_count"] != manifest["chunk_count"]
    ):
        raise ValueError("INCREMENTAL_SNAPSHOT_BINDING_MISMATCH")
    return manifest


def _costs(control):
    try:
        if control is None:
            return 0, PREFETCH_RESERVE_MICRO
        with control._lock:
            value = control._read()
        known, unknown = value.get("settled_cost_micro_cny"), value.get("reserved_cost_micro_cny")
        if any(type(v) is not int or v < 0 for v in (known, unknown)):
            raise ValueError("INVALID_PREFETCH_LEDGER")
        return known, unknown
    except (OSError, ValueError):
        return 0, PREFETCH_RESERVE_MICRO


class SharedPrefetch:
    def __init__(self, snapshot, output, budget, plan_sha256):
        self.snapshot, self.output, self.budget = snapshot, Path(output), budget
        self.path = self.output / "shared-prefetch.json"
        if self.path.exists():
            raise ValueError("ALREADY_STARTED:prefetch is not automatically replayed")
        self.value = {
            "schema_version": "finrisk-shared-prefetch/1.0",
            "plan_sha256": plan_sha256,
            "snapshot_id": snapshot.manifest["snapshot_id"],
            "timing_policy": TIMING_POLICY,
            "search_boundary": SEARCH_BOUNDARY,
            "items": [],
        }
        self.records = {}
        self._search_lock = threading.Lock()
        self._active_searches = 0
        self._boundary_failed = False
        save(self.path, self.value)
        original_search = snapshot.backend.search

        def tracked_search(*args, **kwargs):
            with self._search_lock:
                self._active_searches += 1
            try:
                return original_search(*args, **kwargs)
            finally:
                # Ledger settlement can precede both pair-cache and whole-query-cache writes.
                with self._search_lock:
                    self._active_searches -= 1

        # Installed before the optional experiment's HTTP server starts.
        # Ordinary runs are untouched.
        snapshot.backend.search = tracked_search

    def ensure(self, q):
        sample = q["sample_id"]
        text = canonical({"text": q["text"], "user_context": q["user_context"]})
        identity = obj_sha({"text": q["text"], "user_context": q["user_context"]})
        key = "PREFETCH:" + sample
        backend = self.snapshot.backend
        with self._search_lock:
            if self._active_searches or self._boundary_failed:
                self._boundary_failed = True
                self.value.setdefault(
                    "search_boundary_failure",
                    {
                        "sample_id": sample,
                        "active_searches": self._active_searches,
                        "reason": "PAIRED_PREFETCH_BACKEND_SEARCH_ACTIVE",
                    },
                )
                save(self.path, self.value)
                raise ValueError("PAIRED_PREFETCH_BACKEND_SEARCH_ACTIVE")
            if sample in self.records:
                row = self.records[sample]
                if row["input_sha256"] != identity or row["status"] != "complete":
                    raise ValueError("PAIRED_PREFETCH_FAILED_OR_INPUT_MISMATCH")
                return copy.deepcopy(row)
            # A real search cannot start between the idle check and this clear.
            # Successful-pair writes also use the reranker lock; checkpoints stay on disk.
            with backend.reranker._cache_lock:
                backend.cache.clear()
                backend.reranker._cache.clear()
            self.budget.reserve(key, PREFETCH_RESERVE_MICRO)
        directory = self.output / "prefetch" / obj_sha(sample)[:24]
        ledger = directory / "rerank-ledger.json"
        row = {
            "sample_id": sample,
            "input_sha256": identity,
            "query_text_sha256": identity,
            "status": "started",
            "top_k": 3,
            "ledger_ref": str(ledger),
            "elapsed_seconds": None,
            "known_micro_cny": 0,
            "unknown_micro_cny": PREFETCH_RESERVE_MICRO,
        }
        self.records[sample] = row
        self.value["items"].append(row)
        control = None
        failure = None
        started = time.monotonic()
        try:
            save(self.path, self.value)
            from agent_rag.rerank_backend import RerankTaskControl

            control = RerankTaskControl(ledger, 0.5, 240, timeout_seconds=180)
            result = backend.search(text, 3, control)
            audit_ref = directory / "retrieval-audit.json"
            coverage = result.get("ranking_coverage") or {}
            ranked = (result.get("audit") or {}).get("stages", {}).get("ND", [])
            row.update(
                candidates=result.get("candidates", []),
                ranking=ranked,
                ranking_coverage=coverage,
                audit_ref=str(audit_ref),
                meta=result.get("meta", {}),
                errors=result.get("errors", []),
            )
            save(audit_ref, result.get("audit", {}))
            if (
                result.get("complete") is not True
                or coverage.get("ranking_complete") is not True
                or coverage.get("failed_units") != 0
                or not coverage.get("expected_units")
                or coverage.get("scored_units") != coverage["expected_units"]
                or len(ranked) != coverage["expected_units"]
                or not backend.is_cached(text, 3)
            ):
                raise ValueError("PAIRED_PREFETCH_INCOMPLETE")
            row["status"] = "complete"
        except BaseException as error:
            failure = error
            row.update(status="failed", error=type(error).__name__ + ":" + str(error)[:200])
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            try:
                if control is not None:
                    control.cancel()
            except BaseException as error:
                row.update(
                    status="failed",
                    cancellation_error=type(error).__name__ + ":" + str(error)[:200],
                )
                if failure is None:
                    failure = error
                    raise
            finally:
                known, unknown = _costs(control)
                row.update(known_micro_cny=known, unknown_micro_cny=unknown)
                # Save query evidence before settlement: either artifact may independently fail.
                try:
                    save(self.path, self.value)
                finally:
                    self.budget.finish(key, known, unknown)
        return copy.deepcopy(row)


def task_result_fields(arm, result, shared):
    paired = arm in {"R1", "A"}
    return {
        "shared_prefetch_seconds": shared["elapsed_seconds"],
        "actual_task_seconds": result["wall_seconds"],
        "comparable_e2e_seconds": shared["elapsed_seconds"] + result["wall_seconds"]
        if paired
        else None,
        "timing_interpretation": "composed estimate, not measured cold start"
        if paired
        else "R2 diagnostic only",
        "method_cost_estimate": {
            "known_micro_cny": result["known_micro_cny"]
            + (shared["known_micro_cny"] if paired else 0),
            "unknown_micro_cny": result["unknown_micro_cny"]
            + (shared["unknown_micro_cny"] if paired else 0),
            "not_additive_to_round_actual": True,
        },
    }
