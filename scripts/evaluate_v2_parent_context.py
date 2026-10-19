"""纯离线核验冻结 V2 全部单元及可选历史排名的父上下文投影，不检索、不重排。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_rag import v2_context, v2_snapshot  # noqa: E402
from agent_rag.input_registry import text_hash  # noqa: E402
from agent_rag.knowledge_snapshot import make_evidence  # noqa: E402
from agent_rag.v2_context import ParentContextExpander, strip_parent_context  # noqa: E402
from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash  # noqa: E402

LIMITATIONS = [
    "只核验坐标、原文绑定、可撤销性和机械安全门，不等同于语义归属或人工验收。",
    "全量单元使用 rank=1、score=0 的诊断候选；不是检索结果或召回率实验。",
    "历史排名仅在当前快照绑定同一 unit/file/text 后回放；保留旧排名和分数。",
    "历史快照与当前快照分别记录；未对当前库重新检索，未调用模型、重排或 GPU。",
    "新增字符按各候选内原证据区间的并集去重；跨单元重复字符不作全库去重。",
]


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _union_length(evidence: list[dict[str, Any]]) -> int:
    total, last = 0, -1
    for start, end in sorted((item["start"], item["end"]) for item in evidence):
        total += max(0, end - max(start, last))
        last = max(last, end)
    return total


def _validate_projection(
    snapshot: V2KnowledgeSnapshot,
    base: dict[str, Any],
    enhanced: dict[str, Any],
    policy_sha256: str,
) -> dict[str, Any]:
    if not isinstance(enhanced, dict) or set(enhanced) != set(base) | {"context_expansion"}:
        raise ValueError("CANDIDATE_FIELDS_CHANGED")
    if any(enhanced[key] != value for key, value in base.items() if key != "evidence"):
        raise ValueError("CANDIDATE_FIELDS_CHANGED")
    evidence = enhanced.get("evidence")
    if not isinstance(evidence, list) or evidence[:len(base["evidence"])] != base["evidence"]:
        raise ValueError("ORIGINAL_EVIDENCE_CHANGED")
    ids = [item["evidence_id"] for item in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_EVIDENCE_ID")
    diagnosis = enhanced.get("context_expansion")
    if (
        not isinstance(diagnosis, dict)
        or diagnosis.get("status") not in {"expanded", "skipped"}
        or not isinstance(diagnosis.get("reason"), str)
        or not diagnosis["reason"]
        or diagnosis.get("policy") != v2_context.POLICY
        or diagnosis.get("policy_sha256") != policy_sha256
        or type(diagnosis.get("added_chars")) is not int
        or diagnosis["added_chars"] < 0
    ):
        raise ValueError("INVALID_EXPANSION_DIAGNOSTIC")
    for field in ["added_evidence_ids", "parent_evidence_ids"]:
        values = diagnosis.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(eid, str) or eid not in ids for eid in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("INVALID_EXPANSION_DIAGNOSTIC")
    added = evidence[len(base["evidence"]):]
    if [item["evidence_id"] for item in added] != diagnosis["added_evidence_ids"]:
        raise ValueError("ADDED_EVIDENCE_MISMATCH")
    if any(eid not in diagnosis["parent_evidence_ids"] for eid in diagnosis["added_evidence_ids"]):
        raise ValueError("ADDED_EVIDENCE_MISMATCH")
    if diagnosis["status"] == "skipped":
        if added or diagnosis["parent_evidence_ids"] or diagnosis["added_chars"]:
            raise ValueError("SKIPPED_CANDIDATE_CHANGED")
    elif not diagnosis["parent_evidence_ids"] or diagnosis["reason"] != "VERIFIED_PARENT":
        raise ValueError("INVALID_EXPANSION_DIAGNOSTIC")
    source = snapshot.by_file[base["file_id"]]
    text = source["full_text"]
    if text_hash(text) != source["text_sha256"]:
        raise ValueError("AUDIT_SOURCE_HASH_MISMATCH")
    for item in added:
        start, end = item.get("start"), item.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("ADDED_EVIDENCE_SOURCE_MISMATCH")
        expected = make_evidence(source, start, end, text)
        if (
            any(item.get(key) != value for key, value in expected.items())
            or item.get("source_verified") is not True
            or item.get("evidence_roles") != ["context"]
        ):
            raise ValueError("ADDED_EVIDENCE_SOURCE_MISMATCH")
    unique_added = _union_length(evidence) - _union_length(base["evidence"])
    if unique_added != diagnosis["added_chars"]:
        raise ValueError("ADDED_CHARS_MISMATCH")
    restored = strip_parent_context(enhanced, "audit_round_trip")
    restored.pop("context_expansion", None)
    if restored != base:
        raise ValueError("STRIP_ROUND_TRIP_MISMATCH")
    return copy.deepcopy(diagnosis)


def _audit_candidate(
    snapshot: V2KnowledgeSnapshot,
    expander: ParentContextExpander,
    unit: dict[str, Any],
    rank: int,
    score: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    diagnostic = {"unit_id": unit.get("unit_id"), "file_id": unit.get("file_id")}
    base, enhanced, phase = None, None, "candidate"
    try:
        unit_before = copy.deepcopy(unit)
        base = snapshot.candidate(unit, rank, score)
        base_before = copy.deepcopy(base)
        phase = "expand"
        enhanced = expander.expand(unit, base)
        phase = "validation"
        if base != base_before or unit != unit_before:
            raise ValueError("AUDIT_INPUT_MUTATED")
        diagnosis = _validate_projection(snapshot, base, enhanced, expander.policy_sha256)
        diagnostic.update(diagnosis)
        diagnostic["checks"] = {
            "base_unchanged": True, "required_unchanged": True, "original_fields_unchanged": True,
            "original_evidence_unchanged": True, "added_source_binding": True,
            "unique_added_chars_verified": True, "strip_round_trip": True,
        }
    except Exception as error:
        diagnostic.update(
            status="error", reason=str(error) or type(error).__name__, phase=phase,
            error_type=type(error).__name__, error_message=str(error),
        )
    return diagnostic, base, enhanced


def _validate_ranking(data: Any, units: dict[str, dict[str, Any]]) -> None:
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("query_text"), str)
        or not data["query_text"].strip()
        or data.get("query_text_sha256") != text_hash(data["query_text"])
    ):
        raise ValueError("RANKING_QUERY_HASH_MISMATCH")
    if not isinstance(data.get("snapshot_id"), str) or not data["snapshot_id"]:
        raise ValueError("INVALID_HISTORICAL_SNAPSHOT_ID")
    hits = data.get("top20")
    if not isinstance(hits, list) or not 1 <= len(hits) <= 20:
        raise ValueError("INVALID_FROZEN_TOP20")
    seen, previous_score = set(), math.inf
    for rank, hit in enumerate(hits, 1):
        if not isinstance(hit, dict):
            raise ValueError("INVALID_RANKING_HIT")
        uid, score = hit.get("unit_id"), hit.get("score")
        if not isinstance(uid, str) or uid not in units:
            raise ValueError("RANKING_UNIT_NOT_FOUND")
        if uid in seen:
            raise ValueError("RANKING_DUPLICATE_UNIT")
        seen.add(uid)
        if type(hit.get("rank")) is not int or hit["rank"] != rank:
            raise ValueError("INVALID_FROZEN_RANK")
        if (
            type(score) not in {int, float} or not math.isfinite(score)
            or not 0 <= score <= 100 or score > previous_score
        ):
            raise ValueError("INVALID_FROZEN_SCORE")
        previous_score = score
        unit = units[uid]
        if hit.get("file_id") != unit.get("file_id"):
            raise ValueError("RANKING_FILE_BINDING_MISMATCH")
        if hit.get("unit_text_sha256") != text_hash(unit["text"]):
            raise ValueError("RANKING_TEXT_BINDING_MISMATCH")


def _reject_nonfinite(value: str) -> None:
    raise ValueError("RANKING_NON_FINITE_NUMBER:" + value)


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _reject_nonfinite(value)
    return number


def _replay_ranking(
    path: Path,
    snapshot: V2KnowledgeSnapshot,
    expander: ParentContextExpander,
    units: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "audit_path": str(path.resolve()), "audit_sha256": None, "status": "failed",
        "historical_snapshot_id": None, "current_snapshot_id": snapshot.manifest["snapshot_id"],
        "frozen_top20": None, "candidates": [], "errors": [],
    }
    try:
        result["audit_sha256"] = file_hash(path)
        data = json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_nonfinite, parse_float=_finite_float,
        )
        if isinstance(data, dict):
            result.update(
                query_text=data.get("query_text"), query_text_sha256=data.get("query_text_sha256"),
                historical_snapshot_id=data.get("snapshot_id"),
                frozen_top20=copy.deepcopy(data.get("top20")),
            )
        # Bind the complete frozen list before evaluating any candidate.
        _validate_ranking(data, units)
        for hit in data["top20"]:
            diagnostic, base, enhanced = _audit_candidate(
                snapshot, expander, units[hit["unit_id"]], hit["rank"], hit["score"]
            )
            result["candidates"].append({
                "binding": {k: hit[k] for k in ["unit_id", "file_id", "unit_text_sha256"]},
                "rank": hit["rank"], "score": hit["score"], "diagnostic": diagnostic,
                "base_candidate": base, "enhanced_candidate": enhanced,
            })
            if diagnostic["status"] == "error":
                result["errors"].append(diagnostic)
        if file_hash(path) != result["audit_sha256"]:
            raise ValueError("RANKING_AUDIT_CHANGED_DURING_RUN")
        if not result["errors"]:
            result["status"] = "passed"
    except Exception as error:
        result.update(error_type=type(error).__name__, error_message=str(error))
        result["errors"].append({"error_type": type(error).__name__, "error_message": str(error)})
    return result


def _distribution(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)

    def quantile(fraction: float) -> int | None:
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None

    return {
        "population": "expanded_units", "count": len(values), "total": sum(values),
        "min": ordered[0] if ordered else None, "max": ordered[-1] if ordered else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": quantile(0.5), "p90": quantile(0.9), "p95": quantile(0.95),
        "p99": quantile(0.99), "quantile_method": "nearest_rank",
        "histogram": {str(n): count for n, count in sorted(Counter(values).items())},
    }


def audit_snapshot(
    snapshot_path: Path, destination: Path, ranking_audits: list[Path] | None = None
) -> dict[str, Any]:
    snapshot_path, destination = Path(snapshot_path).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("AUDIT_DESTINATION_EXISTS")
    manifest_path = snapshot_path / "manifest.json"
    manifest_hash = file_hash(manifest_path)
    snapshot = V2KnowledgeSnapshot(snapshot_path)
    manifest = snapshot.manifest
    effective_policy = manifest.get("backend", {}).get("context_expansion")
    if effective_policy is None:
        effective_policy = {"policy": v2_context.POLICY}
    expander = ParentContextExpander(snapshot, effective_policy)
    inputs = {"manifest": {"path": str(manifest_path), "before_sha256": manifest_hash}}
    assets = {**manifest["assets"], **{
        "backend." + name: value
        for name, value in manifest.get("backend", {}).get("assets", {}).items()
    }}
    for name, asset in assets.items():
        actual = file_hash(Path(asset["path"]))
        if actual != asset["sha256"]:
            raise ValueError("AUDIT_ASSET_HASH_MISMATCH:" + name)
        inputs["asset." + name] = {
            "path": str(Path(asset["path"]).resolve()), "before_sha256": actual,
            "manifest_sha256": asset["sha256"],
        }
    code = {
        name: {"path": str(path.resolve()), "sha256": file_hash(path)}
        for name, path in [
            ("script", Path(__file__)), ("v2_context", Path(v2_context.__file__)),
            ("v2_snapshot", Path(v2_snapshot.__file__)),
        ]
    }
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("AUDIT_DESTINATION_EXISTS") from error
    counts = Counter({"expanded": 0, "skipped": 0, "errors": 0})
    reasons: dict[str, Counter] = {key: Counter() for key in counts}
    positive, zero, skipped = [], [], []
    added_chars: list[int] = []
    windows, with_windows = 0, 0
    with (destination / "units.jsonl").open("x", encoding="utf-8") as handle:
        for unit in snapshot.units:
            diagnostic, base, enhanced = _audit_candidate(snapshot, expander, unit, 1, 0.0)
            handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
            status = diagnostic["status"]
            counter_key = "errors" if status == "error" else status
            counts[counter_key] += 1
            reasons[counter_key][diagnostic["reason"]] += 1
            if status == "error":
                continue
            if status == "expanded":
                added_chars.append(diagnostic["added_chars"])
                added_windows = len(diagnostic["added_evidence_ids"])
                windows += added_windows
                with_windows += int(added_windows > 0)
                bucket = positive if diagnostic["added_chars"] > 0 else zero
            else:
                bucket = skipped
            if len(bucket) < (3 if bucket is positive else 2):
                bucket.append({
                    "diagnostic": diagnostic, "base_candidate": base,
                    "enhanced_candidate": enhanced,
                })
    units_by_id = {unit["unit_id"]: unit for unit in snapshot.units}
    replays = [
        _replay_ranking(Path(path), snapshot, expander, units_by_id)
        for path in ranking_audits or []
    ]
    replay_counts = {
        "total": len(replays), "passed": sum(item["status"] == "passed" for item in replays),
        "failed": sum(item["status"] == "failed" for item in replays),
    }
    changed = []
    for name, item in inputs.items():
        try:
            item["after_sha256"] = file_hash(Path(item["path"]))
        except OSError as error:
            item["after_sha256"] = None
            item["error_type"] = type(error).__name__
            item["error_message"] = str(error)
        item["unchanged"] = item["before_sha256"] == item["after_sha256"]
        if not item["unchanged"]:
            changed.append(name)
    summary = {
        "schema_version": "finrisk-v2-parent-context-audit/1.0",
        "status": "failed" if counts["errors"] or replay_counts["failed"] or changed else "passed",
        "snapshot": {
            "path": str(snapshot_path), "snapshot_id": manifest["snapshot_id"],
            "derived_from": manifest.get("derived_from"), "manifest_sha256": manifest_hash,
        },
        "input_integrity": {"unchanged": not changed, "changed_inputs": changed, "files": inputs},
        "code_sha256": code, "policy": expander.policy, "policy_sha256": expander.policy_sha256,
        "denominators": {
            "sources": len(snapshot.sources), "units": len(snapshot.units),
            "sources_with_units": len({unit["file_id"] for unit in snapshot.units}),
        },
        "counts": dict(counts),
        "reason_counts": {key: dict(value) for key, value in reasons.items()},
        "added_context": {
            "added_evidence_windows": windows, "units_with_added_windows": with_windows,
            "positive_added_units": sum(value > 0 for value in added_chars),
            "zero_added_expanded_units": sum(value == 0 for value in added_chars),
            "unique_added_chars": _distribution(added_chars),
        },
        "ranking_audits": replay_counts, "api_calls": 0, "retrieval_calls": 0, "rerank_calls": 0,
        "gpu_used": False, "limitations": LIMITATIONS,
    }
    _write_json(destination / "examples.json", positive + skipped + zero)
    if ranking_audits:
        _write_json(destination / "ranking-replay.json", {
            "schema_version": "finrisk-v2-parent-context-ranking-replay/1.0",
            "retrieval_performed": False, "rerank_performed": False,
            "limitations": LIMITATIONS, "audits": replays,
        })
    _write_json(destination / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--ranking-audit", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        summary = audit_snapshot(args.snapshot, args.destination, args.ranking_audit)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "error_message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({
        "status": summary["status"], "destination": str(args.destination.resolve()),
        "counts": summary["counts"], "ranking_audits": summary["ranking_audits"], "api_calls": 0,
    }, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
