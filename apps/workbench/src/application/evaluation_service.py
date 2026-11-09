"""Serve the explicit published snapshot and flag changes to its evaluated source/context."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SNAPSHOT = "reports/evaluation_dashboard/evaluation.json"


def file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    if path.suffix != ".npz":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_evaluation(root: Path, knowledge_root: Path) -> dict[str, Any]:
    path = root / SNAPSHOT
    if not path.is_file():
        return {"available": False}
    payload: dict[str, Any] = json.loads(path.read_text("utf-8"))
    # Older open tabs can still read the current results until their next page refresh.
    selected = next(
        (run for run in payload.get("runs", []) if run["id"] == payload.get("selected_run")), None
    )
    if selected is not None:
        payload["risk"] = selected["risk"]
        payload["risk_names"] = {"deepseek_adaptive": "DeepSeek · 自适应融合"}
        payload["retrieval"] = {"deepseek": selected["retrieval"]}
    bindings = payload.get("source_bindings", {})
    changed = [name for name, expected in bindings.items() if file_digest(root / name) != expected]
    data = root / "data/eval/competition500_v1"
    context_paths = {
        "knowledge_manifest": knowledge_root / "manifest.json",
        "case_pool": data / "cases.jsonl",
        "test_inputs": data / "test.jsonl",
        "label_reference": data / "reference/labels.jsonl",
        "retrieval_reference": data / "reference/qrels_test.jsonl",
    }
    context_changed = [
        name
        for name, expected in payload.get("context_bindings", {}).items()
        if name not in context_paths or file_digest(context_paths[name]) != expected
    ]
    payload["freshness"] = {
        "status": "unbound"
        if not bindings
        else "stale"
        if changed or context_changed
        else "current",
        "changed_sources": changed,
        "changed_context": context_changed,
        "scope": "Recorded inference source, calibration model, knowledge manifest and test data; "
        "excludes HTTP/UI presentation and does not verify external model service versions.",
    }
    return payload
