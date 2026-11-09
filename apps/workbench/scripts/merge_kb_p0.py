"""Merge the pinned P0 release locally; evaluation material never enters retrieval views."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data/processed/knowledge/finrisk_v2_1_20261018"
OUT = ROOT / "outputs/knowledge_build"
REVISION = "7b065c2712592763ea1c420e00cd239c0fb047ed"
RUNTIME_PIN = "c178df130ac618d5ba6e5a7944648fb6d25d60fb3de2c095b6f833fc16cf96a8"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=KB / "p0_release_20261018")
    source = parser.parse_args().release_root.resolve()
    assert sha(source / "contracts/runtime_manifest.json") == RUNTIME_PIN
    runtime_manifest = json.loads((source / "contracts/runtime_manifest.json").read_text("utf-8"))
    for name in ("contracts/asset_lock.json", "policy/article_scopes.jsonl"):
        meta = runtime_manifest["files"][name]
        path = (source / name).resolve()
        assert path.is_relative_to(source.resolve())
        assert path.stat().st_size == meta["bytes"] and sha(path) == meta["sha256"]
    target = KB / "p0_release_20261018"
    if source != target.resolve():
        shutil.copytree(source, target, dirs_exist_ok=True)
    source_records = {}
    by_file = {}
    for line in (KB / "data/cases.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        by_file.setdefault(row["file_id"], row)
    count = 0
    files = set()
    path = KB / "data/case_chunks_full.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    with (
        sqlite3.connect(KB / "evidence_registry.sqlite3") as db,
        temporary.open("w", encoding="utf-8", newline="\n") as stream,
    ):
        for uid, fid, raw in db.execute("SELECT id,file_id,data FROM units ORDER BY id"):
            unit = json.loads(raw)
            linked = unit.get("application_case_ids", [])
            if not linked:
                source_id = fid + ":SOURCE"
                linked = [source_id]
                base = by_file.get(fid, {})
                source_records[source_id] = {
                    "case_id": source_id,
                    "file_id": fid,
                    "source_file": fid,
                    "source_sha256": unit["source_sha256"],
                    "case_title": "来源级原文：" + str(base.get("case_title") or fid),
                    "violation_behavior": "来源级证据，具体主体、行为及责任须按原文分别核对",
                    "party_name": None,
                    "risk_tags": [],
                    "related_parties": [],
                    "case_origin": "source_document_only",
                    "retrieval_status": "active",
                    "evidence_status": "source_only_attribution_unknown",
                    "quality_flags": ["not_party_attributed", "model_candidate_pending_human"],
                    "data_version": "p0-full-source-view-20261018",
                }
            row = {
                "chunk_id": uid,
                "case_id": linked[0],
                "case_ids": linked,
                "file_id": fid,
                "chunk_type": "evidence",
                "chunk_text": unit["source_text"],
                "is_verbatim": True,
                "retrieval_enabled": True,
                "retrieval_weight": 1.0 if unit.get("application_case_ids") else 0.65,
                "source_span": {
                    "unit_id": uid,
                    "spans": unit["source_spans"],
                    "source_sha256": unit["source_sha256"],
                    "attribution": "registered_case"
                    if unit.get("application_case_ids")
                    else "source_only_no_actor_inference",
                },
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            files.add(fid)
    assert count == 110878 and len(files) == 9639
    temporary.replace(path)
    (KB / "data/source_records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in source_records.values()),
        encoding="utf-8",
    )
    manifest = json.loads((KB / "manifest.json").read_text("utf-8"))
    manifest["upstream_revision"] = REVISION
    manifest["p0_runtime_manifest_sha256"] = RUNTIME_PIN
    manifest["full_projection"] = {
        "units": count,
        "documents": len(files),
        "source_only_records": len(source_records),
        "sha256": sha(path),
        "evaluation_data_indexed": False,
    }
    (KB / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    receipt = {
        "revision": REVISION,
        "runtime_pin": RUNTIME_PIN,
        "base_candidate_tables_changed": False,
        "full_projection": manifest["full_projection"],
        "article_count": len(
            (target / "assets/legal/articles.jsonl").read_text("utf-8").splitlines()
        ),
        "mechanisms": 12,
        "github_push": False,
        "server_deploy": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "merge_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
