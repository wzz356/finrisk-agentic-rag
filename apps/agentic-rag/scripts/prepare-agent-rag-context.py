"""离线派生父段上下文清单；不启动运行时、不修改原清单或当前指针。"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_rag.contracts import canonical_json  # noqa: E402
from agent_rag.input_registry import text_hash  # noqa: E402
from agent_rag.v2_context import POLICY, validate_context_expansion  # noqa: E402
from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash  # noqa: E402


def prepare(
    source_root: Path, destination: Path, context_expansion: dict[str, Any] | None = None
) -> dict[str, Any]:
    source_root, destination = source_root.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError("SNAPSHOT_ALREADY_EXISTS")
    policy = validate_context_expansion(
        {"policy": POLICY} if context_expansion is None else context_expansion
    )
    manifest_path = source_root / "manifest.json"
    source_manifest_sha256 = file_hash(manifest_path)
    snapshot = V2KnowledgeSnapshot(source_root)
    manifest = copy.deepcopy(snapshot.manifest)
    backend = manifest.get("backend")
    if not isinstance(backend, dict) or backend.get("mode") not in {"live", "replay"}:
        raise ValueError("INVALID_V2_BACKEND_CONFIG")
    for asset in backend["assets"].values():
        if file_hash(Path(asset["path"])) != asset["sha256"]:
            raise ValueError("BACKEND_ASSET_HASH_MISMATCH")
    if file_hash(manifest_path) != source_manifest_sha256:
        raise ValueError("SOURCE_MANIFEST_HASH_MISMATCH")
    manifest["derived_from"] = {
        "snapshot_id": manifest["snapshot_id"], "manifest_sha256": source_manifest_sha256
    }
    backend["context_expansion"] = policy
    identity = {"source_manifest_sha256": source_manifest_sha256, "context_expansion": policy}
    manifest["snapshot_id"] = "v2-" + text_hash(canonical_json(identity))[:24]
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("SNAPSHOT_ALREADY_EXISTS") from error
    with (destination / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(manifest))
    return {
        "snapshot_id": manifest["snapshot_id"], "path": str(destination),
        "source_manifest_sha256": source_manifest_sha256, "context_expansion": policy,
        "model_requests_sent": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--max-parent-chars", type=int, default=900)
    parser.add_argument("--max-added-chars", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, {
        "policy": POLICY, "max_parent_chars": args.max_parent_chars,
        "max_added_chars": args.max_added_chars,
    }), ensure_ascii=False))
