"""Validate immutable V2 versions before atomically changing the registry pointer."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash

SCHEMA = "finrisk-knowledge-registry/1.0"
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _snapshot_under_registry(registry: Path, path: Path) -> Path:
    versions = registry.resolve() / "versions"
    resolved = path.resolve()
    if not resolved.is_relative_to(versions):
        raise ValueError("REGISTRY_PATH_OUTSIDE_ROOT")
    parts = resolved.relative_to(versions).parts
    if len(parts) != 2 or parts[1] != "snapshot" or not _VERSION.fullmatch(parts[0]):
        raise ValueError("INVALID_REGISTRY_SNAPSHOT_PATH")
    return resolved


def _read_pointer(registry: Path) -> dict[str, Any]:
    value = json.loads((registry / "current.json").read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA
        or not isinstance(value.get("snapshot_path"), str)
        or not isinstance(value.get("history"), list)
    ):
        raise ValueError("INVALID_REGISTRY_POINTER")
    for item in [value, *value["history"]]:
        if not isinstance(item, dict) or not isinstance(item.get("snapshot_path"), str):
            raise ValueError("INVALID_REGISTRY_POINTER")
        relative = Path(item["snapshot_path"])
        if relative.is_absolute():
            raise ValueError("INVALID_REGISTRY_POINTER_PATH")
        path = _snapshot_under_registry(registry, registry / relative)
        if (
            item.get("version") != path.parent.name
            or not isinstance(item.get("snapshot_id"), str)
            or not isinstance(item.get("manifest_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", item["manifest_sha256"]) is None
        ):
            raise ValueError("INVALID_REGISTRY_POINTER_VERSION")
    return value


def current_snapshot(registry: Path) -> Path:
    """Read one atomic pointer; expensive asset validation happens before publication."""
    registry = Path(registry).resolve()
    pointer = _read_pointer(registry)
    relative = Path(pointer["snapshot_path"])
    if relative.is_absolute():
        raise ValueError("INVALID_REGISTRY_POINTER_PATH")
    path = _snapshot_under_registry(registry, registry / relative)
    if pointer.get("version") != path.parent.name:
        raise ValueError("INVALID_REGISTRY_POINTER_VERSION")
    manifest = path / "manifest.json"
    if file_hash(manifest) != pointer.get("manifest_sha256"):
        raise ValueError("REGISTRY_MANIFEST_HASH_MISMATCH")
    return path


def _validate_backend(snapshot: V2KnowledgeSnapshot) -> None:
    # Imports do not start workers or instantiate a paid scorer.
    from agent_rag.dense_backend import (
        _checked_json,
        _read_config,
        _read_jsonl,
        _validate_cache_metadata,
        _validate_index_rows,
        _validate_model_fingerprint,
        _validate_vector_file,
    )

    backend = snapshot.manifest.get("backend")
    if (
        not isinstance(backend, dict)
        or backend.get("mode") not in {"live", "replay"}
        or not isinstance(backend.get("assets"), dict)
    ):
        raise ValueError("INVALID_V2_BACKEND_CONFIG")
    mode = backend["mode"]
    protocol = backend.get("rerank_protocol", "quote-copy-v1")
    if (
        protocol not in {"quote-copy-v1", "span-ref-v2"}
        or mode == "replay"
        and protocol != "quote-copy-v1"
    ):
        raise ValueError("RERANK_PROTOCOL_NOT_READY")
    required = {"rules", "dense_config"}
    if mode == "replay":
        required |= {"score_inputs", "scores", "candidate_map"}
    if not required <= backend["assets"].keys():
        raise ValueError("INCOMPLETE_V2_BACKEND_ASSETS")
    paths = {}
    for name, asset in backend["assets"].items():
        if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
            raise ValueError("INVALID_BACKEND_ASSET")
        path = Path(asset["path"])
        if not path.is_absolute() or file_hash(path) != asset.get("sha256"):
            raise ValueError("BACKEND_ASSET_HASH_MISMATCH")
        paths[name] = path
    if not paths["rules"].read_text(encoding="utf-8").strip():
        raise ValueError("EMPTY_RERANK_RULES")
    config = _read_config(paths["dense_config"])
    index = Path(config["index_path"])
    if (
        index.resolve() != Path(snapshot.assets["index"]["path"]).resolve()
        or config["index_sha256"] != snapshot.assets["index"]["sha256"]
        or file_hash(index) != config["index_sha256"]
    ):
        raise ValueError("DENSE_SNAPSHOT_INDEX_MISMATCH")
    rows = _read_jsonl(index)
    _validate_index_rows(rows, config)
    meta = _checked_json(Path(config["vector_meta_path"]), config["vector_meta_sha256"])
    _validate_cache_metadata(meta, rows, "document")
    _validate_model_fingerprint(Path(config["model_path"]), meta["binding"]["model"])
    _validate_vector_file(Path(config["vector_path"]), meta)
    if mode == "replay":
        from agent_rag.v2_runtime import FrozenFlashReranker

        FrozenFlashReranker(paths["score_inputs"], paths["scores"], paths["candidate_map"])


def _validate_snapshot(path: Path) -> V2KnowledgeSnapshot:
    snapshot = V2KnowledgeSnapshot(path)
    if (
        snapshot.manifest.get("dictionary_sha256") != snapshot.assets["dictionary"]["sha256"]
        or snapshot.manifest.get("index_sha256") != snapshot.assets["index"]["sha256"]
    ):
        raise ValueError("SNAPSHOT_BINDING_MISMATCH")
    for unit in snapshot.units:
        snapshot.candidate(unit, 1, 0)
    _validate_backend(snapshot)
    return snapshot


def register_base_snapshot(registry: Path, base_path: Path) -> Path:
    """Retain an externally stored, SHA-bound baseline without changing current.json."""
    registry = Path(registry).resolve()
    base_path = Path(base_path).resolve()
    snapshot = _validate_snapshot(base_path)
    raw = (base_path / "manifest.json").read_bytes()
    if json.loads(raw) != snapshot.manifest:
        raise ValueError("BASE_MANIFEST_CHANGED_DURING_VALIDATION")
    if any(not Path(asset["path"]).is_absolute() for asset in snapshot.assets.values()):
        raise ValueError("BASE_ASSET_PATH_NOT_ABSOLUTE")
    snapshot_id = snapshot.manifest.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not _VERSION.fullmatch(snapshot_id):
        raise ValueError("INVALID_REGISTRY_BASE_SNAPSHOT_ID")
    destination = _snapshot_under_registry(
        registry, registry / "versions" / f"bootstrap-{snapshot_id}" / "snapshot"
    )
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        if manifest_path.read_bytes() != raw:
            raise ValueError("REGISTRY_VERSION_IMMUTABLE")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=destination)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # The caller's update lock serializes writers, as for pointer publication.
        if manifest_path.exists():
            if manifest_path.read_bytes() != raw:
                raise ValueError("REGISTRY_VERSION_IMMUTABLE")
        else:
            os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _entry(pointer: dict[str, Any]) -> dict[str, Any]:
    return {
        key: pointer[key]
        for key in ("version", "snapshot_id", "snapshot_path", "manifest_sha256", "published_at")
    }


def _atomic_pointer(registry: Path, value: dict[str, Any]) -> None:
    registry.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".current-", suffix=".tmp", dir=registry)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, registry / "current.json")
    finally:
        temporary.unlink(missing_ok=True)


def publish_snapshot(registry: Path, snapshot_path: Path) -> dict[str, Any]:
    """Publish a complete versions/<directory-version>/snapshot; caller owns run locking."""
    registry = Path(registry).resolve()
    path = _snapshot_under_registry(registry, Path(snapshot_path))
    snapshot = _validate_snapshot(path)
    history = []
    if (registry / "current.json").exists():
        old = _read_pointer(registry)
        history = [_entry(old), *old["history"]]
    version = path.parent.name
    manifest_sha = file_hash(path / "manifest.json")
    for item in history:
        if item["version"] == version and item["manifest_sha256"] != manifest_sha:
            raise ValueError("REGISTRY_VERSION_IMMUTABLE")
    seen = {version}
    retained = []
    for item in history:
        if item["version"] not in seen:
            retained.append(item)
            seen.add(item["version"])
    pointer = {
        "schema_version": SCHEMA,
        "version": version,
        "snapshot_id": snapshot.manifest["snapshot_id"],
        "snapshot_path": path.relative_to(registry).as_posix(),
        "manifest_sha256": manifest_sha,
        "published_at": datetime.now(UTC).isoformat(),
        "history": retained,
    }
    _atomic_pointer(registry, pointer)
    return pointer


def rollback_snapshot(registry: Path, version: str) -> dict[str, Any]:
    """Rollback only to a retained version, validating it again without rebuilding."""
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("INVALID_REGISTRY_VERSION")
    registry = Path(registry).resolve()
    current = _read_pointer(registry)
    matches = [
        item
        for item in [_entry(current), *current["history"]]
        if version in {item["version"], item["snapshot_id"]}
    ]
    if len(matches) != 1:
        raise ValueError("REGISTRY_VERSION_NOT_RETAINED_OR_AMBIGUOUS")
    item = matches[0]
    path = _snapshot_under_registry(registry, registry / item["snapshot_path"])
    if file_hash(path / "manifest.json") != item["manifest_sha256"]:
        raise ValueError("REGISTRY_MANIFEST_HASH_MISMATCH")
    return publish_snapshot(registry, path)
