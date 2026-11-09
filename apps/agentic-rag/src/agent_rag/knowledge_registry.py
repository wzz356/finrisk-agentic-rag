"""V2 mechanical registry plus fail-closed admission for host-managed registries.

Unmanaged historical registries retain their mechanical API for compatibility.
Once a SHA-pinned host baseline or managed usage exists, every extension needs
an injected host admission from a complete approved unified release.
"""

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


def _require_admission(registry, path, *, authority=None, admission=None):
    managed = (registry / "host-baseline.json").is_file() or (registry / "usage.json").exists()
    if not managed:
        return
    marker = registry / "host-baseline.json"
    if not marker.is_file():
        raise ValueError("BASELINE_HOST_PIN_REQUIRED")
    pin = json.loads(marker.read_text(encoding="utf-8"))
    if path.parent.name == pin.get("version") and file_hash(path / "manifest.json") == pin.get(
        "manifest_sha256"
    ):
        return
    from agent_rag.maintenance_publish import HostAdmissionAuthority

    if not isinstance(authority, HostAdmissionAuthority) or not isinstance(admission, dict):
        raise ValueError("HOST_ADMISSION_REQUIRED")
    authority.verify(registry, path, admission)
    return dict(admission)


def _record_admission(registry, path, verified):
    """Host-owned local ledger; never populated from unverified uploaded JSON."""
    if verified is None:
        return
    from agent_rag.knowledge_updates import write_json

    ledger_path = registry / "admission-ledger.json"
    ledger = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.exists()
        else {"schema": "finrisk-host-admission-ledger/1.0", "versions": {}}
    )
    if ledger.get("schema") != "finrisk-host-admission-ledger/1.0" or not isinstance(
        ledger.get("versions"), dict
    ):
        raise ValueError("INVALID_HOST_ADMISSION_LEDGER")
    previous = ledger["versions"].get(path.parent.name)
    if previous and any(
        previous.get(key) != verified.get(key)
        for key in ("registry", "snapshot_path", "manifest_sha256", "release_id")
    ):
        raise ValueError("REGISTRY_VERSION_IMMUTABLE")
    ledger["versions"][path.parent.name] = verified
    write_json(ledger_path, ledger)


def _require_registered_admission(registry, path):
    """Readers trust only the fixed baseline or a prior host-verified admission.

    This ledger shares the trusted local persistence boundary with the registry
    pointers. HMAC verification occurs before its host-only registration write.
    """
    if not (registry / "host-baseline.json").exists() and not (registry / "usage.json").exists():
        return
    marker = registry / "host-baseline.json"
    if not marker.is_file():
        raise ValueError("BASELINE_HOST_PIN_REQUIRED")
    pin = json.loads(marker.read_text(encoding="utf-8"))
    digest = file_hash(path / "manifest.json")
    if path.parent.name == pin.get("version") and digest == pin.get("manifest_sha256"):
        return
    try:
        ledger = json.loads((registry / "admission-ledger.json").read_text(encoding="utf-8"))
        value = ledger["versions"][path.parent.name]
        valid = (
            ledger.get("schema") == "finrisk-host-admission-ledger/1.0"
            and Path(value["registry"]).resolve() == registry.resolve()
            and Path(value["snapshot_path"]).resolve() == path.resolve()
            and value["manifest_sha256"] == digest
            and isinstance(value["release_id"], str)
            and value["authentication"]["algorithm"] == "HMAC-SHA256"
            and re.fullmatch(r"[0-9a-f]{64}", value["authentication"]["signature"]) is not None
        )
    except (OSError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise ValueError("HOST_ADMISSION_REQUIRED")


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
    _require_registered_admission(registry, path)
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
    marker = registry / "host-baseline.json"
    if marker.is_file():
        pin = json.loads(marker.read_text(encoding="utf-8"))
        if file_hash(base_path / "manifest.json") != pin.get("manifest_sha256"):
            raise ValueError("BASELINE_HOST_PIN_MISMATCH")
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


def publish_snapshot(
    registry: Path, snapshot_path: Path, *, authority=None, admission=None
) -> dict[str, Any]:
    """Publish a complete versions/<directory-version>/snapshot; caller owns run locking."""
    registry = Path(registry).resolve()
    path = _snapshot_under_registry(registry, Path(snapshot_path))
    verified = _require_admission(registry, path, authority=authority, admission=admission)
    snapshot = _validate_snapshot(path)
    _record_admission(registry, path, verified)
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


def rollback_snapshot(
    registry: Path, version: str, *, authority=None, admission=None
) -> dict[str, Any]:
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
    return publish_snapshot(registry, path, authority=authority, admission=admission)


USAGE_SCHEMA = "finrisk-managed-usage/1.0"


def _usage_path(registry: Path, version: str, sha256: str, *, verify=True) -> Path:
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("INVALID_USAGE_VERSION")
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("INVALID_USAGE_HASH")
    path = _snapshot_under_registry(registry, registry / "versions" / version / "snapshot")
    if verify and file_hash(path / "manifest.json") != sha256:
        raise ValueError("USAGE_MANIFEST_HASH_MISMATCH")
    return path


def usage_state(registry: Path, *, verify_selected=True) -> dict[str, Any]:
    registry = Path(registry).resolve()
    value = json.loads((registry / "usage.json").read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != USAGE_SCHEMA
        or type(value.get("enabled")) is not bool
        or not isinstance(value.get("history"), list)
    ):
        raise ValueError("INVALID_USAGE_STATE")
    _usage_path(registry, value.get("base_version"), value.get("base_manifest_sha256"))
    selected = value.get("selected_version")
    if selected is not None:
        _usage_path(
            registry,
            selected,
            value.get("selected_manifest_sha256"),
            verify=verify_selected and value["enabled"],
        )
    elif value["enabled"]:
        raise ValueError("USAGE_NO_SELECTED_VERSION")
    for item in value["history"]:
        if not isinstance(item, dict):
            raise ValueError("INVALID_USAGE_HISTORY")
        _usage_path(registry, item.get("version"), item.get("manifest_sha256"), verify=False)
    return value


def _write_usage(registry: Path, value: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".usage-", suffix=".tmp", dir=registry)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, registry / "usage.json")
    finally:
        temporary.unlink(missing_ok=True)


def initialize_usage(
    registry: Path, base_snapshot: Path, *, expected_manifest_sha256=None
) -> dict[str, Any]:
    """Register a read-only baseline and start with extensions excluded from retrieval."""
    from agent_rag.knowledge_updates import update_lock

    registry = Path(registry).resolve()
    if (
        not isinstance(expected_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
    ):
        raise ValueError("BASELINE_HOST_PIN_REQUIRED")
    if file_hash(Path(base_snapshot) / "manifest.json") != expected_manifest_sha256:
        raise ValueError("BASELINE_HOST_PIN_MISMATCH")
    with update_lock(registry):
        if (registry / "usage.json").exists():
            state = usage_state(registry, verify_selected=False)
            if file_hash(Path(base_snapshot) / "manifest.json") != state["base_manifest_sha256"]:
                raise ValueError("USAGE_BASELINE_ALREADY_BOUND")
            from agent_rag.knowledge_updates import write_json

            pin = {"version": state["base_version"], "manifest_sha256": expected_manifest_sha256}
            marker = registry / "host-baseline.json"
            if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != pin:
                raise ValueError("USAGE_BASELINE_ALREADY_BOUND")
            if not marker.exists():
                _validate_snapshot(
                    _usage_path(registry, state["base_version"], expected_manifest_sha256)
                )
                write_json(marker, pin)
            if state["selected_version"]:
                try:
                    selected = _usage_path(
                        registry, state["selected_version"], state["selected_manifest_sha256"]
                    )
                    _require_registered_admission(registry, selected)
                except (ValueError, OSError):
                    state.update(
                        enabled=False,
                        selected_version=None,
                        selected_manifest_sha256=None,
                        history=[],
                    )
                    _write_usage(registry, state)
            return state
        baseline = register_base_snapshot(registry, Path(base_snapshot))
        from agent_rag.knowledge_updates import write_json

        pin = {"version": baseline.parent.name, "manifest_sha256": expected_manifest_sha256}
        marker = registry / "host-baseline.json"
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != pin:
            raise ValueError("USAGE_BASELINE_ALREADY_BOUND")
        write_json(marker, pin)
        if not (registry / "current.json").exists():
            publish_snapshot(registry, baseline)
        state = {
            "schema_version": USAGE_SCHEMA,
            "enabled": False,
            "base_version": baseline.parent.name,
            "base_manifest_sha256": file_hash(baseline / "manifest.json"),
            "selected_version": None,
            "selected_manifest_sha256": None,
            "history": [],
        }
        _write_usage(registry, state)
        return state


def retrieval_snapshot(registry: Path) -> Path:
    """Managed readers follow the explicit selection, never a later automatic publication."""
    registry = Path(registry).resolve()
    if not (registry / "usage.json").exists():
        return current_snapshot(registry)
    state = usage_state(registry)
    prefix = "selected" if state["enabled"] else "base"
    path = _usage_path(registry, state[f"{prefix}_version"], state[f"{prefix}_manifest_sha256"])
    _require_registered_admission(registry, path)
    return path


def select_version(
    registry: Path, version: str, *, authority=None, admission=None
) -> dict[str, Any]:
    from agent_rag.knowledge_updates import update_lock

    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("INVALID_USAGE_VERSION")
    registry = Path(registry).resolve()
    with update_lock(registry):
        state = usage_state(registry)
        path = _snapshot_under_registry(registry, registry / "versions" / version / "snapshot")
        marker = path.parent / "update-complete.json"
        if not marker.is_file() or (path.parent / "update-failed.json").exists():
            raise ValueError("VERSION_NOT_COMPLETE")
        _require_admission(registry, path, authority=authority, admission=admission)
        complete = json.loads(marker.read_text(encoding="utf-8"))
        snapshot = _validate_snapshot(path)
        if (
            complete.get("status") != "built"
            or complete.get("version") != version
            or complete.get("snapshot_id") != snapshot.manifest["snapshot_id"]
            or Path(complete.get("snapshot_path", "")).resolve() != path
        ):
            raise ValueError("VERSION_NOT_COMPLETE")
        previous = _usage_path(
            registry,
            state["selected_version"] or state["base_version"],
            state["selected_manifest_sha256"] or state["base_manifest_sha256"],
        )
        previous_id = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))[
            "snapshot_id"
        ]
        retained = {item["version"]: item["manifest_sha256"] for item in state["history"]}
        digest = file_hash(path / "manifest.json")
        if version != state["selected_version"]:
            if snapshot.manifest.get("parent_snapshot_id") != previous_id:
                raise ValueError("STALE_PARENT_REBUILD_ON_SELECTED_VERSION")
        if state["selected_version"] and state["selected_version"] != version:
            retained[state["selected_version"]] = state["selected_manifest_sha256"]
        retained.pop(version, None)
        # A failed usage write remains safe: managed readers ignore current.json.
        publish_snapshot(registry, path, authority=authority, admission=admission)
        state.update(
            selected_version=version,
            selected_manifest_sha256=digest,
            history=[{"version": key, "manifest_sha256": sha} for key, sha in retained.items()],
        )
        _write_usage(registry, state)
        return state


def set_usage(registry: Path, enabled: bool, *, authority=None, admission=None) -> dict[str, Any]:
    from agent_rag.knowledge_updates import update_lock

    if type(enabled) is not bool:
        raise ValueError("INVALID_USAGE_ENABLED")
    registry = Path(registry).resolve()
    with update_lock(registry):
        state = usage_state(registry, verify_selected=enabled)
        if enabled and not state["selected_version"]:
            raise ValueError("USAGE_NO_SELECTED_VERSION")
        if enabled:
            path = _usage_path(
                registry, state["selected_version"], state["selected_manifest_sha256"]
            )
            _require_admission(registry, path, authority=authority, admission=admission)
            _validate_snapshot(path)
        state["enabled"] = enabled
        _write_usage(registry, state)
        return state
