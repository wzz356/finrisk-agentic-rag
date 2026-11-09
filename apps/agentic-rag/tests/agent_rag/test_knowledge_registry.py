import hashlib
import importlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest
from test_dense_backend import REVISION, metadata
from test_v2_snapshot import fixture_assets

from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash


def registry_module():
    assert importlib.util.find_spec("agent_rag.knowledge_registry"), "版本发布器尚未实现"
    return importlib.import_module("agent_rag.knowledge_registry")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def version_snapshot(registry, version):
    directory = registry / "versions" / version
    directory.mkdir(parents=True)
    sources, units, dictionary, _, _ = fixture_assets(directory)
    snapshot = V2KnowledgeSnapshot.create(sources, units, dictionary, directory / "snapshot")
    vector = directory / "vectors.npy"
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1, 512), }"
    header += b" " * ((16 - (10 + len(header) + 1) % 16) % 16) + b"\n"
    vector.write_bytes(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + struct.pack("<512f", 1.0, *([0.0] * 511))
    )
    model = directory / REVISION
    model.mkdir()
    (model / "config.json").write_bytes(b"model")
    entries = [{"path": "config.json", "size": 5, "sha256": hashlib.sha256(b"model").hexdigest()}]
    meta = metadata(snapshot.units)
    meta["binding"]["model"] = {
        "revision": REVISION,
        "path": str(model.resolve()),
        "files": entries,
        "cache_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
    }
    # The metadata cache key binds the complete model fingerprint.
    text_sha = meta["ordered_unit_text_sha256"]
    meta["key"] = hashlib.sha256(
        json.dumps(
            {"text_sha": text_sha, "config": meta["binding"]}, sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()
    meta["npy_sha256"] = file_hash(vector)
    meta_path = directory / "vectors.json"
    write_json(meta_path, meta)
    config = {
        "python_executable": sys.executable,
        "index_path": str(units.resolve()),
        "index_sha256": file_hash(units),
        "vector_path": str(vector.resolve()),
        "vector_meta_path": str(meta_path.resolve()),
        "vector_meta_sha256": file_hash(meta_path),
        "model_path": str(model.resolve()),
        "model_revision": REVISION,
        "device": "cuda",
        "precision": "float16",
        "corpus_mode": "incremental-v1",
        "document_count": 1,
    }
    config_path = directory / "dense.json"
    write_json(config_path, config)
    rules = directory / "rules.txt"
    rules.write_text("rules", encoding="utf-8")
    manifest = snapshot.manifest
    manifest["snapshot_id"] = "snapshot-" + version
    manifest["backend"] = {
        "mode": "live",
        "rerank_protocol": "span-ref-v2",
        "assets": {
            name: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for name, path in [("rules", rules), ("dense_config", config_path)]
        },
    }
    write_json(snapshot.root / "manifest.json", manifest)
    return snapshot.root


def test_publish_preserves_history_and_rollback_reuses_validated_version(tmp_path, monkeypatch):
    module = registry_module()
    registry = tmp_path / "registry"
    first = version_snapshot(registry, "v1")
    second = version_snapshot(registry, "v2")
    # Publisher validates assets locally; it must never start runtime workers/API clients.
    monkeypatch.setattr(
        "agent_rag.dense_backend.ResidentDenseIndex",
        lambda *args: pytest.fail("发布不得启动GPU或检索"),
    )
    result = module.publish_snapshot(registry, first)
    assert result["version"] == "v1"
    assert module.current_snapshot(registry) == first
    module.publish_snapshot(registry, second)
    pointer = json.loads((registry / "current.json").read_text(encoding="utf-8"))
    assert any(item["version"] == "v1" for item in pointer["history"])
    rolled = module.rollback_snapshot(registry, "v1")
    assert rolled["version"] == "v1"
    assert module.current_snapshot(registry) == first
    assert second.exists()


@pytest.mark.parametrize(
    "damage", ["source", "rules", "dense", "vector", "vector_shape", "model", "quote"]
)
def test_invalid_version_does_not_change_current(tmp_path, damage):
    module = registry_module()
    registry = tmp_path / "registry"
    first = version_snapshot(registry, "good")
    module.publish_snapshot(registry, first)
    before = (registry / "current.json").read_bytes()
    broken = version_snapshot(registry, "bad")
    manifest = json.loads((broken / "manifest.json").read_text(encoding="utf-8"))
    if damage == "source":
        Path(manifest["assets"]["sources"]["path"]).write_text("changed", encoding="utf-8")
    elif damage == "rules":
        del manifest["backend"]["assets"]["rules"]
        write_json(broken / "manifest.json", manifest)
    elif damage == "dense":
        del manifest["backend"]["assets"]["dense_config"]
        write_json(broken / "manifest.json", manifest)
    elif damage == "vector":
        (broken.parent / "vectors.npy").write_bytes(b"corrupted")
    elif damage == "vector_shape":
        # A self-consistent SHA cannot turn malformed NPY bytes into a usable version.
        vector = broken.parent / "vectors.npy"
        vector.write_bytes(b"not-a-npy")
        meta_path = broken.parent / "vectors.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["npy_sha256"] = file_hash(vector)
        write_json(meta_path, meta)
        config_path = broken.parent / "dense.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["vector_meta_sha256"] = file_hash(meta_path)
        write_json(config_path, config)
        manifest["backend"]["assets"]["dense_config"]["sha256"] = file_hash(config_path)
        write_json(broken / "manifest.json", manifest)
    elif damage == "model":
        (broken.parent / REVISION / "config.json").write_bytes(b"corrupted")
    else:
        index = Path(manifest["assets"]["index"]["path"])
        unit = json.loads(index.read_text(encoding="utf-8"))
        unit["source_text"] = "错引原文"
        write_json(index, unit)
        manifest["assets"]["index"]["sha256"] = file_hash(index)
        manifest["index_sha256"] = file_hash(index)
        write_json(broken / "manifest.json", manifest)
    with pytest.raises((ValueError, KeyError, OSError)):
        module.publish_snapshot(registry, broken)
    assert (registry / "current.json").read_bytes() == before


def test_pointer_replace_failure_keeps_existing_version(tmp_path, monkeypatch):
    module = registry_module()
    registry = tmp_path / "registry"
    first = version_snapshot(registry, "v1")
    second = version_snapshot(registry, "v2")
    module.publish_snapshot(registry, first)
    before = (registry / "current.json").read_bytes()

    def fail(*args):
        raise OSError("simulated interrupted atomic replace")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        module.publish_snapshot(registry, second)
    assert (registry / "current.json").read_bytes() == before


def test_registry_rejects_outside_unfinished_and_forged_paths(tmp_path):
    module = registry_module()
    registry = tmp_path / "registry"
    outside = version_snapshot(tmp_path / "outside", "v1")
    with pytest.raises(ValueError, match="PATH"):
        module.publish_snapshot(registry, outside)
    unfinished = registry / "versions" / "unfinished" / "snapshot"
    unfinished.mkdir(parents=True)
    with pytest.raises((ValueError, OSError)):
        module.publish_snapshot(registry, unfinished)
    write_json(registry / "current.json", {"snapshot_path": "../../outside/versions/v1/snapshot"})
    with pytest.raises(ValueError, match="PATH|POINTER"):
        module.current_snapshot(registry)
    with pytest.raises(ValueError):
        module.rollback_snapshot(registry, "../outside")


def test_corrupted_current_manifest_does_not_prevent_rollback_to_valid_history(tmp_path):
    module = registry_module()
    registry = tmp_path / "registry"
    first = version_snapshot(registry, "v1")
    second = version_snapshot(registry, "v2")
    module.publish_snapshot(registry, first)
    module.publish_snapshot(registry, second)
    (second / "manifest.json").write_text("corrupted", encoding="utf-8")
    module.rollback_snapshot(registry, "v1")
    assert module.current_snapshot(registry) == first


def test_external_base_registration_is_read_only_idempotent_and_rollbackable(tmp_path, monkeypatch):
    module = registry_module()
    assert hasattr(module, "register_base_snapshot"), "外部基础快照尚未注册到版本历史"
    registry = tmp_path / "registry"
    external_base = version_snapshot(tmp_path / "external-original-library", "base")
    original = (external_base / "manifest.json").read_bytes()
    monkeypatch.setattr(
        "agent_rag.dense_backend.ResidentDenseIndex",
        lambda *args: pytest.fail("基础注册不得启动GPU"),
    )
    bootstrap = module.register_base_snapshot(registry, external_base)
    assert bootstrap == registry / "versions" / "bootstrap-snapshot-base" / "snapshot"
    assert (bootstrap / "manifest.json").read_bytes() == original
    assert (external_base / "manifest.json").read_bytes() == original
    assert not (registry / "current.json").exists()
    assert module.register_base_snapshot(registry, external_base) == bootstrap
    module.publish_snapshot(registry, bootstrap)
    newer = version_snapshot(registry, "new")
    module.publish_snapshot(registry, newer)
    pointer = json.loads((registry / "current.json").read_text(encoding="utf-8"))
    assert pointer["history"][0]["snapshot_id"] == "snapshot-base"
    module.rollback_snapshot(registry, "snapshot-base")
    assert module.current_snapshot(registry) == bootstrap
    assert (external_base / "manifest.json").read_bytes() == original


def test_base_registration_rejects_same_identity_with_different_manifest(tmp_path):
    module = registry_module()
    assert hasattr(module, "register_base_snapshot"), "外部基础快照尚未注册"
    registry = tmp_path / "registry"
    external_base = version_snapshot(tmp_path / "external", "base")
    bootstrap = module.register_base_snapshot(registry, external_base)
    before = (bootstrap / "manifest.json").read_bytes()
    manifest = json.loads((external_base / "manifest.json").read_text(encoding="utf-8"))
    manifest["knowledge_version"] = "changed-with-same-snapshot-id"
    write_json(external_base / "manifest.json", manifest)
    with pytest.raises(ValueError, match="IMMUTABLE"):
        module.register_base_snapshot(registry, external_base)
    assert (bootstrap / "manifest.json").read_bytes() == before
    assert not (registry / "current.json").exists()


def test_base_registration_validates_external_assets_before_registering(tmp_path):
    module = registry_module()
    assert hasattr(module, "register_base_snapshot"), "外部基础快照尚未注册"
    registry = tmp_path / "registry"
    external_base = version_snapshot(tmp_path / "external", "base")
    (external_base.parent / "vectors.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        module.register_base_snapshot(registry, external_base)
    assert not list(registry.glob("versions/*/snapshot/manifest.json"))
    assert not (registry / "current.json").exists()
