import json

import pytest
from test_knowledge_registry import version_snapshot, write_json

from agent_rag import knowledge_registry as registry_module


@pytest.fixture(autouse=True)
def mechanical_selection_only(monkeypatch):
    # These historical tests isolate parent/atomic-pointer mechanics. Admission
    # is independently exercised without mocks in test_maintenance_publish.py.
    monkeypatch.setattr(registry_module, "_require_admission", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        registry_module, "_require_registered_admission", lambda *args, **kwargs: None
    )


def initialize(module, registry, base):
    return module.initialize_usage(
        registry, base, expected_manifest_sha256=module.file_hash(base / "manifest.json")
    )


def api():
    assert hasattr(registry_module, "retrieval_snapshot"), "缺少独立的手动启用控制"
    return registry_module


def completed(registry, version, parent):
    path = version_snapshot(registry, version)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["parent_snapshot_id"] = parent
    write_json(path / "manifest.json", manifest)
    write_json(
        path.parent / "update-complete.json",
        {
            "status": "built",
            "version": version,
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_path": str(path),
            "new_units": 1,
            "source_count": 1,
        },
    )
    return path


def test_managed_usage_requires_selection_and_does_not_follow_later_publication(tmp_path):
    m = api()
    registry = tmp_path / "registry"
    external = version_snapshot(tmp_path / "original", "base")
    before = (external / "manifest.json").read_bytes()
    initialize(m, registry, external)
    baseline = m.retrieval_snapshot(registry)
    assert m.usage_state(registry)["enabled"] is False
    with pytest.raises(ValueError, match="NO_SELECTED"):
        m.set_usage(registry, True)
    first = completed(registry, "first", "snapshot-base")
    assert m.retrieval_snapshot(registry) == baseline
    m.select_version(registry, "first")
    assert m.usage_state(registry)["enabled"] is False
    assert m.retrieval_snapshot(registry) == baseline
    m.set_usage(registry, True)
    assert m.retrieval_snapshot(registry) == first
    later = completed(registry, "later", "snapshot-first")
    m.publish_snapshot(registry, later)
    assert m.retrieval_snapshot(registry) == first
    m.set_usage(registry, False)
    assert m.retrieval_snapshot(registry) == baseline
    assert m.usage_state(registry)["selected_version"] == "first"
    m.set_usage(registry, True)
    assert m.retrieval_snapshot(registry) == first
    assert (external / "manifest.json").read_bytes() == before


def test_legacy_pointer_unchanged_and_managed_corruption_fails_closed(tmp_path):
    m = api()
    registry = tmp_path / "registry"
    base = version_snapshot(registry, "base")
    m.publish_snapshot(registry, base)
    assert m.retrieval_snapshot(registry) == base
    initialize(m, registry, base)
    (registry / "usage.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="USAGE"):
        m.retrieval_snapshot(registry)


def test_unfinished_failed_and_stale_sibling_cannot_be_enabled(tmp_path):
    m = api()
    registry = tmp_path / "registry"
    base = version_snapshot(tmp_path / "original", "base")
    initialize(m, registry, base)
    version_snapshot(registry, "unfinished")
    with pytest.raises(ValueError, match="COMPLETE"):
        m.select_version(registry, "unfinished")
    a = completed(registry, "a", "snapshot-base")
    completed(registry, "b", "snapshot-base")
    m.select_version(registry, "a")
    m.set_usage(registry, True)
    with pytest.raises(ValueError, match="STALE_PARENT"):
        m.select_version(registry, "b")
    assert m.retrieval_snapshot(registry) == a
    completed(registry, "failed", "snapshot-a")
    write_json(registry / "versions/failed/update-failed.json", {"status": "failed"})
    with pytest.raises(ValueError, match="COMPLETE"):
        m.select_version(registry, "failed")
    with pytest.raises(ValueError, match="VERSION"):
        m.select_version(registry, "../b")


def test_initialize_is_idempotent_but_cannot_replace_baseline(tmp_path):
    m = api()
    registry = tmp_path / "registry"
    base = version_snapshot(tmp_path / "original", "base")
    initialize(m, registry, base)
    completed(registry, "a", "snapshot-base")
    m.select_version(registry, "a")
    before = (registry / "usage.json").read_bytes()
    initialize(m, registry, base)
    assert (registry / "usage.json").read_bytes() == before
    changed = version_snapshot(tmp_path / "original", "other")
    with pytest.raises(ValueError, match="BASELINE"):
        initialize(m, registry, changed)


def test_selected_history_does_not_bypass_cumulative_parent_guard(tmp_path):
    m = api()
    registry = tmp_path / "registry"
    initialize(m, registry, version_snapshot(tmp_path / "original", "base"))
    completed(registry, "a", "snapshot-base")
    completed(registry, "b", "snapshot-a")
    m.select_version(registry, "a")
    m.select_version(registry, "b")
    with pytest.raises(ValueError, match="STALE_PARENT"):
        m.select_version(registry, "a")
