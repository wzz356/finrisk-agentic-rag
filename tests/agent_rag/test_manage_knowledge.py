import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_knowledge_registry import version_snapshot

from agent_rag.collection_service import CollectionService


def cli():
    path = Path(__file__).resolve().parents[2] / "scripts/manage_knowledge.py"
    assert path.is_file(), "隔离采集CLI未实现"
    spec = importlib.util.spec_from_file_location("manage_knowledge_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history(root, *, missing=False):
    root.mkdir()
    (root / "index.jsonl").write_text(
        json.dumps(
            {
                "txt_path": "a.txt",
                "source_url": "https://www.nfra.gov.cn/a",
                "title": "保险销售",
                "column": 4114,
                "doc_id": 1,
            }
        ),
        encoding="utf-8",
    )
    if not missing:
        (root / "a.txt").write_text("某银行代理保险销售误导。", encoding="utf-8")
    return root


def config(root):
    base = version_snapshot(root / "external", "base")
    assets = {}
    for name in ("normalizer", "projector", "contract"):
        path = root / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        from agent_rag.v2_snapshot import file_hash

        assets[name] = {"path": str(path), "sha256": file_hash(path)}
    path = root / "trusted-update.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "finrisk-web-update/1.0",
                "base_snapshot": str(base),
                "base_manifest_sha256": file_hash(base / "manifest.json"),
                "build_assets": assets,
                "registry": str(root / "old-registry"),
                "crawl_dir": str(root / "old-crawl"),
            }
        ),
        encoding="utf-8",
    )
    return path


def output(capsys):
    captured = capsys.readouterr()
    assert not captured.err
    return json.loads(captured.out)


def test_default_root_and_switches_are_isolated_and_disabled(tmp_path, capsys):
    assert cli().main(["status"], project_root=tmp_path) == 0
    status = output(capsys)
    assert status["settings"] == {
        "crawl_enabled": False,
        "usage_enabled": False,
        "daily_hour": 8,
    }
    assert (tmp_path / "runtime/knowledge_updates/inventory/collection.sqlite3").is_file()
    assert not (tmp_path / "inventory").exists()


@pytest.mark.parametrize("candidate", ["../outside", "outside", "runtime/../../outside"])
def test_relative_root_escape_is_rejected_before_writes(tmp_path, capsys, candidate):
    assert cli().main(["--root", candidate, "status"], project_root=tmp_path) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"] == "ROOT_OUTSIDE_PROJECT_RUNTIME"
    assert not captured.out
    assert not (tmp_path / "runtime").exists()


def test_absolute_external_root_is_rejected(tmp_path, capsys):
    candidate = tmp_path / "outside"
    assert cli().main(["--root", str(candidate), "status"], project_root=tmp_path) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ROOT_OUTSIDE_PROJECT_RUNTIME"
    assert not candidate.exists()


def test_root_inside_runtime_is_accepted(tmp_path, capsys):
    assert cli().main(["--root", "runtime/alternative", "status"], project_root=tmp_path) == 0
    output(capsys)
    assert (tmp_path / "runtime/alternative/inventory/collection.sqlite3").is_file()


def test_bind_redirects_writes_and_preserves_external_assets(tmp_path, capsys):
    source = config(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert cli().main(["bind", "--config", str(source)], project_root=tmp_path) == 0
    assert output(capsys) == {"status": "configured", "model_requests_sent": 0}
    managed = tmp_path / "runtime/knowledge_updates"
    bound = json.loads((managed / "build-config.json").read_text(encoding="utf-8"))
    assert Path(bound["registry"]) == managed / "registry"
    assert Path(bound["crawl_dir"]) == managed / "staging"
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not (tmp_path / "old-registry").exists()
    assert not (tmp_path / "old-crawl").exists()


def test_configure_multiple_history_roots_then_import_and_status(tmp_path, capsys):
    module = cli()
    first, second = history(tmp_path / "first"), history(tmp_path / "second")
    assert (
        module.main(
            [
                "configure-history",
                "--source",
                str(first),
                "--source",
                str(second),
                "--since",
                "2026-08-28",
            ],
            project_root=tmp_path,
        )
        == 0
    )
    configured = output(capsys)
    assert configured == {"status": "configured", "source_count": 2, "backfill_since": "2026-08-28"}
    assert module.main(["import-history"], project_root=tmp_path) == 0
    job = output(capsys)
    assert job["kind"] == "import_history" and job["status"] == "completed"
    assert len(job["result"]["batches"]) == 2
    assert str(first) not in json.dumps(job)
    assert module.main(["status"], project_root=tmp_path) == 0
    status = output(capsys)
    assert status["counts"]["documents"] == 1
    assert status["counts"]["sources"] == 2
    assert status["recommended_since"] == "2026-08-28"
    assert status["settings"]["usage_enabled"] is False


def test_incomplete_job_exits_nonzero_without_exposing_source_path(tmp_path, capsys):
    module = cli()
    source = history(tmp_path / "private-history", missing=True)
    CollectionService(tmp_path / "runtime/knowledge_updates").configure_history([source])
    assert module.main(["import-history"], project_root=tmp_path) == 1
    job = output(capsys)
    assert job["status"] == "incomplete"
    assert job["result"]["batches"][0]["errors"][0] == {"code": "HISTORY_ROW_ERROR", "line": 1}
    assert "private-history" not in json.dumps(job)


def test_worker_once_only_collects_even_with_generation_permission(tmp_path, capsys, monkeypatch):
    module = cli()
    calls = []

    def collect(store, batch_id, **kwargs):
        calls.append(kwargs)
        return {"complete": True, "documents": 0}

    def forbidden(*args, **kwargs):
        pytest.fail("worker调用了收费处理")

    class Service(CollectionService):
        def __init__(self, root):
            super().__init__(root, crawler=collect, updater=forbidden)

        def tick(self, now=None):
            return super().tick(datetime(2026, 10, 18, 9, tzinfo=timezone(timedelta(hours=8))))

    monkeypatch.setattr(module, "CollectionService", Service)
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_KB_GENERATION", "1")
    service = Service(tmp_path / "runtime/knowledge_updates")
    service.store.set_value("backfill_since", "2026-08-28")
    assert module.main(["worker", "--once"], project_root=tmp_path) == 0
    assert output(capsys)["status"] == "idle"
    assert not calls
    assert module.main(["set-collection", "--enabled"], project_root=tmp_path) == 0
    assert output(capsys) == {"crawl_enabled": True}
    assert module.main(["worker", "--once"], project_root=tmp_path) == 0
    assert output(capsys)["kind"] == "scheduled_crawl"
    assert calls == [
        {"since": "2026-08-28", "until": "2026-10-18", "max_pages": 2, "baseline_hashes": set()}
    ]
    assert service.status()["settings"]["usage_enabled"] is False


def test_crawl_waits_for_safe_result_and_passes_explicit_range(tmp_path, capsys, monkeypatch):
    module = cli()
    received = []

    class Service(CollectionService):
        def __init__(self, root):
            def collect(store, batch_id, **kwargs):
                received.append(kwargs)
                return {"complete": True, "documents": 0}

            super().__init__(root, crawler=collect)

    monkeypatch.setattr(module, "CollectionService", Service)
    assert (
        module.main(
            ["crawl", "--since", "2026-08-28", "--until", "2026-10-18", "--max-pages", "1"],
            project_root=tmp_path,
        )
        == 0
    )
    job = output(capsys)
    assert job["status"] == "completed" and job["kind"] == "crawl"
    assert received[0]["since"] == "2026-08-28" and received[0]["max_pages"] == 1


def test_process_requires_permission_and_explicit_limits(tmp_path, capsys, monkeypatch):
    module = cli()
    monkeypatch.delenv("FINRISK_AGENT_ALLOW_KB_GENERATION", raising=False)
    args = [
        "process",
        "--batch-id",
        "one",
        "--cap-cny",
        "1",
        "--max-documents",
        "1",
        "--max-requests",
        "1",
    ]
    assert module.main(args, project_root=tmp_path) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "GENERATION_DISABLED"
    with pytest.raises(SystemExit) as error:
        module.main(["process", "--batch-id", "one"], project_root=tmp_path)
    assert error.value.code == 2
    capsys.readouterr()


def test_unreviewed_selection_is_failed_job_and_usage_stays_off(tmp_path, capsys):
    module = cli()
    source = config(tmp_path)
    assert module.main(["bind", "--config", str(source)], project_root=tmp_path) == 0
    output(capsys)
    selected = version_snapshot(tmp_path / "runtime/knowledge_updates/registry", "candidate")
    manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
    manifest["parent_snapshot_id"] = "snapshot-base"
    (selected / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (selected.parent / "update-complete.json").write_text(
        json.dumps(
            {
                "status": "built",
                "version": "candidate",
                "snapshot_id": manifest["snapshot_id"],
                "snapshot_path": str(selected),
            }
        ),
        encoding="utf-8",
    )
    # A completed mechanical build is no longer an approval authority.
    assert module.main(["select", "--version", "candidate"], project_root=tmp_path) == 1
    job = output(capsys)
    assert job["kind"] == "activate" and job["status"] == "failed"
    assert job["error"] == "HOST_ADMISSION_REQUIRED" and job["result"] is None
    assert module.main(["set-usage", "--enabled"], project_root=tmp_path) == 1
    assert output(capsys)["error"] == "USAGE_NO_SELECTED_VERSION"
    assert module.main(["set-usage", "--disabled"], project_root=tmp_path) == 0
    assert output(capsys)["result"]["usage_enabled"] is False


def test_errors_do_not_echo_arbitrary_exception_text(tmp_path, capsys, monkeypatch):
    module = cli()

    def forbidden(*args, **kwargs):
        raise RuntimeError("credential-value C:/private/key.env")

    monkeypatch.setattr(module.CollectionService, "bind", forbidden)
    assert module.main(["bind", "--config", "irrelevant"], project_root=tmp_path) == 1
    assert json.loads(capsys.readouterr().err) == {"error": "RuntimeError"}
