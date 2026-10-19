import importlib
import importlib.util
import json
import threading
from datetime import datetime, timedelta, timezone

import pytest
from test_knowledge_registry import version_snapshot


def module():
    assert importlib.util.find_spec("agent_rag.collection_service"), "管理服务未实现"
    return importlib.import_module("agent_rag.collection_service")


def history(root):
    root.mkdir()
    (root / "a.txt").write_bytes("某银行代理保险销售误导。".encode())
    (root / "index.jsonl").write_text(
        json.dumps(
            {
                "txt_path": "a.txt",
                "source_url": "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1",
                "title": "保险销售",
                "column": 4114,
                "doc_id": 1,
            }
        ),
        encoding="utf-8",
    )


def test_unconfigured_import_is_independent_from_usage_and_is_idempotent(tmp_path):
    m = module()
    old = tmp_path / "history"
    history(old)
    s = m.CollectionService(tmp_path / "managed", history_roots=[old])
    initial = s.status()
    assert initial["settings"] == {"crawl_enabled": False, "usage_enabled": False, "daily_hour": 8}
    assert initial["configuration"]["ready"] is False
    s.submit("import_history", {}).join(5)
    s.submit("import_history", {}).join(5)
    assert s.status()["counts"]["documents"] == 1
    assert s.status()["counts"]["sources"] == 2
    assert s.status()["settings"]["usage_enabled"] is False
    assert s.status()["jobs"][0]["status"] == "completed"
    assert s.store.documents()["items"][0]["contains_insurance"] is True
    with pytest.raises(ValueError, match="NOT_CONFIGURED"):
        s.submit("usage", {"enabled": True})


def test_daily_worker_only_collects_and_does_not_advance_on_partial(tmp_path):
    m = module()
    calls = []

    def collect(store, batch_id, **kwargs):
        calls.append(kwargs)
        return {"complete": False, "documents": 0, "errors": [], "next_pages": {"4113": 2}}

    s = m.CollectionService(tmp_path / "managed", crawler=collect)
    s.store.set_value("backfill_since", "2026-08-29")
    now = datetime(2026, 10, 18, 9, tzinfo=timezone(timedelta(hours=8)))
    assert s.tick(now) is None
    s.set_crawl_enabled(True)
    s.tick(now).join(5)
    assert calls[0]["since"] == "2026-08-29"
    assert s.tick(now) is None
    status = s.status()
    assert status["jobs"][0]["status"] == "incomplete"
    assert status["last_successful_crawl"] is None
    assert status["settings"]["usage_enabled"] is False


def test_jobs_are_exclusive_across_instances(tmp_path):
    m = module()
    entered, release = threading.Event(), threading.Event()

    def collect(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return {"complete": True}

    s = m.CollectionService(tmp_path / "managed", crawler=collect)
    other = m.CollectionService(tmp_path / "managed", crawler=collect)
    job = s.submit("crawl", {"since": "2026-10-01", "until": "2026-10-18", "max_pages": 1})
    assert entered.wait(2)
    try:
        with pytest.raises(ValueError, match="BUSY"):
            other.submit("crawl", {"since": "2026-10-01", "until": "2026-10-18", "max_pages": 1})
    finally:
        release.set()
        job.join(5)


def test_process_requires_explicit_permission_then_builds_without_publication(
    tmp_path, monkeypatch
):
    m = module()
    root = tmp_path / "managed"
    base = version_snapshot(tmp_path / "original", "base")
    assets = {}
    for name in ["normalizer", "projector", "contract"]:
        path = tmp_path / (name + ".txt")
        path.write_text(name, encoding="utf-8")
        assets[name] = {"path": str(path), "sha256": m.file_hash(path)}
    config = {
        "schema_version": "finrisk-web-update/1.0",
        "base_snapshot": str(base),
        "base_manifest_sha256": m.file_hash(base / "manifest.json"),
        "build_assets": assets,
    }
    cfg = tmp_path / "external-config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    calls = []

    def build(config, **kwargs):
        calls.append((config, kwargs))
        return {"status": "no_new_documents", "published": False}

    s = m.CollectionService(root, updater=build)
    s.bind(cfg)
    bound = s._config()
    assert all(m.Path(a["path"]).is_relative_to(root) for a in bound["build_assets"].values())
    batch = s.store.create_batch("history")
    s.store.ingest(
        batch["id"],
        b"\xef\xbb\xbf" + "某银行代理保险销售误导。".encode(),
        {"source_url": "https://www.nfra.gov.cn/a", "column": 4114, "doc_id": 1},
    )
    args = {"batch_id": batch["id"], "cap_cny": 1.0, "max_documents": 2, "max_requests": 2}
    with pytest.raises(ValueError, match="GENERATION_DISABLED"):
        s.submit("process", args)
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_KB_GENERATION", "1")
    s.submit("process", args).join(5)
    assert len(calls) == 1
    config, kwargs = calls[0]
    assert kwargs["publish"] is False and kwargs["generate"] is True
    assert config["include_bank_insurance"] is True
    assert config["build_base_snapshot"].endswith("snapshot")
    assert config["registry"] == str(root / "registry")
    assert (root / "registry/usage.json").is_file()
    assert s.status()["settings"]["usage_enabled"] is False
    staged = list((root / "staging").glob("*/index.jsonl"))
    assert len(staged) == 1 and "保险" in (
        root
        / "staging"
        / batch["id"]
        / "texts"
        / next((root / "staging" / batch["id"] / "texts").iterdir()).name
    ).read_text(encoding="utf-8")


def test_settings_and_job_inputs_strictly_validate(tmp_path):
    s = module().CollectionService(tmp_path / "managed")
    for value in [1, "true", None]:
        with pytest.raises(ValueError):
            s.set_crawl_enabled(value)
    with pytest.raises(ValueError):
        s.submit("crawl", {"since": "2026-02-30", "until": "2026-10-18", "max_pages": 1})
    with pytest.raises(ValueError):
        s.submit("crawl", {"since": "2026-10-18", "until": "2026-10-01", "max_pages": 1})
    with pytest.raises(ValueError):
        s.submit("unknown", {})


def test_status_recovers_orphaned_job_and_exposes_scheduler_error(tmp_path):
    m = module()
    s = m.CollectionService(tmp_path / "managed")
    path = s.root / "jobs/orphan.json"
    m.write_json(
        path, {"id": "orphan", "kind": "crawl", "status": "running", "created_at": "2026-10-18"}
    )
    s.set_crawl_enabled(True)
    s.tick(datetime(2026, 10, 18, 10, tzinfo=m.CHINA))
    status = s.status()
    assert status["jobs"][0]["status"] == "interrupted"
    assert status["scheduler_error"] == "BACKFILL_DATE_NOT_CONFIGURED"


def test_processing_partial_batch_reports_remaining_and_safe_review(tmp_path, monkeypatch):
    m = module()
    s = m.CollectionService(
        tmp_path / "managed",
        updater=lambda *a, **k: {
            "status": "built",
            "published": False,
            "pending_documents": 10,
            "selected": 2,
            "accepted_documents": 1,
            "review_queue": [
                {"file_id": "WEB-a", "code": "NEEDS_REVIEW", "candidate_dir": "C:/private/path"}
            ],
            "generation_budget": {"requests": 2, "settled_cost_cny": 0.02},
        },
    )
    batch = s.store.create_batch("crawl")
    monkeypatch.setattr(
        s, "_config", lambda: {"registry": str(s.registry), "include_bank_insurance": True}
    )
    monkeypatch.setattr(
        m, "usage_state", lambda r: {"selected_version": None, "base_version": "base"}
    )
    result = s._process(
        {"batch_id": batch["id"], "cap_cny": 1, "max_documents": 2, "max_requests": 2}
    )
    assert result["complete"] is False and result["remaining_documents"] == 9
    assert result["generation_budget"]["requests"] == 2
    assert "candidate_dir" not in result["review_queue"][0]
    assert s.store.list_batches()[0]["status"] == "partially_processed"
