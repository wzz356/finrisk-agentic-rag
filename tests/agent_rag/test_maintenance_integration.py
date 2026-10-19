import json

from test_knowledge_registry import version_snapshot

from agent_rag.collection_service import CollectionService
from agent_rag.v2_snapshot import file_hash


def test_old_process_registers_events_without_generation_or_auto_publication(tmp_path, monkeypatch):
    base = version_snapshot(tmp_path / "old", "base")
    assets = {}
    for name in ("normalizer", "projector", "contract"):
        path = tmp_path / (name + ".txt")
        path.write_text(name, encoding="utf-8")
        assets[name] = {"path": str(path), "sha256": file_hash(path)}
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "schema_version": "finrisk-web-update/1.0",
                "base_snapshot": str(base),
                "base_manifest_sha256": file_hash(base / "manifest.json"),
                "build_assets": assets,
            }
        ),
        encoding="utf-8",
    )
    s = CollectionService(tmp_path / "managed")
    s.bind(cfg)
    batch = s.store.create_batch("fixture")
    s.store.ingest(
        batch["id"], "甲保险公司销售误导。".encode(), {"source_url": "https://www.nfra.gov.cn/a"}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("旧处理路径不应调用生成模型")

    monkeypatch.setattr("agent_rag.knowledge_updates.generate_candidate", forbidden)
    result = s._process(
        {"batch_id": batch["id"], "cap_cny": 1.0, "max_documents": 1, "max_requests": 2}
    )
    assert result["maintenance_queued"] is True
    assert result["model_requests_sent"] == 0 and len(result["event_ids"]) == 1
    assert s.status()["settings"]["usage_enabled"] is False
    assert s.store.list_batches()[0]["status"] == "needs_review"
