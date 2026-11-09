import importlib
import json

from test_collection_selection import completed, initialize
from test_knowledge_registry import version_snapshot
from test_registry_runtime import finish, loaded

from agent_rag import knowledge_registry as kr
from agent_rag.knowledge_updates import pending_documents


def test_managed_root_uses_gate_and_active_task_stays_on_old_snapshot(tmp_path, monkeypatch):
    m = importlib.import_module("agent_rag.router")
    managed = tmp_path / "managed"
    registry = managed / "registry"
    base = version_snapshot(tmp_path / "original", "base")
    # Isolate serving-version pinning; real host admission has separate tests.
    monkeypatch.setattr(kr, "_require_admission", lambda *args, **kwargs: None)
    monkeypatch.setattr(kr, "_require_registered_admission", lambda *args, **kwargs: None)
    initialize(kr, registry, base)
    baseline = kr.retrieval_snapshot(registry)
    newer = completed(registry, "new", "snapshot-base")
    monkeypatch.setenv("FINRISK_AGENT_KB_REGISTRY", str(registry))
    monkeypatch.setenv("FINRISK_AGENT_TASK_ROOT", str(tmp_path / "tasks"))
    monkeypatch.setattr(m, "load_v2_snapshot", loaded)
    assert m._snapshot_path() == baseline
    service = m._build_service(lambda *a: None)
    try:
        record, _ = service.admit({"text": "第一题"}, None)
        old = service._current
        kr.select_version(registry, "new")
        assert m._snapshot_path() == baseline
        kr.set_usage(registry, True)
        assert m._snapshot_path() == newer
        service.request_refresh()
        assert service.snapshot.root == baseline
        finish(old, record)
        service.request_refresh()
        if service._refresh_thread:
            service._refresh_thread.join(2)
        service.request_refresh()
        assert service.snapshot.root == newer
        assert service.bundle(record["task_id"])["snapshot"]["snapshot_id"] == "snapshot-base"
        kr.set_usage(registry, False)
        service.request_refresh()
        if service._refresh_thread:
            service._refresh_thread.join(2)
        service.request_refresh()
        assert service.snapshot.root == baseline
    finally:
        service.close()


def test_bank_insurance_is_opt_in_while_noninsurance_is_still_skipped(tmp_path):
    text = "当事人：某银行股份有限公司。代理销售保险时存在销售误导。"
    (tmp_path / "a.txt").write_text(text, encoding="utf-8")
    (tmp_path / "b.txt").write_text("某银行贷款业务违规。", encoding="utf-8")
    (tmp_path / "index.jsonl").write_text(
        "\n".join(
            json.dumps({"txt_path": name, "source_url": "https://www.nfra.gov.cn/a"})
            for name in ["a.txt", "b.txt"]
        ),
        encoding="utf-8",
    )
    assert pending_documents(tmp_path, set()) == []
    assert len(pending_documents(tmp_path, set(), include_bank_insurance=True)) == 1
