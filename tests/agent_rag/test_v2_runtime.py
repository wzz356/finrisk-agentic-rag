import importlib
import importlib.util
import json

import pytest

from agent_rag.input_registry import text_hash


def module():
    assert importlib.util.find_spec("agent_rag.v2_runtime"), "V2运行配置与冻结评分加载尚未实现"
    return importlib.import_module("agent_rag.v2_runtime")


def files(tmp_path):
    packet = {"pair_id": "P1", "query_text": "赠送购物卡", "unit_text": "合同外购物卡"}
    packet.update(
        query_text_sha256=text_hash(packet["query_text"]),
        unit_text_sha256=text_hash(packet["unit_text"]),
    )
    score = {
        "pair_id": "P1",
        "score": 90,
        "level": "specific_direct",
        "source_quote": "合同外购物卡",
        "reason": "同机制",
        "limitation": "需核对当前合同",
        "human_approved": False,
        "query_text_sha256": packet["query_text_sha256"],
        "unit_text_sha256": packet["unit_text_sha256"],
        "origin": "actual_deepseek_flash_API",
        "returned_model": "deepseek-flash",
    }
    mapping = {"pair_id": "P1", "unit_id": "u1", "file_id": "F1"}
    paths = []
    for name, row in [("inputs", packet), ("scores", score), ("mapping", mapping)]:
        p = tmp_path / f"{name}.jsonl"
        p.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def test_frozen_flash_replay_preserves_scores_and_refuses_new_queries(tmp_path):
    reranker = module().FrozenFlashReranker(*files(tmp_path))
    unit = {"unit_id": "u1", "file_id": "F1", "text": "合同外购物卡"}
    result = reranker.score_query("赠送购物卡", [unit], None)
    assert result["complete"] is True
    assert result["scores"][0]["score"] == 90
    assert result["meta"]["rerank_mode"] == "replay"
    assert result["meta"]["external_retrieval_requests"] == 0
    missing = reranker.score_query("另一个查询", [unit], None)
    assert missing["complete"] is False
    assert missing["scores"] == []


def test_replay_text_hash_and_file_identity_cannot_be_rebound(tmp_path):
    reranker = module().FrozenFlashReranker(*files(tmp_path))
    result = reranker.score_query(
        "赠送购物卡", [{"unit_id": "u1", "file_id": "F2", "text": "合同外购物卡"}], None
    )
    assert not result["complete"] and result["scores"] == []
    result = reranker.score_query(
        "赠送购物卡", [{"unit_id": "u1", "file_id": "F1", "text": "合同内购物卡"}], None
    )
    assert not result["complete"] and result["scores"] == []


def test_replay_forged_quote_is_rejected_at_startup(tmp_path):
    paths = files(tmp_path)
    score = json.loads(paths[1].read_text(encoding="utf-8"))
    score["source_quote"] = "不存在的原文"
    paths[1].write_text(json.dumps(score, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        module().FrozenFlashReranker(*paths)


@pytest.mark.parametrize('override', [None, 300])
def test_snapshot_runtime_passes_only_explicit_startup_override(tmp_path, monkeypatch, override):
    from types import SimpleNamespace
    m = module()
    rules, dense_path = tmp_path/'rules', tmp_path/'dense.json'
    rules.write_text('rules', encoding='utf-8')
    dense_path.write_text('{}', encoding='utf-8')
    assets = {name:{'path':str(p),'sha256':m.file_hash(p)}
        for name, p in [('rules',rules),('dense_config',dense_path)]}
    snapshot = SimpleNamespace(manifest={'backend':{'mode':'live','assets':assets}})
    calls = []
    monkeypatch.setenv('FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE', '1')
    monkeypatch.setattr(m, 'V2KnowledgeSnapshot', lambda root: snapshot)
    monkeypatch.setattr(m, 'DeepSeekReranker', lambda *a, **kw: object())
    monkeypatch.setattr(m, 'HybridV2Backend', lambda *a, **kw: object())
    def dense(path, **kwargs):
        calls.append((path, kwargs))
        return object()
    monkeypatch.setattr(m, 'ResidentDenseIndex', dense)
    m.load_v2_snapshot(tmp_path, startup_timeout_seconds=override)
    assert calls == [(dense_path, {} if override is None else {'startup_timeout_seconds':300})]


@pytest.mark.parametrize('value', [True, 0, 601, float('nan'), '300'])
def test_snapshot_runtime_rejects_invalid_wait_before_snapshot_or_key_access(
    tmp_path, monkeypatch, value
):
    m = module()
    monkeypatch.setattr(m, 'V2KnowledgeSnapshot', lambda *a: pytest.fail('snapshot accessed'))
    with pytest.raises(ValueError, match='INVALID_DENSE_STARTUP_TIMEOUT'):
        m.load_v2_snapshot(tmp_path, startup_timeout_seconds=value)
