import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_rerank_backend import backend, control, unit


def reply(request, **changes):
    packet = json.loads(request["messages"][1]["content"])
    opinion = {"pair_id": packet["pair_id"], "score": 85, "level": "specific_direct",
               "support_span": {"start_ref": packet["unit_parts"][0]["ref_id"], "end_ref": packet["unit_parts"][-1]["ref_id"]},
               "reason": "机制直接对应", "limitation": "不能补当前合同事实", "human_approved": False}
    opinion.update(changes)
    return {"model": "deepseek-flash", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(opinion, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def test_span_protocol_reconstructs_quote_without_model_copy_and_audits_reply(tmp_path):
    m = backend(); requests = []

    def send(request, timeout):
        requests.append(request)
        return reply(request)

    scorer = m.DeepSeekReranker("新片段规则", send, protocol="span-ref-v2")
    budget = control(tmp_path)
    text = "甲" * 95 + "😀" + "投保赠购物卡。\n保留合同条件"
    result = scorer.score_query("投保赠礼", [unit(text=text)], budget)
    assert result["complete"] is True
    assert result["scores"][0]["source_quote"] == text
    assert result["scores"][0]["model_support_span"] == {"start_ref": "S0001", "end_ref": "S0003"}
    assert result["meta"]["rerank_protocol"] == "span-ref-v2"
    packet = json.loads(requests[0]["messages"][1]["content"])
    assert set(packet) == {"pair_id", "query_text", "unit_parts"}
    assert "".join(p["text"] for p in packet["unit_parts"]) == text
    assert requests[0]["max_tokens"] == 512
    audit = json.loads(budget.ledger_path.with_name("ledger.json.responses.json").read_text(encoding="utf-8"))
    assert len(audit["records"]) == 1
    assert json.loads(audit["records"][0]["model_output"])["support_span"] == result["scores"][0]["model_support_span"]
    assert audit["records"][0]["ordinal"] == 1


@pytest.mark.parametrize("changes", [
    {"support_span": None}, {"support_span": {"start_ref": "S9999", "end_ref": "S9999"}},
    {"pair_id": "other-unit"}, {"source_quote": "模型不应自行抄写"},
    {"score": True}, {"score": 40, "level": "specific_direct"}, {"human_approved": True},
])
def test_bad_span_opinions_never_get_default_scores_or_unlimited_retries(tmp_path, changes):
    m = backend(); sent = []

    def send(request, timeout):
        sent.append(request)
        return reply(request, **changes)

    scorer = m.DeepSeekReranker("新片段规则", send, protocol="span-ref-v2")
    budget = control(tmp_path)
    result = scorer.score_query("查询", [unit()], budget)
    again = scorer.score_query("查询", [unit()], budget)
    assert result["complete"] is False and result["scores"] == []
    assert len(sent) == 2
    assert again["meta"]["external_retrieval_requests"] == 0


def test_low_score_can_explicitly_have_no_quote(tmp_path):
    m = backend()
    scorer = m.DeepSeekReranker("新规则", lambda p, t: reply(p, score=0, level="none", support_span=None), protocol="span-ref-v2")
    result = scorer.score_query("无关查询", [unit()], control(tmp_path))
    assert result["complete"] is True
    assert result["scores"][0]["source_quote"] == ""


def test_response_audit_write_failure_never_repeats_a_paid_request(tmp_path, monkeypatch):
    m = backend(); sent = []

    def send(request, timeout):
        sent.append(request)
        return reply(request)

    scorer = m.DeepSeekReranker("新规则", send, protocol="span-ref-v2")
    monkeypatch.setattr(scorer, "_record_response", lambda *a: (_ for _ in ()).throw(OSError("offline injected")))
    budget = control(tmp_path)
    result = scorer.score_query("查询", [unit()], budget)
    assert len(sent) == 1 and not result["complete"]
    assert result["errors"][0]["code"] == "AUDIT_WRITE_FAILED"
    assert json.loads(budget.ledger_path.read_text(encoding="utf-8"))["settled_cost_micro_cny"] > 0


def test_protocol_changes_cache_namespace_and_unknown_protocol_never_loads_key():
    m = backend()
    first = m.DeepSeekReranker("相同规则", lambda *a: None)
    second = m.DeepSeekReranker("相同规则", lambda *a: None, protocol="span-ref-v2")
    assert first._cache_key("查询", unit()) != second._cache_key("查询", unit())
    with pytest.raises(ValueError, match="RERANK_PROTOCOL"):
        m.DeepSeekReranker("规则", key_loader=lambda: pytest.fail("cannot load key"), protocol="unknown")


def test_response_audit_failure_blocks_new_pairs_in_same_task(tmp_path, monkeypatch):
    m = backend(); sent = []
    def send(request, timeout):
        sent.append(request)
        return reply(request)
    scorer = m.DeepSeekReranker("新规则", send, protocol="span-ref-v2")
    monkeypatch.setattr(scorer, "_record_response", lambda *a: (_ for _ in ()).throw(OSError("offline disk failure")))
    budget = control(tmp_path)
    scorer.score_query("查询", [unit(1)], budget)
    scorer.score_query("查询", [unit(2)], budget)
    assert len(sent) == 1


@pytest.mark.parametrize("pause_at", ["before_claim", "before_finish"])
def test_concurrent_audit_failure_preserves_failure_code_for_owned_pair(
    tmp_path, monkeypatch, pause_at
):
    m = backend()
    sent = []
    paused = threading.Event()
    release = threading.Event()
    first_unit = unit(1, text="A正文")
    owned_unit = unit(2, text="B正文")

    def send(request, timeout):
        sent.append(request)
        return reply(request)

    scorer = m.DeepSeekReranker("新规则", send, protocol="span-ref-v2")
    budget = control(tmp_path)
    original_payload = scorer._payload
    original_record = scorer._record_response
    original_finish = m.PairCheckpoint.finish
    owned_key = scorer._cache_key("查询", owned_unit)

    def payload(packet, feedback=None):
        if pause_at == "before_claim" and packet["unit_text"] == owned_unit["text"]:
            paused.set()
            assert release.wait(5)
        return original_payload(packet, feedback)

    def record(control, packet, response):
        if packet["unit_text"] == first_unit["text"]:
            raise OSError("private-audit-failure")
        return original_record(control, packet, response)

    def finish(checkpoint, key, *args, **kwargs):
        if pause_at == "before_finish" and key == owned_key:
            paused.set()
            assert release.wait(5)
        return original_finish(checkpoint, key, *args, **kwargs)

    monkeypatch.setattr(scorer, "_payload", payload)
    monkeypatch.setattr(scorer, "_record_response", record)
    monkeypatch.setattr(m.PairCheckpoint, "finish", finish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owned = pool.submit(scorer.score_query, "查询", [owned_unit], budget)
        try:
            assert paused.wait(5)
            first = pool.submit(scorer.score_query, "查询", [first_unit], budget)
            first_result = first.result(timeout=5)
            assert first_result["errors"][0]["code"] == "AUDIT_WRITE_FAILED"
        finally:
            release.set()
        owned_result = owned.result(timeout=5)
    assert not owned_result["complete"]
    assert owned_result["errors"][0]["code"] == "AUDIT_WRITE_FAILED"
    assert owned_result["errors"][0]["terminal"] is True
    expected_owned_requests = 0 if pause_at == "before_claim" else 1
    assert owned_result["meta"]["external_retrieval_requests"] == expected_owned_requests
    assert len(sent) == 1 + expected_owned_requests
    assert budget.meta()["total_external_retrieval_requests"] == len(sent)
    assert "private-audit-failure" not in json.dumps(owned_result)


@pytest.mark.parametrize("pause_at", ["_commit_dispatch", "_start_transport"])
def test_audit_failure_blocks_bound_but_unsent_pair_at_final_transport_boundary(
    tmp_path, monkeypatch, pause_at
):
    m = backend()
    sent = []
    paused = threading.Event()
    release = threading.Event()
    first_unit = unit(1, text="A正文")
    bound_unit = unit(2, text="B正文")

    def send(request, timeout):
        sent.append(request)
        result = reply(request)
        result.pop("usage")
        return result

    scorer = m.DeepSeekReranker("新规则", send, protocol="span-ref-v2")
    budget = control(tmp_path)
    original_boundary = getattr(budget, pause_at)
    original_record = scorer._record_response

    def boundary(ordinal, *args, **kwargs):
        if ordinal == 1:
            paused.set()
            assert release.wait(5)
        return original_boundary(ordinal, *args, **kwargs)

    def record(control, packet, response):
        if packet["unit_text"] == first_unit["text"]:
            raise OSError("private-audit-failure")
        return original_record(control, packet, response)

    monkeypatch.setattr(budget, pause_at, boundary)
    monkeypatch.setattr(scorer, "_record_response", record)
    with ThreadPoolExecutor(max_workers=2) as pool:
        bound = pool.submit(scorer.score_query, "查询", [bound_unit], budget)
        try:
            assert paused.wait(5)
            first = pool.submit(scorer.score_query, "查询", [first_unit], budget)
            first_result = first.result(timeout=5)
            assert first_result["errors"][0]["code"] == "AUDIT_WRITE_FAILED"
            assert first_result["meta"]["external_retrieval_requests"] == 1
        finally:
            release.set()
        bound_result = bound.result(timeout=5)
    assert bound_result["errors"][0]["code"] == "AUDIT_WRITE_FAILED"
    assert bound_result["errors"][0]["audit_ordinals"] == [1]
    assert bound_result["meta"]["external_retrieval_requests"] == 0
    assert len(sent) == 1
    ledger = json.loads(budget.ledger_path.read_text(encoding="utf-8"))
    bound_row, first_row = ledger["request_reservations"]
    assert bound_row["dispatch_state"] == "aborted"
    assert "transport_started_at" not in bound_row
    assert first_row["dispatch_state"] == "transport_started"
    assert ledger["reserved_cost_micro_cny"] == first_row["reserved_micro_cny"]
    meta = budget.meta()
    assert meta["total_external_retrieval_requests"] == 1
    assert meta["aborted_unsent_requests"] == 1
    assert meta["finished_usage_unknown_requests"] == 1


def test_unknown_runtime_protocol_is_rejected_before_loading_assets_or_dense(tmp_path, monkeypatch):
    from agent_rag import v2_runtime

    class Snapshot:
        manifest = {"backend": {"mode": "live", "rerank_protocol": "unknown", "assets": {}}}

    monkeypatch.setattr(v2_runtime, "V2KnowledgeSnapshot", lambda path: Snapshot())
    monkeypatch.setattr(v2_runtime, "ResidentDenseIndex", lambda *a: pytest.fail("cannot load dense"))
    with pytest.raises(ValueError, match="RERANK_PROTOCOL"):
        v2_runtime.load_v2_snapshot(tmp_path)
