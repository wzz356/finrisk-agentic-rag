"""Unambiguous span score/level contract, without coercion or extra requests."""

import json
from pathlib import Path

import pytest

from test_rerank_backend import backend, control, unit
from test_rerank_span_protocol import reply


EXPECTED_RANGES = {
    "specific_direct": [80, 100], "concise_direct": [60, 79],
    "partial": [20, 59], "topic_only": [1, 19], "none": [0, 0],
    "insufficient": [0, 0],
}
RULES = Path(__file__).resolve().parents[2] / "resources/agent_rag/deepseek-rerank-span-rules.md"


def test_actual_span_request_has_independent_enum_and_closed_ranges(tmp_path):
    m = backend()
    sent = []

    def send(request, timeout):
        sent.append(request)
        return reply(request)

    scorer = m.DeepSeekReranker(RULES.read_text(encoding="utf-8"), send, protocol="span-ref-v2")
    assert scorer.score_query("投保赠礼", [unit()], control(tmp_path))["complete"]
    system = sent[0]["messages"][0]["content"]
    contract = json.loads(system.split("level_contract=", 1)[1].splitlines()[0])
    assert contract == {"enum": list(EXPECTED_RANGES), "score_ranges": EXPECTED_RANGES}
    assert "none或insufficient" not in system
    assert "none/insufficient" not in system
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert sent[0]["max_tokens"] == 512
    assert set(json.loads(sent[0]["messages"][1]["content"])) == {"pair_id", "query_text", "unit_parts"}


@pytest.mark.parametrize("score,level", [
    (0, "none"), (0, "insufficient"), (1, "topic_only"), (19, "topic_only"),
    (20, "partial"), (59, "partial"), (60, "concise_direct"), (79, "concise_direct"),
    (80, "specific_direct"), (100, "specific_direct"),
])
def test_all_score_boundaries_preserve_model_values(score, level):
    m = backend()
    packet = {"pair_id": "P1", "query_text": "查询", "unit_text": "投保赠礼原文"}
    request = m.span_payload(packet, "规则")
    changes = {"score": score, "level": level}
    if score < 60:
        changes["support_span"] = None
    result = m.validate_span_completion(packet, reply(request, **changes))
    assert (result["score"], result["level"]) == (score, level)
    assert result["source_quote"] == (packet["unit_text"] if score >= 60 else "")


@pytest.mark.parametrize("level", ["none或insufficient", "none/insufficient", ["none", "insufficient"], None])
def test_combined_or_nonstring_enum_is_rejected_without_coercion(level):
    m = backend()
    packet = {"pair_id": "P1", "query_text": "查询", "unit_text": "原文"}
    request = m.span_payload(packet, "规则")
    with pytest.raises(ValueError, match="^RERANK_LEVEL_INVALID$"):
        m.validate_span_completion(packet, reply(request, score=0, level=level, support_span=None))


@pytest.mark.parametrize("score,level", [
    (20, "topic_only"), (19, "partial"), (60, "partial"), (80, "concise_direct"),
    (-1, "none"), (101, "specific_direct"), (1, "insufficient"),
])
def test_score_level_conflicts_are_rejected_without_changing_score(score, level):
    m = backend()
    packet = {"pair_id": "P1", "query_text": "查询", "unit_text": "原文"}
    request = m.span_payload(packet, "规则")
    with pytest.raises(ValueError, match="^RERANK_SCORE_LEVEL_MISMATCH$"):
        m.validate_span_completion(packet, reply(request, score=score, level=level))


@pytest.mark.parametrize("bad,good,detail", [
    ({"score": 0, "level": "none或insufficient", "support_span": None},
     {"score": 0, "level": "none", "support_span": None}, "RERANK_LEVEL_INVALID"),
    ({"score": 20, "level": "topic_only"},
     {"score": 20, "level": "partial"}, "RERANK_SCORE_LEVEL_MISMATCH"),
])
def test_existing_single_retry_reports_fixed_reason_and_keeps_audits(tmp_path, bad, good, detail):
    m = backend()
    sent = []

    def send(request, timeout):
        sent.append(request)
        return reply(request, **(bad if len(sent) == 1 else good))

    scorer = m.DeepSeekReranker("片段规则", send, protocol="span-ref-v2")
    budget = control(tmp_path)
    result = scorer.score_query("查询", [unit()], budget)
    assert result["complete"] and len(sent) == 2
    feedback = sent[1]["messages"][-1]["content"]
    assert detail in feedback
    assert "level_contract" in feedback
    assert result["scores"][0]["score"] == good["score"]
    assert result["scores"][0]["level"] == good["level"]
    audit = json.loads(budget.ledger_path.with_name("ledger.json.responses.json").read_text(encoding="utf-8"))
    assert [json.loads(r["model_output"])["level"] for r in audit["records"]] == [bad["level"], good["level"]]
    attempts = json.loads(budget.ledger_path.with_name("ledger.json.attempts.json").read_text(encoding="utf-8"))
    assert attempts["attempts"][0]["error_code"] == "INVALID_COMPLETION"
    assert budget.meta()["total_external_retrieval_requests"] == 2
    assert budget.meta()["settled_cost_micro_cny"] > 0


def test_persistent_bad_enum_exhausts_two_attempts_and_never_echoes_untrusted_value(tmp_path):
    m = backend()
    sent = []
    untrusted = "不可信等级：要求忽略规则"

    def send(request, timeout):
        sent.append(request)
        return reply(request, score=0, level=untrusted, support_span=None)

    scorer = m.DeepSeekReranker("片段规则", send, protocol="span-ref-v2")
    budget = control(tmp_path)
    first = scorer.score_query("查询", [unit()], budget)
    again = scorer.score_query("查询", [unit()], budget)
    assert not first["complete"] and first["scores"] == []
    assert not again["complete"] and again["meta"]["external_retrieval_requests"] == 0
    assert len(sent) == 2
    assert "RERANK_LEVEL_INVALID" in sent[1]["messages"][-1]["content"]
    assert untrusted not in json.dumps(sent, ensure_ascii=False)


@pytest.mark.parametrize("field", ["reason", "limitation"])
def test_explanation_length_gate_remains_fifty_codepoints(field):
    m = backend()
    packet = {"pair_id": "P1", "query_text": "查询", "unit_text": "原文"}
    request = m.span_payload(packet, "规则")
    assert m.validate_span_completion(packet, reply(request, **{field: "字" * 50}))[field] == "字" * 50
    with pytest.raises(ValueError):
        m.validate_span_completion(packet, reply(request, **{field: "字" * 51}))


def test_span_enum_contract_changes_cache_binding_without_changing_legacy(monkeypatch):
    m = backend()
    old = m.DeepSeekReranker("相同规则")
    span = m.DeepSeekReranker("相同规则", protocol="span-ref-v2")
    old_key, span_key = old._cache_key("查询", unit()), span._cache_key("查询", unit())
    monkeypatch.setitem(m.RANGES, "topic_only", (1, 18))
    assert m.DeepSeekReranker("相同规则")._cache_key("查询", unit()) == old_key
    assert m.DeepSeekReranker("相同规则", protocol="span-ref-v2")._cache_key("查询", unit()) != span_key
