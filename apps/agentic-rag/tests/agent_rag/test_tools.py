from pathlib import Path

import pytest

from agent_rag.input_registry import InputRegistry
from agent_rag.knowledge_snapshot import KnowledgeSnapshot, make_evidence
from agent_rag.tools import KnowledgeTools


@pytest.fixture
def tools(tmp_path):
    source_root = tmp_path / "official"
    source_root.mkdir()
    (source_root / "001.txt").write_text(
        "处罚决定\n销售人员以购买保险为条件赠送购物卡，给予合同外利益。\n罚款。", encoding="utf-8"
    )
    (source_root / "002.txt").write_text("处罚决定\n虚列费用套取资金。", encoding="utf-8")
    dictionary = (
        Path(__file__).resolve().parents[2] / "contracts/agent_rag/risk_type_dictionary.csv"
    )
    snapshot = KnowledgeSnapshot.build(source_root, dictionary, tmp_path / "snapshot")
    return KnowledgeTools(snapshot, InputRegistry("购买这份保险就送购物卡。", task_id="t1"))


def search_payload(text="购买保险赠送购物卡"):
    return {
        "queries": [
            {
                "text": text,
                "focus_refs": [
                    {"segment_id": "S0001", "quote": "购买这份保险就送购物卡。", "occurrence": 1}
                ],
                "risk_point_ids": [],
                "goal": "mechanism_search",
            }
        ],
        "top_k": 5,
        "filters": {},
    }


def test_read_preview_uses_actual_window_without_registering_or_spending_queries(tools):
    searched = tools.call("search_cases", search_payload())
    anchor = searched["data"]["queries"][0]["candidates"][0]["evidence"][0]
    params = {"source_id": anchor["source_id"], "anchor_evidence_id": anchor["evidence_id"], "max_chars": 500}
    before_evidence = dict(tools.evidence)
    before_cache = dict(tools.snapshot.cache)
    preview = getattr(tools, "preview_read", None)
    assert callable(preview), "source read needs a side-effect-free local preview"
    projected = preview(params)
    assert tools.evidence == before_evidence
    assert tools.snapshot.cache == before_cache
    assert projected["meta"]["backend_query_attempts"] == 0
    actual = tools.call("read_source", params)
    assert projected["data"] == actual["data"]
    assert projected["data"]["evidence"]["quote"] == actual["data"]["evidence"]["quote"]


def test_read_preview_keeps_source_anchor_and_parameter_boundaries(tools):
    preview = getattr(tools, "preview_read", None)
    assert callable(preview), "source read needs a side-effect-free local preview"
    for params, code in [
        ({"source_id": "INPUT:other", "max_chars": 500}, "SOURCE_NOT_ALLOWED"),
        ({"source_id": "INPUT:t1", "anchor_evidence_id": "missing", "max_chars": 500}, "INVALID_EVIDENCE_REFERENCE"),
        ({"source_id": "INPUT:t1", "max_chars": 999999}, "INVALID_SCHEMA"),
    ]:
        result = preview(params)
        assert result["status"] == "error"
        assert result["error"]["code"] == code


def test_real_retrieval_source_and_cache(tools):
    first = tools.call("search_cases", search_payload())
    assert first["status"] == "ok"
    candidate = first["data"]["queries"][0]["candidates"][0]
    evidence = candidate["evidence"][0]
    read = tools.call(
        "read_source",
        {
            "source_id": candidate["source_id"],
            "anchor_evidence_id": evidence["evidence_id"],
            "max_chars": 500,
        },
    )
    assert evidence["quote"] in read["data"]["evidence"]["quote"]
    assert evidence["text_sha256"] == read["data"]["evidence"]["text_sha256"]
    assert first["meta"]["backend_query_attempts"] == 1
    second = tools.call("search_cases", search_payload())
    assert second["meta"]["backend_query_attempts"] == 0
    assert second["meta"]["cache_hits"] == 1


def test_retrieval_window_preserves_nearby_subject_and_document_context(tools):
    result = tools.call("search_cases", search_payload())
    quote = result["data"]["queries"][0]["candidates"][0]["evidence"][0]["quote"]
    assert "处罚决定" in quote
    assert "罚款" in quote


def test_empty_not_tool_failure(tools):
    value = tools.call("search_cases", search_payload("zzzxxyyqq"))
    assert value["status"] == "empty"
    assert value["error"] is None
    assert value["data"]["queries"][0]["complete"] is True


def test_tampered_source_is_error(tools):
    row = next(iter(tools.snapshot.sources.values()))
    Path(row["path"]).write_text("已经改字", encoding="utf-8")
    value = tools.call("read_source", {"source_id": row["source_id"], "max_chars": 500})
    assert value["status"] == "error"
    assert value["error"]["code"] == "SOURCE_HASH_MISMATCH"


def test_dictionary_strict_and_cross_task_input_blocked(tools):
    good = tools.call("get_risk_definition", {"tags": ["赠送利益", "合同外利益"]})
    assert len(good["data"]["definitions"]) == 2
    bad = tools.call("get_risk_definition", {"tags": ["赠送礼物"]})
    assert bad["status"] == "error"
    assert bad["error"]["code"] == "INVALID_RISK_TAG"
    bad_source = tools.call("read_source", {"source_id": "INPUT:t2", "max_chars": 500})
    assert bad_source["error"]["code"] == "SOURCE_NOT_ALLOWED"


def test_batch_partial_does_not_repeat_success(tools, monkeypatch):
    original = tools.snapshot.search

    def search(text, *args):
        if text == "虚列费用":
            raise RuntimeError("temporary failure")
        return original(text, *args)

    monkeypatch.setattr(tools.snapshot, "search", search)
    params = search_payload()
    params["queries"].append({**params["queries"][0], "text": "虚列费用"})
    response = tools.call("search_cases", params)
    assert response["status"] == "partial"
    assert [q["status"] for q in response["data"]["queries"]] == ["ok", "error"]
    assert response["meta"]["backend_query_attempts"] == 2


def test_bad_focus_and_undeclared_profile_rejected(tools):
    params = search_payload()
    params["queries"][0]["focus_refs"][0]["quote"] = "历史案件中有这个"
    assert tools.call("search_cases", params)["error"]["code"] == "INVALID_INPUT_REFERENCE"
    with pytest.raises(ValueError, match="PROFILE_NOT_READY"):
        KnowledgeTools(tools.snapshot, tools.registry, profile="hybrid-bge-rrf-rerank-v1")


def test_long_source_window_is_exact_and_paginated(tools):
    response = tools.call("read_source", {"source_id": tools.registry.source_id, "max_chars": 500})
    evidence = response["data"]["evidence"]
    assert tools.registry.text[evidence["start"] : evidence["end"]] == evidence["quote"]
    assert response["data"]["has_more"] is False


def build_tools_for_texts(tmp_path, texts):
    source_root = tmp_path / "official"
    source_root.mkdir()
    for filename, text in texts.items():
        (source_root / filename).write_bytes(text.encode("utf-8"))
    dictionary = (
        Path(__file__).resolve().parents[2] / "contracts/agent_rag/risk_type_dictionary.csv"
    )
    snapshot = KnowledgeSnapshot.build(source_root, dictionary, tmp_path / "snapshot")
    return KnowledgeTools(snapshot, InputRegistry("购买这份保险就送购物卡。", task_id="t1"))


@pytest.mark.parametrize(
    ("before", "quote", "after", "max_chars"),
    [
        ("长段背景😀𠮷" * 200, "命中事实😀𠮷。" * 10, "后续背景" * 300, 1000),
        ("背景\n\n" * 100, "连续命中\n\n" * 140, "后续\n\n" * 100, 1000),
        ("长段背景😀𠮷" * 200, "末端事实😀𠮷。" * 10, "\n\n", 500),
    ],
    ids=["long-paragraph-unicode", "repeated-newlines", "source-end"],
)
def test_anchor_window_covers_full_anchor_with_exact_unicode_offsets(
    tmp_path, before, quote, after, max_chars
):
    text = before + quote + after
    tools = build_tools_for_texts(tmp_path, {"001.txt": text})
    source = next(iter(tools.snapshot.sources.values()))
    anchor = make_evidence(source, len(before), len(before) + len(quote), text)
    tools.evidence[anchor["evidence_id"]] = anchor

    response = tools.call(
        "read_source",
        {
            "source_id": source["source_id"],
            "anchor_evidence_id": anchor["evidence_id"],
            "max_chars": max_chars,
        },
    )

    assert response["status"] == "ok"
    data = response["data"]
    evidence = data["evidence"]
    assert evidence["start"] <= anchor["start"] < anchor["end"] <= evidence["end"]
    assert quote in evidence["quote"]
    assert text[evidence["start"] : evidence["end"]] == evidence["quote"]
    assert len(evidence["quote"]) <= max_chars
    assert data["anchor_not_fully_covered"] is False


def test_oversized_anchor_is_explicitly_partial_and_can_continue(tmp_path):
    before, quote, after = "背景" * 400, "命中😀𠮷" * 200, "尾文。"
    text = before + quote + after
    tools = build_tools_for_texts(tmp_path, {"001.txt": text})
    source = next(iter(tools.snapshot.sources.values()))
    anchor = make_evidence(source, len(before), len(before) + len(quote), text)
    tools.evidence[anchor["evidence_id"]] = anchor

    response = tools.call(
        "read_source",
        {
            "source_id": source["source_id"],
            "anchor_evidence_id": anchor["evidence_id"],
            "max_chars": 500,
        },
    )

    assert response["status"] == "ok"
    data = response["data"]
    assert data["evidence"]["start"] == anchor["start"]
    assert data["evidence"]["quote"] == quote[:500]
    assert data["anchor_not_fully_covered"] is True
    assert data["has_more"] is True
    follow = tools.call(
        "read_source",
        {"source_id": source["source_id"], "cursor": data["next_cursor"], "max_chars": 500},
    )
    assert follow["data"]["evidence"]["start"] == data["evidence"]["end"]
    assert follow["data"]["evidence"]["quote"] == quote[500:] + after


def test_anchor_window_never_exceeds_internal_4800_character_cap(tmp_path):
    text = "前文" * 3000 + "命中😀𠮷" * 1300 + "尾文。"
    tools = build_tools_for_texts(tmp_path, {"001.txt": text})
    source = next(iter(tools.snapshot.sources.values()))
    anchor = make_evidence(source, 6000, len(text) - 3, text)
    tools.evidence[anchor["evidence_id"]] = anchor
    response = tools.call(
        "read_source",
        {
            "source_id": source["source_id"],
            "anchor_evidence_id": anchor["evidence_id"],
            "max_chars": 6000,
        },
    )
    assert len(response["data"]["evidence"]["quote"]) == 4800
    assert response["data"]["anchor_not_fully_covered"] is True


def test_exact_fact_duplicate_hint_preserves_all_sources_and_punished_parties(tmp_path):
    facts = "销售人员以购买保险为条件赠送购物卡，给予合同外利益，并通过业务宣传费报销该项支出。" * 3
    texts = {
        "机构.txt": (
            f"被处罚当事人：甲保险公司\n主要违法违规事实\n{facts}\n"
            "行政处罚依据\n保险法\n行政处罚决定\n罚款。"
        ),
        "个人.txt": (
            f"被处罚当事人：李某\n主要违法违规事实\n{facts}\n"
            "行政处罚依据\n保险法\n行政处罚决定\n警告。"
        ),
    }
    tools = build_tools_for_texts(tmp_path, texts)
    response = tools.call("search_cases", search_payload())
    candidates = response["data"]["queries"][0]["candidates"]

    assert len(candidates) == 2
    assert len({candidate["source_id"] for candidate in candidates}) == 2
    assert any("甲保险公司" in candidate["evidence"][0]["quote"] for candidate in candidates)
    assert any("李某" in candidate["evidence"][0]["quote"] for candidate in candidates)
    hints = [candidate["duplicate_hint"] for candidate in candidates]
    assert hints[0]["fact_text_sha256"] == hints[1]["fact_text_sha256"]
    for candidate, hint in zip(candidates, hints):
        _, text = tools.snapshot.read(candidate["source_id"])
        assert text[hint["fact_start"] : hint["fact_end"]] == facts
        assert hint["kind"] == "exact_fact_text"
        assert hint["fact_chars"] == len(facts)
        assert hint["event_review_status"] == "not_assessed"
        assert "仅文本重复线索" in hint["caution"]
        assert "不得据此合并文书或主体" in hint["caution"]
        assert candidate["case_id"] is None
        assert candidate["behavior_id"] is None


def test_similar_facts_and_changed_whitespace_do_not_share_duplicate_hash(tmp_path):
    facts = "销售人员以购买保险为条件赠送购物卡，给予合同外利益。" * 4
    texts = {
        "原文.txt": f"违法事实：{facts}\n处罚决定：罚款。",
        "相似.txt": f"违法事实：{facts.replace('购物卡', '礼品卡')}\n处罚决定：罚款。",
        "换行不同.txt": f"违法事实：{facts[:50]}\n{facts[50:]}\n处罚决定：罚款。",
    }
    tools = build_tools_for_texts(tmp_path, texts)
    response = tools.call("search_cases", search_payload())
    candidates = response["data"]["queries"][0]["candidates"]

    assert len(candidates) == 3
    assert len({candidate["duplicate_hint"]["fact_text_sha256"] for candidate in candidates}) == 3


def test_fact_duplicate_hint_preserves_punctuation_before_inline_next_field(tmp_path):
    facts = "销售人员以购买保险为条件赠送购物卡，给予合同外利益。" * 4
    tools = build_tools_for_texts(
        tmp_path,
        {
            "换行字段.txt": f"违法事实：{facts}\n处罚决定：罚款。",
            "同行字段.txt": f"当事人：甲保险公司。违法事实：{facts}处罚决定：罚款。",
        },
    )
    candidates = tools.call("search_cases", search_payload())["data"]["queries"][0]["candidates"]

    assert len(candidates) == 2
    for candidate in candidates:
        hint = candidate["duplicate_hint"]
        _, text = tools.snapshot.read(candidate["source_id"])
        assert text[hint["fact_start"] : hint["fact_end"]] == facts
    assert candidates[0]["duplicate_hint"]["fact_text_sha256"] == candidates[1]["duplicate_hint"][
        "fact_text_sha256"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "违法事实：赠送购物卡。\n处罚决定：罚款。",
        "无事实标题，销售人员赠送购物卡。" * 10,
        "主要违法违规事实\n" + "销售人员赠送购物卡。" * 10,
        "违法事实：" + "销售人员赠送购物卡。" * 10 + "\n违法事实：另有违法行为。\n处罚决定：罚款。",
        "文中提及违法事实：" + "销售人员赠送购物卡。" * 10 + "\n处罚决定：罚款。",
    ],
    ids=["short-facts", "no-heading", "no-end-field", "multiple-headings", "incidental-heading"],
)
def test_uncertain_or_short_fact_sections_have_no_duplicate_hint(tmp_path, text):
    tools = build_tools_for_texts(tmp_path, {"001.txt": text})
    response = tools.call("search_cases", search_payload())
    candidates = response["data"]["queries"][0]["candidates"]
    assert candidates
    assert all(candidate["duplicate_hint"] is None for candidate in candidates)
