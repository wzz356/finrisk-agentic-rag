import importlib
import importlib.util

import pytest


def corpus(tmp_path):
    assert importlib.util.find_spec("agent_rag.maintenance_tools"), "维护证据工具尚未实现"
    return importlib.import_module("agent_rag.maintenance_tools").MaintenanceCorpus(tmp_path)


def test_exact_unicode_evidence_is_immutable_and_bound(tmp_path):
    c = corpus(tmp_path)
    raw = "保险🙂。\r\n未承诺保本。".encode()
    item = c.register_evidence(raw, origin_id="fixture:1")
    assert c.resolve_evidence(item["evidence_id"])["raw_bytes"] == raw
    tools = c.tools("reviewer", "HEAD1", {"scope_id": "public", "classification": "public"})
    result = tools.call(
        "read_evidence",
        {
            "evidence_id": item["evidence_id"],
            "text_sha256": item["text_sha256"],
            "start": 2,
            "end": 3,
        },
    )
    assert result["data"]["quote"] == "🙂"
    with pytest.raises(ValueError, match="TEXT_VERSION"):
        tools.call(
            "read_evidence",
            {"evidence_id": item["evidence_id"], "text_sha256": "f" * 64, "start": 0, "end": 1},
        )
    with pytest.raises(ValueError, match="ARGUMENT"):
        tools.call("read_evidence", {"path": "C:/secret.txt"})


def test_scope_applies_to_search_read_and_independent_evidence_access(tmp_path):
    c = corpus(tmp_path)
    public = c.register_evidence("保险公开案例".encode(), origin_id="fixture:1")
    private = c.register_evidence(
        "保险客户秘密".encode(),
        origin_id="fixture:2",
        scope_id="team-a",
        classification="team_private",
    )
    tools = c.tools("reviewer", "HEAD1", {"scope_id": "public", "classification": "public"})
    result = tools.call("search_evidence", {"query": "保险", "limit": 20})
    assert [r["evidence_id"] for r in result["data"]["items"]] == [public["evidence_id"]]
    with pytest.raises(ValueError, match="NOT_AUTHORIZED"):
        tools.call(
            "read_evidence",
            {
                "evidence_id": private["evidence_id"],
                "text_sha256": private["text_sha256"],
                "start": 0,
                "end": 1,
            },
        )
    assert tools.search_trace_ids and len(tools.trace) == 2


def test_actual_read_ranges_and_roles_are_enforced(tmp_path):
    c = corpus(tmp_path)
    item = c.register_evidence(b"12345", origin_id="fixture:1")
    t = c.tools("proposer", "HEAD1", {"scope_id": "public", "classification": "public"})
    t.call(
        "read_evidence",
        {
            "evidence_id": item["evidence_id"],
            "text_sha256": item["text_sha256"],
            "start": 0,
            "end": 2,
        },
    )
    assert t.read_coverage()[0]["complete"] is False
    t.call(
        "read_evidence",
        {
            "evidence_id": item["evidence_id"],
            "text_sha256": item["text_sha256"],
            "start": 2,
            "end": 5,
        },
    )
    assert t.read_coverage()[0]["complete"] is True
    with pytest.raises(ValueError, match="ROLE_FORBIDDEN"):
        t.call("submit_review", {"decision": "approve"})
    reviewer = c.tools("reviewer", "HEAD1", {"scope_id": "public", "classification": "public"})
    with pytest.raises(ValueError, match="ROLE_FORBIDDEN"):
        reviewer.call("stage_changeset", {"operations": []})


def test_records_are_fixed_by_commit_and_lifecycle_filters_before_ranking(tmp_path):
    c = corpus(tmp_path)
    c.register_record("B1", "behavior", {"summary": "保险旧行为"}, commit="HEAD1")
    c.register_record("B1", "behavior", {"summary": "保险新行为"}, commit="HEAD2", record_version=2)
    c.register_record(
        "L1",
        "law_article",
        {
            "text": "保险旧规则",
            "effective_from": "2020-01-01",
            "effective_to": "2025-01-01",
            "legal_status": "verified",
        },
        commit="HEAD1",
    )
    c.register_record(
        "L2", "law_article", {"text": "保险效力未知", "legal_status": "unknown"}, commit="HEAD1"
    )
    t = c.tools("reviewer", "HEAD1", {"scope_id": "public", "classification": "public"})
    assert (
        t.call("read_record", {"record_id": "B1", "record_version": 1})["data"]["body"]["summary"]
        == "保险旧行为"
    )
    assert [
        r["record_id"]
        for r in c.search_records(
            "保险", commit="HEAD1", scope=t.scope, mode="current_compliance", as_of="2026-10-19"
        )["items"]
    ] == ["B1"]
    history = c.search_records(
        "保险旧规则",
        commit="HEAD1",
        scope=t.scope,
        mode="historical_compliance",
        as_of="2024-01-01",
    )
    assert history["items"][0]["record_id"] == "L1"


def test_search_is_paginated_and_does_not_claim_all_results_read(tmp_path):
    c = corpus(tmp_path)
    for i in range(3):
        c.register_evidence(("保险" + str(i)).encode(), origin_id="fixture:" + str(i))
    t = c.tools("reviewer", "HEAD1", {"scope_id": "public", "classification": "public"})
    first = t.call("search_evidence", {"query": "保险", "limit": 2})["data"]
    last = t.call("search_evidence", {"query": "保险", "limit": 2, "cursor": first["next_cursor"]})[
        "data"
    ]
    assert len(first["items"]) == 2 and len(last["items"]) == 1
    assert t.read_coverage() == []


def test_same_scope_lower_clearance_cannot_read_sensitive_material(tmp_path):
    c = corpus(tmp_path)
    evidence = c.register_evidence(
        "保险客户机密".encode(),
        origin_id="private:1",
        scope_id="team-a",
        classification="business_sensitive",
    )
    c.register_record(
        "PRIVATE1",
        "behavior",
        {"summary": "保险客户机密"},
        commit="HEAD1",
        scope_id="team-a",
        classification="business_sensitive",
    )
    t = c.tools("reviewer", "HEAD1", {"scope_id": "team-a", "classification": "team_private"})
    assert t.call("search_evidence", {"query": "保险"})["data"]["items"] == []
    assert t.call("search_knowledge", {"query": "保险"})["data"]["items"] == []
    with pytest.raises(ValueError, match="NOT_AUTHORIZED"):
        t.call("read_record", {"record_id": "PRIVATE1", "record_version": 1})
    with pytest.raises(ValueError, match="NOT_AUTHORIZED"):
        t.call(
            "read_evidence",
            {
                "evidence_id": evidence["evidence_id"],
                "text_sha256": evidence["text_sha256"],
                "start": 0,
                "end": 1,
            },
        )


def test_knowledge_search_returns_explicit_preview_then_record_can_be_read(tmp_path):
    c = corpus(tmp_path)
    c.register_record("B1", "behavior", {"summary": "保险" + ("业务内容" * 300)}, commit="HEAD1")
    t = c.tools("reviewer", "HEAD1", {"scope_id": "public", "classification": "public"})
    hit = t.call("search_knowledge", {"query": "保险"})["data"]["items"][0]
    assert "body" not in hit and hit["truncated"] is True and len(hit["preview"]) == 300
    assert hit["record_id"] == "B1" and hit["record_version"] == 1
    assert (
        len(
            t.call("read_record", {"record_id": "B1", "record_version": 1})["data"]["body"][
                "summary"
            ]
        )
        > 300
    )
