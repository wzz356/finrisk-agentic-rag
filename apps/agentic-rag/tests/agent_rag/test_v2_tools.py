from test_v2_snapshot import create_snapshot, fixture_assets

from agent_rag.input_registry import InputRegistry
from agent_rag.tools import KnowledgeTools


class Backend:
    def __init__(self, snapshot):
        self.snapshot, self.calls = snapshot, 0

    def is_cached(self, text, top_k):
        return False

    def search(self, text, top_k, control=None):
        self.calls += 1
        return {
            "candidates": [self.snapshot.candidate(self.snapshot.units[0], 1, 90)],
            "complete": True,
            "meta": {
                "external_retrieval_requests": 1,
                "backend_query_attempts": 1,
                "cache_hits": 0,
                "stage_status": {k: "completed" for k in ["bm25", "bge", "rrf", "rerank"]},
            },
        }


def setup(tmp_path):
    sources, units, dictionary, text, _ = fixture_assets(tmp_path)
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    snapshot.backend = Backend(snapshot)
    return KnowledgeTools(snapshot, InputRegistry("赠送购物卡。", task_id="v2-task")), text


def payload():
    return {
        "queries": [
            {
                "text": "赠送购物卡",
                "focus_refs": [{"segment_id": "S0001", "quote": "赠送购物卡。", "occurrence": 1}],
                "risk_point_ids": [],
                "goal": "mechanism_search",
            }
        ],
        "top_k": 5,
        "filters": {},
    }


def test_profile_contracts_are_isolated_and_v2_rejects_two_declared_queries():
    import pytest
    from agent_rag.contracts import tool_schema, validate_schema
    params = payload()
    params['queries'] *= 2
    validate_schema(params, tool_schema('search_cases', 'bm25-source-v1'))
    with pytest.raises(ValueError, match='INVALID_SCHEMA'):
        validate_schema(params, tool_schema('search_cases', 'v2-hybrid-flash-v1'))
    validate_schema(params, tool_schema('search_cases', 'bm25-source-v1'))


def test_v2_search_reads_disjoint_anchors_and_task_local_ids(tmp_path):
    tool, text = setup(tmp_path)
    result = tool.call("search_cases", payload())
    assert result["status"] == "ok", result
    assert result["schema_version"] == "finrisk-tools/1.1"
    candidate = result["data"]["queries"][0]["candidates"][0]
    for ev in candidate["evidence"]:
        read = tool.call(
            "read_source",
            {
                "source_id": candidate["source_id"],
                "anchor_evidence_id": ev["evidence_id"],
                "max_chars": 500,
            },
        )
        assert ev["quote"] in read["data"]["evidence"]["quote"]
    next_result = tool.call("search_cases", payload())
    assert result["data"]["queries"][0]["query_id"] != next_result["data"]["queries"][0]["query_id"]
    assert tool.preview_read(
        {
            "source_id": candidate["source_id"],
            "anchor_evidence_id": candidate["evidence"][0]["evidence_id"],
            "max_chars": 500,
        }
    )["data"]["evidence"]["quote"]


def test_v2_one_query_limit_and_backends_not_called_after_limit(tmp_path):
    tool, _ = setup(tmp_path)
    query = payload()
    query["queries"] *= 2
    result = tool.call("search_cases", query)
    assert result["status"] == "error"
    assert tool.snapshot.backend.calls == 0
    result = tool.call("search_cases", payload(), query_limit=0)
    assert result["status"] == "partial"
    assert tool.snapshot.backend.calls == 0


def test_v2_relation_metadata_has_budget_before_whole_candidate_projection(tmp_path):
    tool, _ = setup(tmp_path)
    backend = tool.snapshot.backend
    original = backend.search

    def search(*args):
        result = original(*args)
        # 多主体、多行为绑定会放大元数据；引文和必要锚点不能因此全部消失。
        association = result['candidates'][0]['associations'][0]
        result['candidates'][0]['associations'] = [dict(association) for _ in range(16)]
        return result

    backend.search = search
    result = tool.call('search_cases', payload())
    assert result['data']['queries'][0]['candidates'], result
    assert result['status'] == 'ok'


def test_private_search_audit_failure_never_keeps_completed_status(tmp_path):
    tool, _ = setup(tmp_path)
    audit_file = tmp_path / 'audit-is-file'
    audit_file.write_text('block mkdir', encoding='utf-8')
    tool.audit_dir = audit_file
    response = tool.call('search_cases', payload())
    row = response['data']['queries'][0]
    assert response['status'] in ('partial', 'error')
    assert row['complete'] is False
    assert row['status'] == 'error'
    assert row['error']['code'] == 'RETRIEVAL_AUDIT_WRITE_FAILED'
    assert row['stage_status']['rerank'] == 'completed'
    assert response['meta']['external_retrieval_requests'] == 1


def test_stopped_control_before_retrieval_is_an_execution_error(tmp_path):
    from agent_rag.v2_backend import HybridV2Backend
    tool, _ = setup(tmp_path)
    class Stopped:
        def check(self):
            return False
        def meta(self):
            return {'stop_reason': 'request_limit'}
    tool.control = Stopped()
    # The production early-stop branch must not need dense/index/scorer at all.
    tool.snapshot.backend = HybridV2Backend.__new__(HybridV2Backend)
    response = tool.call('search_cases', payload())
    row = response['data']['queries'][0]
    assert row['status'] == 'error'
    assert row['complete'] is False
    assert row['error']['code'] == 'STOPPED'
    assert 'ranking_coverage' not in row
    assert response['meta']['external_retrieval_requests'] == 0
    assert response['meta']['stage_status']['rerank'] == 'not_started'
