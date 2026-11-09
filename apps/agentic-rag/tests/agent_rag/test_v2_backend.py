import importlib
import importlib.util

import pytest


def backend_module():
    assert importlib.util.find_spec("agent_rag.v2_backend"), "V2混合检索后端尚未实现"
    return importlib.import_module("agent_rag.v2_backend")


class Snapshot:
    manifest = {"snapshot_id": "v2-test", "index_sha256": "index"}
    units = [
        {"unit_id": "u1", "file_id": "F1", "case_id": "C1", "text": "保险赠送购物卡"},
        {"unit_id": "u2", "file_id": "F1", "case_id": "C1", "text": "合同外购物卡"},
        {"unit_id": "u3", "file_id": "F2", "case_id": "C2", "text": "回访不真实"},
    ]

    def candidate(self, unit, rank, score):
        return {
            "unit_id": unit["unit_id"],
            "file_id": unit["file_id"],
            "rank": rank,
            "score": score,
        }


class Dense:
    def search(self, text, top_k=100):
        return {
            "hits": [
                {"unit_id": uid, "score": score, "rank": rank}
                for rank, (uid, score) in enumerate([("u3", 0.9), ("u2", 0.8), ("u1", 0.7)], 1)
            ],
            "meta": {"mode": "test"},
        }


class Scorer:
    def __init__(self):
        self.calls = 0

    def score_query(self, query, units, control):
        self.calls += 1
        return {
            "scores": [
                {
                    "unit_id": u["unit_id"],
                    "file_id": u["file_id"],
                    "score": {"u1": 90, "u2": 95, "u3": 95}[u["unit_id"]],
                    "source_quote": u["text"],
                }
                for u in units
            ],
            "complete": True,
            "meta": {
                "external_retrieval_requests": len(units),
                "cache_hits": 0,
                "rerank_mode": "test",
            },
        }


def test_rrf_duplicate_route_ids_do_not_get_extra_votes():
    rrf = backend_module().rrf
    ranked = rrf(
        [
            [{"unit_id": "b"}, {"unit_id": "a"}, {"unit_id": "a"}],
            [{"unit_id": "a"}, {"unit_id": "b"}],
        ]
    )
    assert [r["unit_id"] for r in ranked] == ["a", "b"]
    assert ranked[0]["score"] == pytest.approx(1 / 61 + 1 / 62)


def test_file_maximum_aggregation_and_cache_do_not_repeat_scoring(tmp_path):
    scorer = Scorer()
    backend = backend_module().HybridV2Backend(Snapshot(), Dense(), scorer, audit_root=tmp_path)
    result = backend.search("购物卡", 5)
    assert result["complete"] is True
    assert [c["file_id"] for c in result["candidates"]] == ["F1", "F2"]
    assert result["candidates"][0]["unit_id"] == "u2"
    assert len(result["audit"]["stages"]["RRF"]) == 3
    cached = backend.search("购物卡", 5)
    assert scorer.calls == 1
    assert cached["meta"]["external_retrieval_requests"] == 0
    assert cached["meta"]["cache_hits"] >= 1
    assert list(tmp_path.glob("*.json"))


def test_partial_scoring_remains_partial_without_zero_fill():
    class Partial(Scorer):
        def score_query(self, query, units, control):
            result = super().score_query(query, units, control)
            result["scores"] = result["scores"][:1]
            result["complete"] = False
            return result

    backend = backend_module().HybridV2Backend(Snapshot(), Dense(), Partial())
    result = backend.search("购物卡", 5)
    assert result["complete"] is False
    assert len(result["audit"]["stages"]["ND"]) == 1
    assert not backend.is_cached("购物卡", 5)


def test_partial_ranking_carries_counts_and_only_scored_subset_topk():
    class Partial(Scorer):
        def score_query(self, query, units, control):
            result = super().score_query(query, units, control)
            result["scores"] = result["scores"][:1]
            result["complete"] = False
            result["errors"] = [{"unit_id": u["unit_id"], "code": "QUOTE_MISMATCH",
                                 "attempts_used": 2, "remaining_attempts": 0, "terminal": True}
                                for u in units[1:]]
            return result

    value = backend_module().HybridV2Backend(Snapshot(), Dense(), Partial()).search("购物卡", 3)
    coverage = value["ranking_coverage"]
    assert (coverage["expected_units"], coverage["scored_units"], coverage["failed_units"]) == (3, 1, 2)
    assert coverage["ranking_complete"] is False
    assert coverage["topk_scope"] == "scored_subset"
    assert coverage["attempts_exhausted"] is True
    assert coverage["failure_counts"] == {"QUOTE_MISMATCH": 2}
    assert set(value["audit"]["scored_unit_ids"]).isdisjoint(value["audit"]["failed_unit_ids"])


def test_error_rows_cannot_name_successful_or_foreign_units():
    class Forged(Scorer):
        def score_query(self, query, units, control):
            result = super().score_query(query, units, control)
            result["errors"] = [{"unit_id": "u1", "code": "QUOTE_MISMATCH"}]
            return result

    with pytest.raises(ValueError, match="RERANK_BINDING_MISMATCH"):
        backend_module().HybridV2Backend(Snapshot(), Dense(), Forged()).search("购物卡", 3)


def test_production_scorer_receives_stable_snapshot_binding():
    class Bound(Scorer):
        supports_query_binding = True

        def score_query(self, query, units, control, *, query_binding=None):
            self.binding = query_binding
            return super().score_query(query, units, control)

    scorer = Bound()
    backend_module().HybridV2Backend(Snapshot(), Dense(), scorer).search("购物卡", 3)
    assert scorer.binding == {"snapshot_id": "v2-test", "index_sha256": "index", "retrieval_profile": "v2-hybrid-flash-v1"}


def test_query_cache_uses_the_current_tasks_cost_ledger(tmp_path):
    from agent_rag.rerank_backend import RerankTaskControl

    class AccountingScorer(Scorer):
        def score_query(self, query, units, control):
            result = super().score_query(query, units, control)
            result["meta"].update(control.meta())
            return result

    scorer = AccountingScorer()
    backend = backend_module().HybridV2Backend(Snapshot(), Dense(), scorer)
    first = RerankTaskControl(tmp_path / "task-a.json", 1, 5)
    second = RerankTaskControl(tmp_path / "task-b.json", 0.5, 2)
    backend.search("购物卡", 5, first)
    cached = backend.search("购物卡", 5, second)
    assert scorer.calls == 1
    assert cached["meta"]["external_retrieval_requests"] == 0
    assert cached["meta"]["ledger_ref"] == str(second.ledger_path)
    assert cached["meta"]["cap_micro_cny"] == 500000
    assert cached["meta"]["max_requests"] == 2
    assert cached["meta"]["total_external_retrieval_requests"] == 0


def test_dense_unknown_units_and_forged_score_bindings_rejected():
    class BadDense(Dense):
        def search(self, text, top_k=100):
            return {"hits": [{"unit_id": "unknown", "score": 1, "rank": 1}], "meta": {}}

    with pytest.raises(ValueError, match="DENSE_UNIT_MISMATCH"):
        backend_module().HybridV2Backend(Snapshot(), BadDense(), Scorer()).search("购物卡", 5)

    class BadScore(Scorer):
        def score_query(self, query, units, control):
            result = super().score_query(query, units, control)
            result["scores"][0]["file_id"] = "other"
            return result

    with pytest.raises(ValueError, match="RERANK_BINDING_MISMATCH"):
        backend_module().HybridV2Backend(Snapshot(), Dense(), BadScore()).search("购物卡", 5)
