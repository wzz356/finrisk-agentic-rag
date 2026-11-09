"""Integrity, lifecycle and application fallback tests for listwise Pi reranking."""

import json
from types import SimpleNamespace

import pytest
from test_cloud_retrieval import FakeEmbedding, build_service

from llm.types import JsonChatResponse
from retrieval.cloud import CloudRetrievalError
from retrieval.deepseek import DeepSeekRerankClient, validate_ranking


@pytest.mark.parametrize("ranking", [[0, 0], [0], [0, 2], [False, 1], [0.0, 1], "0,1"])
def test_rejects_omitted_invented_or_duplicate_candidates(ranking):
    with pytest.raises(CloudRetrievalError):
        validate_ranking(ranking, 2)


def test_pi_worker_is_reused_and_model_sees_only_query_and_original_candidates():
    calls = []

    class FakeLLM:
        model_name = "fake-pi"

        async def complete_json(self, request):
            calls.append(request)
            return JsonChatResponse(data={"ranking": [2, 0, 1]}, model="fake-pi")

        async def aclose(self):
            calls.append("closed")

    client = DeepSeekRerankClient(client_factory=FakeLLM)
    try:
        first = client.rerank("query", ["原文A", "原文B", "原文C"], top_n=2)
        assert [i for i, _ in first] == [2, 0]
        assert [i for i, _ in client.rerank("query", ["原文A", "原文B", "原文C"])] == [2, 0, 1]
        payload = json.loads(calls[0].messages[1].content)
        assert payload == {
            "query": "query",
            "documents": [
                {"index": 0, "text": "原文A"},
                {"index": 1, "text": "原文B"},
                {"index": 2, "text": "原文C"},
            ],
        }
        assert calls[0].available_tools == []
        assert calls[0].task == "case_rerank"
        assert len(client.events) == 2
    finally:
        client.close()
        client.close()
    assert calls[-1] == "closed"
    assert not client._thread.is_alive()


def test_timeout_cancels_request_and_raises_sanitized_fallback_signal():
    class SlowLLM:
        model_name = "slow-pi"

        async def complete_json(self, request):
            raise TimeoutError("private provider content")

    client = DeepSeekRerankClient(client_factory=SlowLLM)
    try:
        with pytest.raises(CloudRetrievalError) as error:
            client.rerank("query", ["原文"])
        assert "private" not in str(error.value)
    finally:
        client.close()


def test_missing_model_output_falls_back_in_application():
    class MissingRerank:
        model_name = "bad-listwise"

        def rerank(self, query, documents, *, top_n=None):
            return validate_ranking([0, 0], len(documents))

    service = build_service(embedding=FakeEmbedding(), reranker=MissingRerank())
    service._rerank_provider = "deepseek"
    results = service.search("赠送礼品诱导投保", top_k=3)
    assert results
    assert service.last_diagnostics.rerank_channel == "weighted_local"
    assert service.last_diagnostics.warnings


def test_deepseek_permutation_is_preserved_without_interpreting_rank_as_probability():
    class ReverseRerank:
        model_name = "fake-listwise"
        selected_texts = []

        def rerank(self, query, documents, *, top_n=None):
            self.selected_texts = list(reversed(documents))[:top_n]
            return [(i, 0.5) for i in reversed(range(len(documents)))][:top_n]

    ranker = ReverseRerank()
    service = build_service(embedding=FakeEmbedding(), reranker=ranker)
    service._rerank_provider = "deepseek"
    results = service.search("保险销售保证收益并送礼品", top_k=3)
    assert results
    assert service.last_diagnostics.rerank_channel == "deepseek:fake-listwise"
    assert [r.matched_text for r in results] == ranker.selected_texts


def test_application_constructs_deepseek_provider_with_no_qwen_override():
    service = build_service(embedding=FakeEmbedding())
    service._rerank_provider = "deepseek"
    service.settings = SimpleNamespace(deepseek_rerank_model="deepseek-flash")
    client = service._rerank_client()
    assert isinstance(client, DeepSeekRerankClient)
    client.close()
