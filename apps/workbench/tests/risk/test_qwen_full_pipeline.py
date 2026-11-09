"""Regression tests for full indexing, fused reranking and independent risk mechanisms."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from application.retrieval_service import CaseRetrievalService
from domain.common import QueryInputType
from domain.knowledge_models import CaseSummary
from llm import JsonChatResponse
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query
from retrieval.cloud import CloudRetrievalError, CloudSettings
from retrieval.query_planner import RetrievalQueryPlanner
from retrieval.qwen import NativeQwenEmbeddingClient
from retrieval.qwen_dense import QwenDenseIndex
from retrieval.sparse import RetrievalDocument
from risk.detection import RuleRiskDetector, is_explicit_compliance_statement
from risk.pipeline import RiskAssessmentPipeline
from settings import AppSettings
from storage.case_repository import CaseRepository

ROOT = Path(__file__).resolve().parents[2]


def test_native_qwen_separates_query_document_types_and_caches_only_queries(monkeypatch):
    calls = []
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-never-sent-to-network")

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "request_id": "local-mock",
                "output": {
                    "embeddings": [
                        {"text_index": i, "embedding": [1.0, 0.0]}
                        for i, _ in enumerate(payload["input"]["texts"])
                    ]
                },
            },
        )

    client = NativeQwenEmbeddingClient(CloudSettings(embedding_dimension=2))
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        client.embed_documents(["原文一", "原文二"])
        client.embed_queries(["查询一", "查询二"])
        client.embed_query("查询一")
        assert len(calls) == 2
        assert calls[0]["parameters"]["text_type"] == "document"
        assert "instruct" not in calls[0]["parameters"]
        assert calls[1]["parameters"]["text_type"] == "query"
        assert calls[1]["parameters"]["instruct"]
        assert calls[0]["model"] == calls[1]["model"]
        assert "test-key" not in json.dumps(client.events)
    finally:
        client.close()


class TinyEmbedding:
    dimension = 2
    model_name = "local-index-test"
    batch_size = 1
    binding = {"test_model": "same-document-query-space"}
    events = []

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        if self.fail and self.calls >= 2:
            raise CloudRetrievalError("deterministic test interruption")
        return [[1.0, len(text)] for text in texts]

    def embed_query(self, text):
        return [1.0, len(text)]


def test_qwen_checkpoint_resumes_and_serving_never_builds_documents(tmp_path):
    docs = [
        RetrievalDocument("u1", ("c",), "文本甲", "case_chunk"),
        RetrievalDocument("u2", ("c",), "文本乙更长", "case_chunk"),
    ]
    with pytest.raises(CloudRetrievalError, match="未就绪"):
        QwenDenseIndex(docs, embedding=TinyEmbedding(), directory=tmp_path).build()
    first = TinyEmbedding(fail=True)
    with pytest.raises(CloudRetrievalError, match="interruption"):
        QwenDenseIndex(
            docs, embedding=first, directory=tmp_path, allow_build=True, concurrency=1
        ).build()
    resumed = TinyEmbedding()
    index = QwenDenseIndex(
        docs, embedding=resumed, directory=tmp_path, allow_build=True, concurrency=1
    )
    index.build()
    assert resumed.calls == 1
    serving = TinyEmbedding()
    loaded = QwenDenseIndex(docs, embedding=serving, directory=tmp_path)
    loaded.build()
    assert loaded.search("文本甲", top_k=1)
    assert serving.calls == 0
    altered = [*docs[:-1], RetrievalDocument("u2", ("c",), "内容变更", "case_chunk")]
    with pytest.raises(CloudRetrievalError, match="未就绪"):
        QwenDenseIndex(altered, embedding=serving, directory=tmp_path).build()


def test_rewrite_candidates_are_fused_before_exactly_one_rerank():
    docs = [
        RetrievalDocument("u1", ("c1",), "保证收益8%", "case_chunk"),
        RetrievalDocument("u2", ("c2",), "投保赠送礼品", "case_chunk"),
    ]

    class Repository:
        revision = 1

        def summaries(self):
            return tuple(
                CaseSummary(
                    case_id=cid,
                    violation_behavior=text,
                    case_origin="test",
                    retrieval_status="active",
                    evidence_status="test",
                )
                for cid, text in [("c1", "保证收益8%"), ("c2", "投保赠送礼品")]
            )

        def iter_retrieval_documents(self):
            return iter(docs)

    class Reranker:
        model_name = "test-reranker"
        calls = 0

        def rerank(self, query, documents, *, top_n=None):
            self.calls += 1
            return [(i, 0.9 - i * 0.1) for i in range(min(top_n, len(documents)))]

    settings = SimpleNamespace(
        embedding_provider="cloud",
        rerank_provider="cloud",
        query_rewrite_enabled=True,
        type_bridge_enabled=False,
    )
    reranker = Reranker()
    service = CaseRetrievalService(
        Repository(), settings, embedding=TinyEmbedding(), reranker=reranker
    )
    service.query_planner.variants = lambda query: ["投保赠送", "保证收益"]
    assert service.search("收益保证并送礼", top_k=2)
    assert reranker.calls == 1
    assert service.last_diagnostics.rerank_channel == "cloud:test-reranker"


class IndependentReviewer:
    model_name = "offline-test"

    async def complete_json(self, request):
        return JsonChatResponse(
            model=self.model_name,
            data={
                "items": [
                    {
                        "sentence_id": sid,
                        "risk_score": 0.8,
                        "risk_labels": ["承诺收益"],
                        "evidence": ["收益保证8%"],
                        "reasoning": "收益承诺与停售待核分别判断。",
                        "decision_state": "risk",
                        "missing_facts": ["停售公告"],
                        "mechanism_assessments": [
                            {
                                "mechanism_id": "M05",
                                "decision_state": "risk",
                                "risk_labels": ["承诺收益"],
                                "evidence": ["收益保证8%"],
                            },
                            {
                                "mechanism_id": "M09",
                                "decision_state": "insufficient_information",
                                "candidate_labels": ["虚假宣传"],
                                "missing_facts": ["停售公告"],
                            },
                        ],
                    }
                    for sid in request.review_context["target_ids"]
                ]
            },
        )


def test_unknown_mechanism_does_not_erase_an_independent_supported_risk():
    class NoRetrieval:
        def search(self, *args, **kwargs):
            return []

    evidence = SimpleNamespace(
        planner=RetrievalQueryPlanner(), p0=None, rule_context=lambda text: []
    )
    query = asyncio.run(
        prepare_query(
            "收益保证8%，停售公告待核。", QuerySceneClassifier(None), input_type=QueryInputType.TEXT
        )
    )
    pipeline = RiskAssessmentPipeline(
        rule_detector=RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json"),
        retrieval_service=NoRetrieval(),
        taxonomy=("承诺收益", "销售误导", "虚假宣传"),
        llm_client=IndependentReviewer(),
        evidence_service=evidence,
        retrieval_timing="adaptive",
    )
    result = asyncio.run(pipeline.analyze(query))
    assert result.risk_summary.decision_state == "risk"
    assert "承诺收益" in result.risk_summary.risk_labels
    assert "虚假宣传" not in result.risk_summary.risk_labels
    assert result.risk_summary.missing_facts == ["停售公告"]
    assert result.risk_assessments[0].review_status == "needs_review"


def test_web_document_view_keeps_subjects_and_source_only_documents(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    records = [
        {
            "case_id": cid,
            "file_id": "F1",
            "source_file": "F1",
            "party_name": cid,
            "violation_behavior": "各自的事实",
            "retrieval_status": "active",
        }
        for cid in ["p1", "p2"]
    ]
    (directory / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in records), "utf-8")
    source = {
        "case_id": "F2:SOURCE",
        "file_id": "F2",
        "source_file": "F2",
        "violation_behavior": "主体尚未明确",
        "retrieval_status": "active",
    }
    (directory / "source_records.jsonl").write_text(json.dumps(source), "utf-8")
    settings = AppSettings(
        project_root=tmp_path,
        case_kb_root=tmp_path,
        risk_keyword_path=tmp_path / "unused",
        raw_case_upload_dir=tmp_path,
    )
    repo = CaseRepository(settings)
    assert repo.list_cases().total == 2
    assert repo.statistics().total_cases == 2
    assert repo.statistics().subject_case_count == 2
    assert repo.get_case("p1").related_cases[0].case_id == "p2"
    assert repo.get_case("F2:SOURCE").party_name is None


def test_reading_coverage_does_not_truncate_multiple_mechanisms_or_guess_actor_qualification():
    planner = RetrievalQueryPlanner()
    mids = planner.reading_mechanisms(
        "保证收益8%，买保险送卡，客户拒绝后继续外呼，销售替客户确认。"
    )
    assert {"M05", "M04", "M12", "M11"} <= set(mids)
    assert "M10" not in planner.reading_mechanisms("保险代理人介绍了服务流程。")


def test_disclaimer_does_not_bypass_an_affirmative_promise_in_the_same_sentence():
    assert not is_explicit_compliance_statement("未来分红不保证，销售却承诺每年红利一定到账。")
    assert is_explicit_compliance_statement("未来分红不保证，但客户应完整阅读合同。")
    assert is_explicit_compliance_statement("未来分红不保证，但也不能承诺固定收益。")


def test_fixed_cash_flow_is_a_reading_candidate_not_an_automatic_risk():
    assert "M05" in RetrievalQueryPlanner().reading_mechanisms("合同记载固定到账金额。")
