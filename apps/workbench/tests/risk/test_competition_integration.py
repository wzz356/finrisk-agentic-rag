"""Evidence boundaries, unknown decisions and compact batching must hold offline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from application.evidence_service import KnowledgeEvidenceService
from domain.common import QueryInputType
from llm import JsonChatResponse
from query.classification import QuerySceneClassifier
from query.enhancement import prepare_query
from retrieval.cloud import CloudRetrievalError
from retrieval.local_dense import LocalBGEIndex
from retrieval.sparse import RetrievalDocument, SparseRetrievalIndex
from risk.agent_tools import QueryToolBudget, RiskEvidenceTools
from risk.detection import RuleRiskDetector
from risk.pipeline import RiskAssessmentPipeline
from risk.prompt_context import compact_review_payload

ROOT = Path(__file__).resolve().parents[2]


class NoRetrieval:
    calls = 0

    def search(self, *args, **kwargs):
        self.calls += 1
        return []


class CaptureReviewer:
    model_name = "local-test-fixture"

    def __init__(self, unknown=False):
        self.requests = []
        self.unknown = unknown

    async def complete_json(self, request):
        self.requests.append(request)
        return JsonChatResponse(
            data={
                "items": [
                    {
                        "sentence_id": sid,
                        "risk_score": 0.65 if self.unknown else 0.05,
                        "risk_labels": ["虚假宣传"] if self.unknown else [],
                        "evidence": [],
                        "reasoning": "缺少当前公告，保留待核事实。"
                        if self.unknown
                        else "正常业务提示。",
                        "decision_state": "insufficient_information"
                        if self.unknown
                        else "no_clear_risk",
                        "candidate_labels": ["虚假宣传"] if self.unknown else [],
                        "missing_facts": ["产品停售公告及日期"] if self.unknown else [],
                        "condition_assessments": [
                            {"condition_id": "announcement", "status": "unknown", "evidence": []}
                        ]
                        if self.unknown
                        else [],
                    }
                    for sid in request.review_context["target_ids"]
                ]
            },
            model=self.model_name,
        )


def analyze(text, reviewer, **kwargs):
    query = asyncio.run(
        prepare_query(text, QuerySceneClassifier(None), input_type=QueryInputType.TEXT)
    )
    retrieval = NoRetrieval()
    pipeline = RiskAssessmentPipeline(
        rule_detector=RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json"),
        retrieval_service=retrieval,
        taxonomy=("虚假宣传",),
        llm_client=reviewer,
        **kwargs,
    )
    result = asyncio.run(pipeline.analyze(query))
    return query, result, retrieval


def test_unknown_keeps_candidate_but_never_becomes_supported_label():
    _, result, _ = analyze("这款产品下周停止销售，请再了解一下。", CaptureReviewer(unknown=True))
    assert result.risk_summary.decision_state == "insufficient_information"
    assert result.risk_summary.risk_labels == []
    assert result.risk_summary.missing_facts == ["产品停售公告及日期"]
    assert all("虚假宣传" in r.candidate_labels for r in result.risk_assessments)
    assert all(r.review_status == "needs_review" for r in result.risk_assessments)


def test_many_short_sentences_are_bounded_even_under_old_whole_document_char_limit():
    reviewer = CaptureReviewer()
    text = "\n".join(f"顾问：第{i}项产品方案是否值得考虑还要进一步讨论。" for i in range(35))
    query, result, _ = analyze(text, reviewer)
    assert len(text) < 6000 and len(query.sentences) > 12
    assert len(reviewer.requests) > 1
    assert all(len(r.review_context["target_ids"]) <= 12 for r in reviewer.requests)
    assert result.risk_summary.llm_reviewed_sentence_count == len(query.sentences)
    assert len(result.risk_assessments) == len(query.sentences)


def test_shared_sources_keep_scope_stance_and_target_citation_whitelist():
    source = {
        "sentence_id": "source-1",
        "text": "不得向客户承诺收益。",
        "speaker_role": "trainer",
        "is_target": True,
    }
    context = {
        "scope_id": "product-one",
        "stance": "prohibition",
        "speaker_role": "trainer",
        "context_truncated": False,
        "source_sentences": [source],
        "review_reasons": ["培训反例"],
    }
    item = {
        "sentence_id": "source-1",
        "scene": "ST5",
        "context": context,
        "local_stage1_score": 0.7,
        "local_stage2_score": 0.1,
        "local_labels": [],
        "local_evidence": [],
    }
    payload = compact_review_payload([item, {**item, "sentence_id": "target-two"}])
    assert len(payload["source_sentences"]) == 1
    assert all(
        t["source_ids"] == ["s1"] and t["stance"] == "prohibition" for t in payload["targets"]
    )
    assert payload["targets"][0]["speaker_role"] == "trainer"
    assert payload["targets"][0]["scope"] == "p1"


@pytest.fixture
def evidence(tmp_path):
    versions = {
        key: key + "-frozen" for key in ["content", "projection", "articles", "rules", "scope"]
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"knowledge_versions": versions}), encoding="utf-8"
    )
    handoff = tmp_path / "source_bundle/releases/2026-10-18/data/handoff_scope"
    handoff.mkdir(parents=True)
    (handoff / "version_lock.json").write_text(json.dumps(versions), encoding="utf-8")
    source = "甲机构：承诺收益。乙机构：正常说明。"
    sha = hashlib.sha256(source.encode()).hexdigest()
    with sqlite3.connect(tmp_path / "evidence_registry.sqlite3") as connection:
        connection.execute("CREATE TABLE sources(file_id TEXT PRIMARY KEY,text TEXT,sha256 TEXT)")
        connection.execute("CREATE TABLE evidence(id TEXT PRIMARY KEY,file_id TEXT,data TEXT)")
        connection.execute("CREATE TABLE units(id TEXT PRIMARY KEY,file_id TEXT,data TEXT)")
        connection.execute("INSERT INTO sources VALUES (?,?,?)", ("F1", source, sha))
        item = {"start_char": 4, "end_char": 8, "verbatim": "承诺收益", "source_sha256": sha}
        connection.execute("INSERT INTO evidence VALUES (?,?,?)", ("E1", "F1", json.dumps(item)))
    return KnowledgeEvidenceService(tmp_path)


def test_evidence_only_reads_registered_literal_intervals(evidence):
    result = evidence.read("E1")
    assert result["literal_verified"] and result["segments"][0]["text"] == "承诺收益"
    with pytest.raises(ValueError, match="未登记"):
        evidence.read("../../arbitrary-file")
    with pytest.raises(ValueError, match="没有登记"):
        evidence.read("E1", "extended")


def test_source_corruption_is_rejected_before_quote_return(evidence):
    with sqlite3.connect(evidence.database) as connection:
        connection.execute("UPDATE sources SET text=? WHERE file_id='F1'", ("伪造来源",))
    with pytest.raises(ValueError, match="指纹"):
        evidence.read("E1")


def test_five_version_lock_mismatch_is_rejected(evidence):
    path = evidence.root / "manifest.json"
    payload = json.loads(path.read_text())
    payload["knowledge_versions"]["scope"] = "different"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="版本锁"):
        KnowledgeEvidenceService(evidence.root)


def test_shared_search_budget_and_current_fact_conditions(evidence):
    async def execute():
        budget = QueryToolBudget(remaining_searches=1)
        source = [
            {"sentence_id": "S1", "context": {"source_sentences": [{"text": "资格正在申请。"}]}}
        ]
        tools = RiskEvidenceTools(NoRetrieval(), evidence, source, budget)
        first = await tools.dispatch("search_cases", {"query": "销售资格"})
        repeated = await tools.dispatch("search_cases", {"query": "销售资格"})
        exhausted = await tools.dispatch("search_cases", {"query": "销售资格另一问法"})
        assert first["historical_facts_only"]
        assert repeated["status"] == "no_new_evidence"
        assert exhausted["status"] == "budget_exhausted"
        forged = json.dumps([{"sentence_id": "S1", "status": "met", "evidence": ["已经取得资格"]}])
        with pytest.raises(ValueError, match="当前输入原文"):
            await tools.dispatch("check_conditions", {"assessments_json": forged})
        unknown = json.dumps([{"sentence_id": "S1", "status": "unknown", "evidence": []}])
        assert (await tools.dispatch("check_conditions", {"assessments_json": unknown}))[
            "state"
        ] == "unknown"

    asyncio.run(execute())


def test_numpy_bm25_matches_scalar_formula_and_weighted_tie_order():
    import math
    from collections import defaultdict

    from retrieval.tokenization import chinese_unigram_bigram

    docs = [
        RetrievalDocument(str(i), (str(i),), text, "case_chunk", weight)
        for i, (text, weight) in enumerate(
            [
                ("顾问承诺保险收益", 1.0),
                ("保险说明不得承诺收益", 0.55),
                ("核查合同约定及投保人身份", 1.0),
                ("顾问承诺保险收益", 1.0),
            ]
        )
    ]
    index = SparseRetrievalIndex(docs)
    query = "保险收益，承诺收益与投保人身份"
    scalar = defaultdict(float)
    for term in dict.fromkeys(chinese_unigram_bigram(query)):
        postings = index._postings.get(term, [])
        if not postings:
            continue
        idf = math.log(1 + (len(docs) - len(postings) + 0.5) / (len(postings) + 0.5))
        for position, frequency in postings:
            denominator = frequency + index.k1 * (
                1
                - index.b
                + index.b * index._document_lengths[position] / index._average_document_length
            )
            scalar[position] += idf * frequency * (index.k1 + 1) / denominator
    expected = sorted(scalar, key=lambda p: (-scalar[p] * docs[p].weight, p))
    hits = index.search(query, top_k=20)
    assert [h.document.unit_id for h in hits] == [docs[p].unit_id for p in expected]
    for hit, p in zip(hits, expected, strict=True):
        assert hit.score == pytest.approx(scalar[p] * docs[p].weight, abs=1e-12)


def test_local_index_cache_is_bound_to_model_and_text_without_query_time_build(tmp_path):
    class Embedding:
        model_name = "model-one"
        dimension = 2
        batch_size = 2
        query_instruction = "query:"
        truncated_inputs = 0
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]

    docs = [RetrievalDocument("U1", ("C1",), "测试原文", "case_chunk")]
    client = Embedding()
    with pytest.raises(CloudRetrievalError, match="未预构建"):
        LocalBGEIndex(docs, embedding=client, directory=tmp_path).build()
    assert client.calls == 0
    built = LocalBGEIndex(docs, embedding=client, directory=tmp_path, allow_build=True)
    built.build()
    assert client.calls == 1
    reloaded = LocalBGEIndex(docs, embedding=client, directory=tmp_path)
    assert reloaded.build().cache_hits == 1 and client.calls == 1
    client.model_name = "different-model"
    with pytest.raises(CloudRetrievalError, match="未预构建"):
        LocalBGEIndex(docs, embedding=client, directory=tmp_path).build()
    changed = [RetrievalDocument("U1", ("C1",), "已修改原文", "case_chunk")]
    with pytest.raises(CloudRetrievalError, match="未预构建"):
        LocalBGEIndex(changed, embedding=client, directory=tmp_path).build()


def test_hybrid_groups_same_scope_mechanism_before_search():
    class Planner:
        def mechanisms(self, query):
            return ["M09"]

    class Evidence:
        planner = Planner()

        def rule_context(self, text):
            return []

    _, _, retrieval = analyze(
        "顾问：第一项产品方案是否值得考虑还要讨论。顾问：第二项产品安排还需进一步研究。",
        CaptureReviewer(),
        retrieval_timing="hybrid",
        evidence_service=Evidence(),
    )
    assert retrieval.calls == 1


def test_hybrid_strong_negated_gift_is_reviewed_and_can_be_cleared_with_literal_quote():
    class Planner:
        def mechanisms(self, query):
            return ["M04"]

    class Evidence:
        planner = Planner()

        def rule_context(self, text):
            return []

    class Reviewer(CaptureReviewer):
        async def complete_json(self, request):
            result = await super().complete_json(request)
            sources = {
                item["sentence_id"]: item["text"] for item in request.review_context["sentences"]
            }
            for item in result.data["items"]:
                item["evidence"] = [sources[item["sentence_id"]]]
            return result

    reviewer = Reviewer()
    _, result, _ = analyze(
        "我们不另外赠送购物卡或返还保费。",
        reviewer,
        retrieval_timing="hybrid",
        evidence_service=Evidence(),
    )
    assert reviewer.requests
    assert result.risk_assessments[0].llm_reviewed
    assert result.risk_assessments[0].risk_score < 0.2
    assert result.risk_summary.decision_state == "no_clear_risk"


def test_hybrid_tax_keyword_requires_current_fact_review_instead_of_local_direct():
    class Planner:
        def mechanisms(self, query):
            return ["M03"]

    class Evidence:
        planner = Planner()

        def rule_context(self, text):
            return []

    reviewer = CaptureReviewer(unknown=True)
    _, result, _ = analyze(
        "顾问：本次领取免税，具体身份资料尚未补齐。",
        reviewer,
        retrieval_timing="hybrid",
        evidence_service=Evidence(),
    )
    assert reviewer.requests
    assert result.risk_assessments[0].llm_reviewed
    assert result.risk_summary.decision_state == "insufficient_information"
