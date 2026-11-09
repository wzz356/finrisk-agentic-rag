"""云端检索通道、降级路径与检索时机的回归测试。

全部使用桩件，不发起任何真实网络请求，也不需要 API Key；
沿用 tests/risk 既有风格（FakeLLM + 桩检索），可在 CI 离线运行。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from application.retrieval_service import (
    CaseRetrievalService,
    CaseSearchFilters,
    RetrievalDiagnostics,
)
from domain.common import QueryInputType
from domain.knowledge_models import CaseSummary, MatchedCase
from llm.types import JsonChatRequest, JsonChatResponse
from retrieval.cloud import CLOUD_UNAVAILABLE, CloudRetrievalError, DenseVectorCache
from retrieval.sparse import RetrievalDocument

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = ("承诺收益", "销售误导", "夸大收益")


def test_cloud_clients_honor_system_proxy_and_explicit_direct(monkeypatch):
    import retrieval.cloud as cloud

    calls = []
    monkeypatch.delenv("FINRISK_CLOUD_PROXY", raising=False)
    monkeypatch.setattr(cloud, "getproxies", lambda: {"https": "http://127.0.0.1:7890"})
    monkeypatch.setattr(cloud, "proxy_bypass", lambda host: False)
    monkeypatch.setattr(cloud.httpx, "Client", lambda **kwargs: calls.append(kwargs))
    cloud.create_cloud_http_client(cloud.CloudSettings(), "https://dashscope.aliyuncs.com/api/v1")
    assert calls[-1]["proxy"] == "http://127.0.0.1:7890"
    monkeypatch.setenv("FINRISK_CLOUD_PROXY", "direct")
    cloud.create_cloud_http_client(cloud.CloudSettings(), "https://dashscope.aliyuncs.com/api/v1")
    assert calls[-1]["proxy"] is None and calls[-1]["trust_env"] is False


# --------------------------------------------------------------------------
# 桩件
# --------------------------------------------------------------------------


def make_summary(case_id: str, tags: list[str], text: str = "保证年化收益") -> CaseSummary:
    return CaseSummary(
        case_id=case_id,
        case_title=f"{case_id} 案例",
        violation_behavior=text,
        penalty_content="责令改正",
        regulator="国家金融监督管理总局甲监管分局",
        party_name="甲保险公司",
        institution_type="保险公司分支机构",
        penalty_date=date(2026, 6, 18),
        source_url=None,
        risk_tags=tags,
        risk_domains=["销售行为"],
        case_origin="batch_build",
        retrieval_status="active",
        evidence_status="verified",
    )


class FakeRepository:
    """只提供检索服务用到的三个方法。"""

    def __init__(self, summaries: list[CaseSummary], documents: list[RetrievalDocument]) -> None:
        self._summaries = summaries
        self._documents = documents
        self.revision = 1

    def summaries(self) -> tuple[CaseSummary, ...]:
        return tuple(self._summaries)

    def iter_retrieval_documents(self):
        return iter(self._documents)


class FakeEmbedding:
    """确定性向量：按字符出现次数构造，避免测试依赖真实模型。"""

    batch_size = 20

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.document_calls = 0
        self.query_calls = 0

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for index, character in enumerate(" ".join(text.split())):
            vector[ord(character) % self._dimension] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)


class BrokenEmbedding(FakeEmbedding):
    """模拟云端不可用。"""

    def embed_documents(self, texts):
        raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：模拟服务中断")

    def embed_query(self, text: str) -> list[float]:
        raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：模拟服务中断")


class FakeReranker:
    """按文档长度倒序给分，便于断言重排确实生效。"""

    model_name = "fake-rerank"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def rerank(self, query: str, documents, *, top_n: int | None = None):
        self.calls.append((query, tuple(documents)))
        scored = [(index, round(len(text) / 200.0, 4)) for index, text in enumerate(documents)]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: top_n or len(scored)]


class BrokenReranker:
    model_name = "broken-rerank"

    def rerank(self, query: str, documents, *, top_n: int | None = None):
        raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：模拟重排失败")


def build_service(
    *,
    embedding=None,
    reranker=None,
    embedding_provider: str = "cloud",
    rerank_provider: str = "cloud",
    bridge_enabled: bool = False,
) -> CaseRetrievalService:
    documents = [
        RetrievalDocument(
            unit_id="C001_fact_001",
            case_ids=("C001",),
            text="向客户承诺保本，保证年化收益百分之十。",
            source="case_chunk",
            weight=1.0,
        ),
        RetrievalDocument(
            unit_id="C002_fact_001",
            case_ids=("C002",),
            text="赠送礼品诱导客户投保并隐瞒退保损失。",
            source="case_chunk",
            weight=1.0,
        ),
        RetrievalDocument(
            unit_id="C002_exp_001",
            case_ids=("C002",),
            text="送礼投保",
            source="case_expansion",
            weight=0.35,
        ),
    ]
    repository = FakeRepository(
        [
            make_summary("C001", ["承诺收益", "销售误导"]),
            make_summary("C002", ["赠送利益", "销售误导"], "赠送礼品诱导投保"),
        ],
        documents,
    )

    class _Settings:
        dense_cache_dir = None
        type_bridge_top_k = 3
        cloud_batch_size = 20

    service = CaseRetrievalService(
        repository,  # type: ignore[arg-type]
        _Settings(),
        embedding=embedding,
        reranker=reranker,
    )
    service._embedding_provider = embedding_provider
    service._rerank_provider = rerank_provider
    service._bridge_enabled = bridge_enabled
    return service


# --------------------------------------------------------------------------
# 检索通道与降级
# --------------------------------------------------------------------------


def test_cloud_dense_channel_is_used_when_configured() -> None:
    embedding = FakeEmbedding()
    service = build_service(embedding=embedding, reranker=FakeReranker())

    matches = service.search("保证年化收益百分之十", top_k=3, risk_tags={"承诺收益"})

    assert matches
    diagnostics = service.last_diagnostics
    assert diagnostics.vector_channel == "dense_cloud"
    assert diagnostics.rerank_channel.startswith("cloud:")
    assert diagnostics.warnings == []
    assert embedding.query_calls >= 1
    assert all(match.dense_score >= 0.0 for match in matches)


def test_cloud_failure_degrades_to_local_tfidf_with_warning() -> None:
    service = build_service(embedding=BrokenEmbedding(), reranker=BrokenReranker())

    matches = service.search("保证年化收益百分之十", top_k=3, risk_tags={"承诺收益"})

    # 降级后仍必须返回可用结果，不能让检索整体失败。
    assert matches
    assert matches[0].case_id == "C001"
    diagnostics = service.last_diagnostics
    assert diagnostics.vector_channel == "tfidf_local"
    assert diagnostics.rerank_channel == "weighted_local"
    assert diagnostics.degraded is True
    assert any("降级" in warning for warning in diagnostics.warnings)


def test_missing_api_key_never_leaks_secret_into_warnings() -> None:
    """密钥缺失时必须降级而非抛错，且告警不含密钥线索。"""

    service = build_service()  # 未注入 embedding，等价于云端不可用
    matches = service.search("保证年化收益百分之十", top_k=3)

    assert matches
    joined = " ".join(service.last_diagnostics.warnings)
    assert "sk-" not in joined
    assert "Bearer" not in joined


def test_local_provider_matches_legacy_deterministic_ordering() -> None:
    """纯本地模式必须与改造前一致：确定性、可复现、无降级告警。"""

    service = build_service(embedding_provider="local", rerank_provider="none")
    query = "保证年化收益百分之十"

    first = service.search(query, top_k=3, risk_tags={"承诺收益"})
    second = service.search(query, top_k=3, risk_tags={"承诺收益"})

    assert [item.case_id for item in first] == [item.case_id for item in second]
    assert service.last_diagnostics.vector_channel == "tfidf_local"
    assert service.last_diagnostics.warnings == []
    assert first[0].bm25_score > 0
    assert first[0].rrf_score > 0


def test_risk_tag_filter_still_excludes_mismatched_cases() -> None:
    service = build_service(embedding_provider="local", rerank_provider="none")

    assert service.search("保证年化收益", top_k=3, risk_tags={"承诺收益"})[0].case_id == "C001"
    assert service.search("保证年化收益", top_k=3, risk_tags={"虚列费用套取资金"}) == []


def test_metadata_filters_apply_before_scoring() -> None:
    service = build_service(embedding_provider="local", rerank_provider="none")
    filters = CaseSearchFilters(
        risk_tags=frozenset({"承诺收益"}),
        regulator="甲监管分局",
        penalty_date_from=date(2026, 1, 1),
        penalty_date_to=date(2026, 12, 31),
    )

    matches = service.search("保证年化收益", top_k=3, filters=filters)

    assert matches and matches[0].case_id == "C001"
    assert matches[0].match_reason
    assert (
        service.search("保证年化收益", filters=CaseSearchFilters(regulator="不存在的监管机构"))
        == []
    )


def test_reranker_receives_candidate_texts_and_result_is_reproducible() -> None:
    reranker = FakeReranker()
    service = build_service(embedding=FakeEmbedding(), reranker=reranker)

    matches = service.search("赠送礼品诱导投保", top_k=3, risk_tags={"赠送利益"})

    assert reranker.calls, "云端重排未被调用"
    query, documents = reranker.calls[0]
    assert query == "赠送礼品诱导投保"
    assert documents and all(isinstance(text, str) for text in documents)
    assert matches
    assert all(0.0 <= match.score <= 1.0 for match in matches)


def test_top_k_bounds_are_validated() -> None:
    service = build_service(embedding_provider="local", rerank_provider="none")

    with pytest.raises(ValueError):
        service.search("保证收益", top_k=0)
    with pytest.raises(ValueError):
        service.search("保证收益", top_k=21)


# --------------------------------------------------------------------------
# 向量缓存
# --------------------------------------------------------------------------


@pytest.fixture()
def cache_dir():
    """项目内可写的临时目录。

    不依赖 ``tmp_path`` / ``tmp_path_factory``：部分受限执行环境的系统临时目录
    不可写，会让测试报权限错误而非暴露真实失败。夹具退出时自行清理。
    """

    import shutil
    import uuid

    root = PROJECT_ROOT / ".pytest-cache-tmp"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


def test_dense_vector_cache_roundtrip(cache_dir: Path) -> None:
    cache = DenseVectorCache(cache_dir / "dense", "fake-embedding", 4)
    payload = [("U1", "sig-a", [0.1, 0.2, 0.3, 0.4]), ("U2", "sig-b", [0.5, 0.5, 0.0, 0.0])]

    cache.save(payload)
    loaded = cache.load()

    assert set(loaded) == {"U1", "U2"}
    assert loaded["U1"][0] == "sig-a"
    assert loaded["U1"][1] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert (cache_dir / "dense" / "fake-embedding-4.jsonl").is_file()
    # 维度不符的脏数据必须被忽略，不能污染矩阵。
    (cache_dir / "dense" / "fake-embedding-4.jsonl").open("a", encoding="utf-8").write(
        json.dumps({"unit_id": "U3", "signature": "s", "vector": [1.0, 2.0]}) + "\n"
    )
    assert "U3" not in cache.load()


def test_dense_vector_cache_ignores_other_namespace(cache_dir: Path) -> None:
    first = DenseVectorCache(cache_dir / "dense", "model-a", 4)
    first.save([("U1", "sig", [1.0, 0.0, 0.0, 0.0])])

    second = DenseVectorCache(cache_dir / "dense", "model-b", 4)

    # 换模型即换命名空间，旧缓存不得被误用。
    assert second.load() == {}


def test_dense_vector_cache_survives_unreadable_file(cache_dir: Path) -> None:
    """缓存损坏时返回空表而非抛错，保证冷启动不被脏文件阻断。"""

    cache = DenseVectorCache(cache_dir / "dense", "fake-embedding", 4)
    cache.save([("U1", "sig", [1.0, 0.0, 0.0, 0.0])])
    (cache_dir / "dense" / "fake-embedding-4.jsonl").write_text(
        "这不是 JSON\n{损坏的行\n", encoding="utf-8"
    )

    assert cache.load() == {}


def test_dense_vector_cache_missing_directory_is_empty(cache_dir: Path) -> None:
    cache = DenseVectorCache(cache_dir / "not-created", "fake-embedding", 4)

    assert cache.load() == {}


# --------------------------------------------------------------------------
# 检索时机（pipeline 层）
# --------------------------------------------------------------------------


class ScriptedLLM:
    """返回固定风险分的假大模型，并记录收到的提示词。"""

    def __init__(self, score: float = 0.62, labels: tuple[str, ...] = ("销售误导",)) -> None:
        self.requests: list[JsonChatRequest] = []
        self._score = score
        self._labels = list(labels)

    @property
    def model_name(self) -> str:
        return "scripted-model"

    async def complete_json(self, request: JsonChatRequest) -> JsonChatResponse:
        self.requests.append(request)
        content = request.messages[-1].content
        target_ids = json.loads(
            content.split("只需输出这些待复核句子的结果：", 1)[1].splitlines()[0]
        )
        return JsonChatResponse(
            data={
                "items": [
                    {
                        "sentence_id": sentence_id,
                        "risk_score": self._score,
                        "risk_labels": self._labels,
                        "evidence": [],
                        "reasoning": "存在误导性表述",
                    }
                    for sentence_id in target_ids
                ]
            },
            model=self.model_name,
        )


class RecordingRetrieval:
    """记录每次检索使用的标签约束，用于验证前置/后置口径。"""

    def __init__(self, score: float = 0.86) -> None:
        self.calls: list[frozenset[str]] = []
        self._score = score

    def search(self, query: str, *, top_k: int = 5, risk_tags=None, filters=None):
        del query, top_k, filters
        tags = frozenset(risk_tags or ())
        self.calls.append(tags)
        return [
            MatchedCase(
                case_id="C001",
                violation_behavior="承诺保证收益误导投保人",
                risk_tags=["销售误导"],
                score=self._score,
                bm25_score=0.7,
                vector_score=0.6,
                rrf_score=0.8,
                rerank_score=self._score,
                matched_text="承诺保证收益误导投保人",
                matched_source="case_chunk",
                matched_unit_id="C001_fact_001",
                matched_risk_tags=sorted(tags.intersection({"销售误导"})),
                retrieval_channel="bm25+tfidf_local",
            )
        ]


@pytest.fixture(scope="module")
def weak_signal_query():
    """弱信号文本：无强正则命中，必须进入 LLM 复核。"""

    import asyncio

    from query.classification import QuerySceneClassifier
    from query.enhancement import prepare_query

    text = "这款产品的收益表现不错，客户可以放心配置，具体我们见面再聊。"
    return asyncio.run(
        prepare_query(text, QuerySceneClassifier(None), input_type=QueryInputType.TEXT)
    )


def build_pipeline(retrieval, llm, *, timing: str, gate: float = 0.50, top_k: int = 3):
    from risk.detection import RuleRiskDetector
    from risk.pipeline import RiskAssessmentPipeline

    return RiskAssessmentPipeline(
        rule_detector=RuleRiskDetector(
            PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
        ),
        retrieval_service=retrieval,  # type: ignore[arg-type]
        taxonomy=TAXONOMY,
        llm_client=llm,
        retrieval_timing=timing,
        semantic_context_gate=gate,
        retrieval_top_k=top_k,
    )


def test_post_timing_keeps_original_prompt_and_single_tagged_search(weak_signal_query) -> None:
    """后置模式：提示词与原实现逐字一致，仅判定后按最终标签检索一次。"""

    import asyncio

    llm, retrieval = ScriptedLLM(), RecordingRetrieval()
    pipeline = build_pipeline(retrieval, llm, timing="post")

    result = asyncio.run(pipeline.analyze(weak_signal_query))

    assert len(llm.requests) == 1
    prompt = llm.requests[0].messages[-1].content
    system = llm.requests[0].messages[0].content
    assert "reference_cases" not in prompt
    assert "只判断待审文本本身，不得使用或猜测历史案例" in system
    assert llm.requests[0].review_context["case_context_enabled"] is False
    assert retrieval.calls == [frozenset({"销售误导"})]
    assert result.risk_assessments[0].matched_cases


def test_pre_timing_injects_cases_and_researches_with_final_labels(weak_signal_query) -> None:
    """前置模式：注入类案摘要，且判定后仍按最终标签重新检索。"""

    import asyncio

    llm, retrieval = ScriptedLLM(), RecordingRetrieval()
    pipeline = build_pipeline(retrieval, llm, timing="pre")

    result = asyncio.run(pipeline.analyze(weak_signal_query))

    assert len(llm.requests) == 1
    prompt = llm.requests[0].messages[-1].content
    system = llm.requests[0].messages[0].content
    assert "reference_cases" in prompt
    assert "不得使用或猜测历史案例" not in system
    assert llm.requests[0].review_context["case_context_enabled"] is True
    # 判定前无标签语义召回，判定后按最终标签检索。
    assert retrieval.calls == [frozenset(), frozenset({"销售误导"})]
    assert result.risk_assessments[0].matched_cases


def test_case_similarity_never_changes_final_score(weak_signal_query) -> None:
    """案例相似度不得进入风险定性打分：两种时机分数必须完全一致。"""

    import asyncio

    post_llm, post_retrieval = ScriptedLLM(), RecordingRetrieval(score=0.95)
    pre_llm, pre_retrieval = ScriptedLLM(), RecordingRetrieval(score=0.11)

    post_result = asyncio.run(
        build_pipeline(post_retrieval, post_llm, timing="post").analyze(weak_signal_query)
    )
    pre_result = asyncio.run(
        build_pipeline(pre_retrieval, pre_llm, timing="pre").analyze(weak_signal_query)
    )

    # 两次的案例相关分差异极大（0.95 vs 0.11），但风险分必须相同。
    assert post_result.risk_summary.overall_score == pre_result.risk_summary.overall_score
    assert post_result.risk_assessments[0].risk_score == pre_result.risk_assessments[0].risk_score


def test_pre_timing_context_gate_filters_low_relevance_cases(weak_signal_query) -> None:
    """低于注入门槛的案例不得进入提示词，避免不相关案例干扰判断。"""

    import asyncio

    llm, retrieval = ScriptedLLM(), RecordingRetrieval(score=0.10)
    pipeline = build_pipeline(retrieval, llm, timing="pre", gate=0.50)

    asyncio.run(pipeline.analyze(weak_signal_query))

    prompt = llm.requests[0].messages[-1].content
    assert "reference_cases" not in prompt
    assert llm.requests[0].review_context["case_context_enabled"] is False


def test_invalid_timing_is_rejected(weak_signal_query) -> None:
    from risk.detection import RuleRiskDetector
    from risk.pipeline import RiskAssessmentPipeline

    with pytest.raises(ValueError):
        RiskAssessmentPipeline(
            rule_detector=RuleRiskDetector(
                PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
            ),
            retrieval_service=RecordingRetrieval(),  # type: ignore[arg-type]
            taxonomy=TAXONOMY,
            llm_client=None,
            retrieval_timing="before",
        )


def test_retrieval_top_k_is_validated() -> None:
    from risk.detection import RuleRiskDetector
    from risk.pipeline import RiskAssessmentPipeline

    with pytest.raises(ValueError):
        RiskAssessmentPipeline(
            rule_detector=RuleRiskDetector(
                PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
            ),
            retrieval_service=RecordingRetrieval(),  # type: ignore[arg-type]
            taxonomy=TAXONOMY,
            llm_client=None,
            retrieval_top_k=0,
        )


def test_diagnostics_defaults_are_local() -> None:
    diagnostics = RetrievalDiagnostics()

    assert diagnostics.vector_channel == "tfidf_local"
    assert diagnostics.rerank_channel == "weighted_local"
    assert diagnostics.bridge_enabled is False
    assert diagnostics.degraded is False


def test_matched_case_new_fields_have_safe_defaults() -> None:
    """新增字段必须有默认值，保证既有调用方与序列化不受影响。"""

    matched = MatchedCase(
        case_id="C001",
        violation_behavior="保证收益",
        score=0.5,
        matched_text="保证收益",
        matched_source="case_chunk",
        matched_unit_id="C001_fact_001",
    )

    assert matched.dense_score == 0.0
    assert matched.type_bridge_score == 0.0
    assert matched.retrieval_channel == "bm25+tfidf"
    payload = matched.model_dump()
    assert payload["retrieval_channel"] == "bm25+tfidf"
