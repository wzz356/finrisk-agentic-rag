"""案例混合检索应用服务。

三路召回 + RRF 融合 + 元数据过滤 + 可切换重排：

- BM25 词法通道：始终本地，倒排表统计，不上云；
- 稠密语义通道：默认走阿里云百炼 Embedding（``embedding_provider=cloud``），
  失败或未配置时降级为本地 TF-IDF 余弦通道；
- 类型桥接通道：按风险标签收窄候选后用云端向量排序（``type_bridge_enabled``）。

重排同理：默认云端 ``qwen3-rerank``，不可用时退回确定性加权公式。
所有云端失败都不抛给上层，而是记录到 :attr:`CaseRetrievalService.last_diagnostics`
并继续用本地通道出结果，保证断网可用。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

from domain.knowledge_models import CaseSummary, MatchedCase
from retrieval import SparseRetrievalIndex, SparseSearchHit
from retrieval.bridge import TypeBridgeRecall
from retrieval.cloud import (
    CloudRetrievalError,
    DashScopeRerankClient,
    DenseVectorCache,
    EmbeddingProvider,
    NullRerankClient,
    RerankProvider,
    build_cloud_settings,
)
from retrieval.dense import DenseRetrievalIndex
from retrieval.explanation import business_match_reason
from retrieval.query_planner import RetrievalQueryPlanner
from retrieval.sparse import RetrievalDocument
from retrieval.tokenization import chinese_unigram_bigram
from storage.case_repository import CaseRepository

if TYPE_CHECKING:
    from application.evidence_service import KnowledgeEvidenceService

_RRF_K = 60
# 云端精排窗口：候选数超过该值时只对本地分靠前的部分精排，控制单次请求体积。
_RERANK_WINDOW = 20
_LOCAL_VECTOR_CHANNEL = "tfidf_local"
_DENSE_VECTOR_CHANNEL = "dense_cloud"


@dataclass(frozen=True, slots=True)
class CaseSearchFilters:
    """案例组合检索条件；空值表示不限制。"""

    risk_tags: frozenset[str] = frozenset()
    regulator: str | None = None
    party_name: str | None = None
    institution_type: str | None = None
    penalty_date_from: date | None = None
    penalty_date_to: date | None = None


@dataclass(slots=True)
class _ChannelAggregate:
    score: float
    best_hit: SparseSearchHit


@dataclass(slots=True)
class RetrievalDiagnostics:
    """最近一次检索实际生效的通道与降级原因，用于评测归因与告警。"""

    vector_channel: str = _LOCAL_VECTOR_CHANNEL
    rerank_channel: str = "weighted_local"
    bridge_enabled: bool = False
    bridge_candidates: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.warnings)


class CaseRetrievalService:
    """从同一案例库执行多路召回、RRF 融合、组合过滤和可切换重排。

    构造签名保持 ``CaseRetrievalService(repository)`` 可用：未传 ``settings``
    时不启用任何云端通道，行为与纯本地版本完全一致，既有测试与离线部署不受
    影响。
    """

    def __init__(
        self,
        repository: CaseRepository,
        settings: object | None = None,
        *,
        embedding: EmbeddingProvider | None = None,
        reranker: RerankProvider | None = None,
        bridge: TypeBridgeRecall | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self._index: SparseRetrievalIndex | None = None
        self._summaries: dict[str, CaseSummary] = {}
        self._indexed_revision = -1
        self._lock = RLock()
        self.last_diagnostics = RetrievalDiagnostics()
        self._dense_stats: object | None = None

        self._embedding_provider = _provider_value(
            settings, "embedding_provider", "FINRISK_EMBEDDING_PROVIDER", "cloud"
        )
        self._rerank_provider = _provider_value(
            settings, "rerank_provider", "FINRISK_RERANK_PROVIDER", "cloud"
        )
        self._bridge_enabled = _bool_value(
            settings, "type_bridge_enabled", "FINRISK_TYPE_BRIDGE", True
        )

        self._embedding = embedding
        self._reranker = reranker
        self._bridge = bridge
        self._dense: DenseRetrievalIndex | None = None
        self._dense_failed = False
        root = getattr(settings, "case_kb_root", None)
        lexicon = (
            Path(root) / "source_bundle/releases/2026-10-18/data/auxiliary_lexicon.json"
            if root
            else None
        )
        self.query_planner = RetrievalQueryPlanner(lexicon)
        self._rewrite_enabled = bool(getattr(settings, "query_rewrite_enabled", False))
        self._evidence_reader: KnowledgeEvidenceService | None = None
        self._documents_by_id: dict[str, RetrievalDocument] = {}

    @property
    def ready(self) -> bool:
        return self._index is not None and self._indexed_revision == self.repository.revision

    def with_reranker(self, provider: str) -> CaseRetrievalService:
        """Share the ready read-only indexes; keep rerank clients and diagnostics separate."""
        child = CaseRetrievalService(self.repository, self.settings, embedding=self._embedding)
        for name in ("_index", "_indexed_revision", "_dense", "_dense_failed", "_dense_stats"):
            setattr(child, name, getattr(self, name))
        child._summaries = dict(self._summaries)
        child._documents_by_id = dict(self._documents_by_id)
        child._rerank_provider = provider
        return child

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        risk_tags: set[str] | None = None,
        filters: CaseSearchFilters | None = None,
    ) -> list[MatchedCase]:
        """混合检索入口；任何云端异常都降级为纯本地通道，不向上抛出。"""

        if not 1 <= top_k <= 20:
            raise ValueError("top_k 必须在 1 到 20 之间")
        try:
            if self._rewrite_enabled:
                matches = self._search_variants(
                    query, top_k=top_k, risk_tags=risk_tags, filters=filters
                )
            else:
                matches = self._search(query, top_k=top_k, risk_tags=risk_tags, filters=filters)
            return self._verify_matches(matches, query)
        except Exception as error:  # noqa: BLE001 - 检索必须始终可用
            return self._search_local_only(
                query, top_k=top_k, risk_tags=risk_tags, filters=filters, error=error
            )

    def _search_variants(
        self,
        query: str,
        *,
        top_k: int,
        risk_tags: set[str] | None,
        filters: CaseSearchFilters | None,
    ) -> list[MatchedCase]:
        variants = self.query_planner.variants(query)
        if not variants:
            return self._search(query, top_k=top_k, risk_tags=risk_tags, filters=filters)
        queries = [query, *variants[:3]]
        embedding = (
            self._embedding_client() if self._embedding_provider in {"bge", "cloud"} else None
        )
        if embedding is not None and hasattr(embedding, "embed_queries"):
            embedding.embed_queries(queries)
        rankings = [
            self._search(text, top_k=20, risk_tags=risk_tags, filters=filters, rerank=False)
            for text in queries
        ]
        diagnostics = self.last_diagnostics
        scores: dict[str, float] = {}
        best: dict[str, MatchedCase] = {}

        def identity(match: MatchedCase) -> str:
            if bool(getattr(self.settings, "full_corpus", False)):
                return match.file_id or match.case_id
            return match.case_id

        for index, ranking in enumerate(rankings):
            weight = 1.0 if index == 0 else 1.0 / len(variants)
            for rank, match in enumerate(ranking, 1):
                key = identity(match)
                scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
                best.setdefault(key, match)
        ordered = sorted(best, key=lambda cid: (-scores[cid], cid))
        anchors = max(0, min(2, int(getattr(self.settings, "query_rewrite_original_anchors", 1))))
        anchored_ids = (
            [identity(match) for match in rankings[0][:anchors]]
            if (self._rerank_provider not in {"cloud", "deepseek"})
            else []
        )
        ordered = [*anchored_ids, *(cid for cid in ordered if cid not in anchored_ids)]
        fused = [
            best[cid].model_copy(
                update={
                    "match_reason": best[cid].match_reason + "；原句与中性行为查询分组融合",
                    "retrieval_queries": queries,
                }
            )
            for cid in ordered
        ]
        if self._rerank_provider not in {"cloud", "deepseek"}:
            return fused[:top_k]
        maximum = max(scores.values(), default=1.0)
        ranked = [(scores[identity(m)] / maximum, m.rrf_score, m.case_id, m) for m in fused]
        result = self._apply_rerank(query, ranked, top_k, diagnostics)
        self.last_diagnostics = diagnostics
        return result

    def _search(
        self,
        query: str,
        *,
        top_k: int,
        risk_tags: set[str] | None,
        filters: CaseSearchFilters | None,
        rerank: bool = True,
    ) -> list[MatchedCase]:
        normalized_filters = _normalized_filters(filters, risk_tags)
        index = self._ensure_index()
        recall_size = max(120, top_k * 40)
        diagnostics = RetrievalDiagnostics(bridge_enabled=False)

        bm25 = self._aggregate_cases(
            index.search(query, top_k=recall_size),
            normalized_filters,
        )

        # 向量通道：优先云端稠密语义，失败则降级本地 TF-IDF，绝不让检索整体失败。
        vector, dense_scores = self._vector_channel(
            query, index, normalized_filters, recall_size, diagnostics
        )

        # 桥接通道：仅在调用方给出风险标签时启用，用于收窄同标签案例。
        bridge = self._bridge_channel(query, normalized_filters, diagnostics)

        bm25_ranks = self._ranks(bm25)
        vector_ranks = self._ranks(vector)
        bridge_ranks = self._ranks(bridge)
        candidate_ids = set(bm25) | set(vector) | set(bridge)
        query_tokens = _meaningful_tokens(query)
        channel_count = 2 + (1 if bridge else 0)

        pre_ranked: list[tuple[float, float, str, MatchedCase]] = []
        for case_id in candidate_ids:
            summary = self._summaries.get(case_id)
            if summary is None:
                continue
            bm25_item = bm25.get(case_id)
            vector_item = vector.get(case_id)
            bridge_item = bridge.get(case_id)
            best_hit = _choose_hit(bm25_item, vector_item, bridge_item)
            bm25_score = _normalize_bm25(bm25_item.score) if bm25_item else 0.0
            vector_score = min(1.0, vector_item.score) if vector_item else 0.0
            bridge_score = min(1.0, bridge_item.score) if bridge_item else 0.0
            rrf_raw = (
                ((1 / (_RRF_K + bm25_ranks[case_id])) if case_id in bm25_ranks else 0.0)
                + ((1 / (_RRF_K + vector_ranks[case_id])) if case_id in vector_ranks else 0.0)
                + ((1 / (_RRF_K + bridge_ranks[case_id])) if case_id in bridge_ranks else 0.0)
            )
            rrf_score = min(1.0, rrf_raw / (channel_count / (_RRF_K + 1)))
            lexical_score = _token_overlap(query_tokens, best_hit.document.text)
            source_score = _source_score(best_hit.document.source)
            local_score = _weighted_score(
                rrf_score=rrf_score,
                bm25_score=bm25_score,
                vector_score=vector_score,
                lexical_score=lexical_score,
                source_score=source_score,
                bridge_score=bridge_score,
            )
            phrase_match = _contains_phrase(query, best_hit.document.text)
            if phrase_match:
                local_score = min(1.0, local_score + 0.06)
            matched_tags = sorted(normalized_filters.risk_tags.intersection(summary.risk_tags))
            display_text = best_hit.document.text
            provenance: dict[str, Any] = {}
            pre_ranked.append(
                (
                    local_score,
                    rrf_score,
                    case_id,
                    MatchedCase(
                        case_id=case_id,
                        document_kind=summary.document_kind,
                        decision_status=summary.decision_status,
                        source_group=summary.source_group,
                        file_id=summary.document_id
                        if summary.document_id and summary.document_id.startswith("F")
                        else None,
                        case_title=summary.case_title,
                        penalty_doc_no=summary.penalty_doc_no,
                        violation_behavior=summary.violation_behavior,
                        penalty_content=summary.penalty_content,
                        regulator=summary.regulator,
                        party_name=summary.party_name,
                        institution_type=summary.institution_type,
                        penalty_date=summary.penalty_date,
                        source_url=summary.source_url,
                        risk_tags=summary.risk_tags,
                        score=round(local_score, 6),
                        bm25_score=round(bm25_score, 6),
                        vector_score=round(vector_score, 6),
                        rrf_score=round(rrf_score, 6),
                        rerank_score=round(local_score, 6),
                        dense_score=round(dense_scores.get(case_id, 0.0), 6),
                        type_bridge_score=round(bridge_score, 6),
                        retrieval_channel=_channel_label(
                            diagnostics.vector_channel, bool(bridge_item), bool(dense_scores)
                        ),
                        matched_risk_tags=matched_tags,
                        matched_text=display_text,
                        matched_source=best_hit.document.source,
                        matched_unit_id=best_hit.document.unit_id,
                        match_reason=_match_reason(
                            matched_tags=matched_tags,
                            phrase_match=phrase_match,
                            source=best_hit.document.source,
                            filters=normalized_filters,
                            bridge_hit=bridge_item is not None,
                            dense_hit=case_id in dense_scores,
                        ),
                        **provenance,
                    ),
                )
            )

        if rerank:
            final = self._apply_rerank(query, pre_ranked, top_k, diagnostics)
        else:
            pre_ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
            pre_ranked = self._distinct_source_documents(pre_ranked)
            final = [row[3] for row in pre_ranked[:top_k]]
        self.last_diagnostics = diagnostics
        return final

    def _verify_matches(self, matches: list[MatchedCase], query: str = "") -> list[MatchedCase]:
        """Verify only selected evidence, after recall/fusion/reranking, never every candidate."""
        results = []
        for match in matches:
            document = self._documents_by_id.get(match.matched_unit_id)
            if document is None:
                results.append(match)
                continue
            text, provenance = self._hit_provenance(SparseSearchHit(document, match.score, 1))
            results.append(
                match.model_copy(
                    update={
                        "matched_text": text,
                        **provenance,
                        "match_reason": business_match_reason(
                            query,
                            match.violation_behavior,
                            text,
                            match.matched_risk_tags or match.risk_tags,
                        ),
                    }
                )
            )
        return results

    def _search_local_only(
        self,
        query: str,
        *,
        top_k: int,
        risk_tags: set[str] | None,
        filters: CaseSearchFilters | None,
        error: BaseException,
    ) -> list[MatchedCase]:
        """纯本地兜底：只用 BM25 与倒排表 TF-IDF，不触碰任何云端客户端。

        触发条件是 :meth:`_search` 抛出了未被通道级 ``except`` 覆盖的异常（例如
        云端客户端构造期的编程错误）。此时仍必须给出可解释结果，而不是让整条
        Query 分析失败，并把原因写进诊断供排查。
        """

        normalized_filters = _normalized_filters(filters, risk_tags)
        index = self._ensure_index()
        recall_size = max(120, top_k * 40)
        bm25 = self._aggregate_cases(index.search(query, top_k=recall_size), normalized_filters)
        vector = self._aggregate_cases(
            index.vector_search(query, top_k=recall_size), normalized_filters
        )
        bm25_ranks = self._ranks(bm25)
        vector_ranks = self._ranks(vector)
        query_tokens = _meaningful_tokens(query)

        ranked: list[tuple[float, float, str, MatchedCase]] = []
        for case_id in set(bm25) | set(vector):
            summary = self._summaries.get(case_id)
            if summary is None:
                continue
            bm25_item = bm25.get(case_id)
            vector_item = vector.get(case_id)
            best_hit = _choose_hit(bm25_item, vector_item)
            bm25_score = _normalize_bm25(bm25_item.score) if bm25_item else 0.0
            vector_score = min(1.0, vector_item.score) if vector_item else 0.0
            rrf_raw = ((1 / (_RRF_K + bm25_ranks[case_id])) if case_id in bm25_ranks else 0.0) + (
                (1 / (_RRF_K + vector_ranks[case_id])) if case_id in vector_ranks else 0.0
            )
            rrf_score = min(1.0, rrf_raw / (2 / (_RRF_K + 1)))
            phrase_match = _contains_phrase(query, best_hit.document.text)
            local_score = _weighted_score(
                rrf_score=rrf_score,
                bm25_score=bm25_score,
                vector_score=vector_score,
                lexical_score=_token_overlap(query_tokens, best_hit.document.text),
                source_score=_source_score(best_hit.document.source),
                bridge_score=0.0,
            )
            if phrase_match:
                local_score = min(1.0, local_score + 0.06)
            ranked.append(
                (
                    local_score,
                    rrf_score,
                    case_id,
                    MatchedCase(
                        case_id=case_id,
                        document_kind=summary.document_kind,
                        decision_status=summary.decision_status,
                        source_group=summary.source_group,
                        case_title=summary.case_title,
                        penalty_doc_no=summary.penalty_doc_no,
                        violation_behavior=summary.violation_behavior,
                        penalty_content=summary.penalty_content,
                        regulator=summary.regulator,
                        party_name=summary.party_name,
                        institution_type=summary.institution_type,
                        penalty_date=summary.penalty_date,
                        source_url=summary.source_url,
                        risk_tags=summary.risk_tags,
                        score=round(local_score, 6),
                        bm25_score=round(bm25_score, 6),
                        vector_score=round(vector_score, 6),
                        rrf_score=round(rrf_score, 6),
                        rerank_score=round(local_score, 6),
                        retrieval_channel="bm25+tfidf_local_fallback",
                        matched_risk_tags=sorted(
                            normalized_filters.risk_tags.intersection(summary.risk_tags)
                        ),
                        matched_text=best_hit.document.text,
                        matched_source=best_hit.document.source,
                        matched_unit_id=best_hit.document.unit_id,
                        match_reason=_match_reason(
                            matched_tags=sorted(
                                normalized_filters.risk_tags.intersection(summary.risk_tags)
                            ),
                            phrase_match=phrase_match,
                            source=best_hit.document.source,
                            filters=normalized_filters,
                        ),
                    ),
                )
            )

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        self.last_diagnostics = RetrievalDiagnostics(
            vector_channel=_LOCAL_VECTOR_CHANNEL,
            rerank_channel="weighted_local",
            bridge_enabled=False,
            warnings=[f"检索已降级为纯本地模式：{_reason(error)}"],
        )
        return [item[3] for item in ranked[:top_k]]

    def _vector_channel(
        self,
        query: str,
        index: SparseRetrievalIndex,
        filters: CaseSearchFilters,
        recall_size: int,
        diagnostics: RetrievalDiagnostics,
    ) -> tuple[dict[str, _ChannelAggregate], dict[str, float]]:
        """云端稠密语义召回；不可用时降级本地 TF-IDF 并记录告警。"""

        dense_scores: dict[str, float] = {}
        if self._embedding_provider in {"cloud", "bge"} and not self._dense_failed:
            try:
                dense_index = self._ensure_dense_index()
            except (CloudRetrievalError, RuntimeError, OSError, ValueError) as error:
                self._dense_failed = True
                self._dense = None
                diagnostics.vector_channel = _LOCAL_VECTOR_CHANNEL
                diagnostics.warnings.append(
                    f"云端向量通道不可用，已降级本地 TF-IDF：{_reason(error)}"
                )
            else:
                try:
                    hits = dense_index.search(query, top_k=recall_size)
                except (CloudRetrievalError, RuntimeError, OSError, ValueError) as error:
                    diagnostics.vector_channel = _LOCAL_VECTOR_CHANNEL
                    diagnostics.warnings.append(
                        f"云端向量检索失败，已降级本地 TF-IDF：{_reason(error)}"
                    )
                else:
                    diagnostics.vector_channel = (
                        "local_bge_cosine"
                        if self._embedding_provider == "bge"
                        else _DENSE_VECTOR_CHANNEL
                    )
                    aggregates = self._aggregate_cases(hits, filters)
                    for case_id, aggregate in aggregates.items():
                        dense_scores[case_id] = aggregate.best_hit.score
                    return aggregates, dense_scores

        local_hits = index.vector_search(query, top_k=recall_size)
        diagnostics.vector_channel = _LOCAL_VECTOR_CHANNEL
        return self._aggregate_cases(local_hits, filters), dense_scores

    def _bridge_channel(
        self,
        query: str,
        filters: CaseSearchFilters,
        diagnostics: RetrievalDiagnostics,
    ) -> dict[str, _ChannelAggregate]:
        """类型桥接召回；无标签、未启用或云端不可用时返回空。"""

        if not self._bridge_enabled or not filters.risk_tags:
            return {}
        bridge = self._bridge_recall()
        if bridge is None:
            return {}
        try:
            hits = bridge.recall(query, filters.risk_tags, self._summaries.values())
        except (CloudRetrievalError, RuntimeError, OSError, ValueError) as error:
            diagnostics.warnings.append(f"类型桥接召回不可用，已跳过：{_reason(error)}")
            return {}
        stats = bridge.last_stats
        diagnostics.bridge_enabled = bool(hits)
        diagnostics.bridge_candidates = stats.candidate_cases
        return self._aggregate_cases(hits, filters)

    def _apply_rerank(
        self,
        query: str,
        pre_ranked: list[tuple[float, float, str, MatchedCase]],
        top_k: int,
        diagnostics: RetrievalDiagnostics,
    ) -> list[MatchedCase]:
        """云端 Cross-Encoder 精排；不可用时沿用确定性加权分。"""

        pre_ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        pre_ranked = self._distinct_source_documents(pre_ranked)
        if self._rerank_provider not in {"cloud", "deepseek"}:
            diagnostics.rerank_channel = "weighted_local"
            return [item[3] for item in pre_ranked[:top_k]]
        reranker = self._rerank_client()
        if reranker is None or isinstance(reranker, NullRerankClient) or not pre_ranked:
            diagnostics.rerank_channel = "weighted_local"
            return [item[3] for item in pre_ranked[:top_k]]

        window = pre_ranked[: max(top_k, min(len(pre_ranked), _RERANK_WINDOW))]
        verified = self._verify_matches([item[3] for item in window])
        window = [
            (score, rrf, cid, match)
            for (score, rrf, cid, _), match in zip(window, verified, strict=True)
        ]
        documents = [item[3].matched_text for item in window]
        try:
            ordered = reranker.rerank(query, documents, top_n=top_k)
        except (CloudRetrievalError, RuntimeError, OSError, ValueError) as error:
            diagnostics.rerank_channel = "weighted_local"
            diagnostics.warnings.append(f"云端重排不可用，已沿用本地加权分：{_reason(error)}")
            return [item[3] for item in pre_ranked[:top_k]]

        channel = "deepseek" if self._rerank_provider == "deepseek" else "cloud"
        diagnostics.rerank_channel = f"{channel}:{reranker.model_name}"
        results: list[MatchedCase] = []
        seen: set[str] = set()
        for position, relevance in ordered:
            if not 0 <= position < len(window):
                continue
            local_score, rrf_score, case_id, matched = window[position]
            if case_id in seen:
                continue
            seen.add(case_id)
            # 云端相关分为主，本地分作次级排序依据，保证同分时结果可复现。
            results.append(
                matched.model_copy(
                    update={
                        "rerank_score": round(relevance, 6),
                        "score": round(_blend_rerank(relevance, local_score), 6),
                    }
                )
            )
        # 精排窗口外的候选按本地分补位，避免 top_k 被窗口截断。
        for local_score, rrf_score, case_id, matched in pre_ranked:
            if len(results) >= top_k or case_id in seen:
                continue
            seen.add(case_id)
            results.append(matched)
        if self._rerank_provider != "deepseek":
            results.sort(key=lambda item: (-item.score, -item.rrf_score, item.case_id))
        return results[:top_k]

    def _distinct_source_documents(
        self,
        values: list[tuple[float, float, str, MatchedCase]],
    ) -> list[tuple[float, float, str, MatchedCase]]:
        if not bool(getattr(self.settings, "full_corpus", False)):
            return values
        seen: set[str] = set()
        result = []
        for row in values:
            key = row[3].file_id or row[2]
            if key in seen:
                continue
            seen.add(key)
            result.append(row)
        return result

    def rebuild(self) -> None:
        """重建共享倒排表以及 BM25/向量统计。"""

        with self._lock:
            self._summaries = self._load_summaries()
            self._index = SparseRetrievalIndex(self.repository.iter_retrieval_documents())
            self._documents_by_id = {d.unit_id: d for d in self._index.documents}
            self._indexed_revision = self.repository.revision
            self._dense = None
            self._dense_failed = False

    def _ensure_index(self) -> SparseRetrievalIndex:
        if self._index is not None and self._indexed_revision == self.repository.revision:
            return self._index
        self.rebuild()
        if self._index is None:  # pragma: no cover
            raise RuntimeError("案例索引初始化失败")
        return self._index

    def _load_summaries(self) -> dict[str, CaseSummary]:
        loader = getattr(self.repository, "retrieval_summaries", self.repository.summaries)
        return {item.case_id: item for item in loader()}

    def _aggregate_cases(
        self,
        hits: tuple[SparseSearchHit, ...],
        filters: CaseSearchFilters,
    ) -> dict[str, _ChannelAggregate]:
        aggregates: dict[str, _ChannelAggregate] = {}
        for hit in hits:
            for case_id in hit.document.case_ids:
                summary = self._summaries.get(case_id)
                if summary is None or not _matches_filters(summary, filters):
                    continue
                current = aggregates.get(case_id)
                if current is None:
                    aggregates[case_id] = _ChannelAggregate(hit.score, hit)
                    continue
                current.score = max(current.score, hit.score)
                if hit.score > current.best_hit.score:
                    current.best_hit = hit
        return aggregates

    @staticmethod
    def _ranks(values: dict[str, _ChannelAggregate]) -> dict[str, int]:
        ordered = sorted(values.items(), key=lambda item: (-item[1].score, item[0]))
        return {case_id: rank for rank, (case_id, _) in enumerate(ordered, start=1)}

    # ---------- 云端客户端惰性装配 ----------

    def _ensure_dense_index(self, *, allow_build: bool = False) -> DenseRetrievalIndex:
        """构建（或复用）云端稠密索引；revision 变化时随稀疏索引一起失效。"""

        if self._dense is not None and self._dense.ready:
            return self._dense
        with self._lock:
            return self._build_dense_index(allow_build=allow_build)

    def _build_dense_index(self, *, allow_build: bool = False) -> DenseRetrievalIndex:
        if self._dense is not None and self._dense.ready:
            return self._dense
        if self._index is None:
            raise RuntimeError("稀疏索引未就绪，无法构建稠密索引")
        documents = tuple(self._index.documents)
        if not documents:
            raise RuntimeError("案例库无可用检索单元")
        embedding = self._embedding_client()
        if embedding is None:
            raise CloudRetrievalError("未配置云端 Embedding 客户端")
        if self._embedding_provider == "bge":
            from retrieval.local_dense import LocalBGEIndex
            from retrieval.local_embedding import LocalBGEEmbeddingClient

            directory = getattr(self.settings, "dense_cache_dir", None) or Path("data/cache/dense")
            local_dense = LocalBGEIndex(
                documents,
                embedding=cast(LocalBGEEmbeddingClient, embedding),
                directory=Path(directory),
                allow_build=allow_build,
            )
            self._dense_stats = local_dense.build()
            self._dense = local_dense
            return local_dense
        from retrieval.qwen import NativeQwenEmbeddingClient
        from retrieval.qwen_dense import QwenDenseIndex

        if isinstance(embedding, NativeQwenEmbeddingClient):
            directory = getattr(self.settings, "dense_cache_dir", None) or Path("data/cache/dense")
            qwen_dense = QwenDenseIndex(
                documents,
                embedding=embedding,
                directory=Path(directory),
                allow_build=allow_build,
                concurrency=int(getattr(self.settings, "cloud_concurrency", 4)),
            )
            self._dense_stats = qwen_dense.build()
            self._dense = qwen_dense
            return qwen_dense
        cache = None
        cache_dir = getattr(self.settings, "dense_cache_dir", None) if self.settings else None
        if cache_dir is not None:
            cache = DenseVectorCache(Path(cache_dir), embedding.model_name, embedding.dimension)
        dense = DenseRetrievalIndex(documents, embedding=embedding, cache=cache)
        self._dense_stats = dense.build()
        self._dense = dense
        return dense

    def _embedding_client(self) -> EmbeddingProvider | None:
        if self._embedding is not None:
            return self._embedding
        if self.settings is None:
            return None
        if self._embedding_provider == "bge":
            from retrieval.local_embedding import LocalBGEEmbeddingClient

            directory = getattr(self.settings, "bge_model_dir", None)
            if directory is None:
                directory = Path(__file__).resolve().parents[2] / "models/bge-small-zh-v1.5"
            self._embedding = LocalBGEEmbeddingClient(Path(directory))
            return self._embedding
        cloud = build_cloud_settings(self.settings)
        from retrieval.qwen import NativeQwenEmbeddingClient

        client = NativeQwenEmbeddingClient(cloud)
        self._embedding = client
        return self._embedding

    def warmup(self, *, allow_build: bool = False) -> None:
        self._ensure_index()
        if self._embedding_provider in {"bge", "cloud"}:
            self._dense_failed = False
            self._ensure_dense_index(allow_build=allow_build)
            embedding = self._embedding_client()
            if embedding is not None:
                embedding.embed_query("保险营销合规核查")
            self.last_diagnostics = RetrievalDiagnostics(
                vector_channel="local_bge_cosine"
                if self._embedding_provider == "bge"
                else "dense_cloud"
            )

    def _hit_provenance(self, hit: SparseSearchHit) -> tuple[str, dict[str, Any]]:
        metadata_reader = getattr(self.repository, "retrieval_metadata", None)
        metadata = metadata_reader(hit.document.unit_id) if callable(metadata_reader) else {}
        root = getattr(self.settings, "case_kb_root", None)
        if root and (Path(root) / "evidence_registry.sqlite3").is_file():
            if self._evidence_reader is None:
                from application.evidence_service import KnowledgeEvidenceService

                self._evidence_reader = KnowledgeEvidenceService(
                    Path(root), getattr(self.settings, "additional_evidence_databases", ())
                )
            source_span = metadata.get("source_span") or {}
            grounding = metadata.get("grounding_ids") or []
            identifier = (
                grounding[0] if grounding else source_span.get("evidence_id", hit.document.unit_id)
            )
            try:
                evidence = self._evidence_reader.read(identifier)
                segments = evidence["segments"]
                if segments:
                    return segments[0]["text"], {
                        "file_id": evidence.get("file_id"),
                        "evidence_ids": [identifier],
                        "source_sha256": evidence["source_sha256"],
                        "source_spans": [{"start": s["start"], "end": s["end"]} for s in segments],
                        "evidence_origin": "source_verbatim",
                        "literal_verified": True,
                        "knowledge_version": self._evidence_reader.versions["content"],
                    }
            except (ValueError, KeyError):
                pass
        return hit.document.text, {
            "evidence_origin": "generated_retrieval_aid"
            if hit.document.source == "case_expansion"
            else "unverified_case_chunk",
            "literal_verified": False,
        }

    def _rerank_client(self) -> RerankProvider | None:
        if self._reranker is not None:
            return self._reranker
        if self.settings is None:
            return NullRerankClient()
        if self._rerank_provider == "deepseek":
            from retrieval.deepseek import DeepSeekRerankClient

            self._reranker = DeepSeekRerankClient(
                model=str(getattr(self.settings, "deepseek_rerank_model", "deepseek-flash")),
                timeout_seconds=float(
                    getattr(self.settings, "deepseek_rerank_timeout_seconds", 20)
                ),
                concurrency=int(getattr(self.settings, "deepseek_rerank_concurrency", 2)),
            )
            return self._reranker
        client = DashScopeRerankClient(build_cloud_settings(self.settings))
        self._reranker = client
        return self._reranker

    def _bridge_recall(self) -> TypeBridgeRecall | None:
        if self._bridge is not None:
            return self._bridge
        embedding = self._embedding_client()
        if embedding is None:
            return None
        cache_dir = getattr(self.settings, "dense_cache_dir", None) if self.settings else None
        cache_path = (
            Path(cache_dir) / f"bridge-{embedding.model_name}-{embedding.dimension}.jsonl"
            if cache_dir is not None
            else Path("data/cache/dense")
            / f"bridge-{embedding.model_name}-{embedding.dimension}.jsonl"
        )
        top_k = int(getattr(self.settings, "type_bridge_top_k", 3) or 3)
        batch = int(getattr(self.settings, "cloud_batch_size", 20) or 20)
        bridge = TypeBridgeRecall(
            embedding=embedding, cache_path=cache_path, top_k=top_k, batch_size=batch
        )
        self._bridge = bridge
        return self._bridge

    def close(self) -> None:
        """释放云端 HTTP 连接；重复调用安全。"""

        for client in (self._embedding, self._reranker):
            close_method = getattr(client, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:  # pragma: no cover - 关闭失败不应影响主流程
                    pass


def _normalized_filters(
    filters: CaseSearchFilters | None,
    risk_tags: set[str] | None,
) -> CaseSearchFilters:
    current = filters or CaseSearchFilters()
    merged_tags = {
        tag for tag in {*current.risk_tags, *(risk_tags or set())} if tag and tag != "其他"
    }
    if current.penalty_date_from and current.penalty_date_to:
        if current.penalty_date_from > current.penalty_date_to:
            raise ValueError("处罚日期起始值不得晚于结束值")
    return replace(
        current,
        risk_tags=frozenset(merged_tags),
        regulator=_clean_filter(current.regulator),
        party_name=_clean_filter(current.party_name),
        institution_type=_clean_filter(current.institution_type),
    )


def _matches_filters(summary: CaseSummary, filters: CaseSearchFilters) -> bool:
    if filters.risk_tags and not filters.risk_tags.intersection(summary.risk_tags):
        return False
    if not _contains_filter(summary.regulator, filters.regulator):
        return False
    if not _contains_filter(summary.party_name, filters.party_name):
        return False
    if not _contains_filter(summary.institution_type, filters.institution_type):
        return False
    if filters.penalty_date_from:
        if summary.penalty_date is None or summary.penalty_date < filters.penalty_date_from:
            return False
    if filters.penalty_date_to:
        if summary.penalty_date is None or summary.penalty_date > filters.penalty_date_to:
            return False
    return True


def _clean_filter(value: str | None) -> str | None:
    cleaned = (value or "").strip().casefold()
    return cleaned or None


def _contains_filter(value: str | None, needle: str | None) -> bool:
    return needle is None or needle in (value or "").casefold()


def _match_reason(
    *,
    matched_tags: list[str],
    phrase_match: bool,
    source: str,
    filters: CaseSearchFilters,
    bridge_hit: bool = False,
    dense_hit: bool = False,
) -> str:
    reasons: list[str] = []
    if matched_tags:
        reasons.append(f"风险标签一致：{'、'.join(matched_tags)}")
    if phrase_match:
        reasons.append("查询与案例证据存在连续原文匹配")
    elif bridge_hit:
        reasons.append("同风险标签案例经云端语义桥接召回")
    elif dense_hit:
        reasons.append("云端稠密语义召回并经 RRF 融合重排")
    else:
        reasons.append("BM25 与向量双路召回后经 RRF 重排")
    if source == "case_chunk":
        reasons.append("命中可追溯案例证据切片")
    metadata_count = sum(
        value is not None
        for value in (
            filters.regulator,
            filters.party_name,
            filters.institution_type,
            filters.penalty_date_from,
            filters.penalty_date_to,
        )
    )
    if metadata_count:
        reasons.append(f"满足 {metadata_count} 项主体/监管/时间过滤条件")
    return "；".join(reasons)


def _choose_hit(
    bm25: _ChannelAggregate | None,
    vector: _ChannelAggregate | None,
    bridge: _ChannelAggregate | None = None,
) -> SparseSearchHit:
    """挑选可追溯性最好的命中作为证据切片。

    优先级：证据切片(case_chunk) > 扩展表达 > 桥接摘要。同类型下取分数更高者，
    保证 ``matched_text`` 始终尽可能回指处罚原文。
    """

    candidates = [item for item in (bm25, vector, bridge) if item is not None]
    if not candidates:  # pragma: no cover
        raise RuntimeError("混合召回候选缺少来源命中")

    def priority(aggregate: _ChannelAggregate) -> tuple[int, float]:
        source = aggregate.best_hit.document.source
        rank = {"case_chunk": 2, "case_expansion": 1}.get(source, 0)
        strength = (
            _normalize_bm25(aggregate.best_hit.score)
            if aggregate is bm25
            else aggregate.best_hit.score
        )
        return (rank, strength)

    return max(candidates, key=priority).best_hit


def _source_score(source: str) -> float:
    """证据来源可信度：可追溯切片最高，桥接摘要最低。"""

    if source == "case_chunk":
        return 1.0
    if source == "case_expansion":
        return 0.65
    return 0.45


def _weighted_score(
    *,
    rrf_score: float,
    bm25_score: float,
    vector_score: float,
    lexical_score: float,
    source_score: float,
    bridge_score: float,
) -> float:
    """确定性加权重排分（云端重排不可用时的排序依据）。

    权重在原有五项基础上让出 0.05 给桥接分，其余比例保持不变，确保关闭云端
    时与原实现的排序结论一致（桥接未启用时 bridge_score 恒为 0）。
    """

    score = (
        0.26 * rrf_score
        + 0.29 * bm25_score
        + 0.25 * vector_score
        + 0.10 * lexical_score
        + 0.05 * source_score
        + 0.05 * bridge_score
    )
    return min(1.0, score)


def _blend_rerank(relevance: float, local_score: float) -> float:
    """融合云端相关分与本地加权分，云端为主、本地兜底。"""

    return min(1.0, max(0.0, 0.78 * relevance + 0.22 * local_score))


def _channel_label(vector_channel: str, bridge_hit: bool, dense_hit: bool) -> str:
    """生成可读的召回通道标记，供评测归因与前端展示。"""

    parts = ["bm25", "bridge" if bridge_hit else vector_channel]
    if dense_hit and not bridge_hit:
        parts.append("dense")
    return "+".join(parts)


def _reason(error: BaseException) -> str:
    """压缩异常摘要，避免把长响应体或密钥线索写入告警。"""

    text = " ".join(str(error).split())
    return text[:160] or type(error).__name__


def _provider_value(
    settings: object | None,
    attribute: str,
    env_name: str,
    default: str,
) -> str:
    """通道提供方取值：settings 优先，其次环境变量，最后默认值。"""

    if settings is not None:
        value = getattr(settings, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip():
        return env_value.strip().lower()
    return default


def _bool_value(
    settings: object | None,
    attribute: str,
    env_name: str,
    default: bool,
) -> bool:
    """布尔开关取值：settings 优先，其次环境变量，最后默认值。"""

    if settings is not None:
        value = getattr(settings, attribute, None)
        if isinstance(value, bool):
            return value
    env_value = os.getenv(env_name)
    if env_value is None:
        return default
    return env_value.strip().lower() not in {"0", "false", "no", "off", ""}


@lru_cache(maxsize=32768)
def _meaningful_tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in chinese_unigram_bigram(text) if len(token) >= 2)


def _token_overlap(query_tokens: frozenset[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    document_tokens = _meaningful_tokens(text)
    return len(query_tokens.intersection(document_tokens)) / len(query_tokens)


def _contains_phrase(query: str, text: str) -> bool:
    compact_query = "".join(query.split())
    compact_text = "".join(text)
    if min(len(compact_query), len(compact_text)) < 6:
        return False
    return compact_query in compact_text or compact_text in compact_query


def _normalize_bm25(score: float) -> float:
    if score <= 0:
        return 0.0
    return 1 - math.exp(-score / 18.0)


__all__ = ["CaseRetrievalService", "CaseSearchFilters", "RetrievalDiagnostics"]
