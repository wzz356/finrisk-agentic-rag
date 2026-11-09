"""稠密语义向量索引：云端 Embedding + numpy 内积检索。

不引入 faiss：当前案例库进入索引的检索单元约 1.5 万条，1024 维归一化矩阵
一次 ``matmul`` 即毫秒级，用 numpy 足够且零新增依赖。

与 :class:`retrieval.sparse.SparseRetrievalIndex` 返回同构的命中对象，
使 ``CaseRetrievalService._aggregate_cases()`` 无需区分通道即可复用。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import cast

import numpy as np
from numpy.typing import NDArray

from retrieval.cloud import (
    CloudRetrievalError,
    DenseVectorCache,
    EmbeddingProvider,
    text_signature,
)
from retrieval.sparse import RetrievalDocument, SparseSearchHit


@dataclass(slots=True)
class DenseBuildStats:
    """一次索引构建的统计，用于日志与评测归因。"""

    total_units: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    api_documents: int = 0
    elapsed_seconds: float = 0.0
    dimension: int = 0
    model: str = ""

    @property
    def cache_hit_ratio(self) -> float:
        if not self.total_units:
            return 0.0
        return self.cache_hits / self.total_units


class DenseRetrievalIndex:
    """云端向量驱动的稠密召回通道。

    构建流程：读取磁盘缓存 → 只为新增或变更的单元请求 Embedding →
    落盘缓存 → 组装归一化矩阵。任一步云端失败抛
    :class:`CloudRetrievalError`，由调用方降级到本地 TF-IDF 通道。
    """

    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        *,
        embedding: EmbeddingProvider,
        cache: DenseVectorCache | None = None,
    ) -> None:
        if not documents:
            raise ValueError("稠密索引需要至少一条检索单元")
        self.documents = tuple(documents)
        self._weights = np.asarray(
            [math.sqrt(max(0.0, min(1.0, document.weight))) for document in self.documents],
            dtype=np.float32,
        )
        self._embedding = embedding
        self._cache = cache
        self._lock = RLock()
        self._matrix: NDArray[np.float32] | None = None
        self._stats = DenseBuildStats()

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    @property
    def stats(self) -> DenseBuildStats:
        return self._stats

    @property
    def model_name(self) -> str:
        return self._embedding.model_name

    def build(self) -> DenseBuildStats:
        """构建或刷新向量矩阵；线程安全，可重复调用。"""

        with self._lock:
            if self._matrix is not None:
                return self._stats
            cached = self._cache.load() if self._cache is not None else {}
            signatures = [text_signature(document.text) for document in self.documents]
            needed: list[int] = []
            matrix_rows: list[list[float] | None] = [None] * len(self.documents)
            for position, (document, signature) in enumerate(zip(self.documents, signatures)):
                entry = cached.get(document.unit_id)
                if entry is not None and entry[0] == signature:
                    matrix_rows[position] = entry[1]
                else:
                    needed.append(position)

            stats = DenseBuildStats(
                total_units=len(self.documents),
                cache_hits=len(self.documents) - len(needed),
                dimension=self._embedding.dimension,
                model=self._embedding.model_name,
            )
            if needed:
                fetched = self._fetch(needed, stats)
                for position, vector in zip(needed, fetched):
                    matrix_rows[position] = vector

            missing = [index for index, row in enumerate(matrix_rows) if row is None]
            if missing:
                raise CloudRetrievalError(f"稠密索引构建不完整：{len(missing)} 条单元未取得向量")

            matrix = np.asarray(matrix_rows, dtype=np.float32)
            expected = self._embedding.dimension
            if matrix.ndim != 2 or matrix.shape[1] != expected:
                raise CloudRetrievalError(f"稠密索引维度异常：期望 {expected}，实际 {matrix.shape}")
            self._matrix = _l2_normalize(matrix)
            self._stats = stats
            self._persist(matrix_rows, signatures)
            return stats

    def _fetch(self, positions: list[int], stats: DenseBuildStats) -> list[list[float]]:
        """按服务端批次上限分批请求 Embedding。"""

        batch_size = getattr(self._embedding, "batch_size", 20) or 20
        batch_size = max(1, min(int(batch_size), 20))
        vectors: list[list[float]] = []
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            texts = [self.documents[position].text for position in chunk]
            batch_vectors = self._embedding.embed_documents(texts)
            if len(batch_vectors) != len(chunk):
                raise CloudRetrievalError(
                    f"Embedding 返回条数不匹配：请求 {len(chunk)}，返回 {len(batch_vectors)}"
                )
            vectors.extend(batch_vectors)
            stats.api_calls += 1
            stats.api_documents += len(chunk)
        return vectors

    def _persist(self, rows: Iterable[list[float] | None], signatures: Sequence[str]) -> None:
        if self._cache is None:
            return
        entries = []
        for document, signature, row in zip(self.documents, signatures, rows):
            if row is None:
                continue
            entries.append((document.unit_id, signature, row))
        self._cache.save(entries)

    def search(self, query: str, *, top_k: int = 50) -> tuple[SparseSearchHit, ...]:
        """归一化余弦召回；返回结构与稀疏通道一致。"""

        if not query.strip():
            raise ValueError("检索文字不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        with self._lock:
            matrix = self._matrix
        if matrix is None:
            raise CloudRetrievalError("稠密索引尚未构建")

        similarities = self._query_similarities(query, matrix)

        # 与稀疏向量通道保持一致：按 sqrt(weight) 折算证据切片与扩展表达的优先级。
        scores = similarities * self._weights
        scores = np.clip(scores, 0.0, 1.0)

        limit = int(min(top_k, len(self.documents)))
        # 取正分候选；argpartition 后再排序，避免全量排序开销。
        positive_positions = np.flatnonzero(scores > 0.0)
        if positive_positions.size == 0:
            return ()
        if positive_positions.size > limit:
            candidate_positions = positive_positions[
                np.argpartition(-scores[positive_positions], limit - 1)[:limit]
            ]
        else:
            candidate_positions = positive_positions
        candidate_scores = scores[candidate_positions]
        # 分数降序；同分时按文档下标升序，保证结果可复现。
        ordered = candidate_positions[np.lexsort((candidate_positions, -candidate_scores))]

        return tuple(
            SparseSearchHit(
                document=self.documents[position],
                score=float(scores[position]),
                rank=rank,
            )
            for rank, position in enumerate(ordered, start=1)
        )

    def _query_similarities(
        self,
        query: str,
        matrix: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        query_vector = np.asarray(self._embedding.embed_query(query), dtype=np.float32).reshape(
            1, -1
        )
        if query_vector.shape[1] != matrix.shape[1]:
            raise CloudRetrievalError("查询向量与索引维度不一致")
        return cast(NDArray[np.float32], (matrix @ _l2_normalize(query_vector).T).ravel())

    def similarity_matrix(self, query: str, texts: Sequence[str]) -> list[float]:
        """对给定文本批量算与 query 的余弦相似度（type_bridge 用）。"""

        if not texts:
            return []
        vectors = self._embedding.embed_documents(list(texts))
        query_vector: NDArray[np.float32] = np.asarray(
            self._embedding.embed_query(query), dtype=np.float32
        ).reshape(1, -1)
        query_vector = _l2_normalize(query_vector)
        matrix = _l2_normalize(np.asarray(vectors, dtype=np.float32))
        return [float(value) for value in (matrix @ query_vector.T).ravel()]


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """按行做 L2 归一化；零向量保持为零，避免除零产生 NaN。"""

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return cast(NDArray[np.float32], (matrix / safe).astype(np.float32))


__all__ = ["DenseBuildStats", "DenseRetrievalIndex"]
