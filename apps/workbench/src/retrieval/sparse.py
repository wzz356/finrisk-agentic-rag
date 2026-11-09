"""面向当前案例库的内存 BM25 索引。"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from retrieval.tokenization import chinese_unigram_bigram


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    """参与检索的案例切片或扩展表达。"""

    unit_id: str
    case_ids: tuple[str, ...]
    text: str
    source: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class SparseSearchHit:
    document: RetrievalDocument
    score: float
    rank: int


class SparseRetrievalIndex:
    """倒排表驱动的 BM25，只为包含 Query 词的文档计分。"""

    def __init__(self, documents: Iterable[RetrievalDocument], *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 参数不合法")
        self.k1 = k1
        self.b = b
        self.documents = tuple(documents)
        self._weights = np.asarray([document.weight for document in self.documents])
        self._document_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._vector_idf: dict[str, float] = {}
        self._vector_norms: list[float] = []
        self._bm25_arrays: dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]] = {}
        self._build()

    def _build(self) -> None:
        seen_ids: set[str] = set()
        total_length = 0
        for document_index, document in enumerate(self.documents):
            if not document.unit_id or document.unit_id in seen_ids:
                raise ValueError(f"重复或为空的检索单元 ID: {document.unit_id}")
            if not document.case_ids or not document.text.strip():
                raise ValueError(f"检索单元缺少案例或正文: {document.unit_id}")
            if not 0 < document.weight <= 1:
                raise ValueError(f"检索权重不合法: {document.unit_id}")
            seen_ids.add(document.unit_id)
            frequencies = Counter(chinese_unigram_bigram(document.text))
            length = sum(frequencies.values())
            if length == 0:
                raise ValueError(f"检索单元无法分词: {document.unit_id}")
            self._document_lengths.append(length)
            total_length += length
            for token, frequency in frequencies.items():
                self._postings[token].append((document_index, frequency))
        self._average_document_length = (
            total_length / len(self.documents) if self.documents else 0.0
        )
        self._build_vector_statistics()
        self._build_bm25_arrays()

    def _build_bm25_arrays(self) -> None:
        """Precompute identical BM25 contributions, removing Python loops over every posting."""
        count = len(self.documents)
        if not count:
            return
        lengths = np.asarray(self._document_lengths, dtype=np.float64)
        for term, postings in self._postings.items():
            positions = np.asarray([p[0] for p in postings], dtype=np.int64)
            frequencies = np.asarray([p[1] for p in postings], dtype=np.float64)
            idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            denominator = frequencies + self.k1 * (
                1 - self.b + self.b * lengths[positions] / self._average_document_length
            )
            contribution = idf * frequencies * (self.k1 + 1) / denominator
            self._bm25_arrays[term] = (positions, contribution)

    def _build_vector_statistics(self) -> None:
        """复用倒排表构建本地 TF-IDF 向量通道，不额外加载模型权重。"""

        document_count = len(self.documents)
        norm_squares = [0.0] * document_count
        if not document_count:
            self._vector_norms = norm_squares
            return
        for term, postings in self._postings.items():
            idf = math.log((document_count + 1) / (len(postings) + 1)) + 1.0
            self._vector_idf[term] = idf
            for document_index, frequency in postings:
                weight = (1.0 + math.log(frequency)) * idf
                norm_squares[document_index] += weight * weight
        self._vector_norms = [math.sqrt(value) for value in norm_squares]

    def search(self, query: str, *, top_k: int = 50) -> tuple[SparseSearchHit, ...]:
        """返回文档级候选；同案例聚合由应用层完成。"""

        if not query.strip():
            raise ValueError("检索文字不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        query_terms = tuple(dict.fromkeys(chinese_unigram_bigram(query)))
        scores = np.zeros(len(self.documents), dtype=np.float64)
        document_count = len(self.documents)
        if not document_count:
            return ()

        for term in query_terms:
            arrays = self._bm25_arrays.get(term)
            if arrays is None:
                continue
            positions, contribution = arrays
            scores[positions] += contribution
        positive = np.flatnonzero(scores > 0)
        weighted = scores[positive] * self._weights[positive]
        ranked = positive[np.lexsort((positive, -weighted))[:top_k]]
        return tuple(
            SparseSearchHit(
                document=self.documents[document_index],
                score=float(scores[document_index] * self.documents[document_index].weight),
                rank=rank,
            )
            for rank, document_index in enumerate(ranked, start=1)
        )

    def vector_search(self, query: str, *, top_k: int = 50) -> tuple[SparseSearchHit, ...]:
        """使用同一中文词元空间执行归一化 TF-IDF 余弦向量召回。"""

        if not query.strip():
            raise ValueError("检索文字不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        frequencies = Counter(chinese_unigram_bigram(query))
        query_weights: dict[str, float] = {}
        query_norm_square = 0.0
        for term, frequency in frequencies.items():
            idf = self._vector_idf.get(term)
            if idf is None:
                continue
            weight = (1.0 + math.log(frequency)) * idf
            query_weights[term] = weight
            query_norm_square += weight * weight
        if not query_weights or query_norm_square <= 0:
            return ()

        dot_products: dict[int, float] = defaultdict(float)
        for term, query_weight in query_weights.items():
            idf = self._vector_idf[term]
            for document_index, frequency in self._postings[term]:
                document_weight = (1.0 + math.log(frequency)) * idf
                dot_products[document_index] += query_weight * document_weight

        query_norm = math.sqrt(query_norm_square)
        cosine_scores = {
            document_index: dot_product
            / (query_norm * self._vector_norms[document_index])
            * math.sqrt(self.documents[document_index].weight)
            for document_index, dot_product in dot_products.items()
            if self._vector_norms[document_index] > 0
        }
        ranked = heapq.nlargest(
            min(top_k, len(cosine_scores)),
            cosine_scores.items(),
            key=lambda item: (item[1], -item[0]),
        )
        return tuple(
            SparseSearchHit(
                document=self.documents[document_index],
                score=max(0.0, min(1.0, score)),
                rank=rank,
            )
            for rank, (document_index, score) in enumerate(ranked, start=1)
        )
