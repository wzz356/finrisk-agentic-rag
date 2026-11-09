"""类型桥接召回：按风险标签召回案例，用云端语义相似度排序。

对齐 risk_src 的「方案 A」动机——处罚文书与营销话术语体差异大，纯语义检索
容易漂移，因此先用风险标签把候选收窄到同标签案例，再在窄候选内比语义。

与 risk_src 的关键差异：risk_src 按 ``case_id`` 编号升序取前 top_k（其注释
承认该策略在 gold 上命中率更高），那是对评测集编号偏置的拟合，不是真实相关
性。这里改为按云端向量相似度排序，gold 口径变化时结论依然成立。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import cast

import numpy as np
from numpy.typing import NDArray

from domain.knowledge_models import CaseSummary
from retrieval.cloud import EmbeddingProvider, text_signature
from retrieval.sparse import RetrievalDocument, SparseSearchHit

BRIDGE_SOURCE = "type_bridge"


@dataclass(slots=True)
class BridgeStats:
    """桥接召回的调用统计，供评测归因与成本核对。"""

    candidate_cases: int = 0
    cached_vectors: int = 0
    api_calls: int = 0
    api_documents: int = 0
    returned: int = 0


class _VectorStore:
    """案例向量的磁盘缓存；以 sha256(文本) 判断失效。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = RLock()
        self._data: dict[str, tuple[str, list[float]]] | None = None

    def load(self) -> dict[str, tuple[str, list[float]]]:
        with self._lock:
            if self._data is not None:
                return self._data
            loaded: dict[str, tuple[str, list[float]]] = {}
            if self._path.is_file():
                try:
                    with self._path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            case_id = item.get("case_id")
                            signature = item.get("signature")
                            vector = item.get("vector")
                            if (
                                isinstance(case_id, str)
                                and isinstance(signature, str)
                                and isinstance(vector, list)
                                and vector
                            ):
                                loaded[case_id] = (signature, [float(v) for v in vector])
                except OSError:
                    loaded = {}
            self._data = loaded
            return loaded

    def save(self, entries: Iterable[tuple[str, str, Sequence[float]]]) -> None:
        with self._lock:
            if self._data is None:
                self.load()
            assert self._data is not None
            for case_id, signature, vector in entries:
                self._data[case_id] = (signature, [float(v) for v in vector])
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._path.with_suffix(".jsonl.tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    for case_id, (signature, vector) in self._data.items():
                        handle.write(
                            json.dumps(
                                {
                                    "case_id": case_id,
                                    "signature": signature,
                                    "vector": [round(value, 6) for value in vector],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                temporary.replace(self._path)
            except OSError:
                return


class TypeBridgeRecall:
    """同标签案例的语义桥接召回。"""

    def __init__(
        self,
        *,
        embedding: EmbeddingProvider,
        cache_path: Path,
        top_k: int = 3,
        candidate_limit: int = 200,
        batch_size: int = 20,
    ) -> None:
        if top_k < 1:
            raise ValueError("桥接 top_k 必须大于 0")
        if candidate_limit < 1:
            raise ValueError("桥接 candidate_limit 必须大于 0")
        self._embedding = embedding
        self._store = _VectorStore(cache_path)
        self._top_k = top_k
        self._candidate_limit = candidate_limit
        self._batch_size = max(1, min(int(batch_size), 20))
        self._last_stats = BridgeStats()

    @property
    def last_stats(self) -> BridgeStats:
        return self._last_stats

    def recall(
        self,
        query: str,
        risk_tags: Iterable[str],
        summaries: Iterable[CaseSummary],
    ) -> tuple[SparseSearchHit, ...]:
        """返回桥接命中；无标签或无候选时返回空元组。

        候选排序优先级：标签命中数降序 → 案例文本长度升序（短文本更聚焦违规
        事实），保证在 ``candidate_limit`` 截断时保留信息密度高的案例。
        """

        tags = {tag for tag in risk_tags if tag and tag != "其他"}
        stats = BridgeStats()
        self._last_stats = stats
        if not tags or not query.strip():
            return ()

        tagged: list[tuple[int, str, CaseSummary]] = []
        for summary in summaries:
            overlap = len(tags.intersection(summary.risk_tags))
            if overlap <= 0:
                continue
            text = (summary.violation_behavior or summary.case_title or "").strip()
            if not text:
                continue
            tagged.append((overlap, text, summary))
        if not tagged:
            return ()

        tagged.sort(key=lambda item: (-item[0], len(item[1]), item[2].case_id))
        chosen = tagged[: self._candidate_limit]
        stats.candidate_cases = len(chosen)

        vectors = self._ensure_vectors(chosen, stats)
        if not vectors:
            return ()

        query_vector = _normalize(
            np.asarray(self._embedding.embed_query(query), dtype=np.float32).reshape(1, -1)
        )
        matrix = _normalize(np.asarray(vectors, dtype=np.float32))
        similarities = (matrix @ query_vector.T).ravel()

        order = np.argsort(-similarities, kind="stable")[: self._top_k]
        hits: list[SparseSearchHit] = []
        for rank, position in enumerate(order, start=1):
            similarity = float(similarities[position])
            if similarity <= 0.0:
                continue
            _, text, summary = chosen[position]
            hits.append(
                SparseSearchHit(
                    document=RetrievalDocument(
                        unit_id=f"bridge:{summary.case_id}",
                        case_ids=(summary.case_id,),
                        text=text,
                        source=BRIDGE_SOURCE,
                        weight=1.0,
                    ),
                    score=max(0.0, min(1.0, similarity)),
                    rank=rank,
                )
            )
        stats.returned = len(hits)
        self._last_stats = stats
        return tuple(hits)

    def _ensure_vectors(
        self, chosen: Sequence[tuple[int, str, CaseSummary]], stats: BridgeStats
    ) -> list[list[float]]:
        """补齐缺失向量；命中的直接复用缓存。"""

        cached = self._store.load()
        signatures = [text_signature(text) for _, text, _ in chosen]
        vectors: list[list[float] | None] = [None] * len(chosen)
        needed: list[int] = []
        for position, (summary, signature) in enumerate(
            zip((item[2] for item in chosen), signatures)
        ):
            entry = cached.get(summary.case_id)
            if entry is not None and entry[0] == signature:
                vectors[position] = entry[1]
                stats.cached_vectors += 1
            else:
                needed.append(position)

        if needed:
            fetched: list[tuple[str, str, list[float]]] = []
            for start in range(0, len(needed), self._batch_size):
                batch_positions = needed[start : start + self._batch_size]
                texts = [chosen[position][1] for position in batch_positions]
                returned = self._embedding.embed_documents(texts)
                if len(returned) != len(batch_positions):
                    raise RuntimeError(
                        f"桥接 Embedding 返回条数不匹配：请求 {len(batch_positions)}，"
                        f"返回 {len(returned)}"
                    )
                for position, vector in zip(batch_positions, returned):
                    vectors[position] = vector
                    fetched.append((chosen[position][2].case_id, signatures[position], vector))
                stats.api_calls += 1
                stats.api_documents += len(batch_positions)
            if fetched:
                self._store.save(fetched)

        return [vector for vector in vectors if vector is not None]


def _normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return cast(NDArray[np.float32], (matrix / safe).astype(np.float32))


__all__ = ["BRIDGE_SOURCE", "BridgeStats", "TypeBridgeRecall"]
