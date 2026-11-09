"""Offline-only full Qwen index, resumable per-text checkpoints and mmap serving."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from retrieval.cloud import CloudRetrievalError, text_signature
from retrieval.dense import DenseBuildStats, DenseRetrievalIndex
from retrieval.qwen import NativeQwenEmbeddingClient
from retrieval.sparse import RetrievalDocument


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class QwenDenseIndex(DenseRetrievalIndex):
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        *,
        embedding: NativeQwenEmbeddingClient,
        directory: Path,
        allow_build: bool = False,
        concurrency: int = 4,
    ) -> None:
        super().__init__(documents, embedding=embedding)
        self.client = embedding
        self.directory = directory
        self.allow_build = allow_build
        self.concurrency = max(1, min(concurrency, 4))
        self.binding = {
            "provider": embedding.binding,
            "ordered_units": [(d.unit_id, text_signature(d.text)) for d in documents],
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(self.binding, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        self.path = directory / f"qwen-{self.fingerprint[:24]}.npy"
        self.receipt = self.path.with_suffix(".json")
        self._query_scores: OrderedDict[str, NDArray[np.float32]] = OrderedDict()

    def clear_query_cache(self) -> None:
        with self._lock:
            self._query_scores.clear()

    def _query_similarities(
        self,
        query: str,
        matrix: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        with self._lock:
            cached = self._query_scores.get(query)
            if cached is not None:
                self._query_scores.move_to_end(query)
                return cached
        scores = super()._query_similarities(query, matrix)
        with self._lock:
            self._query_scores[query] = scores
            while len(self._query_scores) > 64:
                self._query_scores.popitem(last=False)
        return scores

    def build(self) -> DenseBuildStats:
        with self._lock:
            if self.ready:
                return self.stats
            if self.path.is_file() and self.receipt.is_file():
                meta = json.loads(self.receipt.read_text(encoding="utf-8"))
                if meta["fingerprint"] != self.fingerprint or file_sha(self.path) != meta["sha256"]:
                    raise CloudRetrievalError("Qwen index fingerprint mismatch")
                matrix = np.load(self.path, mmap_mode="r", allow_pickle=False)
                if matrix.shape != (len(self.documents), self.client.dimension):
                    raise CloudRetrievalError("Qwen index shape mismatch")
                self._matrix = matrix
                self._stats = DenseBuildStats(
                    total_units=len(self.documents),
                    cache_hits=len(self.documents),
                    dimension=self.client.dimension,
                    model=self.client.model_name,
                )
                return self.stats
            if not self.allow_build:
                raise CloudRetrievalError("全量 Qwen 索引未就绪，请先运行离线构建脚本")
            return self._build_offline()

    def _build_offline(self) -> DenseBuildStats:
        began = perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        groups: dict[str, list[int]] = defaultdict(list)
        for position, document in enumerate(self.documents):
            groups[text_signature(document.text)].append(position)
        db_path = self.directory / (self.client.model_name + ".sqlite3")
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (signature TEXT PRIMARY KEY, data BLOB)")
            existing = {row[0] for row in db.execute("SELECT signature FROM vectors")}
            missing = [key for key in groups if key not in existing]
            chunks = [
                missing[i : i + self.client.batch_size]
                for i in range(0, len(missing), self.client.batch_size)
            ]
            print(
                json.dumps(
                    {
                        "units": len(self.documents),
                        "unique_texts": len(groups),
                        "cached_texts": len(groups) - len(missing),
                        "pending_api_batches": len(chunks),
                    }
                ),
                flush=True,
            )

            def fetch(keys: list[str]) -> list[list[float]]:
                return self.client.embed_documents(
                    [self.documents[groups[k][0]].text for k in keys]
                )

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                jobs = {pool.submit(fetch, keys): keys for keys in chunks}
                for done, future in enumerate(as_completed(jobs), 1):
                    keys = jobs[future]
                    vectors = future.result()
                    db.executemany(
                        "INSERT OR REPLACE INTO vectors VALUES (?,?)",
                        [
                            (key, np.asarray(vector, dtype=np.float32).tobytes())
                            for key, vector in zip(keys, vectors, strict=True)
                        ],
                    )
                    db.commit()
                    if done % 40 == 0 or done == len(chunks):
                        print(
                            json.dumps(
                                {
                                    "completed_batches": done,
                                    "total_batches": len(chunks),
                                    "elapsed_seconds": round(perf_counter() - began, 2),
                                }
                            ),
                            flush=True,
                        )
            temporary = self.path.with_suffix(".building.npy")
            matrix = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(len(self.documents), self.client.dimension),
            )
            for key, positions in groups.items():
                row = db.execute("SELECT data FROM vectors WHERE signature=?", (key,)).fetchone()
                if row is None:
                    raise CloudRetrievalError("Incomplete Qwen checkpoint")
                vector = np.frombuffer(row[0], dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if (
                    vector.size != self.client.dimension
                    or not np.isfinite(vector).all()
                    or norm == 0
                ):
                    raise CloudRetrievalError("Invalid cached Qwen vector")
                matrix[positions] = vector / norm
            matrix.flush()
            del matrix
            temporary.replace(self.path)
        stats = DenseBuildStats(
            total_units=len(self.documents),
            cache_hits=len(self.documents) - sum(len(groups[key]) for key in missing),
            api_calls=len(chunks),
            api_documents=len(missing),
            elapsed_seconds=perf_counter() - began,
            dimension=self.client.dimension,
            model=self.client.model_name,
        )
        meta = {
            "fingerprint": self.fingerprint,
            "provider": self.client.binding,
            "unit_count": len(self.documents),
            "unique_texts": len(groups),
            "sha256": file_sha(self.path),
            "seconds": stats.elapsed_seconds,
            "api_events": self.client.events,
            "evaluation_data_indexed": False,
        }
        temporary_receipt = self.receipt.with_suffix(".json.tmp")
        temporary_receipt.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_receipt.replace(self.receipt)
        self._matrix = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self._stats = stats
        return stats
