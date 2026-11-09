"""Binary, fingerprinted local BGE index with deduplicated offline encoding."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from retrieval.cloud import CloudRetrievalError, text_signature
from retrieval.dense import DenseBuildStats, DenseRetrievalIndex
from retrieval.local_embedding import LocalBGEEmbeddingClient
from retrieval.sparse import RetrievalDocument


class LocalBGEIndex(DenseRetrievalIndex):
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        *,
        embedding: LocalBGEEmbeddingClient,
        directory: Path,
        allow_build: bool = False,
    ) -> None:
        super().__init__(documents, embedding=embedding)
        self._local_embedding = embedding
        self.directory = directory
        self.allow_build = allow_build
        self._binding = {
            "model": embedding.model_name,
            "dimension": embedding.dimension,
            "query_instruction": embedding.query_instruction,
            "max_length": 512,
            "ordered_units": [(d.unit_id, text_signature(d.text)) for d in self.documents],
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(self._binding, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        self.path = directory / f"bge-{self.fingerprint[:24]}.npy"

    def build(self) -> DenseBuildStats:
        with self._lock:
            if self._matrix is not None:
                return self._stats
            metadata = self.path.with_suffix(".json")
            if self.path.is_file() and metadata.is_file():
                receipt = json.loads(metadata.read_text(encoding="utf-8"))
                if receipt.get("fingerprint") == self.fingerprint:
                    matrix = np.load(self.path, mmap_mode="r", allow_pickle=False)
                    if (
                        matrix.shape == (len(self.documents), self._embedding.dimension)
                        and matrix.dtype == np.float32
                    ):
                        self._matrix = matrix
                        self._stats = DenseBuildStats(
                            total_units=len(self.documents),
                            cache_hits=len(self.documents),
                            dimension=self._embedding.dimension,
                            model=self._embedding.model_name,
                        )
                        return self._stats
            if not self.allow_build:
                raise CloudRetrievalError(
                    "BGE 索引未预构建；请离线运行 scripts/build_competition_index.py"
                )
            self.directory.mkdir(parents=True, exist_ok=True)
            positions: dict[str, list[int]] = defaultdict(list)
            texts: dict[str, str] = {}
            for i, document in enumerate(self.documents):
                signature = text_signature(document.text)
                positions[signature].append(i)
                texts.setdefault(signature, document.text)
            keys = list(texts)
            matrix = np.empty((len(self.documents), self._embedding.dimension), dtype=np.float32)
            started = time.perf_counter()
            batch = self._local_embedding.batch_size
            for start in range(0, len(keys), batch):
                group = keys[start : start + batch]
                vectors = self._embedding.embed_documents([texts[k] for k in group])
                for key, vector in zip(group, vectors, strict=True):
                    matrix[positions[key], :] = vector
                if start % (batch * 20) == 0:
                    print(
                        f"BGE offline encoding {min(start + batch, len(keys))}/{len(keys)} "
                        "unique texts",
                        flush=True,
                    )
            temporary = self.path.with_suffix(".building")
            with temporary.open("wb") as stream:
                np.save(stream, matrix, allow_pickle=False)
            temporary.replace(self.path)
            metadata.write_text(
                json.dumps(
                    {
                        "fingerprint": self.fingerprint,
                        "model": self._embedding.model_name,
                        "units": len(self.documents),
                        "unique_encoded_texts": len(keys),
                        "dimension": self._embedding.dimension,
                        "pooling": "CLS",
                        "normalization": "L2",
                        "truncated_inputs": self._local_embedding.truncated_inputs,
                        "build_seconds": round(time.perf_counter() - started, 3),
                        "paid_api_calls": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._matrix = matrix
            self._stats = DenseBuildStats(
                total_units=len(self.documents),
                cache_hits=0,
                dimension=self._embedding.dimension,
                model=self._embedding.model_name,
            )
            return self._stats
