"""Native Qwen query/document embeddings, bounded query cache and safe API audit."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from threading import RLock
from time import perf_counter
from typing import Any

import numpy as np

from retrieval.cloud import (
    CloudRetrievalError,
    CloudSettings,
    _post_with_retry,
    _read_api_key,
    create_cloud_http_client,
)

QUERY_INSTRUCTION = (
    "Given an insurance marketing or training statement, retrieve relevant enforcement "
    "case evidence describing the same concrete behavior."
)


class NativeQwenEmbeddingClient:
    """All document and query vectors share this exact model/endpoint/dimension binding."""

    def __init__(self, settings: CloudSettings) -> None:
        self.config = settings
        self.batch_size = min(settings.batch_size, 20)
        self.dimension = settings.embedding_dimension
        self.query_instruction = QUERY_INSTRUCTION
        self.binding = {
            "model": settings.embedding_model,
            "dimension": self.dimension,
            "endpoint": settings.rerank_base_url.rstrip("/"),
            "document_text_type": "document",
            "query_text_type": "query",
            "query_instruction": QUERY_INSTRUCTION,
            "output_type": "dense",
        }
        import json

        signature = hashlib.sha256(json.dumps(self.binding, sort_keys=True).encode()).hexdigest()
        self.model_name = settings.embedding_model + "-native-" + signature[:16]
        self._client = create_cloud_http_client(
            settings,
            settings.rerank_base_url,
            max_connections=8,
            max_keepalive_connections=8,
        )
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = RLock()
        self.events: list[dict[str, Any]] = []

    def _request(self, texts: list[str], kind: str) -> list[list[float]]:
        parameters: dict[str, object] = {
            "text_type": kind,
            "dimension": self.dimension,
            "output_type": "dense",
        }
        if kind == "query":
            parameters["instruct"] = QUERY_INSTRUCTION
        started = perf_counter()
        data = _post_with_retry(
            self._client,
            self.config.rerank_base_url.rstrip("/")
            + "/services/embeddings/text-embedding/text-embedding",
            _read_api_key(),
            {
                "model": self.config.embedding_model,
                "input": {"texts": texts},
                "parameters": parameters,
            },
            self.config,
            "embedding_" + kind,
        )
        output = data.get("output", {})
        items = output.get("embeddings", []) if isinstance(output, dict) else []
        if not isinstance(items, list) or len(items) != len(texts):
            raise CloudRetrievalError("Qwen embedding response count mismatch")
        items.sort(key=lambda row: row["text_index"])
        if [r.get("text_index") for r in items] != list(range(len(texts))):
            raise CloudRetrievalError("Qwen embedding response indices mismatch")
        vectors = [[float(value) for value in row["embedding"]] for row in items]
        if any(len(v) != self.dimension for v in vectors) or not np.isfinite(vectors).all():
            raise CloudRetrievalError("Qwen embedding dimension/nonfinite error")
        with self._lock:
            self.events.append(
                {
                    "stage": "embedding_" + kind,
                    "items": len(texts),
                    "characters": sum(map(len, texts)),
                    "seconds": perf_counter() - started,
                    "request_id": data.get("request_id"),
                    "usage": data.get("usage", {}),
                }
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        result = []
        for start in range(0, len(texts), self.batch_size):
            result.extend(self._request(list(texts[start : start + self.batch_size]), "document"))
        return result

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        if any(not text for text in cleaned):
            raise CloudRetrievalError("Query must not be empty")
        with self._lock:
            found = {text: self._cache[text] for text in cleaned if text in self._cache}
        missing = list(dict.fromkeys(text for text in cleaned if text not in found))
        for start in range(0, len(missing), self.batch_size):
            group = missing[start : start + self.batch_size]
            vectors = self._request(group, "query")
            found.update(zip(group, vectors, strict=True))
            with self._lock:
                for text, vector in zip(group, vectors, strict=True):
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                while len(self._cache) > 2048:
                    self._cache.popitem(last=False)
        return [found[text] for text in cleaned]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def clear_query_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def close(self) -> None:
        self._client.close()
