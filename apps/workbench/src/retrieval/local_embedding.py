"""Pinned local BGE inference with batched documents and a bounded query cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Any

from retrieval.cloud import CloudRetrievalError


class LocalBGEEmbeddingClient:
    """Load only local official weights; model loading never downloads during a query."""

    dimension = 512
    batch_size = 64
    query_instruction = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, directory: Path, *, batch_size: int = 64) -> None:
        self.directory = directory
        self.batch_size = batch_size
        self._lock = RLock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._queries: OrderedDict[str, list[float]] = OrderedDict()
        manifest = directory / "finrisk_model_manifest.json"
        self._manifest = (
            json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        )
        self.calls = 0
        self.query_cache_hits = 0
        self.truncated_inputs = 0

    @property
    def model_name(self) -> str:
        revision = self._manifest.get("revision", "missing")
        return f"bge-small-zh-v1.5-{revision}-cls-l2-512"

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self._manifest or not (self.directory / "model.safetensors").is_file():
            raise CloudRetrievalError("本地 BGE 权重未准备，请先运行 scripts/prepare_bge.py")
        for entry in self._manifest.get("files", []):
            path = self.directory / entry["path"]
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise CloudRetrievalError("本地 BGE 文件指纹失配，请重新准备固定版本权重")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise CloudRetrievalError("请先安装 local-embedding 可选依赖") from error

        torch.set_num_threads(max(1, min(8, int(os.getenv("FINRISK_BGE_THREADS", "4")))))
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.directory), local_files_only=True)
        self._model = AutoModel.from_pretrained(
            str(self.directory), local_files_only=True, use_safetensors=True
        ).eval()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            self._load()
            import torch

            vectors = []
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                lengths = self._tokenizer(batch, truncation=False, add_special_tokens=True)[
                    "input_ids"
                ]
                self.truncated_inputs += sum(len(value) > 512 for value in lengths)
                inputs = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                with torch.inference_mode():
                    output = self._model(**inputs).last_hidden_state[:, 0]
                    output = torch.nn.functional.normalize(output, p=2, dim=1)
                vectors.extend(output.cpu().tolist())
                self.calls += 1
            return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            keys = [
                hashlib.sha256(
                    (self.model_name + self.query_instruction + text).encode()
                ).hexdigest()
                for text in texts
            ]
            missing = list(dict.fromkeys(key for key in keys if key not in self._queries))
            by_key = dict(zip(keys, texts, strict=True))
            if missing:
                returned = self._encode([self.query_instruction + by_key[key] for key in missing])
                self._queries.update(zip(missing, returned, strict=True))
            self.query_cache_hits += len(keys) - len(missing)
            result = [self._queries[key] for key in keys]
            while len(self._queries) > 4096:
                self._queries.popitem(last=False)
            return result

    def close(self) -> None:
        self._queries.clear()
