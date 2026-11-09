"""阿里云百炼 Embedding / Rerank 云端适配器。

设计约束：

- API Key 只从环境变量读取，绝不写入日志、异常消息或返回值；
- 任何云端失败都抛 :class:`CloudRetrievalError`，由调用方决定降级，本模块静默吞错；
- 保持同步接口，与 ``CaseRetrievalService.search()`` 的同步签名一致，
  上层继续用 ``asyncio.to_thread`` 包装，不改动 API 与测试的构造方式；
- 稠密向量落盘缓存，避免每次启动重跑上万次请求。

端点差异（易错点，勿混用）：

- Embedding：OpenAI 兼容 ``{base}/embeddings``；
- Rerank：DashScope 原生 ``{base}/services/rerank/text-rerank/text-rerank``，
  其 base 与 Embedding 不同（``api/v1`` 而非 ``compatible-mode/v1``）。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse
from urllib.request import getproxies, proxy_bypass

import httpx

CLOUD_UNAVAILABLE = "云端检索不可用"


class CloudRetrievalError(RuntimeError):
    """云端 Embedding 或 Rerank 调用失败；调用方应据此降级到本地通道。"""


@dataclass(frozen=True, slots=True)
class CloudSettings:
    """云端调用参数；不含任何密钥。"""

    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimension: int = 1024
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rerank_model: str = "qwen3-rerank"
    rerank_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    batch_size: int = 20
    rerank_max_documents: int = 20
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_base_seconds: float = 0.5


def create_cloud_http_client(
    settings: CloudSettings,
    url: str,
    *,
    max_connections: int = 4,
    max_keepalive_connections: int = 2,
) -> httpx.Client:
    """Honor the configured system proxy on Windows, as urllib/browser access does."""
    override = os.getenv("FINRISK_CLOUD_PROXY", "").strip()
    proxy = None
    if override and override.lower() != "direct":
        proxy = override
    elif not override and not proxy_bypass(urlparse(url).hostname or ""):
        proxy = getproxies().get(urlparse(url).scheme)
    return httpx.Client(
        timeout=httpx.Timeout(settings.timeout_seconds),
        limits=httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_keepalive_connections
        ),
        proxy=proxy,
        trust_env=override.lower() != "direct",
    )


class EmbeddingProvider(Protocol):
    """文本向量化的最小接口，云端与本地实现共用。"""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RerankProvider(Protocol):
    """候选文档重排序的最小接口。"""

    @property
    def model_name(self) -> str: ...

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]: ...


def _read_api_key() -> str:
    """按项目既有约定读取密钥；不缓存、不打印。"""

    key = os.getenv("FINRISK_DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key or not key.strip():
        raise CloudRetrievalError(
            f"{CLOUD_UNAVAILABLE}：未配置 FINRISK_DASHSCOPE_API_KEY 或 DASHSCOPE_API_KEY"
        )
    return key.strip()


class DashScopeEmbeddingClient:
    """阿里云百炼文本向量客户端（OpenAI 兼容 /embeddings）。"""

    def __init__(self, settings: CloudSettings | None = None) -> None:
        self._settings = settings or CloudSettings()
        if not 1 <= self._settings.batch_size <= 20:
            raise ValueError("embedding batch_size 必须在 1 到 20 之间（服务端上限 20）")
        self._client: httpx.Client | None = None

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    @property
    def dimension(self) -> int:
        return self._settings.embedding_dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """分批向量化；任一批失败即抛错，由调用方整体降级。"""

        cleaned = ["" if text is None else str(text) for text in texts]
        if not cleaned:
            return []
        vectors: list[list[float]] = []
        batch_size = self._settings.batch_size
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start : start + batch_size]
            # 服务端不接受空串，用占位符替换并在结果里保留位置，避免整批失败。
            payload = [item if item.strip() else " " for item in batch]
            vectors.extend(self._request(payload))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：查询文本为空")
        return self._request([text.strip()])[0]

    def _request(self, texts: list[str]) -> list[list[float]]:
        api_key = _read_api_key()
        url = self._settings.embedding_base_url.rstrip("/") + "/embeddings"
        body = {
            "model": self._settings.embedding_model,
            "input": texts,
            "dimensions": self._settings.embedding_dimension,
            "encoding_format": "float",
        }
        data = _post_with_retry(
            self._ensure_client(),
            url,
            api_key,
            body,
            self._settings,
            stage="embedding",
        )
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：embedding 响应缺少 data 数组")
        ordered = sorted(
            (item for item in items if isinstance(item, dict)),
            key=lambda item: int(item.get("index") or 0),
        )
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：embedding 元素格式非法")
            floats = [float(value) for value in vector]
            if len(floats) != self._settings.embedding_dimension:
                raise CloudRetrievalError(
                    f"{CLOUD_UNAVAILABLE}：向量维度 {len(floats)} 与配置 "
                    f"{self._settings.embedding_dimension} 不一致"
                )
            vectors.append(floats)
        if len(vectors) != len(texts):
            raise CloudRetrievalError(
                f"{CLOUD_UNAVAILABLE}：期望 {len(texts)} 条向量，实际返回 {len(vectors)} 条"
            )
        return vectors

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = create_cloud_http_client(
                self._settings,
                self._settings.embedding_base_url,
                max_connections=8,
                max_keepalive_connections=4,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class DashScopeRerankClient:
    """阿里云百炼重排序客户端（DashScope 原生 text-rerank 端点）。"""

    # 与 Embedding 不同：走原生 api/v1，且请求体是 input/parameters 嵌套结构。
    _RERANK_PATH = "/services/rerank/text-rerank/text-rerank"

    def __init__(self, settings: CloudSettings | None = None) -> None:
        self._settings = settings or CloudSettings()
        if self._settings.rerank_max_documents < 1:
            raise ValueError("rerank_max_documents 必须大于 0")
        self._client: httpx.Client | None = None

    @property
    def model_name(self) -> str:
        return self._settings.rerank_model

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """返回 ``(原始下标, 相关分)`` 列表，按相关分降序。

        文档数超过服务端上限时分片调用，再按分数归并，保证不丢候选。
        """

        if not query or not query.strip():
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：rerank 查询为空")
        texts = [str(item) for item in documents]
        if not texts:
            return []
        limit = self._settings.rerank_max_documents
        limit = max(1, min(limit, 500))  # qwen3-rerank 单次最多 500 文档
        wanted = top_n if top_n is not None else len(texts)
        scored: list[tuple[int, float]] = []
        for start in range(0, len(texts), limit):
            shard = texts[start : start + limit]
            shard_top_n = max(1, min(len(shard), wanted))
            for index, score in self._request(query.strip(), shard, shard_top_n):
                scored.append((start + index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:wanted]

    def _request(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        api_key = _read_api_key()
        url = self._settings.rerank_base_url.rstrip("/") + self._RERANK_PATH
        # 显式标注为 dict[str, object]：dict 的值类型不变，嵌套 list[str] 会被
        # 推断成 Collection[str] 而与 _post_with_retry 的形参不兼容。
        body: dict[str, object] = {
            "model": self._settings.rerank_model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        data = _post_with_retry(self._ensure_client(), url, api_key, body, self._settings, "rerank")
        output = data.get("output") if isinstance(data.get("output"), dict) else data
        results = output.get("results") if isinstance(output, dict) else None
        if not isinstance(results, list) or not results:
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：rerank 响应缺少 results 数组")
        pairs: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if raw_index is None or score is None:
                continue
            try:
                pairs.append((int(raw_index), _clamp01(float(score))))
            except (TypeError, ValueError):
                continue
        if not pairs:
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：rerank 结果无可解析条目")
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs[: max(1, top_n)]

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = create_cloud_http_client(self._settings, self._settings.rerank_base_url)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class NullRerankClient:
    """``rerank_provider=none`` 时使用；让调用方走确定性加权重排。"""

    model_name = "none"

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        del query, documents, top_n
        return []


def _post_with_retry(
    client: httpx.Client,
    url: str,
    api_key: str,
    body: dict[str, object],
    settings: CloudSettings,
    stage: str,
) -> dict[str, object]:
    """带指数退避的 POST；异常消息不包含密钥与请求正文。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_reason = "未知原因"
    for attempt in range(settings.max_retries + 1):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            response = client.post(url, headers=headers, content=payload)
        except httpx.HTTPError as error:
            last_reason = f"网络错误 {type(error).__name__}"
            if attempt < settings.max_retries:
                time.sleep(settings.retry_base_seconds * (2**attempt))
                continue
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：{stage} {last_reason}") from None

        if response.status_code == 429 or response.status_code >= 500:
            last_reason = f"服务端返回 {response.status_code}"
            if attempt < settings.max_retries:
                time.sleep(settings.retry_base_seconds * (2**attempt))
                continue
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：{stage} {last_reason}")

        if response.status_code in (401, 403):
            # 明确区分鉴权问题，便于定位；不回显密钥内容。
            raise CloudRetrievalError(
                f"{CLOUD_UNAVAILABLE}：{stage} 鉴权失败({response.status_code})，"
                "请检查 API Key 与地域"
            )

        if response.status_code != 200:
            raise CloudRetrievalError(
                f"{CLOUD_UNAVAILABLE}：{stage} 返回 {response.status_code} "
                f"{_short_error(response.text)}"
            )

        try:
            data = response.json()
        except ValueError as error:
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：{stage} 响应不是合法 JSON") from error
        if not isinstance(data, dict):
            raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：{stage} 响应不是 JSON 对象")
        return data

    raise CloudRetrievalError(f"{CLOUD_UNAVAILABLE}：{stage} 重试耗尽（{last_reason}）")


def _short_error(text: str) -> str:
    """截取服务端错误摘要，避免把长响应体带进异常消息。"""

    compact = " ".join((text or "").split())
    return compact[:200]


def _clamp01(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def text_signature(text: str) -> str:
    """文本指纹，用于向量缓存命中判断。"""

    return hashlib.sha256((" ".join(str(text).split())).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DenseVectorCache:
    """稠密向量磁盘缓存。

    以 ``unit_id`` 为键、``sha256(text)`` 为失效判据，模型或维度变化时整体作废。
    存储为 JSON Lines，读取后由 numpy 组装矩阵；写入失败只影响下次冷启动。
    """

    directory: Path
    model: str
    dimension: int

    def __post_init__(self) -> None:
        # slots 数据类不能新增实例属性，这里只做入参归一化。
        self.directory = Path(self.directory)

    @property
    def namespace(self) -> str:
        """模型与维度共同决定缓存命名空间，换模型即自然失效。"""

        return f"{self.model}-{self.dimension}"

    @property
    def _path(self) -> Path:
        return self.directory / f"{self.namespace}.jsonl"

    def load(self) -> dict[str, tuple[str, list[float]]]:
        """读取缓存；文件缺失或损坏时返回空表，不阻断启动。"""

        path = self._path
        if not path.is_file():
            return {}
        cached: dict[str, tuple[str, list[float]]] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    unit_id = item.get("unit_id")
                    signature = item.get("signature")
                    vector = item.get("vector")
                    if not isinstance(unit_id, str) or not isinstance(signature, str):
                        continue
                    if not isinstance(vector, list) or len(vector) != self.dimension:
                        continue
                    cached[unit_id] = (signature, [float(value) for value in vector])
        except OSError:
            return {}
        return cached

    def save(self, entries: Iterable[tuple[str, str, Sequence[float]]]) -> None:
        """整体重写缓存文件；写入失败只影响下次冷启动，不抛给调用方。"""

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for unit_id, signature, vector in entries:
                    handle.write(
                        json.dumps(
                            {
                                "unit_id": unit_id,
                                "signature": signature,
                                "vector": [round(float(value), 6) for value in vector],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            temporary.replace(self._path)
        except OSError:
            return

    def clear_stale(self, valid_namespace: bool = True) -> None:
        """删除与当前模型/维度不匹配的旧缓存文件，防止目录无限膨胀。"""

        del valid_namespace
        if not self.directory.is_dir():
            return
        try:
            for path in self.directory.glob("*.jsonl"):
                if path.name != self._path.name:
                    path.unlink()
        except OSError:
            return


def build_cloud_settings(settings: object) -> CloudSettings:
    """从 :class:`settings.AppSettings` 提取云端参数；不复制任何密钥。"""

    return CloudSettings(
        embedding_model=str(getattr(settings, "embedding_model", "qwen3.7-text-embedding")),
        embedding_dimension=int(getattr(settings, "embedding_dimension", 1024)),
        embedding_base_url=str(
            getattr(
                settings,
                "embedding_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        ),
        rerank_model=str(getattr(settings, "rerank_model", "qwen3-rerank")),
        rerank_base_url=str(
            getattr(settings, "rerank_base_url", "https://dashscope.aliyuncs.com/api/v1")
        ),
        batch_size=int(getattr(settings, "cloud_batch_size", 20)),
        rerank_max_documents=int(getattr(settings, "cloud_rerank_max_documents", 20)),
        timeout_seconds=float(getattr(settings, "cloud_timeout_seconds", 30.0)),
        max_retries=int(getattr(settings, "cloud_max_retries", 2)),
    )


__all__ = [
    "CLOUD_UNAVAILABLE",
    "CloudRetrievalError",
    "CloudSettings",
    "DenseVectorCache",
    "DashScopeEmbeddingClient",
    "DashScopeRerankClient",
    "EmbeddingProvider",
    "NullRerankClient",
    "RerankProvider",
    "build_cloud_settings",
    "text_signature",
]
