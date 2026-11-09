"""FinRisk 运行路径配置。

所有业务模块通过 :class:`AppSettings` 获取数据位置，避免在案例、检索、风险和
API 层分别拼接路径。环境变量只用于覆盖部署路径，默认值均相对项目根目录。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 阿里云百炼公共端点。若使用业务空间专属域名（形如
# https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com），用环境变量覆盖即可。
# 注意：Embedding 走 compatible-mode，Rerank 走原生 api/v1，两者路径不同。
DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_RERANK_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"


def _resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = (os.getenv(name) or default).strip().lower()
    return value if value in allowed else default


@dataclass(frozen=True, slots=True)
class AppSettings:
    """应用共用的数据、上传限制与云端检索配置。"""

    project_root: Path
    case_kb_root: Path
    risk_keyword_path: Path
    raw_case_upload_dir: Path
    max_upload_size_bytes: int = 25 * 1024 * 1024
    risk_display_threshold: float = 0.20
    retrieval_risk_threshold: float = 0.40
    # 稠密语义通道：cloud 走阿里云百炼，local 退回倒排表 TF-IDF。
    embedding_provider: str = "cloud"
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimension: int = 1024
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    # 重排序：默认 Pi/DeepSeek；cloud 用百炼，none 保留融合排序。
    rerank_provider: str = "deepseek"
    rerank_model: str = "qwen3-rerank"
    rerank_base_url: str = DEFAULT_RERANK_BASE_URL
    deepseek_rerank_model: str = "deepseek-flash"
    deepseek_rerank_timeout_seconds: float = 20.0
    deepseek_rerank_concurrency: int = 2
    cloud_batch_size: int = 20
    cloud_rerank_max_documents: int = 20
    cloud_timeout_seconds: float = 30.0
    cloud_max_retries: int = 2
    cloud_concurrency: int = 4
    # 默认自适应前后融合；post 判定后检索，pre 判定前注入。
    retrieval_timing: str = "adaptive"
    longtext_screening: bool = True
    semantic_context_gate: float = 0.50
    type_bridge_enabled: bool = False
    type_bridge_top_k: int = 3
    retrieval_top_k: int = 5
    dense_cache_dir: Path | None = None
    bge_model_dir: Path | None = None
    query_rewrite_enabled: bool = True
    query_rewrite_original_anchors: int = 0
    agent_retrieval_enabled: bool = True
    risk_llm_concurrency: int = 6
    full_corpus: bool = True
    collection_enabled: bool = False
    collection_start_date: str = ""
    collection_daily_time: str = "04:30"
    collection_overlap_days: int = 3
    collection_delay_seconds: float = 3.0
    collection_jitter_seconds: float = 2.0
    collection_max_pages: int = 3
    collection_max_details: int = 12
    collection_max_requests: int = 32
    collection_columns: str = "4113,4114,4115"

    @classmethod
    def from_env(cls) -> AppSettings:
        """读取路径与云端检索覆盖；不在此处加载或暴露任何 API 密钥。"""

        knowledge_root = PROJECT_ROOT / "data" / "processed" / "knowledge"
        cache_dir = os.getenv("FINRISK_DENSE_CACHE_DIR")
        return cls(
            project_root=PROJECT_ROOT,
            case_kb_root=_resolve_path(
                os.getenv("FINRISK_CASE_KB_ROOT"),
                knowledge_root / "finrisk_full_20261024"
                if (knowledge_root / "finrisk_full_20261024/manifest.json").is_file()
                else knowledge_root / "finrisk_v2_1_20261018",
            ),
            risk_keyword_path=_resolve_path(
                os.getenv("FINRISK_RISK_KEYWORD_PATH"),
                PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json",
            ),
            raw_case_upload_dir=_resolve_path(
                os.getenv("FINRISK_RAW_CASE_UPLOAD_DIR"),
                PROJECT_ROOT / "data" / "raw" / "cases" / "uploads",
            ),
            max_upload_size_bytes=int(
                os.getenv("FINRISK_MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024))
            ),
            risk_display_threshold=float(os.getenv("FINRISK_RISK_DISPLAY_THRESHOLD", "0.20")),
            retrieval_risk_threshold=float(os.getenv("FINRISK_RETRIEVAL_RISK_THRESHOLD", "0.40")),
            embedding_provider=_env_choice(
                "FINRISK_EMBEDDING_PROVIDER", "cloud", ("bge", "cloud", "local")
            ),
            embedding_model=os.getenv("FINRISK_EMBEDDING_MODEL", "qwen3.7-text-embedding"),
            embedding_dimension=int(os.getenv("FINRISK_EMBEDDING_DIMENSION", "1024")),
            embedding_base_url=(
                os.getenv("FINRISK_EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL
            ),
            rerank_provider=_env_choice(
                "FINRISK_RERANK_PROVIDER", "deepseek", ("cloud", "deepseek", "none")
            ),
            rerank_model=os.getenv("FINRISK_RERANK_MODEL", "qwen3-rerank"),
            rerank_base_url=os.getenv("FINRISK_RERANK_BASE_URL") or DEFAULT_RERANK_BASE_URL,
            deepseek_rerank_model=os.getenv("FINRISK_DEEPSEEK_RERANK_MODEL", "deepseek-flash"),
            deepseek_rerank_timeout_seconds=float(
                os.getenv("FINRISK_DEEPSEEK_RERANK_TIMEOUT", "20")
            ),
            deepseek_rerank_concurrency=int(os.getenv("FINRISK_DEEPSEEK_RERANK_CONCURRENCY", "2")),
            cloud_batch_size=int(os.getenv("FINRISK_CLOUD_BATCH_SIZE", "20")),
            cloud_rerank_max_documents=int(os.getenv("FINRISK_CLOUD_RERANK_MAX_DOCS", "20")),
            cloud_timeout_seconds=float(os.getenv("FINRISK_CLOUD_TIMEOUT", "30")),
            cloud_max_retries=int(os.getenv("FINRISK_CLOUD_MAX_RETRIES", "2")),
            cloud_concurrency=int(os.getenv("FINRISK_CLOUD_CONCURRENCY", "4")),
            retrieval_timing=_env_choice(
                "FINRISK_RETRIEVAL_TIMING", "adaptive", ("post", "pre", "hybrid", "adaptive")
            ),
            longtext_screening=os.getenv("FINRISK_LONGTEXT_SCREENING", "true").lower() == "true",
            semantic_context_gate=float(os.getenv("FINRISK_SEMANTIC_CONTEXT_GATE", "0.50")),
            type_bridge_enabled=_env_bool("FINRISK_TYPE_BRIDGE", False),
            type_bridge_top_k=int(os.getenv("FINRISK_TYPE_BRIDGE_TOP_K", "3")),
            retrieval_top_k=int(os.getenv("FINRISK_RETRIEVAL_TOP_K", "5")),
            dense_cache_dir=(
                _resolve_path(cache_dir, PROJECT_ROOT / "data" / "cache" / "dense")
                if cache_dir
                else PROJECT_ROOT / "data" / "cache" / "dense"
            ),
            bge_model_dir=_resolve_path(
                os.getenv("FINRISK_BGE_MODEL_DIR"), PROJECT_ROOT / "models/bge-small-zh-v1.5"
            ),
            query_rewrite_enabled=_env_bool("FINRISK_QUERY_REWRITE", True),
            query_rewrite_original_anchors=max(
                0, min(2, int(os.getenv("FINRISK_QUERY_ANCHORS", "0")))
            ),
            agent_retrieval_enabled=_env_bool("FINRISK_AGENT_RETRIEVAL", True),
            risk_llm_concurrency=int(os.getenv("FINRISK_RISK_LLM_CONCURRENCY", "6")),
            full_corpus=_env_bool("FINRISK_FULL_CORPUS", True),
            collection_enabled=_env_bool("FINRISK_COLLECTION_ENABLED", False),
            collection_start_date=os.getenv("FINRISK_COLLECTION_START_DATE", "").strip(),
            collection_daily_time=os.getenv("FINRISK_COLLECTION_DAILY_TIME", "04:30"),
            collection_overlap_days=max(1, int(os.getenv("FINRISK_COLLECTION_OVERLAP_DAYS", "3"))),
            collection_delay_seconds=max(3.0, float(os.getenv("FINRISK_COLLECTION_DELAY", "3"))),
            collection_jitter_seconds=max(0.0, float(os.getenv("FINRISK_COLLECTION_JITTER", "2"))),
            collection_max_pages=max(
                1, min(5, int(os.getenv("FINRISK_COLLECTION_MAX_PAGES", "3")))
            ),
            collection_max_details=max(
                1, min(30, int(os.getenv("FINRISK_COLLECTION_MAX_DETAILS", "12")))
            ),
            collection_max_requests=max(
                4, min(80, int(os.getenv("FINRISK_COLLECTION_MAX_REQUESTS", "32")))
            ),
            collection_columns=os.getenv("FINRISK_COLLECTION_COLUMNS", "4113,4114,4115"),
        )

    @property
    def case_data_dir(self) -> Path:
        return self.case_kb_root / "data"

    @property
    def cases_path(self) -> Path:
        return self.case_data_dir / "cases.jsonl"

    @property
    def chunks_path(self) -> Path:
        full = self.case_data_dir / "case_chunks_full.jsonl"
        if self.full_corpus and full.is_file():
            return full
        return self.case_data_dir / "case_chunks.jsonl"

    @property
    def expansions_path(self) -> Path:
        return self.case_data_dir / "case_expansions.jsonl"

    @property
    def risk_dictionary_path(self) -> Path:
        return self.case_kb_root / "dictionary" / "risk_type_dictionary.csv"

    @property
    def query_reviews_path(self) -> Path:
        return self.project_root / "data" / "processed" / "reviews" / "query_reviews.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.case_kb_root / "manifest.json"

    @property
    def incremental_cases_root(self) -> Path:
        return self.project_root / "data" / "processed" / "case_sources"

    def source_root(self, origin: str) -> Path:
        if origin not in {"official_crawl", "web_ingestion"}:
            raise ValueError("新增案例来源不合法")
        return self.incremental_cases_root / origin

    @property
    def collection_root(self) -> Path:
        return self.project_root / "data" / "raw" / "cases" / "official_crawl"

    @property
    def additional_evidence_databases(self) -> tuple[Path, ...]:
        return tuple(
            [self.case_kb_root / "partitions/official_crawl/evidence_registry.sqlite3"]
            + [
                self.source_root(origin) / "evidence_registry.sqlite3"
                for origin in ("official_crawl", "web_ingestion")
            ]
        )
