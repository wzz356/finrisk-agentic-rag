"""Prebuild BGE or full Qwen vectors; never encode the corpus inside an HTTP query."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from application.retrieval_service import CaseRetrievalService  # noqa: E402
from settings import AppSettings  # noqa: E402
from storage.case_repository import CaseRepository  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env()
    if "--qwen" in sys.argv:
        settings = replace(settings, embedding_provider="cloud", rerank_provider="none")
    print("Building index for", settings.case_kb_root, flush=True)
    started = time.perf_counter()
    service = CaseRetrievalService(CaseRepository(settings), settings)
    service.warmup(allow_build=True)
    print(
        json.dumps(
            {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "provider": settings.embedding_provider,
                "cache_directory": str(settings.dense_cache_dir),
                "stats": asdict(service._dense.stats) if service._dense is not None else {},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
