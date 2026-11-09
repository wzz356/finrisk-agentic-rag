"""有日期边界和请求预算的每日官网增量更新（与 Web 共用进度和锁）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.case_service import CaseManagementService  # noqa: E402
from application.collection_service import CollectionService  # noqa: E402
from application.retrieval_service import CaseRetrievalService  # noqa: E402
from llm import create_llm_client_from_env  # noqa: E402
from risk.detection import RuleRiskDetector  # noqa: E402
from settings import AppSettings  # noqa: E402
from storage.case_repository import CaseRepository  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="读取任务状态")
    action.add_argument("--dry-run", action="store_true", help="只查看日期窗口，不发官网或模型请求")
    action.add_argument("--activate", action="store_true", help="固定首次起始日期，不立即采集")
    parser.add_argument("--start-date", help="首次启用的 YYYY-MM-DD；已有进度不会重置")
    parser.add_argument(
        "--collect-only", action="store_true", help="留存官网原件，暂不调用模型入库"
    )
    parser.add_argument(
        "--through-today", action="store_true", help="手动检查时包含今天，默认更新至昨天"
    )
    parser.add_argument("--columns", nargs="+", type=int, choices=[4113, 4114, 4115])
    parser.add_argument("--max-pages", type=int, choices=range(1, 6))
    parser.add_argument("--max-details", type=int, choices=range(1, 31))
    parser.add_argument("--max-requests", type=int, choices=range(4, 81))
    parser.add_argument("--delay", type=float, help="请求间隔，最少3秒")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    settings = AppSettings.from_env()
    overrides = {}
    for option, field in (
        ("start_date", "collection_start_date"),
        ("max_pages", "collection_max_pages"),
        ("max_details", "collection_max_details"),
        ("max_requests", "collection_max_requests"),
    ):
        if getattr(args, option) is not None:
            overrides[field] = getattr(args, option)
    if args.columns:
        overrides["collection_columns"] = ",".join(map(str, args.columns))
    if args.delay is not None:
        overrides["collection_delay_seconds"] = max(3.0, args.delay)
    settings = replace(settings, **overrides)
    client = create_llm_client_from_env()
    repository = CaseRepository(settings)
    cases = CaseManagementService(
        settings=settings,
        repository=repository,
        rule_detector=RuleRiskDetector(settings.risk_keyword_path),
        llm_client=client,
    )
    retrieval = CaseRetrievalService(repository, settings)
    service = CollectionService(settings, cases, retrieval)
    try:
        if args.status:
            result = service.status()
        elif args.dry_run:
            result = service.plan(through_today=args.through_today)
        elif args.activate:
            service.activate()
            result = service.status()
        else:
            if client is not None and not args.collect_only:
                await client.start()
            result = await service.run_once(
                collect_only=args.collect_only, through_today=args.through_today
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if client is not None:
            await client.aclose()
        retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
