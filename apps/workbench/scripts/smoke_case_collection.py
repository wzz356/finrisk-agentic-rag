"""对已小流量获取的官网原件，隔离验证真实 Pi/DeepSeek + Qwen 入库检索链。"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.case_service import CaseManagementService  # noqa: E402
from application.collection_service import SHANGHAI, CollectionService  # noqa: E402
from application.evidence_service import KnowledgeEvidenceService  # noqa: E402
from application.retrieval_service import CaseRetrievalService  # noqa: E402
from llm import create_llm_client_from_env  # noqa: E402
from risk.detection import RuleRiskDetector  # noqa: E402
from settings import AppSettings  # noqa: E402
from storage.case_repository import CaseRepository  # noqa: E402


class ReceiptReplay:
    """Replay unmodified responses already obtained from the official HTTPS site."""

    def __init__(self, probe: Path, doc_id: int):
        self.listing = json.loads((probe / "latest_list.json").read_text("utf-8"))
        self.detail = json.loads((probe / f"latest_detail_{doc_id}.json").read_text("utf-8"))
        self.doc_id = doc_id
        self.requests = self.cache_hits = 0

    def get_json(self, url: str) -> dict:
        self.requests += 1
        if "SelectDocByItemIdAndChild" in url:
            if "pageIndex=1," not in url:
                raise RuntimeError("隔离测试只允许读取已保存的首页")
            return self.listing
        if f"data_docId={self.doc_id}.json" in url:
            return self.detail
        raise RuntimeError("隔离测试没有此地址的官网收据")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--doc-id", type=int, required=True)
    parser.add_argument("--resume-workspace", type=Path, help="续跑已结构化、等待向量的隔离测试")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    production = AppSettings.from_env()
    replay = ReceiptReplay(args.receipts, args.doc_id)
    selected = next(
        row for row in replay.listing["data"]["rows"] if int(row["docId"]) == args.doc_id
    )
    published = selected["publishDate"][:10]
    now = datetime.now(SHANGHAI)
    if not 0 <= (now.date() - datetime.fromisoformat(published).date()).days <= 7:
        raise ValueError("实测仅允许最近7天的官网原件")
    output_root = PROJECT_ROOT / "outputs/crawl_integration_20261019"
    workspace = (
        args.resume_workspace.resolve()
        if args.resume_workspace
        else output_root / now.strftime("live_%H%M%S")
    )
    if not workspace.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("隔离测试目录必须位于采集测试输出内")
    base = workspace / "base"
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "dictionary").mkdir(exist_ok=True)
    for name in ("cases.jsonl", "case_chunks.jsonl", "case_expansions.jsonl"):
        (base / "data" / name).write_text("", encoding="utf-8")
    for name in ("risk_type_dictionary.csv", "risk_domain_mapping.json"):
        shutil.copy2(production.case_kb_root / "dictionary" / name, base / "dictionary" / name)
    (base / "manifest.json").write_text(
        json.dumps({"data_version": "live-collection-smoke"}), encoding="utf-8"
    )
    config = replace(
        production,
        project_root=workspace,
        case_kb_root=base,
        full_corpus=False,
        raw_case_upload_dir=workspace / "web_raw",
        dense_cache_dir=workspace / "dense",
        rerank_provider="none",
        collection_enabled=False,
        collection_start_date=published,
        collection_columns="4114",
        collection_max_pages=1,
        collection_max_details=1,
    )
    repository = CaseRepository(config)
    client = create_llm_client_from_env()
    if client is None:
        raise RuntimeError("本地没有配置真实模型 API")
    cases = CaseManagementService(
        settings=config,
        repository=repository,
        rule_detector=RuleRiskDetector(config.risk_keyword_path),
        llm_client=client,
    )
    retrieval = CaseRetrievalService(repository, config)
    service = CollectionService(config, cases, retrieval)
    try:
        await client.start()
        started = perf_counter()
        first = await service.run_once(
            now=now, through_today=True, client=replay, force=bool(args.resume_workspace)
        )
        elapsed = perf_counter() - started
        assert repository.statistics().crawled_cases >= 1 and first["index_status"] == "ready", (
            first
        )
        stats = repository.statistics()
        assert stats.base_cases == 0 and stats.user_cases == 0 and stats.crawled_cases >= 1
        case = repository.list_cases(origin="official_crawl").items[0]
        assert case.publish_date.isoformat() == published
        assert case.source_url and str(args.doc_id) in case.source_url
        matches = await asyncio.to_thread(retrieval.search, "虚构保险中介业务", top_k=3)
        assert matches and matches[0].case_id == case.case_id and matches[0].dense_score > 0
        index_stats = asdict(retrieval._dense_stats)
        registry = config.source_root("official_crawl") / "evidence_registry.sqlite3"
        with sqlite3.connect(registry) as connection:
            ids = [row[0] for row in connection.execute("SELECT id FROM units")]
        assert ids
        reader = KnowledgeEvidenceService(production.case_kb_root, (registry,))
        evidence = reader.read(ids[0])
        assert evidence["literal_verified"] and evidence["segments"]
        again = await service.run_once(
            now=now,
            through_today=True,
            force=True,
            client=ReceiptReplay(args.receipts, args.doc_id),
        )
        assert again["created_cases"] == 0 and again["downloaded_documents"] == 0
        state = json.loads(service.state_path.read_text("utf-8"))
        assert any(
            part.get("llm_used") for doc in state["documents"].values() for part in doc["parts"]
        )
        result = {
            "tested_at": now.isoformat(),
            "mode": "real_official_response_replay_with_live_model_apis",
            "official_http_requests_for_capture": 3,
            "official_doc_id": args.doc_id,
            "publish_date": published,
            "live_models": {
                "agent": "Pi",
                "extraction": production.deepseek_rerank_model,
                "embedding": config.embedding_model,
                "dimension": config.embedding_dimension,
            },
            "first_ingestion_seconds": round(elapsed, 3),
            "created_cases": stats.crawled_cases,
            "resumed_index_after_connection_fix": bool(args.resume_workspace),
            "field_check": {
                "party_name": case.party_name,
                "penalty_doc_no": case.penalty_doc_no,
                "source_url": case.source_url,
                "risk_tags": case.risk_tags,
            },
            "source_partitions": {
                "original": stats.base_cases,
                "official_crawl": stats.crawled_cases,
                "web_ingestion": stats.user_cases,
            },
            "index": index_stats,
            "qwen_api_events": retrieval._embedding.events,
            "retrieval_top1_correct": True,
            "dense_score": matches[0].dense_score,
            "literal_evidence_verified": True,
            "repeat_created_cases": again["created_cases"],
            "repeat_downloaded_documents": again["downloaded_documents"],
            "scope": "单份近期官方文书的链路冒烟，不作为召回率/准确率评测",
            "test_workspace": str(workspace.relative_to(PROJECT_ROOT)),
            "bounded_listing_status": first["status"],
        }
        report_dir = PROJECT_ROOT / "reports" / "collection"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "latest_smoke.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await client.aclose()
        retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
