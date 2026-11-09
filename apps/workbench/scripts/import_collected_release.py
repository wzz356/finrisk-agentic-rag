"""Import structured official cases from a verified release without crawling or chat APIs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from application.case_service import CaseManagementService  # noqa: E402
from application.collection_service import SHANGHAI, CollectionService  # noqa: E402
from application.retrieval_service import CaseRetrievalService  # noqa: E402
from offline.penalty_crawler.polite_http import CollectionHTTPError, atomic_json  # noqa: E402
from offline.release_import import case_payload, load_new_sources, validate_source  # noqa: E402
from risk.detection import RuleRiskDetector  # noqa: E402
from settings import AppSettings  # noqa: E402
from storage.case_repository import CaseRepository, DuplicateCaseError  # noqa: E402


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--release", default="knowledge-v2-20261019")
    parser.add_argument("--apply", action="store_true", help="缺省只核对去重计划，不写入")
    parser.add_argument("--no-refresh-index", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/collection/repository_import.json"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.release):
        parser.error("发布标识格式不合法")
    if file_sha(args.archive) != args.sha256.lower():
        raise ValueError("发布包 SHA256 与已核对的上游清单不一致")
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env()
    repository = CaseRepository(settings)
    existing_ids: set[str] = set()
    for path in (
        settings.cases_path,
        *(
            settings.source_root(origin) / "cases.jsonl"
            for origin in ("official_crawl", "web_ingestion")
        ),
    ):
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                existing_ids.add(row["case_id"])
                if row.get("upstream_case_id"):
                    existing_ids.add(row["upstream_case_id"])
    groups, counts = load_new_sources(args.archive, existing_ids)
    print(
        json.dumps(
            {"plan": counts, "chat_api_requests": 0, "official_http_requests": 0},
            ensure_ascii=False,
        ),
        flush=True,
    )
    service = CaseManagementService(
        settings=settings,
        repository=repository,
        rule_detector=RuleRiskDetector(settings.risk_keyword_path),
        llm_client=None,
    )
    taxonomy = set(repository.risk_dictionary())
    results = []
    created = 0
    job_state = settings.collection_root / "job_state.json"
    schedule_hash_before = file_sha(job_state) if job_state.is_file() else None
    for group in groups:
        try:
            url = validate_source(group)
            document = group["document"]
            text = document["full_text"]
            payloads = [case_payload(item, text, taxonomy) for item in group["cases"]]
            if not args.apply:
                results.append(
                    {
                        "file_id": group["file_id"],
                        "status": "ready",
                        "cases": len(payloads),
                        "source_url": url,
                    }
                )
                continue
            directory = settings.collection_root / "imports" / args.release
            directory.mkdir(parents=True, exist_ok=True)
            original = Path(
                group["source"].get("original_filename") or document["text_sha256"]
            ).name
            filename = f"{Path(original).stem}.txt"
            path = directory / f"{document['text_sha256'][:16]}_{filename}"
            path.write_bytes(text.encode("utf-8"))
            receipt = path.with_suffix(".provenance.json")
            atomic_json(receipt, group)
            params = parse_qs(urlparse(url).query)
            metadata: dict[str, object] = {
                "source_url": url,
                "official_doc_id": params.get("docId", [""])[0],
                "official_column": int(params.get("itemId", ["0"])[0]),
                "upstream_repository": "wzz356/finrisk-agentic-rag",
                "upstream_release": args.release,
                "upstream_archive_sha256": args.sha256,
                "upstream_file_id": group["file_id"],
                "upstream_document_text_sha256": document["text_sha256"],
                "source_snapshot_file": str(receipt.relative_to(settings.project_root)),
                "imported_at": datetime.now(SHANGHAI).isoformat(),
            }
            outcome = await service.ingest_file(
                path,
                original_filename=filename,
                origin="official_crawl",
                source_metadata=metadata,
                structured_cases=payloads,
            )
            assert outcome.llm_used is False
            created += len(outcome.cases)
            results.append(
                {
                    "file_id": group["file_id"],
                    "status": "imported",
                    "case_ids": [case.case_id for case in outcome.cases],
                    "source_url": url,
                    "chat_api_requests": 0,
                }
            )
        except DuplicateCaseError:
            results.append({"file_id": group["file_id"], "status": "duplicate"})
        except (CollectionHTTPError, ValueError, KeyError) as error:
            results.append(
                {"file_id": group["file_id"], "status": "rejected", "reason": str(error)}
            )
    index = None
    index_status = "pending" if created else "unchanged"
    if created and not args.no_refresh_index:
        retrieval = CaseRetrievalService(repository, settings)
        try:
            updater = CollectionService(settings, service, retrieval)
            ready = await updater.refresh_indexes()
            index_status = "ready" if ready else "pending"
            index = asdict(retrieval._dense_stats) if ready else None
        finally:
            retrieval.close()
    schedule_hash_after = file_sha(job_state) if job_state.is_file() else None
    assert schedule_hash_before == schedule_hash_after, "历史文件导入不应改变每日增量日期或水位"
    report = {
        "tested_at": datetime.now(SHANGHAI).isoformat(),
        "release": args.release,
        "archive_sha256": args.sha256,
        "apply": args.apply,
        **counts,
        "created_cases": created,
        "results": results,
        "chat_api_requests": 0,
        "official_http_requests": 0,
        "index_status": index_status,
        "index": index,
        "daily_collection_checkpoint_unchanged": True,
        "statistics": repository.statistics().model_dump(mode="json"),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
