"""Batch-import existing collected files with literal evidence and no chat API calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from application.case_service import CaseManagementService  # noqa: E402
from application.collection_service import SHANGHAI, CollectionService  # noqa: E402
from application.retrieval_service import CaseRetrievalService  # noqa: E402
from domain.common import FilePurpose  # noqa: E402
from offline.collected_import import (  # noqa: E402
    archive_documents,
    collected_payload,
    parse_fields,
)
from offline.penalty_crawler.polite_http import atomic_json  # noqa: E402
from risk.detection import RuleRiskDetector  # noqa: E402
from settings import AppSettings  # noqa: E402
from storage.case_repository import CaseRepository, DuplicateCaseError  # noqa: E402


def file_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def business_key(record: dict[str, object]) -> tuple[str, str] | None:
    values = tuple(
        re.sub(r"\s+", "", str(record.get(k) or "")) for k in ("penalty_doc_no", "party_name")
    )
    return (values[0], values[1]) if all(values) else None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no-refresh-index", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/collection/archived_collection_import.json"
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env()
    repository = CaseRepository(settings)
    detector = RuleRiskDetector(settings.risk_keyword_path)
    service = CaseManagementService(
        settings=settings, repository=repository, rule_detector=detector, llm_client=None
    )
    taxonomy = set(repository.risk_dictionary())
    archive_sha = file_sha(args.archive)
    state_path = settings.collection_root / "job_state.json"
    state_before = file_sha(state_path)
    existing = []
    for path in (
        settings.cases_path,
        *(settings.source_root(s) / "cases.jsonl" for s in ("official_crawl", "web_ingestion")),
    ):
        if path.is_file():
            existing.extend(
                json.loads(line) for line in path.read_text("utf-8").splitlines() if line
            )
    hashes = {row.get("source_content_sha256") for row in existing}
    keys = {business_key(row) for row in existing} - {None}
    seen_text: set[str] = set()
    results = []
    artifacts = []
    source_texts = {}
    new_files: list[Path] = []
    prepared = []
    directory = settings.collection_root / "imports" / "legacy-delivery-20261018"
    for document in archive_documents(args.archive):
        text = document.raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip()
        result: dict[str, object] = {
            "member": document.member,
            "source_url": document.metadata["source_url"],
        }
        try:
            payload = collected_payload(document, text, taxonomy, detector)
            if payload is None:
                result["status"] = "non_insurance"
                results.append(result)
                continue
            content_sha = hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest()
            key = business_key(payload)
            if content_sha in hashes or content_sha in seen_text or (key and key in keys):
                result["status"] = "duplicate"
                results.append(result)
                continue
            seen_text.add(content_sha)
            if key:
                keys.add(key)
            result.update(
                status="ready",
                label_source=payload["import_kind"],
                party_name=payload["party_name"],
            )
            if not args.apply:
                results.append(result)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            path = (
                directory
                / f"{hashlib.sha256(document.raw).hexdigest()[:16]}_{Path(document.member).name}"
            )
            if not path.is_file():
                path.write_bytes(document.raw)
                new_files.append(path)
            extraction = service.extractor.extract(
                path, FilePurpose.CASE, source_uri=str(document.metadata["source_url"])
            )
            # Build against exactly the text that will be saved in the evidence registry.
            payload = collected_payload(document, extraction.full_text, taxonomy, detector)
            assert payload is not None
            logical_path = path.relative_to(settings.project_root).as_posix()
            built = await service.builder.build_imported(
                extraction, source_file=logical_path, payload=payload
            )
            assert built.llm_used is False
            fields = parse_fields(extraction.full_text)
            grouped = bool(re.search(r"责任人|责任人员|及其|、|\n", fields.get("party_name", "")))
            built.record.update(
                case_origin="official_crawl",
                source_group="official_crawl",
                raw_source_file=logical_path,
                source_content_sha256=content_sha,
                source_url=document.metadata["source_url"],
                official_doc_id=str(document.metadata["doc_id"]),
                official_column=int(document.metadata["column"]),
                publish_date=str(document.metadata.get("publish_date") or "")[:10] or None,
                imported_at=datetime.now(SHANGHAI).isoformat(),
                source_archive_sha256=archive_sha,
                source_archive_member=document.member,
                legacy_file_sha1=document.metadata["sha1"],
                source_subject_scope="grouped_notice" if grouped else "single_subject",
                source_party_text=fields.get("party_name"),
                source_table_part=document.metadata.get("part"),
                import_provenance="original_delivery_collected_files",
            )
            if grouped:
                built.record["quality_flags"] = [
                    *built.record["quality_flags"],
                    "grouped_notice_preserved",
                ]
            artifacts.append((built.record, built.chunks, built.expansions))
            source_texts[str(built.record["file_id"])] = extraction.full_text
            prepared.append((result, path, str(built.record["case_id"])))
        except (ValueError, KeyError) as error:
            result.update(status="rejected", reason=str(error))
        results.append(result)
    print(
        json.dumps(
            {"planned": dict(Counter(str(r["status"]) for r in results)), "chat_api_requests": 0},
            ensure_ascii=False,
        ),
        flush=True,
    )
    details = []
    committed = False
    try:
        if artifacts:
            try:
                details = repository.add_cases(
                    artifacts, source_texts=source_texts, origin="official_crawl"
                )
            except DuplicateCaseError:
                details = []
        committed = True
        ids = {detail.case_id for detail in details}
        for result, path, cid in prepared:
            result.update(status="imported" if cid in ids else "duplicate", case_id=cid)
            if cid not in ids and path in new_files:
                path.unlink(missing_ok=True)
    finally:
        if not committed:
            for path in new_files:
                path.unlink(missing_ok=True)
    index_stats = None
    index_status = "pending" if details else "unchanged"
    if details and not args.no_refresh_index:
        retrieval = CaseRetrievalService(repository, settings)
        try:
            ready = await CollectionService(settings, service, retrieval).refresh_indexes()
            index_status = "ready" if ready else "pending"
            index_stats = asdict(retrieval._dense_stats) if ready else None
        finally:
            retrieval.close()
    assert state_before == file_sha(state_path), "历史文件导入不应改变每天增量的起始日期和水位"
    report = {
        "tested_at": datetime.now(SHANGHAI).isoformat(),
        "archive_sha256": archive_sha,
        "apply": args.apply,
        "counts": dict(Counter(str(r["status"]) for r in results)),
        "label_sources": dict(
            Counter(str(r["label_source"]) for r in results if r.get("status") == "imported")
        ),
        "created_cases": len(details),
        "results": results,
        "chat_api_requests": 0,
        "official_http_requests": 0,
        "daily_collection_checkpoint_unchanged": True,
        "index_status": index_status,
        "index": index_stats,
        "statistics": repository.statistics().model_dump(mode="json"),
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
