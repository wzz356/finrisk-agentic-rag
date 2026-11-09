"""同一案例知识库的读取、增量写入与状态管理。"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, tzinfo
from hashlib import sha256
from itertools import chain
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any
from uuid import uuid4

from domain.knowledge_models import (
    CaseDetail,
    CaseListPage,
    CaseStatistics,
    CaseSummary,
    EvidenceChunk,
)
from retrieval import RetrievalDocument
from settings import AppSettings
from storage.file_lock import FileLock


class CaseNotFoundError(KeyError):
    """案例不存在。"""


class DuplicateCaseError(RuntimeError):
    """相同 ID 的案例已经存在。"""


class CaseRepository:
    """联合读取原始库、官网采集和网页录入；新增只写对应的独立分区。"""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._lock = RLock()
        self._records: dict[str, dict[str, object]] = {}
        self._source_records: dict[str, dict[str, object]] = {}
        self._cases_by_file: dict[str, list[str]] = {}
        self._chunks_by_case: dict[str, list[dict[str, object]]] = {}
        self._retrieval_metadata: dict[str, dict[str, object]] = {}
        self._risk_tags: set[str] = set()
        self._data_version: str | None = None
        self._latest_crawl_at: str | None = None
        self._latest_penalty_date: str | None = None
        self._revision = 0
        self._loaded_stamp: tuple[tuple[int, int] | None, ...] = ()
        self._last_check = 0.0
        with FileLock(settings.incremental_cases_root / ".write.lock"):
            for origin in ("official_crawl", "web_ingestion"):
                self._recover_write(settings.source_root(origin))
            self.reload()

    def _mark_index_pending(self) -> None:
        _write_json_atomic(
            self.settings.incremental_cases_root / "index_state.json",
            {"status": "pending", "updated_at": datetime.now(UTC).isoformat()},
        )

    def _recover_write(self, root: Path) -> None:
        journal = root / ".pending_write.json"
        if not journal.is_file():
            return
        entry = json.loads(journal.read_text("utf-8"))
        committed = False
        registry = root / "evidence_registry.sqlite3"
        if registry.is_file():
            with sqlite3.connect(registry) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='ingestion_commits'"
                ).fetchone()
                committed = bool(
                    exists
                    and connection.execute(
                        "SELECT 1 FROM ingestion_commits WHERE id=?", (entry["id"],)
                    ).fetchone()
                )
        if committed:
            self._refresh_manifest(root)
            self._mark_index_pending()
        else:
            for name, snapshot in entry["files"].items():
                if name not in {"cases.jsonl", "case_chunks.jsonl", "case_expansions.jsonl"}:
                    raise ValueError("入库恢复日志包含未知文件")
                _restore_file(root / name, existed=snapshot[0], size=snapshot[1])
            manifest = root / "manifest.json"
            if entry["manifest"] is None:
                manifest.unlink(missing_ok=True)
            else:
                manifest.write_text(entry["manifest"], encoding="utf-8")
        journal.unlink()

    @property
    def revision(self) -> int:
        self._ensure_fresh()
        return self._revision

    def _stamp(self) -> tuple[tuple[int, int] | None, ...]:
        paths = [
            self.settings.source_root(origin) / "manifest.json"
            for origin in ("official_crawl", "web_ingestion")
        ]
        paths.append(self.settings.incremental_cases_root / "status_overrides.json")
        return tuple(
            (path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None
            for path in paths
        )

    def _ensure_fresh(self) -> None:
        now = monotonic()
        if now - self._last_check >= 1:
            self._last_check = now
            if self._stamp() != self._loaded_stamp:
                self.reload()

    @property
    def risk_tags(self) -> tuple[str, ...]:
        return tuple(sorted(self._risk_tags))

    def _paths(self, filename: str) -> tuple[Path, ...]:
        base = (
            self.settings.chunks_path
            if filename == "case_chunks.jsonl"
            else self.settings.case_data_dir / filename
        )
        return (
            base,
            self.settings.case_kb_root / "partitions/official_crawl" / filename,
            *(
                self.settings.source_root(origin) / filename
                for origin in ("official_crawl", "web_ingestion")
            ),
        )

    def reload(self) -> None:
        with self._lock:
            self._read_snapshot()

    def _read_snapshot(self) -> None:
        if not self.settings.cases_path.is_file():
            raise FileNotFoundError(f"案例主表不存在: {self.settings.cases_path}")
        aliases = self.settings.case_kb_root / "source_aliases.json"
        alias_data = json.loads(aliases.read_text("utf-8")) if aliases.is_file() else {}
        suppressed = set(alias_data.get("suppressed_file_ids", []))
        self._case_aliases = alias_data.get("case_ids", {})
        records = {
            str(record["case_id"]): record
            for path in self._paths("cases.jsonl")
            for record in _read_jsonl(path)
            if record.get("file_id") not in suppressed
        }
        overrides = self.settings.incremental_cases_root / "status_overrides.json"
        if overrides.is_file():
            for cid, status in json.loads(overrides.read_text("utf-8")).items():
                if cid in records:
                    records[cid] = {**records[cid], "retrieval_status": status}
        with self._lock:
            self._records = records
            self._cases_by_file = {}
            for cid, record in records.items():
                fid = str(record.get("file_id") or record.get("source_file") or cid)
                self._cases_by_file.setdefault(fid, []).append(cid)
            self._source_records = (
                {
                    str(row["case_id"]): row
                    for path in self._paths("source_records.jsonl")
                    for row in _read_jsonl(path)
                    if row.get("file_id") not in suppressed
                }
                if self.settings.full_corpus
                else {}
            )
            self._chunks_by_case = {}
            self._retrieval_metadata = {}
            self._risk_tags = {
                str(tag)
                for record in records.values()
                for tag in _as_list(record.get("risk_tags"))
                if tag
            }
            manifest_meta = _read_manifest_meta(self.settings.manifest_path)
            self._data_version = manifest_meta.get("data_version")
            self._latest_crawl_at = manifest_meta.get("latest_crawl_at")
            self._latest_penalty_date = max(
                [manifest_meta.get("latest_penalty_date", "")]
                + [str(value) for record in records.values()
                   if (value := _as_date(record.get("penalty_date"))) is not None]
            ) or None
            self._revision += 1
            self._loaded_stamp = self._stamp()

    def list_cases(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        query: str | None = None,
        risk_tag: str | None = None,
        status: str | None = None,
        origin: str | None = None,
        regulator: str | None = None,
        party_name: str | None = None,
        institution_type: str | None = None,
        penalty_date_from: date | None = None,
        penalty_date_to: date | None = None,
    ) -> CaseListPage:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("分页参数不合法")
        self._ensure_fresh()
        normalized_query = (query or "").strip().casefold()
        records = self._listing_records()
        if normalized_query:
            records = [record for record in records if _record_contains(record, normalized_query)]
        if risk_tag:
            records = [
                record for record in records if risk_tag in _as_list(record.get("risk_tags"))
            ]
        if status:
            records = [record for record in records if record.get("retrieval_status") == status]
        if origin:
            records = [
                record
                for record in records
                if _source_group(record) == origin or record.get("case_origin") == origin
            ]
        if regulator:
            needle = regulator.strip().casefold()
            records = [
                record
                for record in records
                if needle in str(record.get("regulator") or "").casefold()
            ]
        if party_name:
            needle = party_name.strip().casefold()
            records = [
                record
                for record in records
                if needle in str(record.get("party_name") or "").casefold()
            ]
        if institution_type:
            needle = institution_type.strip().casefold()
            records = [
                record
                for record in records
                if needle in str(record.get("institution_type") or "").casefold()
            ]
        if penalty_date_from:
            records = [
                record
                for record in records
                if (value := _as_date(record.get("penalty_date"))) is not None
                and value >= penalty_date_from
            ]
        if penalty_date_to:
            records = [
                record
                for record in records
                if (value := _as_date(record.get("penalty_date"))) is not None
                and value <= penalty_date_to
            ]
        records.sort(key=_case_sort_key)
        if self._document_listing:
            grouped: dict[str, dict[str, object]] = {}
            for record in records:
                fid = str(record.get("file_id") or record.get("source_file") or record["case_id"])
                grouped.setdefault(fid, record)
            records = list(grouped.values())
        total = len(records)
        start = (page - 1) * page_size
        return CaseListPage(
            items=[self._to_summary(record) for record in records[start : start + page_size]],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size) if total else 0,
        )

    def get_case(self, case_id: str) -> CaseDetail:
        self._ensure_fresh()
        record = self._record(case_id)
        return self._to_detail(record, self._case_chunks(str(record["case_id"])))

    def get_summary(self, case_id: str) -> CaseSummary:
        return self._to_summary(self._record(case_id))

    def summaries(self) -> tuple[CaseSummary, ...]:
        """一次性生成检索所需摘要，避免通过分页反复排序主表。"""

        self._ensure_fresh()
        records = tuple(self._records.values())
        return tuple(self._to_summary(record) for record in records)

    def statistics(self) -> CaseStatistics:
        self._ensure_fresh()
        records = self._listing_records()
        subject_count = len(self._records)
        if self._document_listing:
            grouped: dict[str, list[dict[str, object]]] = {}
            for record in records:
                fid = str(record.get("file_id") or record.get("source_file") or record["case_id"])
                grouped.setdefault(fid, []).append(record)
            records = [
                {
                    **members[0],
                    "retrieval_status": "active"
                    if any(r.get("retrieval_status") == "active" for r in members)
                    else "inactive",
                }
                for members in grouped.values()
            ]
        return CaseStatistics(
            total_cases=len(records),
            base_cases=sum(_source_group(record) == "original" for record in records),
            crawled_cases=sum(_source_group(record) == "official_crawl" for record in records),
            user_cases=sum(_source_group(record) == "web_ingestion" for record in records),
            active_cases=sum(1 for record in records if record.get("retrieval_status") == "active"),
            risk_tag_count=len(self._risk_tags),
            data_version=self._data_version,
            latest_crawl_at=max(
                [
                    self._latest_crawl_at or "",
                    *(
                        str(record.get("collected_at") or "")
                        for record in records
                        if _source_group(record) == "official_crawl"
                    ),
                ]
            )
            or None,
            latest_penalty_date=self._latest_penalty_date,
            subject_case_count=subject_count,
            listing_unit="document" if self._document_listing else "case",
        )

    def count_collected_documents(self, on: date, *, zone: tzinfo) -> int:
        """Count persisted official documents by their collection date, across subjects."""
        self._ensure_fresh()
        documents: set[str] = set()
        with self._lock:
            for record in self._records.values():
                if _source_group(record) != "official_crawl":
                    continue
                try:
                    collected_at = datetime.fromisoformat(str(record.get("collected_at") or ""))
                except ValueError:
                    continue
                if collected_at.tzinfo is None:
                    collected_at = collected_at.replace(tzinfo=zone)
                if collected_at.astimezone(zone).date() == on:
                    documents.add(
                        str(record.get("file_id") or record.get("source_file") or record["case_id"])
                    )
        return len(documents)

    @property
    def _document_listing(self) -> bool:
        return bool(self._source_records) and self.settings.full_corpus

    def _listing_records(self) -> list[dict[str, object]]:
        records = list(self._records.values())
        if self._document_listing:
            records.extend(
                record
                for record in self._source_records.values()
                if str(record["file_id"]) not in self._cases_by_file
            )
        return records

    def add_case(
        self,
        record: dict[str, object],
        chunks: list[dict[str, object]],
        expansions: list[dict[str, object]],
    ) -> CaseDetail:
        return self.add_cases([(record, chunks, expansions)])[0]

    def add_cases(
        self,
        artifacts: list[tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]],
        *,
        source_text: str | None = None,
        source_texts: dict[str, str] | None = None,
        origin: str | None = None,
    ) -> list[CaseDetail]:
        if source_text is not None and source_texts is not None:
            raise ValueError("单源全文和多源全文不能同时提供")
        records = [item[0] for item in artifacts]
        origin = origin or (str(records[0].get("case_origin")) if records else "web_ingestion")
        if origin not in {"official_crawl", "web_ingestion"}:
            origin = "web_ingestion"
        for record in records:
            record["case_origin"] = origin
            record["source_group"] = origin
        ids = [str(r.get("case_id") or "") for r in records]
        if not ids or any(not cid for cid in ids) or len(set(ids)) != len(ids):
            raise ValueError("案例ID为空或重复")
        with self._lock, FileLock(self.settings.incremental_cases_root / ".write.lock"):
            for group in ("official_crawl", "web_ingestion"):
                self._recover_write(self.settings.source_root(group))
            self.reload()
            existing_keys = {_business_key(r) for r in self._records.values()} - {None}
            existing_hashes = {r.get("source_content_sha256") for r in self._records.values()} - {
                None,
                "",
            }
            artifacts = [
                item
                for item in artifacts
                if str(item[0]["case_id"]) not in self._records
                and _business_key(item[0]) not in existing_keys
                and item[0].get("source_content_sha256") not in existing_hashes
            ]
            if not artifacts:
                raise DuplicateCaseError(ids[0])
            records = [item[0] for item in artifacts]
            root = self.settings.source_root(origin)
            root.mkdir(parents=True, exist_ok=True)
            paths = (
                root / "case_chunks.jsonl",
                root / "case_expansions.jsonl",
                root / "cases.jsonl",
            )
            manifest_path = root / "manifest.json"
            snapshots = {
                path: (path.is_file(), path.stat().st_size if path.is_file() else 0)
                for path in paths
            }
            manifest_original = manifest_path.read_bytes() if manifest_path.is_file() else None
            transaction_id = uuid4().hex
            journal = root / ".pending_write.json"
            _write_json_atomic(
                journal,
                {
                    "id": transaction_id,
                    "files": {p.name: value for p, value in snapshots.items()},
                    "manifest": manifest_original.decode("utf-8") if manifest_original else None,
                },
            )
            connection = None
            try:
                registry = root / "evidence_registry.sqlite3"
                connection = sqlite3.connect(registry)
                connection.executescript(
                    "CREATE TABLE IF NOT EXISTS sources("
                    "file_id TEXT PRIMARY KEY,text TEXT,sha256 TEXT);"
                    "CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,file_id TEXT,data TEXT);"
                    "CREATE TABLE IF NOT EXISTS evidence("
                    "id TEXT PRIMARY KEY,file_id TEXT,data TEXT);"
                    "CREATE TABLE IF NOT EXISTS ingestion_commits(id TEXT PRIMARY KEY);"
                )
                if source_text is not None or source_texts is not None:
                    for record, chunks, _ in artifacts:
                        fid = str(record.get("file_id") or record["case_id"])
                        text = source_texts[fid] if source_texts is not None else source_text
                        if not text:
                            raise ValueError(f"缺少来源全文: {fid}")
                        source_hash = sha256(text.encode()).hexdigest()
                        connection.execute(
                            "INSERT OR REPLACE INTO sources VALUES (?,?,?)",
                            (fid, text, source_hash),
                        )
                        for chunk in chunks:
                            quote = str(chunk.get("chunk_text") or "")
                            start = text.find(quote)
                            if not quote or start < 0:
                                continue
                            unit = {
                                "source_text": quote,
                                "source_spans": [{"start": start, "end": start + len(quote)}],
                                "source_sha256": source_hash,
                                "application_case_ids": [record["case_id"]],
                            }
                            connection.execute(
                                "INSERT INTO units VALUES (?,?,?)",
                                (chunk["chunk_id"], fid, json.dumps(unit, ensure_ascii=False)),
                            )
                _append_jsonl(paths[0], [c for _, chunks, _ in artifacts for c in chunks])
                _append_jsonl(
                    paths[1],
                    [e for _, _, expansions in artifacts for e in expansions],
                )
                _append_jsonl(paths[2], records)
                self._refresh_manifest(root)
                connection.execute("INSERT INTO ingestion_commits VALUES (?)", (transaction_id,))
                connection.commit()
            except Exception:
                if connection is not None:
                    connection.rollback()
                for path, (existed, size) in snapshots.items():
                    _restore_file(path, existed=existed, size=size)
                if manifest_original is not None:
                    manifest_path.write_bytes(manifest_original)
                else:
                    manifest_path.unlink(missing_ok=True)
                manifest_path.with_suffix(".json.tmp").unlink(missing_ok=True)
                journal.unlink(missing_ok=True)
                raise
            finally:
                if connection is not None:
                    connection.close()
            self._mark_index_pending()
            journal.unlink(missing_ok=True)
            for record, chunks, _ in artifacts:
                case_id = str(record["case_id"])
                self._records[case_id] = dict(record)
                fid = str(record.get("file_id") or record.get("source_file") or case_id)
                self._cases_by_file.setdefault(fid, []).append(case_id)
                self._chunks_by_case[case_id] = [dict(chunk) for chunk in chunks]
                self._risk_tags.update(str(tag) for tag in _as_list(record.get("risk_tags")) if tag)
            self._revision += 1
            self._loaded_stamp = self._stamp()
        return [self._to_detail(record, chunks) for record, chunks, _ in artifacts]

    def set_retrieval_status(self, case_id: str, *, active: bool) -> CaseDetail:
        """停用或恢复案例，不删除任何记录与原始材料。"""

        with self._lock, FileLock(self.settings.incremental_cases_root / ".write.lock"):
            self.reload()
            record = dict(self._record(case_id))
            case_id = str(record["case_id"])
            if active and record.get("retrieval_disposition") in {"archive_only", "scope_review"}:
                raise ValueError("该来源需先完成文书类型或保险范围审核，不能直接启用检索")
            record["retrieval_status"] = "active" if active else "inactive"
            record["updated_at"] = datetime.now(UTC).date().isoformat()
            flags = [str(value) for value in _as_list(record.get("quality_flags"))]
            marker = "web_reactivated" if active else "web_deactivated"
            record["quality_flags"] = list(dict.fromkeys([*flags, marker]))
            updated = dict(self._records)
            updated[case_id] = record
            path = self.settings.incremental_cases_root / "status_overrides.json"
            overrides = json.loads(path.read_text("utf-8")) if path.is_file() else {}
            overrides[case_id] = record["retrieval_status"]
            _write_json_atomic(path, overrides)
            self._records = updated
            self._revision += 1
            self._loaded_stamp = self._stamp()
        return self.get_case(case_id)

    def iter_retrieval_documents(self) -> Iterator[RetrievalDocument]:
        active_ids = {
            case_id
            for case_id, record in {**self._records, **self._source_records}.items()
            if record.get("retrieval_status") == "active"
        }
        for chunk in chain.from_iterable(
            _read_jsonl(path) for path in self._paths("case_chunks.jsonl")
        ):
            case_id = str(chunk.get("case_id") or "")
            linked_ids = tuple(
                str(value)
                for value in (_as_list(chunk.get("case_ids")) or [case_id])
                if str(value) in active_ids
            )
            if not linked_ids or not chunk.get("retrieval_enabled", True):
                continue
            if chunk.get("chunk_type") not in {"fact", "evidence"}:
                continue
            text = str(chunk.get("chunk_text") or "").strip()
            if text:
                self._retrieval_metadata[str(chunk.get("chunk_id"))] = {
                    "file_id": chunk.get("file_id"),
                    "source_span": chunk.get("source_span"),
                    "origin": "case_chunk",
                }
                yield RetrievalDocument(
                    unit_id=str(chunk.get("chunk_id")),
                    case_ids=linked_ids,
                    text=text,
                    source="case_chunk",
                    weight=_as_float(chunk.get("retrieval_weight"), 1.0),
                )
        for expansion in chain.from_iterable(
            _read_jsonl(path) for path in self._paths("case_expansions.jsonl")
        ):
            if not expansion.get("retrieval_enabled", False):
                continue
            case_ids = tuple(
                dict.fromkeys(
                    str(link.get("case_id"))
                    for link in _as_dict_list(expansion.get("case_links"))
                    if str(link.get("case_id") or "") in active_ids
                )
            )
            text = str(expansion.get("expansion_text") or "").strip()
            if text and case_ids:
                self._retrieval_metadata[str(expansion.get("expansion_id"))] = {
                    "origin": "case_expansion",
                    "grounding_ids": [
                        str(eid)
                        for link in _as_dict_list(expansion.get("case_links"))
                        for eid in _as_list(link.get("grounding_chunk_ids"))
                    ],
                }
                yield RetrievalDocument(
                    unit_id=str(expansion.get("expansion_id")),
                    case_ids=case_ids,
                    text=text,
                    source="case_expansion",
                    weight=_as_float(expansion.get("retrieval_weight"), 0.35),
                )

    def retrieval_metadata(self, unit_id: str) -> dict[str, object]:
        return self._retrieval_metadata.get(unit_id, {})

    def retrieval_summaries(self) -> tuple[CaseSummary, ...]:
        """Include explicitly unattributed source views without changing party case counts."""
        return (*self.summaries(), *(self._to_summary(r) for r in self._source_records.values()))

    def risk_dictionary(self) -> tuple[str, ...]:
        if not self.settings.risk_dictionary_path.is_file():
            return self.risk_tags
        labels: list[str] = []
        with self.settings.risk_dictionary_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                value = next(
                    (
                        row.get(key)
                        for key in ("risk_type", "risk_tag", "label_name", "风险类型", "标签名称")
                        if row.get(key)
                    ),
                    None,
                )
                if value:
                    labels.append(value.strip())
        return tuple(dict.fromkeys(labels)) or self.risk_tags

    def _record(self, case_id: str) -> dict[str, object]:
        case_id = self._case_aliases.get(case_id, case_id)
        record = self._records.get(case_id) or self._source_records.get(case_id)
        if record is None:
            raise CaseNotFoundError(case_id)
        return record

    def _case_chunks(self, case_id: str) -> list[dict[str, object]]:
        cached = self._chunks_by_case.get(case_id)
        if cached is not None:
            return cached
        chunks = [
            chunk
            for path in self._paths("case_chunks.jsonl")
            for chunk in _read_jsonl(path)
            if chunk.get("case_id") == case_id or case_id in _as_list(chunk.get("case_ids"))
        ]
        self._chunks_by_case[case_id] = chunks
        return chunks

    def _to_summary(self, record: dict[str, object]) -> CaseSummary:
        case_id = str(record["case_id"])
        fid = str(record.get("file_id") or record.get("source_file") or case_id)
        return CaseSummary(
            case_id=case_id,
            case_title=_optional_string(record.get("case_title")),
            party_name=_optional_string(record.get("party_name")),
            penalty_doc_no=_optional_string(record.get("penalty_doc_no")),
            violation_behavior=str(record.get("violation_behavior") or "未提供违规事实"),
            penalty_content=_optional_string(record.get("penalty_content")),
            regulator=_optional_string(record.get("regulator")),
            penalty_date=_as_date(record.get("penalty_date")),
            publish_date=_as_date(record.get("publish_date")),
            institution_type=_optional_string(record.get("institution_type")),
            source_url=_optional_string(record.get("source_url")),
            risk_tags=[str(value) for value in _as_list(record.get("risk_tags"))],
            risk_domains=[str(value) for value in _as_list(record.get("risk_domains"))],
            case_origin=str(record.get("case_origin") or "unknown"),
            source_group=_source_group(record),
            retrieval_status=str(record.get("retrieval_status") or "inactive"),
            evidence_status=str(record.get("evidence_status") or "limited"),
            is_user_managed=_source_group(record) != "original",
            document_id=fid,
            subject_case_count=max(1, len(self._cases_by_file.get(fid, []))),
            document_kind=str(record.get("document_kind") or "penalty"),
            decision_status=str(record.get("decision_status") or "not_individually_verified"),
            insurance_relevance=True
            if record.get("insurance_relevance") is True
            else False
            if record.get("insurance_relevance") is False
            else None,
            review_status=str(record.get("review_status") or "pending"),
            retrieval_disposition=_optional_string(record.get("retrieval_disposition")),
            scope_assessment=dict(_as_dict(record.get("scope_assessment")) or {}),
        )

    def _to_detail(
        self,
        record: dict[str, object],
        chunks: list[dict[str, object]],
    ) -> CaseDetail:
        return CaseDetail(
            **self._to_summary(record).model_dump(),
            case_summary=_optional_string(record.get("case_summary")),
            related_parties=_as_dict_list(record.get("related_parties")),
            entity_relations=_as_dict_list(record.get("entity_relations")),
            legal_basis_refs=_as_dict_list(record.get("legal_basis_refs")),
            source_file=_optional_string(record.get("source_file")),
            source_sha256=_optional_string(record.get("source_sha256")),
            raw_source_file=_optional_string(record.get("raw_source_file")),
            party_type=_optional_string(record.get("party_type")),
            penalty_amounts=_as_dict_list(record.get("penalty_amounts")),
            field_sources=dict(_as_dict(record.get("field_sources")) or {}),
            field_quality={
                str(k): str(v) for k, v in (_as_dict(record.get("field_quality")) or {}).items()
            },
            quality_flags=[str(value) for value in _as_list(record.get("quality_flags"))],
            data_version=str(record.get("data_version") or "case_kb_incremental_v1"),
            evidence_chunks=[
                EvidenceChunk(
                    chunk_id=str(chunk.get("chunk_id")),
                    chunk_type=str(chunk.get("chunk_type") or "fact"),
                    chunk_text=str(chunk.get("chunk_text") or ""),
                    supports_risk_tags=[
                        str(value) for value in _as_list(chunk.get("supports_risk_tags"))
                    ],
                    is_verbatim=bool(chunk.get("is_verbatim", False)),
                    source_span=_as_dict(chunk.get("source_span")),
                )
                for chunk in chunks
            ],
            related_cases=[
                self._to_summary(self._records[cid])
                for cid in self._cases_by_file.get(
                    str(record.get("file_id") or record.get("source_file") or record["case_id"]), []
                )
                if cid != record["case_id"]
            ],
        )

    def _refresh_manifest(self, root: Path) -> None:
        path_manifest = root / "manifest.json"
        manifest: dict[str, Any] = (
            json.loads(path_manifest.read_text(encoding="utf-8"))
            if path_manifest.is_file()
            else {"source_group": root.name}
        )
        counts = manifest.setdefault("counts", {})
        files = manifest.setdefault("files", {})
        for key, path in (
            ("cases", root / "cases.jsonl"),
            ("case_chunks", root / "case_chunks.jsonl"),
            ("case_expansions", root / "case_expansions.jsonl"),
        ):
            rows = _line_count(path)
            counts[key.removeprefix("case_") if key != "cases" else key] = rows
            files[key] = {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "rows": rows,
            }
        manifest["locally_extended"] = True
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        temporary = path_manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path_manifest)


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            yield value


def _append_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _write_jsonl_atomic(path: Path, records: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for record in sorted(records, key=_case_sort_key):
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_manifest_meta(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return {}
    return {
        key: str(text)
        for key in ("data_version", "latest_crawl_at", "latest_penalty_date")
        if (text := value.get(key)) not in (None, "")
    }


def _record_contains(record: dict[str, object], normalized_query: str) -> bool:
    values = (
        record.get("case_id"),
        record.get("case_title"),
        record.get("party_name"),
        record.get("penalty_doc_no"),
        record.get("violation_behavior"),
        record.get("penalty_content"),
        record.get("regulator"),
        " ".join(str(value) for value in _as_list(record.get("risk_tags"))),
    )
    return any(normalized_query in str(value or "").casefold() for value in values)


def _case_sort_key(record: dict[str, object]) -> tuple[int, str]:
    case_id = str(record.get("case_id") or "")
    return (0 if _source_group(record) != "original" else 1, case_id)


def _source_group(record: dict[str, object]) -> str:
    origin = str(record.get("case_origin") or "")
    return origin if origin in {"official_crawl", "web_ingestion"} else "original"


def _business_key(record: dict[str, object]) -> tuple[str, str] | None:
    document = re.sub(r"\s+", "", str(record.get("penalty_doc_no") or ""))
    party = re.sub(r"\s+", "", str(record.get("party_name") or ""))
    return (document, party) if document and party else None


def _optional_string(value: object) -> str | None:
    return str(value) if value not in (None, "") else None


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _as_dict_list(value: object) -> list[dict[str, object]]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _as_dict(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _line_count(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for line in file if line.strip())


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_file(path: Path, *, existed: bool, size: int) -> None:
    """把一次多文件追加恢复到写入前大小。"""

    if not existed:
        path.unlink(missing_ok=True)
        return
    with path.open("r+b") as file:
        file.truncate(size)
        os.fsync(file.fileno())
