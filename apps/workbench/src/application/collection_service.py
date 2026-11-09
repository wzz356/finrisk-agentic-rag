"""Daily incremental collection, durable checkpoints, ingestion and index refresh."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from application.case_service import CaseManagementService
from application.retrieval_service import CaseRetrievalService
from offline.insurance_relevance import evaluate_insurance_scope
from offline.penalty_crawler.nfra import collect_window
from offline.penalty_crawler.polite_http import PoliteHTTPClient, atomic_json
from settings import AppSettings
from storage.case_repository import DuplicateCaseError
from storage.file_lock import FileLock

SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")
LOGGER = logging.getLogger(__name__)
FINISHED = {"ingested", "duplicate", "non_insurance", "bank_party", "scope_review"}
JsonObject = dict[str, Any]


def _read(path: Path) -> JsonObject:
    import json

    return json.loads(path.read_text("utf-8")) if path.is_file() else {}


class CollectionService:
    def __init__(
        self,
        settings: AppSettings,
        case_service: CaseManagementService,
        retrieval_service: CaseRetrievalService | None = None,
    ) -> None:
        self.settings, self.case_service = settings, case_service
        self.retrieval_service = retrieval_service
        self.root = settings.collection_root
        self.state_path = self.root / "job_state.json"
        self.index_state_path = settings.incremental_cases_root / "index_state.json"
        hour, minute = (int(value) for value in settings.collection_daily_time.split(":"))
        self.daily_time = time(hour, minute, tzinfo=SHANGHAI)
        self.columns = tuple(
            dict.fromkeys(int(c.strip()) for c in settings.collection_columns.split(","))
        )
        if not self.columns or set(self.columns) - {4113, 4114, 4115}:
            raise ValueError("采集栏目仅支持 4113/4114/4115")
        self._index_lock = asyncio.Lock()
        self._stop = threading.Event()
        self._task: asyncio.Task[None] | None = None

    def activate(self, now: datetime | None = None) -> JsonObject:
        now = (now or datetime.now(SHANGHAI)).astimezone(SHANGHAI)
        with FileLock(self.root / ".job.lock", timeout=0):
            state = _read(self.state_path)
            if not state:
                start = (
                    date.fromisoformat(self.settings.collection_start_date)
                    if self.settings.collection_start_date
                    else now.date()
                )
                state = {
                    "schema_version": 1,
                    "start_date": start.isoformat(),
                    "activated_at": now.isoformat(),
                    "documents": {},
                    "pending_details": {},
                    "processed_hashes": {},
                    "column_watermarks": {},
                    "status": "waiting",
                    "index_status": "ready",
                }
                atomic_json(self.state_path, state)
            return state

    def _next_run(self, state: JsonObject, now: datetime) -> datetime:
        scheduled = datetime.combine(now.date(), self.daily_time)
        attempted = str(state.get("last_attempt_date") or "") >= now.date().isoformat()
        starts = date.fromisoformat(state.get("start_date") or now.date().isoformat())
        if attempted or starts >= now.date():
            scheduled = datetime.combine(
                max(starts + timedelta(days=1), now.date() + timedelta(days=1)), self.daily_time
            )
        else:
            scheduled = max(scheduled, now)
        retry = datetime.fromtimestamp(float(state.get("retry_at") or 0), SHANGHAI)
        return max(scheduled, retry)

    def status(self, now: datetime | None = None) -> JsonObject:
        now = (now or datetime.now(SHANGHAI)).astimezone(SHANGHAI)
        state = _read(self.state_path)
        return {
            key: value
            for key, value in {
                "enabled": self.settings.collection_enabled,
                "start_date": state.get("start_date"),
                "daily_time": self.settings.collection_daily_time,
                "timezone": "Asia/Shanghai",
                "today": now.date().isoformat(),
                "today_crawled_cases": self.case_service.repository.count_collected_documents(
                    now.date(), zone=SHANGHAI
                ),
                "next_run_at": self._next_run(state, now).isoformat()
                if self.settings.collection_enabled
                else None,
                "status": state.get("status", "not_activated"),
                "last_attempt_at": state.get("last_attempt_at"),
                "last_finished_at": state.get("last_finished_at"),
                "successful_until": state.get("successful_until"),
                "column_watermarks": state.get("column_watermarks", {}),
                "last_result": state.get("last_result", {}),
                "index_status": _read(self.index_state_path).get(
                    "status", state.get("index_status", "ready")
                ),
                "pending_parts": self._pending(state),
                "pending_documents": len(state.get("pending_details", {})),
                "scope_review_parts": sum(
                    part.get("status") == "scope_review"
                    for document in state.get("documents", {}).values()
                    for part in document.get("parts", [])
                ),
                "retry_at": state.get("retry_at"),
                "limits": {
                    "pages_per_column": self.settings.collection_max_pages,
                    "details": self.settings.collection_max_details,
                    "requests": self.settings.collection_max_requests,
                    "delay_seconds": self.settings.collection_delay_seconds,
                    "jitter_seconds": self.settings.collection_jitter_seconds,
                },
            }.items()
        }

    @staticmethod
    def _pending(state: JsonObject) -> int:
        return sum(
            part.get("status") not in FINISHED
            for doc in state.get("documents", {}).values()
            for part in doc.get("parts", [])
        )

    def plan(self, now: datetime | None = None, *, through_today: bool = False) -> JsonObject:
        now = (now or datetime.now(SHANGHAI)).astimezone(SHANGHAI)
        state = _read(self.state_path)
        start = date.fromisoformat(
            state.get("start_date") or self.settings.collection_start_date or now.date().isoformat()
        )
        watermark = state.get("successful_until")
        since = (
            max(
                start,
                date.fromisoformat(watermark)
                - timedelta(days=self.settings.collection_overlap_days - 1),
            )
            if watermark
            else start
        )
        until = now.date() - timedelta(days=0 if through_today else 1)
        return {
            "start_date": start.isoformat(),
            "since": since.isoformat(),
            "until": until.isoformat(),
            "has_window": since <= until,
            "columns": self.columns,
            "network_requests": 0,
            "reason": "日期窗口内采增量；不访问起始日期前的详情",
        }

    async def refresh_indexes(self) -> bool:
        if self.retrieval_service is None:
            return True
        async with self._index_lock:
            lock = FileLock(self.settings.incremental_cases_root / ".index.lock", timeout=120)
            acquired = False
            try:
                await asyncio.to_thread(lock.__enter__)
                acquired = True
                atomic_json(
                    self.index_state_path,
                    {"status": "building", "updated_at": datetime.now(SHANGHAI).isoformat()},
                )
                await asyncio.to_thread(self.case_service.repository.reload)
                await asyncio.to_thread(self.retrieval_service.rebuild)
                await asyncio.to_thread(self.retrieval_service.warmup, allow_build=True)
                if (
                    self.retrieval_service._indexed_revision
                    != self.case_service.repository.revision
                ):
                    raise RuntimeError("构建期间新增了案例，等待下一轮刷新")
                atomic_json(
                    self.index_state_path,
                    {"status": "ready", "updated_at": datetime.now(SHANGHAI).isoformat()},
                )
                return True
            except Exception:
                atomic_json(
                    self.index_state_path,
                    {"status": "pending", "updated_at": datetime.now(SHANGHAI).isoformat()},
                )
                LOGGER.exception("新增案例已存储，向量索引等待下一轮更新")
                return False
            finally:
                if acquired:
                    lock.__exit__(None, None, None)

    async def run_once(
        self,
        *,
        now: datetime | None = None,
        through_today: bool = False,
        collect_only: bool = False,
        force: bool = False,
        client: PoliteHTTPClient | None = None,
    ) -> JsonObject:
        now = (now or datetime.now(SHANGHAI)).astimezone(SHANGHAI)
        try:
            self.activate(now)
            lock = FileLock(self.root / ".job.lock", timeout=0)
            lock.__enter__()
        except TimeoutError:
            return {"status": "already_running", "requests": 0}
        try:
            state = _read(self.state_path)
            if not force and state.get("last_attempt_date") == now.date().isoformat():
                return {"status": "already_attempted_today", "requests": 0}
            if float(state.get("retry_at") or 0) > now.timestamp():
                return {"status": "site_paused", "requests": 0, "retry_at": state["retry_at"]}
            state.update(
                last_attempt_date=now.date().isoformat(),
                last_attempt_at=now.isoformat(),
                status="running",
            )
            atomic_json(self.state_path, state)
            # Download receipts survive an interruption before the job checkpoint was written.
            for receipt in sorted((self.root / "documents").glob("*/metadata.json")):
                doc = _read(receipt)
                if doc.get("publish_date", "") >= state["start_date"]:
                    state["documents"].setdefault(f"nfra:{doc['doc_id']}", doc)
            plan = self.plan(now, through_today=through_today)
            saved_pending = _read(self.root / "pending_details.json")
            state["pending_details"] = {**state.get("pending_details", {}), **saved_pending}
            state["pending_details"] = {
                key: value
                for key, value in state["pending_details"].items()
                if key not in state["documents"]
            }
            client = client or PoliteHTTPClient(
                self.root / "http_cache",
                delay=self.settings.collection_delay_seconds,
                jitter=self.settings.collection_jitter_seconds,
                max_requests=self.settings.collection_max_requests,
                cancelled=self._stop.is_set,
            )
            errors: list[str] = []
            batch = None
            if plan["has_window"]:
                batch = await asyncio.to_thread(
                    collect_window,
                    client,
                    self.root,
                    since=date.fromisoformat(plan["since"]),
                    until=date.fromisoformat(plan["until"]),
                    columns=self.columns,
                    known_documents=state["documents"],
                    pending_details=state["pending_details"],
                    max_pages=self.settings.collection_max_pages,
                    max_details=self.settings.collection_max_details,
                )
                state["documents"].update(batch.documents)
                state["pending_details"] = batch.pending_details
                errors.extend(batch.errors)
                if batch.retry_at:
                    state["retry_at"] = batch.retry_at
                atomic_json(self.state_path, state)
            created = duplicates = filtered = attempts = 0
            if not collect_only:
                for doc in sorted(
                    state["documents"].values(), key=lambda d: d["publish_date"], reverse=True
                ):
                    for part in doc["parts"]:
                        if (
                            self._stop.is_set()
                            or attempts >= self.settings.collection_max_details * 2
                        ):
                            break
                        if part.get("status") in FINISHED:
                            continue
                        if part["sha256"] in state["processed_hashes"]:
                            part["status"] = "duplicate"
                            duplicates += 1
                            continue
                        path = (self.root / part["path"]).resolve()
                        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
                            errors.append("待入库文书原件缺失")
                            continue
                        text = path.read_text("utf-8")
                        scope = evaluate_insurance_scope(text)
                        part["scope_assessment"] = scope
                        if scope["route"] == "scope_review":
                            part["status"] = "scope_review"
                            filtered += 1
                        else:
                            attempts += 1
                            try:
                                result = await self.case_service.ingest_file(
                                    path,
                                    original_filename=f"nfra_{doc['doc_id']}_{path.name}",
                                    origin="official_crawl",
                                    source_metadata={
                                        "source_url": doc["source_url"],
                                        "publish_date": doc["publish_date"],
                                        "official_doc_id": str(doc["doc_id"]),
                                        "official_column": doc["column"],
                                        "source_snapshot_file": str(
                                            (self.root / doc["raw_path"]).relative_to(
                                                self.settings.project_root
                                            )
                                        ),
                                        "source_content_sha256": part["sha256"],
                                        "collected_at": now.isoformat(),
                                        "scope_assessment": scope,
                                    },
                                )
                                part.update(
                                    status="ingested", case_ids=[c.case_id for c in result.cases]
                                )
                                part["llm_used"] = bool(getattr(result, "llm_used", False))
                                created += len(result.cases)
                                state["index_status"] = "pending"
                            except DuplicateCaseError:
                                part["status"] = "duplicate"
                                state["index_status"] = "pending"
                                duplicates += 1
                            except ValueError as error:
                                if "未识别到保险相关处罚案例" in str(error):
                                    part["status"] = "non_insurance"
                                    filtered += 1
                                else:
                                    errors.append(str(error)[:300])
                                    part["last_error"] = str(error)[:300]
                            except Exception as error:
                                errors.append(
                                    f"文书 {doc['doc_id']} 入库失败: {type(error).__name__}"
                                )
                                part["last_error"] = str(error)[:300]
                        if part.get("status") in FINISHED:
                            state["processed_hashes"][part["sha256"]] = part.get("case_ids", [])
                        atomic_json(self.state_path, state)
            if batch is not None:
                for column, info in batch.columns.items():
                    unfinished = any(
                        part.get("status") not in FINISHED
                        for doc in state["documents"].values()
                        if str(doc["column"]) == column
                        for part in doc["parts"]
                    )
                    unfinished = unfinished or any(
                        str(doc["column"]) == column for doc in state["pending_details"].values()
                    )
                    if info.get("complete") and not unfinished:
                        state["column_watermarks"][column] = plan["until"]
                if all(str(c) in state["column_watermarks"] for c in self.columns):
                    state["successful_until"] = min(
                        state["column_watermarks"][str(c)] for c in self.columns
                    )
            if (
                state.get("index_status") == "pending"
                or _read(self.index_state_path).get("status") in {"pending", "building"}
            ) and not collect_only:
                if await self.refresh_indexes():
                    state["index_status"] = "ready"
            pending = self._pending(state)
            status = (
                "paused"
                if batch and batch.paused
                else "partial"
                if errors or pending or state["pending_details"] or (batch and not batch.complete)
                else "collected"
                if collect_only
                else "success"
                if batch
                else "waiting"
            )
            state.update(
                status=status,
                last_finished_at=datetime.now(SHANGHAI).isoformat(),
                last_result={
                    **plan,
                    "requests": client.requests,
                    "cache_hits": client.cache_hits,
                    "downloaded_documents": len(batch.documents) if batch else 0,
                    "skipped_known": batch.skipped_known if batch else 0,
                    "skipped_old": batch.skipped_old if batch else 0,
                    "created_cases": created,
                    "duplicates": duplicates,
                    "filtered": filtered,
                    "pending_parts": pending,
                    "pending_documents": len(state["pending_details"]),
                    "columns": batch.columns if batch else {},
                    "errors": errors,
                },
            )
            atomic_json(self.state_path, state)
            return {"status": status, **state["last_result"], "index_status": state["index_status"]}
        except BaseException:
            state = _read(self.state_path)
            state["status"] = "interrupted"
            atomic_json(self.state_path, state)
            raise
        finally:
            lock.__exit__(None, None, None)

    async def _loop(self) -> None:
        try:
            self.activate()
        except TimeoutError:
            pass
        while not self._stop.is_set():
            try:
                now = datetime.now(SHANGHAI)
                state = _read(self.state_path)
                if self._next_run(state, now) <= now:
                    await self.run_once(now=now)
            except Exception:
                LOGGER.exception("每日案例采集未完成，已保留进度")
            await asyncio.to_thread(self._stop.wait, 30)

    def start(self) -> None:
        if self.settings.collection_enabled and self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop(), name="finrisk-daily-collection")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
