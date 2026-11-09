"""Persistent collection jobs. The scheduler collects only; building and use are explicit."""

from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from agent_rag.collection_crawler import crawl_range, import_history
from agent_rag.collection_store import CollectionStore
from agent_rag.knowledge_registry import initialize_usage, select_version, set_usage, usage_state
from agent_rag.knowledge_updates import run_update, update_lock, write_json
from agent_rag.v2_snapshot import file_hash, rows

CHINA = timezone(timedelta(hours=8))


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("INVALID_DATE")
    return date.fromisoformat(value)


def _code(error):
    # Never expose provider errors, credentials, source paths, or arbitrary exception text.
    text = str(error).split(":", 1)[0]
    return text if re.fullmatch(r"[A-Z][A-Z0-9_]{0,120}", text) else type(error).__name__


class CollectionService:
    def __init__(self, root: Path, *, history_roots=None, crawler=None, updater=None):
        self.root = Path(root).resolve()
        self.store = CollectionStore(self.root / "inventory")
        self.registry = self.root / "registry"
        self.history_roots = [Path(p).resolve() for p in history_roots or []]
        self.crawler, self.updater = crawler or crawl_range, updater or run_update
        self._stop = threading.Event()
        self._scheduler = None
        self._threads = []
        self._thread_lock = threading.RLock()
        self._hash_cache = None
        self._recover_jobs()

    def _inside(self, path):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("MANAGED_PATH_OUTSIDE_ROOT")
        return resolved

    def _recover_jobs(self):
        try:
            with update_lock(self.root / "operations"):
                for path in (self.root / "jobs").glob("*.json"):
                    job = json.loads(path.read_text(encoding="utf-8"))
                    if job.get("status") in {"queued", "running"}:
                        job.update(
                            status="interrupted", error="PROCESS_INTERRUPTED_REVIEW_BEFORE_RESUME"
                        )
                        write_json(path, job)
        except ValueError as error:
            if str(error) != "UPDATE_LOCKED":
                raise

    def configure_history(self, paths, *, backfill_since=None):
        roots = [Path(p).resolve() for p in paths]
        if not roots or any(not (p / "index.jsonl").is_file() for p in roots):
            raise ValueError("HISTORY_INDEX_NOT_FOUND")
        if backfill_since is not None:
            _date(backfill_since)
        self.store.set_value("history_roots", [str(p) for p in roots])
        if backfill_since is not None:
            self.store.set_value("backfill_since", backfill_since)
            self.store.set_value("scheduler_error", None)

    def bind(self, config_path: Path):
        """Read trusted build assets; redirect every writable path into this copy."""
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if config.get("schema_version") != "finrisk-web-update/1.0":
            raise ValueError("INVALID_UPDATE_CONFIG")
        base = Path(config["base_snapshot"])
        if not base.is_absolute() or file_hash(base / "manifest.json") != config.get(
            "base_manifest_sha256"
        ):
            raise ValueError("BASE_MANIFEST_HASH_MISMATCH")
        if set(config.get("build_assets", {})) != {"normalizer", "projector", "contract"}:
            raise ValueError("INCOMPLETE_BUILD_ASSETS")
        for asset in config["build_assets"].values():
            if (
                not Path(asset["path"]).is_absolute()
                or file_hash(Path(asset["path"])) != asset["sha256"]
            ):
                raise ValueError("BUILD_ASSET_HASH_MISMATCH")
        # Binding is administrative, serialized with builds/selections and collection.
        with update_lock(self.root / "operations"):
            self._inside(self.registry)
            initialize_usage(
                self.registry, base, expected_manifest_sha256=config["base_manifest_sha256"]
            )
            clean = {
                key: config[key]
                for key in (
                    "schema_version",
                    "base_snapshot",
                    "base_manifest_sha256",
                    "build_assets",
                )
            }
            local_assets = {}
            for name, asset in config["build_assets"].items():
                original = Path(asset["path"])
                target = self._inside(
                    self.root
                    / "build_assets"
                    / (name + "-" + asset["sha256"][:16] + original.suffix)
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = original.read_bytes()
                if target.exists() and target.read_bytes() != raw:
                    raise ValueError("BUILD_ASSET_IMMUTABLE")
                target.write_bytes(raw)
                local_assets[name] = {"path": str(target), "sha256": asset["sha256"]}
            clean["build_assets"] = local_assets
            clean.update(
                registry=str(self.registry),
                crawl_dir=str(self.root / "staging"),
                include_bank_insurance=True,
            )
            write_json(self.root / "build-config.json", clean)
            self._hash_cache = None
        return {"status": "configured", "model_requests_sent": 0}

    def _config(self):
        path = self.root / "build-config.json"
        if not path.is_file():
            raise ValueError("KNOWLEDGE_NOT_CONFIGURED")
        config = json.loads(path.read_text(encoding="utf-8"))
        if self._inside(config["registry"]) != self.registry:
            raise ValueError("INVALID_MANAGED_REGISTRY")
        self._inside(config["crawl_dir"])
        if (
            file_hash(Path(config["base_snapshot"]) / "manifest.json")
            != config["base_manifest_sha256"]
        ):
            raise ValueError("BASE_MANIFEST_HASH_MISMATCH")
        for asset in config["build_assets"].values():
            if file_hash(Path(asset["path"])) != asset["sha256"]:
                raise ValueError("BUILD_ASSET_HASH_MISMATCH")
        return config

    def _baseline_hashes(self):
        if not (self.root / "build-config.json").exists():
            return set()
        config = self._config()
        if self._hash_cache is None:
            manifest = json.loads(
                (Path(config["base_snapshot"]) / "manifest.json").read_text(encoding="utf-8")
            )
            asset = manifest["assets"]["sources"]
            source = Path(asset["path"])
            if file_hash(source) != asset["sha256"]:
                raise ValueError("BASE_SOURCE_HASH_MISMATCH")
            self._hash_cache = {row["text_sha256"] for row in rows(source)}
        return self._hash_cache

    def set_crawl_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("INVALID_CRAWL_ENABLED")
        self.store.set_value("crawl_enabled", enabled)
        return {"crawl_enabled": enabled}

    def jobs(self):
        jobs = [
            json.loads(p.read_text(encoding="utf-8")) for p in (self.root / "jobs").glob("*.json")
        ]
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:50]

    def status(self):
        self._recover_jobs()
        ready, reason, state = False, "尚未绑定基础V2快照及构建资产；可先导入和采集原件。", None
        try:
            self._config()
            state = usage_state(self.registry)
            ready, reason = True, ""
        except (ValueError, OSError, KeyError, TypeError):
            if (self.root / "build-config.json").exists():
                reason = "知识库配置或版本校验未通过，请检查本机配置。"
        allowed = os.environ.get("FINRISK_AGENT_ALLOW_KB_GENERATION") == "1"
        versions = []
        for marker in sorted((self.registry / "versions").glob("*/update-complete.json")):
            try:
                value = json.loads(marker.read_text(encoding="utf-8"))
                manifest = json.loads(
                    (marker.parent / "snapshot/manifest.json").read_text(encoding="utf-8")
                )
                versions.append(
                    {
                        "version": marker.parent.name,
                        "status": "ready"
                        if value.get("status") == "built"
                        and not (marker.parent / "update-failed.json").exists()
                        else "incomplete",
                        "source_count": value.get("source_count"),
                        "new_units": value.get("new_units"),
                        "parent_snapshot_id": manifest.get("parent_snapshot_id"),
                    }
                )
            except (OSError, ValueError):
                continue
        # Local operational results intentionally contain counts/codes, never raw config paths.
        return {
            "settings": {
                "crawl_enabled": self.store.get_value("crawl_enabled", False),
                "usage_enabled": bool(state and state["enabled"]),
                "daily_hour": 8,
            },
            "counts": self.store.counts(),
            "batches": self.store.list_batches(),
            "versions": versions,
            "jobs": self.jobs(),
            "configuration": {
                "ready": ready,
                "generation_allowed": allowed,
                "reason": reason
                or ("未开启收费处理权限；采集不会调用生成模型。" if not allowed else ""),
            },
            "selection": {
                "selected_version": state["selected_version"] if state else None,
                "base_version": state["base_version"] if state else None,
            },
            "last_successful_crawl": self.store.get_value("crawl_success_until"),
            "recommended_since": self.store.get_value("crawl_success_until")
            or self.store.get_value("backfill_since"),
            "scheduler_error": self.store.get_value("scheduler_error"),
        }

    def _validate(self, kind, args):
        if kind not in {
            "import_history",
            "crawl",
            "scheduled_crawl",
            "process",
            "activate",
            "usage",
        }:
            raise ValueError("UNKNOWN_JOB")
        if kind in {"crawl", "scheduled_crawl"}:
            if _date(args.get("since")) > _date(args.get("until")):
                raise ValueError("INVALID_DATE_RANGE")
            if type(args.get("max_pages")) is not int or not 1 <= args["max_pages"] <= 100:
                raise ValueError("INVALID_PAGE_LIMIT")
        if kind == "import_history" and not self._history_roots():
            raise ValueError("HISTORY_NOT_CONFIGURED")
        if kind in {"process", "activate", "usage"}:
            self._config()
        if kind == "usage" and type(args.get("enabled")) is not bool:
            raise ValueError("INVALID_USAGE_ENABLED")
        if kind == "process":
            if os.environ.get("FINRISK_AGENT_ALLOW_KB_GENERATION") != "1":
                raise ValueError("GENERATION_DISABLED")
            cap = args.get("cap_cny")
            if (
                isinstance(cap, bool)
                or not isinstance(cap, (int, float))
                or not math.isfinite(cap)
                or not 0 < cap <= 100
                or type(args.get("max_documents")) is not int
                or not 1 <= args["max_documents"] <= 100
                or type(args.get("max_requests")) is not int
                or not 1 <= args["max_requests"] <= 200
            ):
                raise ValueError("INVALID_UPDATE_LIMITS")
            if args.get("batch_id") not in {b["id"] for b in self.store.list_batches()}:
                raise ValueError("UNKNOWN_BATCH")

    def _history_roots(self):
        return self.history_roots or [
            Path(p).resolve() for p in self.store.get_value("history_roots", [])
        ]

    def submit(self, kind, args):
        self._validate(kind, args)
        lock = update_lock(self.root / "operations")
        try:
            lock.__enter__()
        except ValueError:
            raise ValueError("KNOWLEDGE_JOB_BUSY") from None
        job = {
            "id": uuid4().hex,
            "kind": kind,
            "status": "queued",
            "created_at": datetime.now(UTC).isoformat(),
            "result": None,
            "error": None,
        }
        try:
            write_json(self._inside(self.root / "jobs" / (job["id"] + ".json")), job)
        except BaseException:
            lock.__exit__(None, None, None)
            raise
        thread = threading.Thread(target=self._run, args=(job, args, lock), daemon=True)
        thread.job_id = job["id"]
        with self._thread_lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
            thread.start()
        return thread

    def _run(self, job, args, lock):
        path = self.root / "jobs" / (job["id"] + ".json")
        try:
            job["status"] = "running"
            write_json(path, job)
            result = self._execute(job["kind"], args)
            job.update(
                result=result,
                status="incomplete" if result.get("complete") is False else "completed",
            )
        except Exception as error:
            job.update(status="failed", error=_code(error))
        finally:
            try:
                job["finished_at"] = datetime.now(UTC).isoformat()
                write_json(path, job)
            finally:
                lock.__exit__(None, None, None)

    def _execute(self, kind, args):
        if kind == "import_history":
            batches = []
            for source in self._history_roots():
                batch = self.store.create_batch("legacy_import", {"source_kind": "initial_round"})
                result = import_history(
                    self.store, source, batch["id"], baseline_hashes=self._baseline_hashes()
                )
                batches.append({"batch_id": batch["id"], **self._safe_result(result)})
            return {"complete": all(b["complete"] for b in batches), "batches": batches}
        if kind in {"crawl", "scheduled_crawl"}:
            batch = self.store.create_batch(
                kind, {k: args[k] for k in ("since", "until", "max_pages")}
            )
            result = self.crawler(
                self.store, batch["id"], **args, baseline_hashes=self._baseline_hashes()
            )
            return {"batch_id": batch["id"], **self._safe_result(result)}
        if kind == "activate":
            state = select_version(self.registry, args["version"])
            return {
                "selected_version": state["selected_version"],
                "usage_enabled": state["enabled"],
            }
        if kind == "usage":
            state = set_usage(self.registry, args["enabled"])
            return {
                "usage_enabled": state["enabled"],
                "selected_version": state["selected_version"],
            }
        return self._process(args)

    @staticmethod
    def _safe_result(result):
        result = dict(result)
        result["errors"] = [
            {
                "code": e.get("code", "COLLECTION_ERROR"),
                **{k: e[k] for k in ("column", "page", "line", "doc_id") if k in e},
            }
            for e in result.get("errors", [])
        ]
        return result

    def _process(self, args):
        if self.updater is run_update:
            from agent_rag.maintenance_service import MaintenanceService

            config = self._config()
            maintenance = MaintenanceService(self.root / "maintenance")
            maintenance.bind_snapshot(config["base_snapshot"], config["base_manifest_sha256"])
            registered = maintenance.import_inventory(
                self.store, args["batch_id"], max_documents=args["max_documents"]
            )
            result = {
                "status": "maintenance_queued",
                "maintenance_queued": True,
                **registered,
                "model_requests_sent": 0,
                "published": False,
                "complete": False,
                "message": "已登记待审事件；未调用模型、未生成可使用版本。",
            }
            self.store.set_batch_status(args["batch_id"], "needs_review", result)
            return result
        config = self._config()
        state = usage_state(self.registry)
        base_version = state["selected_version"] or state["base_version"]
        config["build_base_snapshot"] = str(self.registry / "versions" / base_version / "snapshot")
        staging = self._inside(self.root / "staging" / args["batch_id"])
        texts = self._inside(staging / "texts")
        texts.mkdir(parents=True, exist_ok=True)
        index = []
        for offset in range(0, self.store.documents(args["batch_id"], limit=1)["total"], 100):
            for item in self.store.documents(args["batch_id"], offset=offset, limit=100)["items"]:
                if item["baseline_duplicate"]:
                    continue
                document = self.store.read_document(item["digest"])
                path = self._inside(texts / (item["digest"] + ".txt"))
                raw = document["text"].encode("utf-8")
                if path.exists() and path.read_bytes() != raw:
                    raise ValueError("STAGING_ORIGINAL_CHANGED")
                path.write_bytes(raw)
                metadata = next(
                    s["metadata"] for s in item["sources"] if s["batch_id"] == args["batch_id"]
                )
                index.append(
                    {**metadata, "txt_path": str(path), "collection_digest": item["digest"]}
                )
        (staging / "index.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index), encoding="utf-8"
        )
        config["crawl_dir"] = str(staging)
        held_before = sum(
            (self.registry / "candidates" / item["collection_digest"] / "disposition.json").exists()
            for item in index
        )
        self.store.set_batch_status(args["batch_id"], "processing")
        try:
            result = self.updater(
                config,
                generate=True,
                publish=False,
                **{k: args[k] for k in ("cap_cny", "max_documents", "max_requests")},
            )
        except Exception as error:
            self.store.set_batch_status(args["batch_id"], "needs_review", {"error": _code(error)})
            raise
        safe = {
            key: result.get(key)
            for key in (
                "status",
                "version",
                "snapshot_id",
                "new_units",
                "source_count",
                "pending_documents",
                "selected",
                "accepted_documents",
                "held_documents",
                "published",
                "human_review_status",
                "semantic_coverage_verified",
                "generation_budget",
            )
        }
        safe["review_queue"] = [
            {k: item.get(k) for k in ("file_id", "code", "status")}
            for item in result.get("review_queue", [])
        ]
        safe["remaining_documents"] = (
            max(0, (result.get("pending_documents") or 0) - (result.get("accepted_documents") or 0))
            + held_before
        )
        safe["held_documents_in_batch"] = held_before
        safe["complete"] = not safe["remaining_documents"] and not safe["review_queue"]
        batch_status = (
            ("processed" if result.get("status") == "built" else "no_new_documents")
            if safe["complete"]
            else "partially_processed"
        )
        self.store.set_batch_status(args["batch_id"], batch_status, safe)
        return safe

    def tick(self, now=None):
        now = (now or datetime.now(CHINA)).astimezone(CHINA)
        today = now.date().isoformat()
        if (
            not self.store.get_value("crawl_enabled", False)
            or now.hour < 8
            or self.store.get_value("daily_attempt") == today
        ):
            return None
        # Freeze one resumable range: moving until on every day would abandon page cursors.
        current = self.store.get_value("daily_range")
        if not current:
            since = self.store.get_value("crawl_success_until") or self.store.get_value(
                "backfill_since"
            )
            if not since:
                self.store.set_value("scheduler_error", "BACKFILL_DATE_NOT_CONFIGURED")
                return None
            current = {"since": since, "until": today, "max_pages": 2}
        try:
            thread = self.submit("scheduled_crawl", current)
        except ValueError as error:
            if str(error) == "KNOWLEDGE_JOB_BUSY":
                return None
            raise
        self.store.set_value("daily_attempt", today)
        self.store.set_value("daily_range", current)
        self.store.set_value("scheduler_error", None)
        return thread

    def start(self):
        if self._scheduler and self._scheduler.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    current = self.store.get_value("daily_range")
                    if (
                        current
                        and self.store.get_value("crawl_success_until", "") >= current["until"]
                    ):
                        self.store.set_value("daily_range", None)
                    self.tick()
                except Exception as error:
                    self.store.set_value("scheduler_error", _code(error))
                self._stop.wait(30)

        self._scheduler = threading.Thread(target=loop, daemon=True)
        self._scheduler.start()

    def close(self):
        self._stop.set()
        if self._scheduler:
            self._scheduler.join(2)
