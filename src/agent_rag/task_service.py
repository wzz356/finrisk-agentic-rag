"""Python单实例原子准入；持久化幂等记录，读取旧任务不启动新调用。"""

import asyncio
import hashlib
import json
import os
import re
import threading
import time
import uuid
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from agent_rag.budget_profiles import (
    LEGACY_V2_PROFILE,
    MAX_SAFE_INTEGER,
    budget_profile_sha256,
    load_budget_profile,
    validate_budget_config,
)
from agent_rag.contracts import (
    PROJECT_ROOT,
    canonical_json,
    load_contract,
    tool_schemas_for_profile,
    validate_schema,
)
from agent_rag.input_registry import InputRegistry, text_hash
from agent_rag.knowledge_snapshot import KnowledgeSnapshot
from agent_rag.planned_execution_contract import PLANNED_PROFILE, planned_execution_contract
from agent_rag.rerank_backend import RerankTaskControl
from agent_rag.tools import KnowledgeTools
from agent_rag.v2_snapshot import PROFILE as V2_PROFILE

TERMINAL = {"completed", "partial", "failed", "interrupted", "rejected"}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


class TaskService:
    def __init__(self, root: Path, snapshot: KnowledgeSnapshot) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot = snapshot
        self.lock = threading.Lock()
        self.records: dict[str, dict[str, Any]] = {}
        self.tools: dict[str, KnowledgeTools] = {}
        self.controls: dict[str, RerankTaskControl] = {}
        self.active_task_id: str | None = None
        self.jobs: set[asyncio.Task[None]] = set()
        self.baseline_config = json.loads(
            (PROJECT_ROOT / "configs/agent_rag/mock-baseline.json").read_text(encoding="utf-8")
        )
        self.snapshot_manifest = json.loads(canonical_json(snapshot.manifest))
        self.profile = snapshot.manifest.get("retrieval_profile", "bm25-source-v1")
        self.rerank_mode = snapshot.manifest.get("rerank_mode", "not_enabled")
        default_profile_id = self.baseline_config["profile_id"]
        if self.profile == V2_PROFILE:
            if self.rerank_mode not in {"replay", "live"}:
                raise ValueError("RERANK_MODE_NOT_READY")
            default_profile_id = f"mock-v2-flash-{self.rerank_mode}-v1"
        self.config = self._profile_config(default_profile_id)
        for path in self.root.glob("*/admission.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            task_id = record["task_id"]
            if not re.fullmatch(r"[a-f0-9]{32}", task_id):
                continue
            if record["task_status"] not in TERMINAL:
                report_path = path.parent / "report.json"
                report = (
                    json.loads(report_path.read_text(encoding="utf-8"))
                    if report_path.exists()
                    else {}
                )
                record["task_status"] = (
                    report.get("task_status")
                    if report.get("task_status") in TERMINAL
                    else "interrupted"
                )
                record["restart_action"] = "read_only_no_auto_resume"
                atomic_json(path, record)
            self.records[task_id] = record

    def _profile_config(self, profile_id: str) -> dict[str, Any]:
        catalog = json.loads(
            (PROJECT_ROOT / "configs/agent_rag/retrieval-profiles.json").read_text(encoding="utf-8")
        )["profiles"]
        entry = catalog.get(profile_id)
        if not entry or entry["retrieval_profile"] != self.profile:
            raise ValueError("PROFILE_NOT_READY")
        if self.profile == V2_PROFILE and entry["rerank_mode"] != self.rerank_mode:
            raise ValueError("RERANK_MODE_MISMATCH")
        overrides = json.loads(
            (PROJECT_ROOT / "configs/agent_rag" / entry["config"]).read_text(encoding="utf-8")
        )
        overrides.pop("extends", None)
        config = {**self.baseline_config, **overrides}
        budget_id = config.get("budget_profile_id")
        if budget_id != entry.get("budget_profile_id"):
            raise ValueError("INVALID_BUDGET_CONFIG: unregistered budget profile binding")
        if budget_id is not None:
            budget = load_budget_profile(budget_id)
            config.update({key: value for key, value in budget.items() if key != "profile_id"})
            config.update(
                budget_profile_sha256=budget_profile_sha256(budget),
                max_external_retrieval_requests=budget["rerank_max_requests"],
            )
        validate_budget_config(config)
        schemas = tool_schemas_for_profile(self.profile)
        if (
            config["max_queries_per_search"]
            != schemas["search_cases"]["properties"]["queries"]["maxItems"]
        ):
            raise ValueError("TOOL_CONTRACT_CONFIG_MISMATCH")
        return config

    @staticmethod
    def _rerank_limits(budget_profile_id: str = LEGACY_V2_PROFILE) -> tuple[float, int]:
        budget = load_budget_profile(budget_profile_id)
        cap = os.environ.get("FINRISK_AGENT_RERANK_CAP_CNY")
        requests = os.environ.get("FINRISK_AGENT_RERANK_MAX_REQUESTS")
        if cap is None or requests is None:
            raise ValueError("RERANK_LIMITS_REQUIRED")
        try:
            micro_cny = Decimal(cap) * 1_000_000
            if (
                not micro_cny.is_finite()
                or micro_cny != micro_cny.to_integral_value()
                or not 1 <= micro_cny <= budget["rerank_cap_micro_cny"]
                or not re.fullmatch(r"[1-9][0-9]*", requests.strip())
            ):
                raise ValueError("INVALID_RERANK_LIMITS")
            max_requests = int(requests)
            if max_requests > min(budget["rerank_max_requests"], MAX_SAFE_INTEGER):
                raise ValueError("INVALID_RERANK_LIMITS")
        except (DecimalException, ValueError, OverflowError) as error:
            raise ValueError("INVALID_RERANK_LIMITS") from error
        return int(micro_cny) / 1_000_000, max_requests

    def admit(
        self, request: dict[str, Any], idempotency_key: str | None
    ) -> tuple[dict[str, Any], bool]:
        admission_started = time.perf_counter()
        profile_id = request.get("experiment_profile_id") or self.config["profile_id"]
        task_config = self._profile_config(profile_id)
        rerank_limits = None
        if self.profile == V2_PROFILE and task_config["rerank_mode"] == "live":
            if os.environ.get("FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE") != "1":
                raise ValueError("RETRIEVAL_LIVE_MODE_NOT_ENABLED")
            rerank_limits = self._rerank_limits(
                task_config.get("budget_profile_id", LEGACY_V2_PROFILE)
            )
            task_config.update(
                rerank_cap_cny=rerank_limits[0],
                rerank_max_requests=rerank_limits[1],
                max_external_retrieval_requests=rerank_limits[1],
            )
            if task_config.get("budget_profile_id"):
                task_config["rerank_cap_micro_cny"] = int(
                    Decimal(str(rerank_limits[0])) * 1_000_000
                )
                task_config["task_total_cost_cap_micro_cny"] = (
                    task_config["rerank_cap_micro_cny"] + task_config["model_trial_cap_micro_cny"]
                )
        if task_config["execution_mode"] == "live":
            if os.environ.get("FINRISK_AGENT_ALLOW_LIVE") != "1":
                raise ValueError("LIVE_MODE_NOT_ENABLED")
            if self.profile == V2_PROFILE:
                if not os.environ.get("FINRISK_AGENT_TRIAL_LEDGER"):
                    raise ValueError("MODEL_TRIAL_LEDGER_REQUIRED")
                task_config.update(
                    model_trial_cap_cny=1, task_total_cost_cap_cny=task_config["rerank_cap_cny"] + 1
                )
        if task_config.get("budget_profile_id"):
            task_config.update(
                model_trial_cap_cny=task_config["model_trial_cap_micro_cny"] / 1_000_000,
                task_total_cost_cap_cny=task_config["task_total_cost_cap_micro_cny"] / 1_000_000,
            )
        if request.get("engine", "controlled_agent") not in {"controlled_agent", "fixed_rag"}:
            raise ValueError("INVALID_ENGINE")
        scenario = request.get("mock_scenario", "generic")
        if scenario not in {"generic", "gift-example"}:
            raise ValueError("INVALID_MOCK_SCENARIO")
        if task_config["execution_mode"] == "live" and scenario != "generic":
            raise ValueError("LIVE_CANNOT_USE_SIMULATED_JUDGMENT")
        if scenario == "gift-example" and request["text"] != "购买这份保险就送购物卡。":
            raise ValueError("MOCK_GIFT_REQUIRES_EXACT_SYNTHETIC_INPUT")
        if idempotency_key is not None and (
            not idempotency_key.strip() or len(idempotency_key) > 128
        ):
            raise ValueError("INVALID_IDEMPOTENCY_KEY")
        task_id = uuid.uuid4().hex
        registry = InputRegistry(request["text"], request.get("user_context"), task_id)
        task_config["engine"] = request.get("engine", "controlled_agent")
        if task_config.get("budget_profile_id"):
            if task_config["engine"] == "fixed_rag":
                task_config["orchestration_mode"] = "planned_fixed"
            validate_budget_config(task_config)
        binding = text_hash(
            canonical_json(
                {
                    "input_sha256": registry.input_sha256,
                    "profile_id": profile_id,
                    "engine": request.get("engine", "controlled_agent"),
                    "mock_scenario": scenario,
                    "snapshot_id": self.snapshot.manifest["snapshot_id"],
                    "config_sha256": text_hash(canonical_json(task_config)),
                }
            )
        )
        with self.lock:
            if idempotency_key:
                prior = next(
                    (
                        r
                        for r in self.records.values()
                        if r.get("idempotency_key") == idempotency_key
                    ),
                    None,
                )
                if prior:
                    if prior["binding_sha256"] != binding:
                        raise ValueError("IDEMPOTENCY_CONFLICT")
                    return dict(prior), True
            if self.active_task_id is not None:
                raise ValueError("CAPACITY_EXCEEDED")
            directory = self.root / task_id
            directory.mkdir()
            record = {
                "task_id": task_id,
                "task_status": "prepared",
                "idempotency_key": idempotency_key,
                "binding_sha256": binding,
                "engine": request.get("engine", "controlled_agent"),
                "execution_mode": task_config["execution_mode"],
                "profile_id": profile_id,
                "retrieval_profile": self.profile,
                "rerank_mode": self.rerank_mode,
                "mock_scenario": scenario,
                "created_at": time.time(),
                "admission_ms": round((time.perf_counter() - admission_started) * 1000, 2),
            }
            manifest = {
                "schema_version": "finrisk-manifest/1.0",
                "task_id": task_id,
                "snapshot_id": self.snapshot.manifest["snapshot_id"],
                "dictionary_sha256": self.snapshot.manifest["dictionary_sha256"],
                "input_sha256": registry.input_sha256,
                "text_sha256": registry.text_sha256,
                "text_version": registry.text_version,
                "config_sha256": text_hash(canonical_json(task_config)),
                "execution_mode": task_config["execution_mode"],
                "profile_id": profile_id,
                "retrieval_profile": self.profile,
                "rerank_mode": self.rerank_mode,
                "snapshot": self.snapshot_manifest,
                "config": task_config,
                "prompt_sha256": hashlib.sha256(
                    (PROJECT_ROOT / "configs/agent_rag/system-prompt.txt").read_bytes()
                ).hexdigest(),
                "contract_version": "finrisk-model/1.0",
                "tool_contract_sha256": text_hash(
                    canonical_json(tool_schemas_for_profile(self.profile))
                ),
                "request_admission_policy_sha256": text_hash(
                    canonical_json(
                        load_contract("request-admission-policy.json")["policies"][
                            task_config.get("request_admission_policy", "legacy-v1")
                        ]
                    )
                ),
                "model_wire_version": "finrisk-model-wire/1.2",
                "model_wire_sha256": hashlib.sha256(
                    (PROJECT_ROOT / "contracts/agent_rag/model-wire.schema.json").read_bytes()
                ).hexdigest(),
                "criteria_sha256": hashlib.sha256(
                    (PROJECT_ROOT / "contracts/agent_rag/risk-criteria.json").read_bytes()
                ).hexdigest(),
            }
            if task_config.get("budget_profile_sha256"):
                manifest["budget_profile_sha256"] = task_config["budget_profile_sha256"]
            if profile_id == PLANNED_PROFILE:
                manifest.update(planned_execution_contract(task_config, manifest))
            atomic_json(directory / "input.json", registry.export())
            atomic_json(directory / "segments.json", registry.segments)
            atomic_json(directory / "manifest.json", manifest)
            atomic_json(directory / "admission.json", record)
            control = None
            if rerank_limits is not None:
                remaining = task_config["task_deadline_seconds"] - (
                    time.perf_counter() - admission_started
                )
                if remaining <= 0:
                    raise ValueError("TASK_DEADLINE_EXCEEDED")
                control = RerankTaskControl(
                    directory / "rerank-ledger.json", *rerank_limits, timeout_seconds=remaining,
                    expected_budget_profile_id=task_config.get("budget_profile_id"),
                    expected_budget_profile_sha256=task_config.get("budget_profile_sha256"),
                )
                self.controls[task_id] = control
            self.records[task_id] = record
            self.tools[task_id] = KnowledgeTools(
                self.snapshot, registry, control=control, audit_dir=directory / "retrieval-audits"
            )
            self.active_task_id = task_id
            return dict(record), False

    def get(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.records:
            raise KeyError("TASK_NOT_FOUND")
        record = dict(self.records[task_id])
        # 中断任务的迟到旧进程结果仅留在诊断文件，不能重开或覆盖对外状态。
        if record["task_status"] == "interrupted":
            return {
                **record,
                "report": None,
                "message": "运行已中断，未自动恢复模型调用。",
                **self._progress(task_id, "interrupted"),
            }
        path = self.root / task_id / "report.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            return {
                **record,
                "task_status": report["task_status"],
                "report": report,
                **self._progress(task_id, report["task_status"]),
            }
        return {**record, "report": None, **self._progress(task_id, record["task_status"])}

    def _progress(self, task_id: str, task_status: str) -> dict[str, Any]:
        from datetime import datetime

        directory = self.root / task_id
        contract = load_contract("progress.schema.json")

        def valid(value: Any) -> dict[str, Any]:
            validate_schema(value, contract)
            if value["task_id"] != task_id:
                raise ValueError("PROGRESS_TASK_MISMATCH")
            return value

        snapshot = None
        try:
            path = directory / "progress.json"
            if path.stat().st_size <= 64_000:
                snapshot = valid(json.loads(path.read_text(encoding="utf-8")))
                updated = datetime.fromisoformat(snapshot["updated_at"].replace("Z", "+00:00"))
                if updated.tzinfo is None:
                    raise ValueError("PROGRESS_TIMEZONE_MISSING")
                elapsed = max(0, time.time() - updated.timestamp())
                snapshot["time"]["remaining_seconds"] = max(
                    0, snapshot["time"]["remaining_seconds"] - int(elapsed)
                )
                if task_status in {"failed", "interrupted", "rejected"}:
                    labels = {
                        "failed": "执行失败",
                        "interrupted": "执行中断",
                        "rejected": "任务被拒绝",
                    }
                    label = labels[task_status]
                    snapshot.update(
                        task_status=task_status,
                        phase=task_status,
                        action="incomplete",
                        stage_label=label,
                        message=f"{label}；未自动重新运行。",
                    )
                if task_status in TERMINAL:
                    snapshot["time"]["remaining_seconds"] = 0
        except (OSError, ValueError, KeyError, TypeError):
            snapshot = None
        history = []
        try:
            path = directory / "progress_events.jsonl"
            if path.stat().st_size <= 1_000_000:
                for line in path.read_text(encoding="utf-8").splitlines()[-10:]:
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict) and event.get("type") == "progress.snapshot":
                            history.append(valid(event["data"]))
                    except (ValueError, KeyError, TypeError):
                        continue
        except (OSError, UnicodeError):
            pass
        return {"progress": snapshot, "progress_history": history}

    def bundle(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.records:
            raise KeyError("TASK_NOT_FOUND")
        record = self.records[task_id]
        manifest = json.loads((self.root / task_id / "manifest.json").read_text(encoding="utf-8"))
        return {
            "input": json.loads((self.root / task_id / "input.json").read_text(encoding="utf-8")),
            "snapshot": manifest.get("snapshot", self.snapshot_manifest),
            "definitions": list(self.snapshot.definitions.values()),
            "config": manifest["config"],
            "mock_scenario": record["mock_scenario"],
            "execution_contract": (
                {
                    key: manifest[key]
                    for key in (
                        "model_wire_sha256",
                        "criteria_sha256",
                        "prompt_sha256",
                        "tool_contract_sha256",
                        "request_admission_policy_sha256",
                        "budget_profile_sha256",
                        "agent_loop_schema_sha256",
                        "agent_loop_protocol_sha256",
                        "profile_sha256",
                    )
                    if key in manifest
                }
                if manifest.get("model_wire_sha256")
                else None
            ),
        }

    def tool(self, task_id: str) -> KnowledgeTools:
        if task_id not in self.tools:
            if task_id not in self.records:
                raise KeyError("TASK_NOT_FOUND")
            directory = self.root / task_id
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["snapshot_id"] != self.snapshot.manifest["snapshot_id"]:
                raise ValueError("SOURCE_SNAPSHOT_MISMATCH")
            registered = json.loads((directory / "input.json").read_text(encoding="utf-8"))
            registry = InputRegistry(registered["text"], registered["user_context"], task_id)
            if (
                registry.input_sha256 != manifest["input_sha256"]
                or registry.text_sha256 != manifest["text_sha256"]
            ):
                raise ValueError("INPUT_HASH_MISMATCH")
            restored = KnowledgeTools(self.snapshot, registry)
            restored.search_disabled = True
            report_path = directory / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                for evidence in report.get("retrieved_evidence", []):
                    restored.evidence[evidence["evidence_id"]] = evidence
            self.tools[task_id] = restored
        return self.tools[task_id]

    def cancel_retrieval(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.records:
            raise KeyError("TASK_NOT_FOUND")
        control = self.controls.get(task_id)
        if control is not None:
            control.cancel()
            return {"cancelled": True, **control.meta()}
        # Historical tasks stay read-only: never construct a fresh billable control.
        path = self.root / task_id / "rerank-ledger.json"
        summary: dict[str, Any] = {
            "external_retrieval_requests": 0,
            "total_external_retrieval_requests": 0,
        }
        if path.exists():
            ledger = json.loads(path.read_text(encoding="utf-8"))
            count = sum(
                "transport_started_at" in row for row in ledger.get("request_reservations", [])
            )
            summary.update(
                external_retrieval_requests=count,
                total_external_retrieval_requests=count,
                **{
                    key: ledger[key]
                    for key in (
                        "cap_cny",
                        "cap_micro_cny",
                        "max_requests",
                        "stop_reason",
                        "reserved_cost_micro_cny",
                        "settled_cost_micro_cny",
                    )
                    if key in ledger
                },
            )
        return {"cancelled": True, **summary}

    def verify(self, task_id: str, evidence_ids: list[str]) -> dict[str, Any]:
        tool = self.tool(task_id)
        valid, rejected = [], []
        for evidence_id in evidence_ids:
            evidence = tool.evidence.get(evidence_id)
            try:
                if evidence is None:
                    raise ValueError("UNKNOWN_EVIDENCE")
                if evidence["source_kind"] == "input":
                    text = tool.registry.read_input(evidence["source_id"])
                else:
                    _, text = self.snapshot.read(evidence["source_id"])
                if (
                    text[evidence["start"] : evidence["end"]] != evidence["quote"]
                    or text_hash(text) != evidence["text_sha256"]
                ):
                    raise ValueError("EVIDENCE_MISMATCH")
                valid.append(evidence_id)
            except (ValueError, OSError):
                rejected.append(evidence_id)
        return {"valid_ids": valid, "rejected_ids": rejected}

    def launch(self, task_id: str, node_url: str, token: str) -> None:
        url = urlparse(node_url)
        if url.hostname not in {"127.0.0.1", "localhost"} or not token:
            raise ValueError("LOCAL_AUTHENTICATED_HARNESS_REQUIRED")
        job = asyncio.create_task(self._dispatch(task_id, node_url, token))
        self.jobs.add(job)
        job.add_done_callback(self.jobs.discard)

    async def _dispatch(self, task_id: str, node_url: str, token: str) -> None:
        record = self.records[task_id]
        record["task_status"] = "running"
        atomic_json(self.root / task_id / "admission.json", record)
        try:
            manifest = json.loads(
                (self.root / task_id / "manifest.json").read_text(encoding="utf-8")
            )
            async with httpx.AsyncClient(
                timeout=manifest["config"]["task_deadline_seconds"] + 15,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.post(
                    f"{node_url}/internal/harness/run",
                    json={"task_id": task_id},
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                report = response.json()["report"]
                if report["task_id"] != task_id:
                    raise ValueError("TASK_ID_MISMATCH")
                record["task_status"] = report["task_status"]
                record["total_e2e_ms"] = round((time.time() - record["created_at"]) * 1000)
        except Exception as error:
            record["task_status"] = "failed"
            record["error"] = {"code": "HARNESS_DISPATCH_FAILED", "detail": type(error).__name__}
        finally:
            self.cancel_retrieval(task_id)
            atomic_json(self.root / task_id / "admission.json", record)
            with self.lock:
                if self.active_task_id == task_id:
                    self.active_task_id = None
