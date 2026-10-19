"Frozen finrisk-loop-evaluation/1.0. prepare/preflight never load keys or live workers."

from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import html
import importlib.util
import itertools
import json
import math
import os
import re
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path

VERSION = "finrisk-loop-evaluation/1.0"
ARMS = ("R1", "R2", "A")
STAGE_LIMITS = {"smoke": 3_000_000, "development": 15_000_000, "independent": 20_000_000}
TASK_RESERVE = 2_500_000
PROFILE = "live-deepseek-flash-planned-v1"
BUDGET_ID = "planned-evidence-loop-v1"
IDS = {
    "smoke": ("S01", "C15", "C01"),
    "development": ("S02", "C15", "S01", "C07", "C18", "C19", "C06", "C01"),
    "independent": tuple(f"N{i:02}" for i in range(1, 11)),
}
SCOPE = (
    "selected text, explicit user context and retrieved penalty excerpts to official DeepSeek API"
)
STATISTICS_RULES = {
    "denominator": "all planned rows",
    "invalid_empty_gold_correct": False,
    "business_valid": (
        "processed segments + current accepted points + no hard/global blocker;"
        " auditable local_partial allowed"
    ),
    "semantic_checks": (
        "point coverage, necessary unknown and case applicability pending human review"
    ),
    "summaries": ["wins_ties_losses", "mean", "median", "slowest"],
    "timing": {
        "basis": "submission_before_admit_to_final_report_persistence",
        "receipt": "last report.persisted event; exact report bytes/task/input/profile",
        "same_host_clock_tolerance_ms": 50,
        "timestamp_unit": "epoch_milliseconds; clock resolution not asserted",
        "missing_or_untrusted": "unverifiable; never substitute runner wall or zero",
        "shared_gpu_warmup": "separate",
        "post_report_cleanup": "diagnostic only",
        "simple_speed_population": (
            "per-arm correct_valid simple rows; every included row requires verified e2e receipt"
        ),
    },
    "simple_target": (
        "independent A >=3/4 correct valid with exactly1 model and <=1 query; "
        "accuracy>=R1; median(correct-valid A/R1 only)<=max(1.25*R1,R1+2sec); "
        "all correct-valid simple A/R1 e2e receipts verified"
    ),
    "complex_target": (
        "independent A correct-valid count>=R2+1 with no increase in no-risk fa"
        "lse positives; exploratory small sample only"
    ),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def obj_sha(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    return [
        json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()
    ]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf8")
    temporary.replace(path)


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_imports(project):
    source = str(Path(project) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def evaluation_query(value):
    text = canonical({"text": value["text"], "user_context": value.get("user_context") or {}})
    if len(text) > 500:
        raise ValueError("EVALUATION_QUERY_TOO_LONG")
    return text


def schedule(stage, queries, references):
    if stage not in IDS:
        raise ValueError("INVALID_STAGE")
    qmap = {q["sample_id"]: q for q in queries}
    rmap = {r["review_id"]: r for r in references}
    if len(qmap) != len(queries) or len(rmap) != len(references):
        raise ValueError("DUPLICATE_MATERIAL_ID")
    tasks, orders = [], list(itertools.permutations(ARMS))
    # Explicit balanced Latin order; reverse orders complete the six-way cycle.
    orders = [
        ("R1", "R2", "A"),
        ("R2", "A", "R1"),
        ("A", "R1", "R2"),
        ("A", "R2", "R1"),
        ("R2", "R1", "A"),
        ("R1", "A", "R2"),
    ]
    for i, rid in enumerate(IDS[stage]):
        r = rmap[rid]
        q = qmap[r["sample_id"]]
        if r.get("human_approved") is not True:
            raise ValueError("REFERENCE_NOT_APPROVED")
        if set(q) - {"sample_id", "text", "user_context"}:
            raise ValueError("UNEXPECTED_INPUT_FIELDS")
        query = evaluation_query(q)
        inp = {"text": q["text"], "user_context": q.get("user_context") or {}}
        digest = obj_sha(inp)
        if r.get("input_sha256", digest) != digest:
            raise ValueError("REFERENCE_INPUT_MISMATCH")
        for arm in ("A",) if stage == "smoke" else orders[i % 6]:
            tasks.append(
                {
                    "task_key": rid + ":" + arm,
                    "review_id": rid,
                    "sample_id": q["sample_id"],
                    "difficulty": r["difficulty"],
                    "family_id": r.get("family_id"),
                    "arm": arm,
                    "input": q,
                    "input_sha256": digest,
                    "evaluation_query": query,
                    "reference": r["reference"],
                }
            )
    return tasks


def file_records(paths):
    return [
        {"path": str(p), "sha256": file_sha(p)} for p in sorted({Path(p).resolve() for p in paths})
    ]


def verify_files(records):
    for row in records:
        if not Path(row["path"]).is_file() or file_sha(row["path"]) != row["sha256"]:
            raise ValueError("FROZEN_FILE_CHANGED:" + row["path"])


def describe(project, node):
    result = subprocess.run(
        [str(node), str(Path(project) / "agent_harness/dist/src/agent_loop_cli.js"), "--describe"],
        capture_output=True,
        check=True,
        timeout=60,
        encoding="utf8",
    )
    value = json.loads(result.stdout)
    if value.get("schema_version") != VERSION:
        raise ValueError("LOOP_DESCRIPTION_VERSION_MISMATCH")
    return value


def freeze_runtime(project):
    project = Path(project)
    files = []
    for directory, suffixes in [
        ("agent_harness/src", {".ts"}),
        ("agent_harness/dist/src", {".js"}),
        ("src", {".py"}),
        ("configs/agent_rag", {".json", ".txt"}),
        ("contracts/agent_rag", {".json", ".csv"}),
        ("scripts/agentic_v3", {".py"}),
        ("resources/deepseek-v41", None),
    ]:
        files.extend(
            p
            for p in (project / directory).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and (suffixes is None or p.suffix in suffixes)
        )
    files.extend(
        [
            project / "agent_harness/package.json",
            project / "agent_harness/package-lock.json",
            project / "agent_harness/tsconfig.json",
            project / "scripts/agentic_dev30/experiment.py",
            project / "scripts/agentic_dev30/retrieval_reliability.py",
        ]
    )
    files.extend(project.glob("requirements*.txt"))
    return files


def service_admission_bindings(description, stage):
    result = {}
    for arm in ARMS:
        config = dict(description["stages"][stage][arm]["base_config"])
        config.update(
            rerank_cap_cny=config["rerank_cap_micro_cny"] / 1_000_000,
            model_trial_cap_cny=config["model_trial_cap_micro_cny"] / 1_000_000,
            task_total_cost_cap_cny=config["task_total_cost_cap_micro_cny"] / 1_000_000,
        )
        result[arm] = {
            "config": config,
            "config_sha256": obj_sha(config),
            "hash_kind": "Python contracts.canonical_json actual admission",
        }
    return result


def node_obj_sha(value):
    def normalize(v):
        if isinstance(v, dict):
            return {k: normalize(x) for k, x in v.items()}
        if isinstance(v, list):
            return [normalize(x) for x in v]
        return int(v) if type(v) is float and v.is_integer() else v

    return obj_sha(normalize(value))


def snapshot_assets(snapshot_dir, manifest):
    paths = [Path(snapshot_dir) / "manifest.json"]
    for asset in list(manifest["assets"].values()) + list(manifest["backend"]["assets"].values()):
        paths.append(Path(asset["path"]))
    dense = load(manifest["backend"]["assets"]["dense_config"]["path"])
    for key in ("vector_path", "vector_meta_path"):
        paths.append(Path(dense[key]))
    meta = load(dense["vector_meta_path"])
    paths.extend(
        Path(dense["model_path"]) / row["path"] for row in meta["binding"]["model"]["files"]
    )
    return paths


def runtime_snapshot(manifest):
    return manifest | {
        "rerank_mode": manifest["backend"]["mode"],
        "rerank_protocol": manifest["backend"].get("rerank_protocol", "quote-copy-v1"),
    }


def prepare(
    project_root,
    stage,
    output,
    round_cap_cny,
    fast_path,
    node,
    stage_budget_register,
    review_bundle=None,
    development_score=None,
):
    project, output = Path(project_root).resolve(), Path(output).resolve()
    stage = "development" if stage == "dev" else stage
    if stage not in STAGE_LIMITS or fast_path != ("off" if stage == "smoke" else "on"):
        raise ValueError("FROZEN_STAGE_FAST_POLICY_MISMATCH")
    cap = int(Decimal(str(round_cap_cny)) * 1_000_000)
    if cap < TASK_RESERVE or cap > STAGE_LIMITS[stage]:
        raise ValueError("INVALID_ROUND_CAP")
    if output.exists():
        raise ValueError("PLAN_ALREADY_EXISTS")
    source_imports(project)
    from agent_rag.input_registry import InputRegistry
    from agent_rag.knowledge_registry import _validate_snapshot, current_snapshot

    snapshot_dir = current_snapshot(project / "runtime/agent_rag/knowledge-registry")
    snapshot = _validate_snapshot(
        snapshot_dir
    )  # standard-library validators; no resident GPU/scorer
    manifest = snapshot.manifest
    if (
        manifest.get("corpus_scope") != "incremental-v1"
        or manifest["backend"].get("context_expansion") is not None
    ):
        raise ValueError("SNAPSHOT_SCOPE_OR_PARENT_POLICY_MISMATCH")
    frozen = freeze_runtime(project) + snapshot_assets(snapshot_dir, manifest) + [Path(node)]
    if stage == "independent":
        if not review_bundle:
            raise ValueError("INDEPENDENT_REVIEW_BUNDLE_REQUIRED")
        review = module_file("loop_review", project / "scripts/agentic_v3/review.py")
        bundle = load(review_bundle)
        review.verify_review_bundle(bundle)
        queries, refs = bundle["queries"], bundle["references"]
        material = {"kind": "independent", "review_bundle": str(Path(review_bundle).resolve())}
        frozen.extend(
            [Path(review_bundle), Path(bundle["source_proposal"]), Path(bundle["source_approval"])]
        )
    else:
        original = project / "outputs/Agentic_RAG30条人工参照_20261016"
        approval = load(original / "approval.json")
        if (
            approval.get("authority") != "human_user_in_current_conversation"
            or approval.get("decision") != "accept_all_30_proposals"
        ):
            raise ValueError("DEVELOPMENT_APPROVAL_MISMATCH")
        paths = [
            original / "inputs/queries.jsonl",
            original / "evaluation/references.jsonl",
            original / "approval.json",
        ]
        queries, refs = lines(paths[0]), lines(paths[1])
        material = {
            "kind": "development",
            "queries": str(paths[0]),
            "references": str(paths[1]),
            "approval": str(paths[2]),
        }
        frozen.extend(paths)
    tasks = schedule(stage, queries, refs)
    for row in tasks:
        InputRegistry(row["input"]["text"], row["input"].get("user_context"))
    description = describe(project, node)
    register_path = Path(stage_budget_register).resolve()
    validate_register(load(register_path))
    plan = {
        "schema_version": VERSION,
        "project_root": str(project),
        "stage": stage,
        "node": str(Path(node).resolve()),
        "round_cap_micro_cny": cap,
        "stage_budget_register": str(register_path),
        "stage_limits_micro_cny": STAGE_LIMITS,
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": file_sha(snapshot_dir / "manifest.json"),
        "snapshot_manifest": manifest,
        "runtime_snapshot_projection": runtime_snapshot(manifest),
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_human_approved": manifest.get("human_approved"),
        "snapshot_semantic_coverage_verified": manifest.get("semantic_coverage_verified"),
        "description": description,
        "service_admission_bindings": service_admission_bindings(description, stage),
        "material_sources": material,
        "evaluation_protocol_sha256": description["evaluation_protocol_sha256"],
        "tasks": tasks,
        "statistics_rules": STATISTICS_RULES,
        "fast_path": fast_path,
        "local_cache_policy": {
            "result_cache_size": 0,
            "rerank_cache_size": 0,
            "shared_dense_index": True,
            "provider_prompt_cache": "no disabling guarantee; actual usage recorded",
        },
        "fresh_build_requirement": (
            "tsc must succeed before preparing this plan; source/dist hashes separa"
            "tely freeze bytes and are not proof of provenance"
        ),
        "development_score": str(Path(development_score).resolve()) if development_score else None,
    }
    if development_score:
        frozen.append(Path(development_score))
    plan["frozen_files"] = file_records(frozen)
    save(output, plan)
    return {
        "plan": str(output),
        "plan_sha256": file_sha(output),
        "tasks": len(tasks),
        "stage": stage,
        "api_calls": 0,
        "execution_created": False,
    }


def preflight(plan_path, *, offline=True):
    plan = load(plan_path)
    if (
        not offline
        or plan.get("schema_version") != VERSION
        or plan.get("statistics_rules") != STATISTICS_RULES
        or plan.get("stage_limits_micro_cny") != STAGE_LIMITS
        or plan.get("local_cache_policy", {}).get("result_cache_size") != 0
        or plan.get("local_cache_policy", {}).get("rerank_cache_size") != 0
    ):
        raise ValueError("LOOP_PLAN_POLICY_MISMATCH")
    stage = plan["stage"]
    if stage not in STAGE_LIMITS or plan["fast_path"] != ("off" if stage == "smoke" else "on"):
        raise ValueError("FROZEN_STAGE_FAST_POLICY_MISMATCH")
    verify_files(plan["frozen_files"])
    if (
        file_sha(Path(plan["snapshot_dir"]) / "manifest.json") != plan["snapshot_manifest_sha256"]
        or load(Path(plan["snapshot_dir"]) / "manifest.json") != plan["snapshot_manifest"]
        or runtime_snapshot(plan["snapshot_manifest"]) != plan["runtime_snapshot_projection"]
        or plan["snapshot_manifest"]["snapshot_id"] != plan["snapshot_id"]
    ):
        raise ValueError("SNAPSHOT_IDENTITY_MISMATCH")
    if describe(plan["project_root"], plan["node"]) != plan["description"]:
        raise ValueError("LOOP_DESCRIPTION_CHANGED")
    if plan["evaluation_protocol_sha256"] != plan["description"]["evaluation_protocol_sha256"]:
        raise ValueError("LOOP_PROTOCOL_MISMATCH")
    expected_ids = IDS[stage]
    if len(plan["tasks"]) != len(expected_ids) * (1 if stage == "smoke" else 3):
        raise ValueError("PLAN_TASK_DENOMINATOR_MISMATCH")
    material = plan["material_sources"]
    if material["kind"] == "independent":
        review = module_file(
            "loop_review", Path(plan["project_root"]) / "scripts/agentic_v3/review.py"
        )
        bundle = load(material["review_bundle"])
        review.verify_review_bundle(bundle)
        queries, refs = bundle["queries"], bundle["references"]
    else:
        queries, refs = lines(material["queries"]), lines(material["references"])
    if plan["tasks"] != schedule(stage, queries, refs):
        raise ValueError("PLAN_TASK_SCHEDULE_OR_MATERIAL_MISMATCH")
    if plan["service_admission_bindings"] != service_admission_bindings(plan["description"], stage):
        raise ValueError("BASE_ADMISSION_CONFIGURATION_MISMATCH")
    validate_register(load(plan["stage_budget_register"]))
    return plan


def validate_register(value):
    if (
        value.get("schema_version") != "finrisk-loop-stage-budget-register/1.0"
        or value.get("hard_cap_micro_cny") != 40_000_000
        or value.get("stage_limits_micro_cny") != STAGE_LIMITS
        or not isinstance(value.get("executions"), list)
    ):
        raise ValueError("INVALID_STAGE_BUDGET_REGISTER")
    known = unknown = 0
    seen = set()
    for execution in value["executions"]:
        if execution["plan_sha256"] in seen or execution["stage"] not in STAGE_LIMITS:
            raise ValueError("INVALID_STAGE_BUDGET_REGISTER")
        seen.add(execution["plan_sha256"])
        keys = set()
        for row in execution["tasks"]:
            if (
                row["task_key"] in keys
                or row["status"] not in {"reserved", "settled"}
                or any(
                    type(row[k]) is not int or row[k] < 0
                    for k in ("known_micro_cny", "unknown_micro_cny")
                )
            ):
                raise ValueError("INVALID_STAGE_BUDGET_REGISTER")
            if row["status"] == "reserved" and (
                row["known_micro_cny"] != 0 or row["unknown_micro_cny"] != TASK_RESERVE
            ):
                raise ValueError("INVALID_STAGE_BUDGET_REGISTER")
            keys.add(row["task_key"])
            known += row["known_micro_cny"]
            unknown += row["unknown_micro_cny"]
    if (
        type(value.get("known_micro_cny")) is not int
        or type(value.get("unknown_reserved_micro_cny")) is not int
        or value["known_micro_cny"] != known
        or value["unknown_reserved_micro_cny"] != unknown
    ):
        raise ValueError("INVALID_STAGE_BUDGET_TOTALS")
    return value


class StageBudget:
    "One shared durable register; pending reservations survive failure and diagnostic rounds."

    def __init__(self, path, stage, plan_sha, cap):
        self.path, self.stage, self.plan_sha, self.cap = Path(path), stage, plan_sha, cap
        if (
            stage not in STAGE_LIMITS
            or type(cap) is not int
            or not TASK_RESERVE <= cap <= STAGE_LIMITS[stage]
        ):
            raise ValueError("INVALID_ROUND_BUDGET")
        validate_register(load(self.path))

    @contextlib.contextmanager
    def transaction(self):
        lock = self.path.with_name(self.path.name + ".lock")
        try:
            handle = lock.open("x")
        except FileExistsError as error:
            raise ValueError("STAGE_BUDGET_LOCKED") from error
        try:
            value = validate_register(load(self.path))
            yield value
            validate_register(value)
            save(self.path, value)
        finally:
            handle.close()
            lock.unlink()

    def execution(self, value):
        existing = next((r for r in value["executions"] if r["plan_sha256"] == self.plan_sha), None)
        if existing is None:
            existing = {
                "plan_sha256": self.plan_sha,
                "stage": self.stage,
                "round_cap_micro_cny": self.cap,
                "tasks": [],
            }
            value["executions"].append(existing)
        if existing["stage"] != self.stage or existing["round_cap_micro_cny"] != self.cap:
            raise ValueError("STAGE_BUDGET_BINDING_MISMATCH")
        return existing

    def reserve(self, key):
        with self.transaction() as value:
            execution = self.execution(value)
            if any(t["task_key"] == key for t in execution["tasks"]):
                raise ValueError("TASK_ALREADY_RESERVED")
            stage_total = sum(
                t["known_micro_cny"] + t["unknown_micro_cny"]
                for r in value["executions"]
                if r["stage"] == self.stage
                for t in r["tasks"]
            )
            round_total = sum(
                t["known_micro_cny"] + t["unknown_micro_cny"] for t in execution["tasks"]
            )
            if (
                stage_total + TASK_RESERVE > STAGE_LIMITS[self.stage]
                or round_total + TASK_RESERVE > self.cap
                or value["known_micro_cny"] + value["unknown_reserved_micro_cny"] + TASK_RESERVE
                > 40_000_000
            ):
                raise ValueError("STAGE_ROUND_OR_TOTAL_BUDGET_EXHAUSTED")
            execution["tasks"].append(
                {
                    "task_key": key,
                    "status": "reserved",
                    "known_micro_cny": 0,
                    "unknown_micro_cny": TASK_RESERVE,
                }
            )
            value["unknown_reserved_micro_cny"] += TASK_RESERVE

    def finish(self, key, known, unknown):
        if any(type(v) is not int or v < 0 for v in (known, unknown)):
            raise ValueError("INVALID_TASK_COST")
        with self.transaction() as value:
            row = next(t for t in self.execution(value)["tasks"] if t["task_key"] == key)
            if row["status"] != "reserved":
                raise ValueError("TASK_ALREADY_SETTLED")
            value["unknown_reserved_micro_cny"] += unknown - row["unknown_micro_cny"]
            value["known_micro_cny"] += known
            row.update(status="settled", known_micro_cny=known, unknown_micro_cny=unknown)


def create_execution(path):
    path = Path(path)
    try:
        path.mkdir()
    except FileExistsError as error:
        raise ValueError("ALREADY_STARTED:read existing evidence; no automatic rerun") from error
    return path


def authorization_record(plan_path, plan):
    return {
        "schema_version": "finrisk-loop-local-authorization/1.0",
        "plan_sha256": file_sha(plan_path),
        "authority": "executor_record_of_existing_human_authorization_in_current_conversation",
        "new_user_click_claimed": False,
        "machine": socket.gethostname(),
        "local_user": getpass.getuser(),
        "project_root": plan["project_root"],
        "stage": plan["stage"],
        "scope": SCOPE,
        "round_cap_micro_cny": plan["round_cap_micro_cny"],
        "stage_limits_micro_cny": STAGE_LIMITS,
        "total_cap_micro_cny": 40_000_000,
        "unallocated_micro_cny_not_usable": 2_000_000,
        "maximum_model_requests": 12 * len(plan["tasks"]),
        "maximum_rerank_requests": 800 * len(plan["tasks"]),
        "next_task_reserve_micro_cny": TASK_RESERVE,
        "unknown_rule": ("retain reserved usage; no refund on failure; count toward all caps"),
        "supplier_actual_debit": "unknown; peak usage estimate only",
    }


def physical_counts(events, rerank_ledger):
    requests, failures, anonymous = set(), {}, []
    for event in events:
        data = event.get("data", {})
        rid = data.get("request_id")
        if event.get("type") == "model.request" and data.get("physical_request") is True:
            requests.add(rid if rid is not None else "anonymous:" + str(len(requests)))
        elif event.get("type") == "model.failure":
            count = data.get("physical_requests", 0)
            if type(count) is not int or count < 0:
                raise ValueError("INVALID_MODEL_PHYSICAL_COUNT")
            if rid is None:
                anonymous.append(count)
            else:
                failures[rid] = max(failures.get(rid, 0), count)
    physical = len(requests) + sum(n for rid, n in failures.items() if rid not in requests)
    physical = max([physical] + anonymous)
    return physical, sum(
        isinstance(r.get("transport_started_at"), str)
        for r in rerank_ledger.get("request_reservations", [])
    )


def read_costs(model_path, rerank_path, plan):
    known = unknown = 0
    details = {}
    for name, path, cap in (
        ("model", Path(model_path), 1_000_000),
        ("rerank", Path(rerank_path) if rerank_path is not None else None, 1_500_000),
    ):
        try:
            if path is None or not path.is_file():
                raise ValueError("LEDGER_MISSING")
            if name == "model":
                checked = subprocess.run(
                    [
                        plan["node"],
                        str(
                            Path(plan["project_root"]) / "agent_harness/dist/src/agent_loop_cli.js"
                        ),
                        "--audit-model-ledger",
                        str(path),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=60,
                    encoding="utf8",
                )
                ledger = json.loads(checked.stdout)
            else:
                source_imports(plan["project_root"])
                from agent_rag.rerank_backend import RerankTaskControl

                ledger = load(path)
                # Reuse the pure validator, without the writing constructor.
                checker = object.__new__(RerankTaskControl)
                checker.expected_budget_profile_id = BUDGET_ID
                checker.expected_budget_profile_sha256 = plan["description"][
                    "budget_profile_sha256"
                ]
                checker.cap_micro_cny, checker.max_requests = cap, 800
                checker._validate(ledger)
                checker.attempt_audit_path = path.with_name(path.name + ".attempts.json")
                attempts = checker._read_attempts()["attempts"]
                by_ordinal = {a["ordinal"]: a for a in attempts}
                for row in ledger["request_reservations"]:
                    if "transport_started_at" not in row:
                        continue
                    attempt = by_ordinal.get(row["ordinal"])
                    if (
                        attempt is None
                        or attempt.get("pair_id") != row["pair_id"]
                        or attempt.get("transport_started_at") != row["transport_started_at"]
                    ):
                        raise ValueError("RERANK_ATTEMPT_LEDGER_BINDING_MISMATCH")
                if any(
                    a["ordinal"] > len(ledger["request_reservations"])
                    or a.get("pair_id")
                    != ledger["request_reservations"][a["ordinal"] - 1]["pair_id"]
                    for a in attempts
                ):
                    raise ValueError("RERANK_ATTEMPT_LEDGER_BINDING_MISMATCH")
            k, u = ledger["settled_cost_micro_cny"], ledger["reserved_cost_micro_cny"]
            if any(type(v) is not int or v < 0 for v in (k, u)):
                raise ValueError("INVALID_LEDGER_COST")
            known += k
            unknown += u
            details[name] = {
                "valid": True,
                "known_micro_cny": k,
                "unknown_micro_cny": u,
                "reservations": len(ledger["request_reservations"]),
                "supplier_actual_debit": None,
            }
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            unknown += cap
            details[name] = {
                "valid": False,
                "known_micro_cny": 0,
                "unknown_micro_cny": cap,
                "error": type(error).__name__,
            }
    return known, unknown, details


def independent_gate(plan):
    if plan["stage"] != "independent":
        return
    path = plan.get("development_score")
    if not path:
        raise ValueError("DEVELOPMENT_GATE_EVIDENCE_REQUIRED")
    saved = load(path)
    if (
        saved.get("stage") != "development"
        or saved.get("schema_version") != "finrisk-loop-score/1.0"
    ):
        raise ValueError("DEVELOPMENT_GATE_IDENTITY_MISMATCH")
    actual = score(saved["plan_path"], write=False)
    if saved != actual or not actual["development_gate"]["passed"]:
        raise ValueError("DEVELOPMENT_GATE_NOT_PASSED")
    # The frozen protocol binds the same rules, model, criteria and stage fast policy.
    if actual["evaluation_protocol_sha256"] != plan["evaluation_protocol_sha256"]:
        raise ValueError("DEVELOPMENT_GATE_PROTOCOL_MISMATCH")


def timing_anchor():
    return {"epoch_ms": int(time.time() * 1000), "monotonic_ms": time.monotonic() * 1000}


def report_timing(task_id, input_sha, profile_sha, report_sha, events, anchors):
    """Cross-process report time uses a same-host clock checked against parent monotonic time."""
    result = {
        "schema_version": "finrisk-loop-task-timing/1.0",
        "status": "unverifiable",
        "anchors": anchors,
        "errors": [],
        "receipt_event_id": None,
        "task_e2e_seconds": None,
        "cli_to_report_seconds": None,
        "post_report_cleanup_seconds": None,
        "cli_wall_seconds": None,
        "parent_cleanup_seconds": None,
    }
    try:
        names = ("submitted", "cli_started", "cli_returned", "cleanup_completed")
        points = [anchors[name] for name in names]
        if any(
            type(point[k]) not in (int, float) or not math.isfinite(point[k])
            for point in points
            for k in ("epoch_ms", "monotonic_ms")
        ):
            raise ValueError("INVALID_TIMING_ANCHORS")
        result["cli_wall_seconds"] = (points[2]["monotonic_ms"] - points[1]["monotonic_ms"]) / 1000
        result["parent_cleanup_seconds"] = (
            points[3]["monotonic_ms"] - points[2]["monotonic_ms"]
        ) / 1000
        for left, right in list(zip(points, points[1:])) + [(points[0], points[-1])]:
            epoch = right["epoch_ms"] - left["epoch_ms"]
            monotonic = right["monotonic_ms"] - left["monotonic_ms"]
            if epoch < 0 or monotonic < 0 or abs(epoch - monotonic) > 50:
                raise ValueError("CLOCK_ROLLBACK_OR_DISCONTINUITY")
        receipts = [e for e in events if e.get("type") == "report.persisted"]
        if not receipts:
            raise ValueError("FINAL_REPORT_RECEIPT_MISSING")
        receipt = receipts[-1]  # Never fall back to an earlier overwritten report.
        result["receipt_event_id"] = receipt.get("event_id")
        data = receipt["data"]
        if (
            data.get("schema_version") != "finrisk-loop-report-persistence/1.0"
            or data.get("task_id") != task_id
            or data.get("input_sha256") != input_sha
            or data.get("profile_sha256") != profile_sha
            or data.get("report_sha256") != report_sha
            or not re.fullmatch(
                re.escape(task_id) + r":[1-9][0-9]*", str(receipt.get("event_id", ""))
            )
        ):
            raise ValueError("FINAL_REPORT_RECEIPT_IDENTITY_MISMATCH")
        end = data["persisted_at_epoch_ms"]
        if type(end) is not int or not points[1]["epoch_ms"] <= end <= points[2]["epoch_ms"]:
            raise ValueError("FINAL_REPORT_TIME_OUTSIDE_EXECUTION")
        result.update(
            status="verified",
            task_e2e_seconds=(end - points[0]["epoch_ms"]) / 1000,
            cli_to_report_seconds=(end - points[1]["epoch_ms"]) / 1000,
            post_report_cleanup_seconds=(points[3]["epoch_ms"] - end) / 1000,
        )
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result["errors"] = [str(error).split(":")[0]]
    return result


def verified_task_e2e(record):
    timing = record.get("task_timing")
    value = record.get("task_e2e_seconds")
    return (
        value
        if (
            isinstance(timing, dict)
            and timing.get("status") == "verified"
            and timing.get("task_e2e_seconds") == value
            and type(value) in (int, float)
            and math.isfinite(value)
            and value >= 0
        )
        else None
    )


def execute(plan_path, authorization, node, key_file):
    plan_path = Path(plan_path).resolve()
    plan = preflight(plan_path)
    if load(authorization) != authorization_record(plan_path, plan):
        raise ValueError("LOCAL_AUTHORIZATION_BINDING_MISMATCH")
    if str(Path(node).resolve()) != plan["node"]:
        raise ValueError("FROZEN_NODE_BINDING_MISMATCH")
    independent_gate(plan)
    # Key-file metadata and live modules are only touched after all frozen/gate checks.
    if not Path(key_file).is_file():
        raise ValueError("KEY_FILE_MISSING")
    output = create_execution(plan_path.parent / "execution")
    save(output / "authorization.json", load(authorization))
    budget = StageBudget(
        plan["stage_budget_register"],
        plan["stage"],
        file_sha(plan_path),
        plan["round_cap_micro_cny"],
    )
    infra = module_file(
        "loop_infrastructure", Path(plan["project_root"]) / "scripts/agentic_dev30/experiment.py"
    )
    circuit = infra.RoundCircuit(output / "round-health.json")
    old_env = dict(os.environ)
    snapshot = service = server = thread = sock = None
    active = None
    try:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.update(
            FINRISK_AGENT_ALLOW_LIVE="1",
            FINRISK_AGENT_ALLOW_LOOP_EVALUATION="1",
            FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE="1",
            FINRISK_AGENT_RERANK_CAP_CNY="1.5",
            FINRISK_AGENT_RERANK_MAX_REQUESTS="800",
            DEEPSEEK_API_KEY_FILE=str(Path(key_file).resolve()),
            FINRISK_AGENT_DEEPSEEK_KEY_FILE=str(Path(key_file).resolve()),
            FINRISK_AGENT_SERVICE_TOKEN=secrets.token_urlsafe(32),
        )
        source_imports(plan["project_root"])
        import uvicorn

        from agent_rag.task_service import TaskService, atomic_json
        from agent_rag.v2_runtime import load_v2_snapshot

        started = time.monotonic()
        snapshot = load_v2_snapshot(
            Path(plan["snapshot_dir"]),
            result_cache_size=0,
            rerank_cache_size=0,
            audit_root=output / "retrieval-audits",
        )
        save(
            output / "warmup.json",
            {
                "model_gpu_warmup_seconds": time.monotonic() - started,
                "shared_dense_index": True,
                "model_generation_warmup": "none",
                "included_in_task_wall": False,
            },
        )
        if snapshot.manifest != plan["runtime_snapshot_projection"]:
            raise ValueError("RUNTIME_SNAPSHOT_PROJECTION_MISMATCH")
        service = TaskService(output / "tasks", snapshot)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        server = uvicorn.Server(uvicorn.Config(infra.local_tool_app(service), log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 20
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise ValueError("LOCAL_TOOL_SERVER_NOT_READY")
            time.sleep(0.05)
        for task in plan["tasks"]:
            circuit.before_task()
            key = task["task_key"]
            budget.reserve(key)
            active = {"task_key": key}
            model_ledger = output / "model-ledgers" / (key.replace(":", "_") + ".json")
            os.environ["FINRISK_AGENT_TRIAL_LEDGER"] = str(model_ledger)
            task_anchors = {"submitted": timing_anchor()}
            record, reused = service.admit(
                {
                    "text": task["input"]["text"],
                    "user_context": task["input"].get("user_context") or {},
                    "engine": "fixed_rag" if task["arm"] == "R2" else "controlled_agent",
                    "experiment_profile_id": PROFILE,
                },
                key,
            )
            if reused:
                raise ValueError("UNEXPECTED_TASK_REUSE")
            directory = service.root / record["task_id"]
            active.update(task_id=record["task_id"], directory=directory, model_ledger=model_ledger)
            binding = plan["description"]["stages"][plan["stage"]][task["arm"]]
            base = plan["service_admission_bindings"][task["arm"]]
            admission = load(directory / "manifest.json")
            if (
                admission["config"] != base["config"]
                or admission["config_sha256"] != base["config_sha256"]
                or admission["input_sha256"] != task["input_sha256"]
                or admission["snapshot"] != plan["runtime_snapshot_projection"]
            ):
                raise ValueError("BASE_ADMISSION_BINDING_MISMATCH")
            bundle = service.bundle(record["task_id"])
            bundle.update(
                config=binding["config"],
                execution_contract=binding["planned_execution_contract"],
                task_dir=str(directory),
                loop_evaluation={
                    "protocol_sha256": plan["evaluation_protocol_sha256"],
                    "stage": plan["stage"],
                    "arm": task["arm"],
                    "base_config_node_sha256": binding["base_config_node_sha256"],
                    "base_admission_config_sha256": base["config_sha256"],
                    "plan_sha256": file_sha(plan_path),
                    "admission_binding_sha256": record["binding_sha256"],
                },
            )
            save(directory / "loop-bundle.json", bundle)
            save(
                output / "current-task.json",
                {"task_key": key, "task_id": record["task_id"], "state": "starting"},
            )
            task_anchors["cli_started"] = timing_anchor()
            started = task_anchors["cli_started"]["monotonic_ms"] / 1000
            error_name, returncode = None, None
            try:
                with (
                    (directory / "cli.stdout").open("wb") as stdout,
                    (directory / "cli.stderr").open("wb") as stderr,
                ):
                    result = subprocess.run(
                        [
                            str(node),
                            str(
                                Path(plan["project_root"])
                                / "agent_harness/dist/src/agent_loop_cli.js"
                            ),
                            "--bundle",
                            str(directory / "loop-bundle.json"),
                            "--arm",
                            task["arm"],
                            "--stage",
                            plan["stage"],
                            "--base-url",
                            f"http://127.0.0.1:{sock.getsockname()[1]}",
                        ],
                        env=dict(os.environ),
                        stdout=stdout,
                        stderr=stderr,
                        timeout=630,
                        check=False,
                    )
                    returncode = result.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                error_name = type(error).__name__
            finally:
                task_anchors["cli_returned"] = timing_anchor()
                try:
                    service.cancel_retrieval(record["task_id"])
                finally:
                    task_anchors["cleanup_completed"] = timing_anchor()
            wall = task_anchors["cleanup_completed"]["monotonic_ms"] / 1000 - started
            known, unknown, cost_detail = read_costs(
                model_ledger, directory / "rerank-ledger.json", plan
            )
            budget.finish(key, known, unknown)
            active = None
            events = (
                lines(directory / "events.jsonl") if (directory / "events.jsonl").exists() else []
            )
            report_path = directory / "report.json"
            report = load(report_path) if report_path.exists() else None
            if report is not None:
                copy = output / "reports" / (key.replace(":", "_") + ".json")
                copy.parent.mkdir(exist_ok=True)
                shutil.copyfile(report_path, copy)
            physical = physical_counts(events, load(directory / "rerank-ledger.json"))
            timing = report_timing(
                record["task_id"],
                task["input_sha256"],
                binding["config_sha256"],
                file_sha(report_path) if report_path.is_file() else None,
                events,
                task_anchors,
            )
            save(
                output / (key.replace(":", "_") + ".json"),
                {
                    "task_key": key,
                    "task_id": record["task_id"],
                    "plan_sha256": file_sha(plan_path),
                    "returncode": returncode,
                    "error": error_name,
                    "wall_seconds": wall,
                    "task_timing": timing,
                    "task_e2e_seconds": timing["task_e2e_seconds"],
                    "known_micro_cny": known,
                    "unknown_micro_cny": unknown,
                    "cost_detail": cost_detail,
                    "model_physical_requests": physical[0],
                    "rerank_physical_requests": physical[1],
                    "report": report,
                    "actual_response_model": "unknown_not_captured",
                    "actual_system_fingerprint": "unknown_not_captured",
                },
            )
            service.records[record["task_id"]]["task_status"] = (report or {}).get(
                "task_status", "failed"
            )
            atomic_json(directory / "admission.json", service.records[record["task_id"]])
            service.active_task_id = None
            circuit.observe(key, report, events)
            if error_name or returncode != 0:
                raise ValueError("TASK_FAILED_STOP_NO_AUTOMATIC_REPLAY")
        save(output / "status.json", {"status": "finished", "plan_sha256": file_sha(plan_path)})
    except BaseException as error:
        # A reservation preceding an admission/launch failure stays conservative; never reset it.
        save(
            output / "status.json",
            {
                "status": "stopped",
                "reason": type(error).__name__ + ":" + str(error).split(":")[0],
                "pending_task": active.get("task_key") if active else None,
                "restart_policy": "no automatic resume",
            },
        )
        raise
    finally:
        errors = infra.cleanup_resources(service, snapshot, server, thread, sock, old_env)
        if errors:
            save(output / "cleanup-errors.json", errors)


def business_valid(report, assessment, retrieval):
    reasons = []
    coverage = report.get("coverage", {})
    if not coverage.get(
        "all_submitted_segments_reviewed", not coverage.get("unprocessed_segment_ids", [1])
    ):
        reasons.append("unprocessed_segments")
    validation = report.get("validation", {})
    if (
        not validation.get("schema_valid")
        or not validation.get("references_valid")
        or validation.get("semantic_review_status") != "model_assessed"
    ):
        reasons.append("invalid_or_unassessed_report")
    if assessment.get("active_blockers"):
        reasons.append("active_hard_blockers")
    if any(
        e not in {"CASE_QUERY_GOAL_UNFINISHED", "CASE_SEARCH_NOT_COMPLETED"}
        for e in validation.get("errors", [])
    ):
        reasons.append("hard_validation_error")
    stop_reason = str(report.get("stop", {}).get("reason", "")).lower()
    if (
        "stopped" in stop_reason
        or "circuit" in stop_reason
        or stop_reason
        in {
            "model_failure",
            "tool_failure",
            "deadline_exceeded",
            "diagnostic_write_failed",
            "budget_exhausted",
        }
    ):
        reasons.append("global_failure")
    points = {p["risk_point_id"]: p for p in assessment.get("points", [])}
    rp = {p["risk_point_id"]: p for p in report.get("risk_points", [])}
    if set(points) != set(rp):
        reasons.append("point_set_mismatch")
    for p in points.values():
        if (
            p.get("processing_state") != "assessed"
            or not p.get("accepted_assessment_id")
            or p.get("last_assessed_revision") != p.get("state_revision")
            or p.get("additional_mechanisms_uncovered")
            or p.get("retrieval_state") not in {"completed", "local_partial", "not_run"}
        ):
            reasons.append("point_not_current_accepted")
    for p in rp.values():
        for tag in p.get("tag_checks", []):
            if tag.get("status") == "supported" and any(
                c.get("status") == "unknown" for c in tag.get("conditions", [])
            ):
                reasons.append("necessary_unknown_marked_supported")
    if retrieval.get("retrieval_status") == "error" or any(
        p.get("assessment_status") in {"blocked", "not_assessed"}
        for p in retrieval.get("points", [])
    ):
        reasons.append("retrieval_globally_failed_or_not_assessed")
    local = any(p.get("retrieval_state") == "local_partial" for p in points.values())
    return not reasons, {
        "reasons": sorted(set(reasons)),
        "local_partial": local,
        "semantic_review": "pending_human_review",
        "case_applicability": "pending_human_review",
    }


def predicted_tags(report):
    return {
        tag for p in (report or {}).get("risk_points", []) for tag in p.get("supported_tags", [])
    }


def aggregate(tasks, records):
    rows, arms = [], {}
    for task in tasks:
        record = records.get(task["task_key"], {})
        report = record.get("report")
        valid = record.get("business_valid") is True
        gold = set(task["reference"].get("supported_tags", []))
        predicted = predicted_tags(report)
        expected = task["reference"].get("overall_judgment")
        expected = "no_clear_risk" if expected == "normal" else expected
        correct = (
            valid
            and report is not None
            and gold == predicted
            and (expected is None or report["overall_judgment"] == expected)
        )
        rows.append(
            {
                "task_key": task["task_key"],
                "review_id": task["review_id"],
                "arm": task["arm"],
                "difficulty": task["difficulty"],
                "business_valid": valid,
                "strict_completed": bool(valid and report.get("task_status") == "completed"),
                "correct_valid": correct,
                "tag_set_exact_valid": valid and gold == predicted,
                "no_risk_invalid_or_missing": not gold and not valid,
                "gold_tags": sorted(gold),
                "predicted_tags": sorted(predicted),
                "no_risk_false_positive": not gold and bool(predicted),
                "status": record.get("audit_status", "missing"),
                "wall_seconds": record.get("wall_seconds"),
                "task_e2e_seconds": verified_task_e2e(record),
                "timing_status": (record.get("task_timing") or {}).get("status", "unverifiable"),
                "model_physical_requests": record.get("model_physical_requests"),
                "query_count": record.get("query_count"),
                "point_coverage_review": "pending_human_review",
                "necessary_unknown_review": "pending_human_review",
                "case_applicability_review": "pending_human_review",
            }
        )
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        tp = sum(
            len(set(r["gold_tags"]) & set(r["predicted_tags"]))
            for r in selected
            if r["business_valid"]
        )
        fp = sum(len(set(r["predicted_tags"]) - set(r["gold_tags"])) for r in selected)
        fn = sum(
            len(set(r["gold_tags"]) - (set(r["predicted_tags"]) if r["business_valid"] else set()))
            for r in selected
        )
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        timing = [r["task_e2e_seconds"] for r in selected if r["task_e2e_seconds"] is not None]
        arms[arm] = {
            "denominator": len(selected),
            "correct_valid": sum(r["correct_valid"] for r in selected),
            "tag_set_exact_valid": sum(r["tag_set_exact_valid"] for r in selected),
            "no_risk_invalid_or_missing": sum(r["no_risk_invalid_or_missing"] for r in selected),
            "business_valid": sum(r["business_valid"] for r in selected),
            "strict_completed": sum(r["strict_completed"] for r in selected),
            "no_risk_false_positives": sum(r["no_risk_false_positive"] for r in selected),
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "timing_basis": "submission_to_final_report_persistence",
            "measured_timing_rows": len(timing),
            "mean_seconds": statistics.mean(timing) if timing else None,
            "median_seconds": statistics.median(timing) if timing else None,
            "slowest_seconds": max(timing) if timing else None,
        }
    pairs = {}
    for baseline in ("R1", "R2"):
        wins = ties = losses = 0
        by_key = {(r["review_id"], r["arm"]): r for r in rows}
        for rid in {r["review_id"] for r in rows}:
            if (rid, "A") in by_key and (rid, baseline) in by_key:
                a, b = by_key[rid, "A"]["correct_valid"], by_key[rid, baseline]["correct_valid"]
                wins += a > b
                ties += a == b
                losses += a < b
        pairs["A_vs_" + baseline] = {"wins": wins, "ties": ties, "losses": losses}
    return {"rows": rows, "arms": arms, "paired": pairs}


def verify_references(report, bundle, snapshot):
    evidence = {e["evidence_id"]: e for e in report["retrieved_evidence"]}
    if len(evidence) != len(report["retrieved_evidence"]):
        raise ValueError("DUPLICATE_EVIDENCE")
    inp = bundle["input"]

    def walk(value):
        if isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, dict):
            if {"source_id", "source_kind", "start", "end", "quote", "text_sha256"} <= value.keys():
                if value["source_kind"] == "input":
                    text = inp["text"]
                    if (
                        value["source_id"] != inp["source_id"]
                        or value["text_version"] != inp["text_version"]
                    ):
                        raise ValueError("INPUT_REFERENCE_IDENTITY_MISMATCH")
                else:
                    source, text = snapshot.read(value["source_id"])
                    if value["text_version"] != source["text_version"]:
                        raise ValueError("SOURCE_REFERENCE_VERSION_MISMATCH")
                if (
                    not 0 <= value["start"] < value["end"] <= len(text)
                    or text[value["start"] : value["end"]] != value["quote"]
                    or hashlib.sha256(text.encode("utf8")).hexdigest() != value["text_sha256"]
                ):
                    raise ValueError("REFERENCE_TEXT_OR_OFFSET_MISMATCH")
            if "unit_id" in value and "source_id" in value and "evidence_ids" in value:
                units = {u["unit_id"]: u for u in snapshot.units}
                unit = units.get(value["unit_id"])
                if not unit:
                    raise ValueError("CASE_UNIT_IDENTITY_MISMATCH")
                candidate = snapshot.candidate(unit, 1, 0)
                if candidate["source_id"] != value["source_id"]:
                    raise ValueError("CASE_UNIT_SOURCE_MISMATCH")
                selected_windows = [
                    evidence[eid] for eid in value["evidence_ids"] if eid in evidence
                ]
                for anchor in candidate["evidence"]:
                    if anchor["evidence_id"] not in candidate["required_evidence_ids"]:
                        continue
                    if not any(
                        w["source_id"] == anchor["source_id"]
                        and w["text_sha256"] == anchor["text_sha256"]
                        and w["text_version"] == anchor["text_version"]
                        and w["start"] <= anchor["start"]
                        and w["end"] >= anchor["end"]
                        for w in selected_windows
                    ):
                        raise ValueError("CASE_REQUIRED_ANCHOR_NOT_READ")
                for eid in value["evidence_ids"]:
                    e = evidence.get(eid)
                    if not e or not e.get("was_read") or e["source_id"] != value["source_id"]:
                        raise ValueError("CASE_UNREAD_OR_FOREIGN_EVIDENCE")
            for child in value.values():
                walk(child)

    walk(report)


def gain_candidates(events, messages=None, gold_tags=None):
    """Only a complete observed chain with tag-reference improvement is a candidate."""
    if messages is None or gold_tags is None:
        return []
    outputs = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        try:
            content = message.get("content", [])
            text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
            value = json.loads(text)
            if value.get("kind") == "assess":
                outputs[value["assessment_id"]] = value
        except (ValueError, KeyError, TypeError):
            continue
    candidates, gold = [], set(gold_tags)
    for index, event in enumerate(events):
        data = event.get("data", {})
        if (
            event.get("type") != "action.proposed"
            or data.get("origin") != "adaptive_model"
            or data.get("action_type") != "search_cases"
        ):
            continue
        before, after = events[:index], events[index + 1 :]
        observation = data.get("preceding_observation_event_id")
        preceding = next(
            (
                e
                for e in before
                if e.get("event_id") == observation and e.get("type") == "progress.observed"
            ),
            None,
        )
        points = set(data.get("point_ids", []))
        gap = next(
            (
                e
                for e in reversed(before)
                if e.get("type") == "gap.derived"
                and e.get("data", {}).get("point_id") in points
                and e.get("data", {}).get("gap_id") in data.get("target_gap_ids", [])
            ),
            None,
        )
        earlier = {
            e.get("data", {}).get("query_hash")
            for e in before
            if e.get("type") == "action.proposed"
            and e.get("data", {}).get("action_type") == "search_cases"
        }
        search = next(
            (
                e
                for e in after
                if e.get("type") == "action.completed"
                and e.get("data", {}).get("action_id") == data["action_id"]
                and e.get("data", {}).get("status") in {"completed", "partial"}
            ),
            None,
        )
        if (
            not preceding
            or not gap
            or not search
            or not data.get("query_hash")
            or data["query_hash"] in earlier
        ):
            continue
        old = next(
            (
                e
                for e in reversed(before)
                if e.get("type") == "assessment.accepted"
                and e.get("data", {}).get("point_id") in points
            ),
            None,
        )
        if not old or old["data"].get("assessment_id") not in outputs:
            continue
        # A newly read required anchor must follow this query, before the next query.
        search_index = events.index(search)
        reads = []
        for e in events[search_index + 1 :]:
            d = e.get("data", {})
            if e.get("type") == "action.proposed" and d.get("action_type") == "search_cases":
                break
            if (
                e.get("type") == "progress.observed"
                and d.get("observation_kind") == "source_read"
                and d.get("evidence_gain") is True
                and points.intersection(d.get("affected_point_ids", []))
                and any(w.get("anchor_evidence_id") for w in d.get("windows", []))
            ):
                reads.append(e)
        new_ids = {eid for e in reads for eid in e["data"].get("evidence_ids", [])}
        later = next(
            (
                e
                for e in after
                if e.get("type") == "assessment.accepted"
                and e.get("data", {}).get("point_id") in points
                and new_ids.intersection(e.get("data", {}).get("used_evidence_ids", []))
                and e.get("data", {}).get("assessment_id") in outputs
            ),
            None,
        )
        if not later or not new_ids or events.index(later) <= max(events.index(e) for e in reads):
            continue
        previous = outputs[old["data"]["assessment_id"]]
        current = outputs[later["data"]["assessment_id"]]
        old_tags = {t["tag"] for t in previous["tag_checks"] if t["status"] == "supported"}
        new_tags = {t["tag"] for t in current["tag_checks"] if t["status"] == "supported"}
        if len(new_tags.symmetric_difference(gold)) >= len(old_tags.symmetric_difference(gold)):
            continue
        candidates.append(
            {
                "action_id": data["action_id"],
                "query_hash": data["query_hash"],
                "point_ids": sorted(points),
                "observation_event_id": observation,
                "gap_event_id": gap["event_id"],
                "new_required_evidence_ids": sorted(new_ids),
                "accepted_assessment_id": later["data"]["assessment_id"],
                "old_supported_tags": sorted(old_tags),
                "new_supported_tags": sorted(new_tags),
                "tag_reference_distance_improved": True,
                "case_applicability": "pending_human_review",
                "confirmed_benefit": False,
            }
        )
    return candidates


def validate_audit_schema(value, name):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.exceptions import NoSuchResource, Unresolvable
    from referencing.jsonschema import DRAFT202012

    from agent_rag.contracts import load_contract

    allowed = ("final-report.schema.json", "agent-assessment.schema.json",
               "retrieval-assessment.schema.json", "model-wire.schema.json")
    if name not in allowed:
        raise ValueError("AUDIT_SCHEMA_NOT_ALLOWED")
    schemas = {filename: load_contract(filename) for filename in allowed}

    def closed(uri):
        raise NoSuchResource(ref=uri)

    registry = Registry(retrieve=closed).with_resources(
        (schema["$id"], Resource.from_contents(schema, default_specification=DRAFT202012))
        for schema in schemas.values() if "$id" in schema
    )
    try:
        errors = list(Draft202012Validator(schemas[name], registry=registry).iter_errors(value))
    except Unresolvable as error:
        raise ValueError("AUDIT_SCHEMA_REFERENCE_NOT_ALLOWED") from error
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path)
        raise ValueError(f"INVALID_SCHEMA: {location}: {first.message[:800]}")


def audit_task(plan, plan_path, task, snapshot, stage_execution):
    output = Path(plan_path).parent / "execution"
    path = output / (task["task_key"].replace(":", "_") + ".json")
    if not path.exists():
        return {"audit_status": "missing", "business_valid": False, "hard_engineering_error": False}
    result = load(path)
    if (
        result.get("plan_sha256") != file_sha(plan_path)
        or result.get("task_key") != task["task_key"]
    ):
        raise ValueError("RESULT_PLAN_IDENTITY_MISMATCH")
    if not re.fullmatch(r"[a-f0-9]{32}", result["task_id"]):
        raise ValueError("INVALID_RESULT_TASK_ID")
    directory = output / "tasks" / result["task_id"]
    report_path = directory / "report.json"
    copy_path = output / "reports" / path.name
    if report_path.read_bytes() != copy_path.read_bytes() or load(report_path) != result["report"]:
        raise ValueError("REPORT_COPY_OR_RESULT_MISMATCH")
    report, bundle, manifest = (
        result["report"],
        load(directory / "loop-bundle.json"),
        load(directory / "manifest.json"),
    )
    description = plan["description"]
    binding = description["stages"][plan["stage"]][task["arm"]]
    base = plan["service_admission_bindings"][task["arm"]]
    if (
        manifest["config"] != base["config"]
        or manifest["config_sha256"] != base["config_sha256"]
        or bundle["config"] != binding["config"]
        or bundle["execution_contract"] != binding["planned_execution_contract"]
        or bundle["snapshot"] != plan["runtime_snapshot_projection"]
        or manifest["snapshot"] != bundle["snapshot"]
        or bundle["input"]["input_sha256"] != task["input_sha256"]
        or obj_sha({k: bundle["input"][k] for k in ("text", "user_context")})
        != task["input_sha256"]
        or bundle["loop_evaluation"]["plan_sha256"] != file_sha(plan_path)
    ):
        raise ValueError("TASK_BASE_DERIVED_OR_INPUT_BINDING_MISMATCH")
    from agent_rag.input_registry import InputRegistry

    admission = load(directory / "admission.json")
    expected_input = InputRegistry(
        task["input"]["text"], task["input"].get("user_context") or {}, result["task_id"]
    ).export()
    expected_admission_binding = obj_sha(
        {
            "input_sha256": task["input_sha256"],
            "profile_id": PROFILE,
            "engine": base["config"]["engine"],
            "mock_scenario": "generic",
            "snapshot_id": plan["snapshot_id"],
            "config_sha256": base["config_sha256"],
        }
    )
    expected_meta = {
        "protocol_sha256": plan["evaluation_protocol_sha256"],
        "stage": plan["stage"],
        "arm": task["arm"],
        "base_config_node_sha256": binding["base_config_node_sha256"],
        "base_admission_config_sha256": base["config_sha256"],
        "plan_sha256": file_sha(plan_path),
        "admission_binding_sha256": expected_admission_binding,
    }
    if (
        bundle["input"] != expected_input
        or bundle["loop_evaluation"] != expected_meta
        or admission["task_id"] != result["task_id"]
        or admission["binding_sha256"] != expected_admission_binding
        or manifest["task_id"] != result["task_id"]
        or manifest["input_sha256"] != task["input_sha256"]
        or manifest["snapshot_id"] != plan["snapshot_id"]
        or bundle["definitions"] != list(snapshot.definitions.values())
    ):
        raise ValueError("ADMISSION_BUNDLE_OR_MATERIAL_IDENTITY_MISMATCH")
    v = report["versions"]
    if (
        report["task_id"] != result["task_id"]
        or report["execution_mode"] != "live"
        or report.get("simulated")
        or v["input_sha256"] != task["input_sha256"]
        or v["snapshot_id"] != plan["snapshot_id"]
        or v["dictionary_sha256"] != bundle["snapshot"]["dictionary_sha256"]
        or v["text_sha256"] != bundle["input"]["text_sha256"]
        or v["retrieval_profile"] != binding["config"]["retrieval_profile"]
        or v["config_sha256"] != binding["config_sha256"]
        or v["prompt_sha256"] != binding["planned_execution_contract"]["prompt_sha256"]
        or v["criteria_sha256"] != description["loop_contract_inventory"]["criteria_sha256"]
    ):
        raise ValueError("REPORT_IDENTITY_MISMATCH")
    from agent_rag.contracts import load_contract

    validate_audit_schema(report, "final-report.schema.json")
    a, r = load(directory / "agent-assessment.json"), load(directory / "retrieval-assessment.json")
    validate_audit_schema(a, "agent-assessment.schema.json")
    validate_audit_schema(r, "retrieval-assessment.schema.json")
    identity = {
        "task_id": result["task_id"],
        "profile_id": PROFILE,
        "input_sha256": task["input_sha256"],
        "snapshot_sha256": node_obj_sha(plan["runtime_snapshot_projection"]),
        "criteria_sha256": v["criteria_sha256"],
        "budget_profile_sha256": description["budget_profile_sha256"],
        "profile_sha256": binding["config_sha256"],
        "loop_schema_sha256": description["loop_contract_inventory"]["loop_schema_sha256"],
        "report_sha256": file_sha(report_path),
    }
    if (
        a["identity"] != identity
        or a["orchestration_mode"] != binding["config"]["orchestration_mode"]
        or r["report_sha256"] != file_sha(report_path)
        or r["task_id"] != result["task_id"]
        or r["input_sha256"] != task["input_sha256"]
        or r["snapshot_id"] != plan["snapshot_id"]
        or r["evaluation_protocol_sha256"] is not None
    ):
        raise ValueError("SIDECAR_IDENTITY_OR_EXACT_REPORT_HASH_MISMATCH")
    segment_ids = [s["segment_id"] for s in bundle["input"]["segments"]]
    if report["coverage"]["provided_segment_ids"] != segment_ids:
        raise ValueError("REPORT_SEGMENT_IDENTITY_MISMATCH")
    point_ids = [p["risk_point_id"] for p in report["risk_points"]]
    if (
        len(point_ids) != len(set(point_ids))
        or set(point_ids) != {p["risk_point_id"] for p in r["points"]}
        or len(r["points"]) != len(point_ids)
        or len(a["points"]) != len(point_ids)
    ):
        raise ValueError("SIDECAR_POINT_SET_MISMATCH")
    criteria = {
        t["tag"]: {c["id"] for c in t["criteria"]}
        for t in load_contract("risk-criteria.json")["tags"]
    }
    for point in report["risk_points"]:
        for check in point.get("tag_checks", []):
            ids = [c["condition_id"] for c in check.get("conditions", [])]
            if len(ids) != len(set(ids)) or set(ids) != criteria[check["tag"]]:
                raise ValueError("REPORT_CRITERIA_INCOMPLETE")
    coverage_check = module_file(
        "loop_ranking_audit",
        Path(plan["project_root"]) / "scripts/agentic_dev30/retrieval_reliability.py",
    )
    for query in r["queries"]:
        coverage_check._coverage(query, bundle["snapshot"]["rerank_protocol"])
    verify_references(report, bundle, snapshot)
    events = lines(directory / "events.jsonl")
    if any(e.get("event_id", "").split(":")[0] != result["task_id"] for e in events):
        raise ValueError("EVENT_TASK_IDENTITY_MISMATCH")
    known, unknown, detail = read_costs(
        output / "model-ledgers" / path.name, directory / "rerank-ledger.json", plan
    )
    cost_row = next(
        (t for t in stage_execution["tasks"] if t["task_key"] == task["task_key"]), None
    )
    if (
        cost_row is None
        or cost_row["status"] != "settled"
        or cost_row["known_micro_cny"] != known
        or cost_row["unknown_micro_cny"] != unknown
        or result["known_micro_cny"] != known
        or result["unknown_micro_cny"] != unknown
        or result["cost_detail"] != detail
    ):
        raise ValueError("TASK_COMPONENT_OR_STAGE_COST_MISMATCH")
    valid, limits = business_valid(report, a, r)
    if any(e["type"] in {"loop.execution_failed", "loop.sidecar_failed"} for e in events):
        valid = False
        limits["reasons"].append("global_execution_failure")
    if not all(v["valid"] for v in detail.values()):
        valid = False
        limits["reasons"].append("invalid_cost_ledger")
    model_n, rerank_n = physical_counts(events, load(directory / "rerank-ledger.json"))
    if (
        result["model_physical_requests"] != model_n
        or result["rerank_physical_requests"] != rerank_n
    ):
        raise ValueError("PHYSICAL_REQUEST_COUNT_MISMATCH")
    queries = report["resources"]["search_query_attempts"]
    config = binding["config"]
    if (
        model_n > config["max_model_requests"]
        or rerank_n > config["rerank_max_requests"]
        or queries > config["max_search_query_attempts"]
        or report["resources"]["tool_actions"] > config["max_tool_actions"]
        or report["resources"]["search_calls"] > config["max_search_calls"]
        or report["resources"]["model_request_units"] > config["max_model_requests"]
        or known + unknown > config["task_total_cost_cap_micro_cny"]
    ):
        valid = False
        limits["reasons"].append("task_budget_limit_breached")
    if task["arm"] != "A" and any(
        e["type"] == "action.proposed"
        and e.get("data", {}).get("origin") == "adaptive_model"
        and e.get("data", {}).get("action_type") == "search_cases"
        for e in events
    ):
        valid = False
        limits["reasons"].append("fixed_arm_generated_adaptive_query")
    if task["arm"] == "R1" and (
        model_n != 1 or queries != 1 or report["resources"].get("json_repairs", 0) != 0
    ):
        valid = False
        limits["reasons"].append("one_shot_execution_contract_violated")
    saved_timing = result.get("task_timing")
    anchors = saved_timing.get("anchors", {}) if isinstance(saved_timing, dict) else {}
    audited_timing = report_timing(
        result["task_id"],
        task["input_sha256"],
        binding["config_sha256"],
        file_sha(report_path),
        events,
        anchors,
    )
    if (
        saved_timing != audited_timing
        or result.get("task_e2e_seconds") != audited_timing["task_e2e_seconds"]
    ):
        audited_timing["status"] = "unverifiable"
        audited_timing["task_e2e_seconds"] = None
        audited_timing["errors"].append("STORED_TIMING_BINDING_MISMATCH")
    return result | {
        "task_timing": audited_timing,
        "task_e2e_seconds": audited_timing["task_e2e_seconds"],
        "audit_status": "valid" if valid else "business_invalid",
        "business_valid": valid,
        "hard_engineering_error": bool(
            [
                v
                for v in limits["reasons"]
                if v not in {"point_not_current_accepted", "unprocessed_segments"}
            ]
        ),
        "limitations": limits,
        "query_count": queries,
        "gain_candidates": gain_candidates(
            events,
            lines(directory / "model_messages.jsonl")
            if (directory / "model_messages.jsonl").exists()
            else [],
            task["reference"].get("supported_tags", []),
        ),
        "trace": events,
        "raw_replies": [load(p) for p in sorted(directory.glob("model-response-*.json"))],
    }


def raw_task_evidence(plan, output, task):
    """Retain diagnostic originals separately from accepted reports and settled accounting."""

    def artifact(path, jsonl=False):
        if path is None:
            return {"path": None, "status": "missing", "reason": "task directory unresolved"}
        record = {"path": str(path)}
        if not path.is_file():
            return record | {"status": "missing"}
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            return record | {"status": "read_error", "error": type(error).__name__}
        try:
            value = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if jsonl
                else json.loads(text)
            )
            return record | {"status": "available", "value": value}
        except (ValueError, TypeError) as error:
            return record | {
                "status": "parse_error",
                "error": type(error).__name__,
                "raw_text": text,
            }

    name = task["task_key"].replace(":", "_") + ".json"
    original_result = artifact(output / name)
    value = original_result.get("value")
    value = value if isinstance(value, dict) else {}
    task_id = value.get("task_id")
    directory = (
        output / "tasks" / task_id
        if isinstance(task_id, str) and re.fullmatch(r"[a-f0-9]{32}", task_id)
        else None
    )

    def task_path(name):
        return directory / name if directory is not None else None

    files = {
        "result": original_result,
        "report": artifact(task_path("report.json")),
        "report_copy": artifact(output / "reports" / name),
        "model_messages": artifact(task_path("model_messages.jsonl"), jsonl=True),
        "trace": artifact(task_path("events.jsonl"), jsonl=True),
        "model_ledger": artifact(output / "model-ledgers" / name),
        "rerank_ledger": artifact(task_path("rerank-ledger.json")),
        "rerank_attempts": artifact(task_path("rerank-ledger.json.attempts.json")),
    }
    replies = (
        [artifact(p) for p in sorted(directory.glob("model-response-*.json"))]
        if directory is not None
        else []
    )
    files["raw_replies"] = {
        "status": "available" if replies else "missing",
        "items": replies,
        "path_pattern": str(directory / "model-response-*.json") if directory is not None else None,
    }
    report = files["report"].get("value")
    report = report if isinstance(report, dict) else {}
    stop = (
        {"status": "available", "source_path": files["report"]["path"], "value": report["stop"]}
        if "stop" in report
        else {"status": "missing"}
    )
    cost_fields = ("known_micro_cny", "unknown_micro_cny", "cost_detail")
    recorded_cost = {key: value[key] for key in cost_fields if key in value}
    known, unknown, detail = read_costs(
        output / "model-ledgers" / name, task_path("rerank-ledger.json"), plan
    )
    return {
        "status": "raw_unverified",
        "notice": "原始/未验收；仅供人工诊断，不作为有效报告或费用结算依据。",
        "files": files,
        "stop": stop,
        "recorded_cost": {
            "status": "available" if recorded_cost else "missing",
            "source_path": original_result["path"],
            "value": recorded_cost,
            "missing_fields": [k for k in cost_fields if k not in value],
        },
        "ledger_cost_review": {
            "known_micro_cny": known,
            "unknown_micro_cny": unknown,
            "components": detail,
            "used_for_scoring": False,
        },
    }


def score(plan_path, *, write=True):
    plan_path = Path(plan_path).resolve()
    plan = preflight(plan_path)
    source_imports(plan["project_root"])
    from agent_rag.v2_snapshot import V2KnowledgeSnapshot

    snapshot = V2KnowledgeSnapshot(Path(plan["snapshot_dir"]))
    output = plan_path.parent / "execution"
    stage = validate_register(load(plan["stage_budget_register"]))
    execution = next(
        (r for r in stage["executions"] if r["plan_sha256"] == file_sha(plan_path)), {"tasks": []}
    )
    records = {}
    authorization = output / "authorization.json"
    globally_authorized = authorization.is_file() and load(authorization) == authorization_record(
        plan_path, plan
    )
    for task in plan["tasks"]:
        if not output.is_dir():
            records[task["task_key"]] = {
                "audit_status": "missing",
                "business_valid": False,
                "hard_engineering_error": False,
            }
            continue
        try:
            if not globally_authorized:
                raise ValueError("EXECUTION_AUTHORIZATION_MISSING")
            records[task["task_key"]] = audit_task(plan, plan_path, task, snapshot, execution)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            records[task["task_key"]] = {
                "audit_status": "invalid",
                "business_valid": False,
                "hard_engineering_error": True,
                "error": str(error).split(":")[0],
            }
    for task in plan["tasks"]:
        record = records[task["task_key"]]
        if record["audit_status"] in {"invalid", "missing"}:
            record["raw_evidence"] = raw_task_evidence(plan, output, task)
    result = aggregate(plan["tasks"], records)
    a_rows = [r for r in result["rows"] if r["arm"] == "A"]
    engineering = sum(r.get("hard_engineering_error", False) for r in records.values())
    status_path = output / "status.json"
    status = load(status_path) if status_path.is_file() else {}
    if status.get("status") == "stopped" and "BUDGET_EXHAUSTED" not in status.get("reason", ""):
        engineering += 1
    all_rows = [t for r in stage["executions"] for t in r["tasks"]]
    exceeded = (
        sum(t["known_micro_cny"] + t["unknown_micro_cny"] for t in all_rows) > 40_000_000
        or any(
            sum(
                t["known_micro_cny"] + t["unknown_micro_cny"]
                for r in stage["executions"]
                if r["stage"] == name
                for t in r["tasks"]
            )
            > cap
            for name, cap in STAGE_LIMITS.items()
        )
        or sum(t["known_micro_cny"] + t["unknown_micro_cny"] for t in execution["tasks"])
        > plan["round_cap_micro_cny"]
    )
    if exceeded or any(
        t["task_key"] not in {p["task_key"] for p in plan["tasks"]} for t in execution["tasks"]
    ):
        engineering += 1
    gate = {
        "business_valid": sum(r["business_valid"] for r in a_rows),
        "complex_business_valid": sum(
            r["business_valid"] for r in a_rows if r["difficulty"] == "complex"
        ),
        "hard_engineering_errors": engineering,
    }
    gate["passed"] = (
        plan["stage"] == "development"
        and gate["business_valid"] >= 7
        and gate["complex_business_valid"] >= 5
        and engineering == 0
    )
    result.update(
        schema_version="finrisk-loop-score/1.0",
        plan_path=str(plan_path),
        plan_sha256=file_sha(plan_path),
        stage=plan["stage"],
        evaluation_protocol_sha256=plan["evaluation_protocol_sha256"],
        development_gate=gate,
        evidence_status="mechanical checks only; semantic point/condition/case review pending",
        supplier_actual_debit=None,
        costs={
            "known_micro_cny": sum(t["known_micro_cny"] for t in execution["tasks"]),
            "unknown_micro_cny": sum(t["unknown_micro_cny"] for t in execution["tasks"]),
        },
    )
    simple_a = [r for r in result["rows"] if r["arm"] == "A" and r["difficulty"] == "simple"]
    simple_r1 = [r for r in result["rows"] if r["arm"] == "R1" and r["difficulty"] == "simple"]
    target_count = sum(
        r["correct_valid"] and r["model_physical_requests"] == 1 and r["query_count"] <= 1
        for r in simple_a
    )

    def median_measured(rows):
        values = [r["task_e2e_seconds"] for r in rows if r["correct_valid"]]
        return (statistics.median(values)
                if values and all(value is not None for value in values) else None)

    ma, mr = median_measured(simple_a), median_measured(simple_r1)
    simple_pass = (
        len(simple_a) == 4
        and target_count >= 3
        and sum(r["correct_valid"] for r in simple_a) >= sum(r["correct_valid"] for r in simple_r1)
        and ma is not None
        and mr is not None
        and ma <= max(1.25 * mr, mr + 2)
    )
    complex_a = sum(
        r["correct_valid"]
        for r in result["rows"]
        if r["arm"] == "A" and r["difficulty"] == "complex"
    )
    complex_r2 = sum(
        r["correct_valid"]
        for r in result["rows"]
        if r["arm"] == "R2" and r["difficulty"] == "complex"
    )
    result["independent_targets"] = {
        "applicable": plan["stage"] == "independent",
        "simple_pass": simple_pass,
        "simple_correct_valid_one_model_one_query": target_count,
        "complex_exploratory_candidate": plan["stage"] == "independent"
        and complex_a >= complex_r2 + 1
        and result["arms"]["A"]["no_risk_invalid_or_missing"] == 0
        and result["arms"]["R2"]["no_risk_invalid_or_missing"] == 0
        and result["arms"]["A"]["no_risk_false_positives"]
        <= result["arms"]["R2"]["no_risk_false_positives"],
        "confirmed_case_applicability_benefit": False,
        "semantic_confirmation": "pending_human_review",
    }
    if write:
        review_output = output if output.is_dir() else plan_path.parent / "offline-review"
        save(review_output / "score.json", result)
        save(
            review_output / "review-data.json",
            {"plan_sha256": file_sha(plan_path), "tasks": plan["tasks"], "records": records},
        )
        content = [
            (
                "<!doctype html><meta charset='utf-8'><title>FinRisk逐题审核</title><style>"
                "body{font:16px sans-serif;max-width:1400px;margin:auto}pre{white-space"
                ":pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ccc;padding"
                ":12px;margin:10px}</style>"
            ),
            "<h1>逐题输入、背景、人工参照与三组审计</h1><p>案例适用、条件业务含义与补查收益均待真人核定。费用为峰价usage估算，供应商实际扣款未知。</p>",
        ]
        for rid in IDS[plan["stage"]]:
            selected = [t for t in plan["tasks"] if t["review_id"] == rid]
            content.append(
                "<h2>"
                + html.escape(rid)
                + "</h2><pre>"
                + html.escape(
                    json.dumps(
                        {"input": selected[0]["input"], "gold": selected[0]["reference"]},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre>"
            )
            for task in selected:
                content.append(
                    "<details><summary>"
                    + html.escape(task["arm"] + " / " + records[task["task_key"]]["audit_status"])
                    + "</summary><pre>"
                    + html.escape(
                        json.dumps(records[task["task_key"]], ensure_ascii=False, indent=2)
                    )
                    + "</pre></details>"
                )
        (review_output / "review.html").write_text("\n".join(content), encoding="utf8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="freeze and validate; no execution, keys or GPU")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--stage", choices=["smoke", "development", "dev", "independent"], required=True)
    p.add_argument("--output", type=Path, required=True, help="new plan JSON path")
    p.add_argument("--round-cap-cny", required=True)
    p.add_argument(
        "--fast-path",
        choices=["off", "on"],
        required=True,
        help="smoke off; development (dev alias)/independent on; only A uses fast",
    )
    p.add_argument(
        "--node",
        type=Path,
        default=shutil.which("node"),
        help="offline Node runtime; required when absent from PATH",
    )
    p.add_argument(
        "--stage-budget-register",
        type=Path,
        required=True,
        help="existing cross-round 3/15/20/40 CNY register",
    )
    p.add_argument("--review-bundle", type=Path)
    p.add_argument(
        "--development-score",
        type=Path,
        help="real score.json required for independent execute; independent prepare may omit",
    )
    p = commands.add_parser("preflight", help="read-only offline freeze checks")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--offline", action="store_true", required=True)
    p = commands.add_parser(
        "authorize", help="record existing conversation authorization; does not claim a new click"
    )
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("execute", help="one exclusive live execution; never auto rerun")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--node", type=Path, required=True)
    p.add_argument("--key-file", type=Path, required=True)
    p = commands.add_parser(
        "score",
        help="mechanical audit and complete planned denominator; semantic review remains human",
    )
    p.add_argument("--plan", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        if args["node"] is None:
            parser.error("--node is required when Node is absent from PATH")
        result = prepare(**args)
    elif command == "preflight":
        preflight(args["plan"], offline=args["offline"])
        result = {"status": "offline_preflight_passed", "api_calls": 0, "execution_created": False}
    elif command == "authorize":
        plan = preflight(args["plan"])
        if args["output"].exists():
            raise ValueError("AUTHORIZATION_FILE_ALREADY_EXISTS")
        result = authorization_record(args["plan"], plan)
        save(args["output"], result)
    elif command == "execute":
        execute(args["plan"], args["authorization"], args["node"], args["key_file"])
        result = {"status": "execution_returned"}
    else:
        result = score(args["plan"])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
