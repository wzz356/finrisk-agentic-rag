"""Offline-first maintenance orchestration; fixtures can never enter production."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from agent_rag.contracts import PROJECT_ROOT, load_contract
from agent_rag.knowledge_updates import update_lock, write_json
from agent_rag.maintenance_bridge import NodeMaintenanceRunner
from agent_rag.maintenance_budget import BudgetLedger
from agent_rag.maintenance_contracts import (
    attestation_sha256,
    canonical_bytes,
    canonical_sha256,
    changeset_sha256,
    issue_receipt,
    validate_changeset,
)
from agent_rag.maintenance_store import MaintenanceStore
from agent_rag.maintenance_tools import MaintenanceCorpus

_EVENTS = {
    "new_source",
    "source_revision",
    "extraction_correction",
    "duplicate_relation",
    "law_revision",
    "deactivation",
    "conflict_report",
    "periodic_consolidation",
}
_SCOPE = {"scope_id": "public", "classification": "public"}
_REQUIRED_FIELDS = {
    "case": {"party_name", "case_summary"},
    "behavior": {
        "summary",
        "actor_role",
        "action",
        "finding_quote",
        "risk_tags",
        "conditions",
        "negations",
        "exceptions",
        "uncertainties",
    },
    "law_article": {"text", "legal_status", "effective_from", "effective_to", "applicability"},
    "guide": {"text", "depends_on"},
    "relation": {"relation_type", "source_id", "target_id"},
}


def business_validator(operation):
    """Validate new canonical record semantics; all supplied fields require claims."""
    body, kind = operation["after"], operation["record_kind"]
    required = _REQUIRED_FIELDS[kind]
    if not required.issubset(body):
        raise ValueError("BUSINESS_FIELDS_REQUIRED")
    canonical_bytes(body)
    if kind == "behavior":
        official = set(load_contract("model-output.schema.json")["$defs"]["tag"]["enum"])
        tags = body["risk_tags"]
        if not isinstance(tags, list) or len(tags) > 3:
            raise ValueError("INVALID_RISK_TAGS")
        for tag in tags:
            if (
                not isinstance(tag, dict)
                or tag.get("tag") not in official
                or tag.get("assignment") not in {"retrieval_hypothesis", "confirmed_in_source"}
            ):
                raise ValueError("INVALID_RISK_TAGS")
        if len({t["tag"] for t in tags}) != len(tags):
            raise ValueError("INVALID_RISK_TAGS")
        for field in ("conditions", "negations", "exceptions", "uncertainties"):
            if not isinstance(body[field], list):
                raise ValueError("QUALIFIERS_REQUIRED")
    if kind == "law_article":
        from datetime import date

        if body["legal_status"] not in {"verified", "unknown"} or not isinstance(
            body["applicability"], dict
        ):
            raise ValueError("INVALID_LAW_SCOPE")
        for field in ("effective_from", "effective_to"):
            if body[field] is not None:
                date.fromisoformat(body[field])
        if (
            body["effective_to"]
            and body["effective_from"]
            and body["effective_to"] <= body["effective_from"]
        ):
            raise ValueError("INVALID_LAW_DATES")
    if operation["kind"] == "deactivate" and body.get("active") is not False:
        raise ValueError("DEACTIVATION_REQUIRES_INACTIVE")
    return ["/" + key.replace("~", "~0").replace("/", "~1") for key in body]


class MaintenanceService:
    def __init__(self, root, *, runner=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = MaintenanceStore(self.root / "state")
        self.corpus = MaintenanceCorpus(self.root / "corpus")
        self.runner = runner or NodeMaintenanceRunner(PROJECT_ROOT)
        self.policy = json.loads(
            (PROJECT_ROOT / "configs/knowledge_maintenance/policy.json").read_text(encoding="utf-8")
        )
        self._cancel = threading.Event()
        self._threads = []
        self._active_run_id = None
        self._api_lock = threading.Lock()
        self._role_slots = {
            "proposer": threading.BoundedSemaphore(self.policy["proposer_concurrency"]),
            "reviewer": threading.BoundedSemaphore(self.policy["reviewer_concurrency"]),
        }
        from agent_rag.maintenance_publish import KnowledgePublisher

        self.publisher = KnowledgePublisher(self.root / "publication", self.corpus)
        self.base_commit = self.publisher.head()
        key_file = self.root / "receipt.key"
        if not key_file.exists():
            try:
                with key_file.open("xb") as handle:
                    handle.write(secrets.token_bytes(32))
            except FileExistsError:
                pass
        self._key = key_file.read_bytes()
        self._scope = dict(_SCOPE)

    def status(self):
        return {
            "live_ready": False,
            "execution_mode": "fixture",
            "usage_enabled": False,
            "policy_id": self.policy["policy_id"],
            "events": self.store.list_events(),
            "jobs": self.store.list_jobs(),
            "runs": self.list_runs(),
            "knowledge_commit": self.base_commit,
            "baseline_knowledge_bound": (self.root / "baseline-bound.json").is_file(),
            "limits": {
                k: self.policy[k]
                for k in ("max_repair_rounds", "proposer_concurrency", "reviewer_concurrency")
            },
        }

    def create_event(
        self, event_type, evidence_ids, target_record_ids, reason, *, idempotency_key=None
    ):
        if event_type not in _EVENTS or not evidence_ids or len(evidence_ids) > 3:
            raise ValueError("INVALID_MAINTENANCE_EVENT")
        for eid in evidence_ids:
            source = self.corpus.resolve_evidence(eid)
            if source["classification"] != "public":
                raise ValueError("PRIVATE_MAINTENANCE_NOT_ADMITTED")
        for rid in target_record_ids:
            self.corpus.read_record(rid, self.base_commit, self._scope)
        if idempotency_key is not None and (
            type(idempotency_key) is not str or not idempotency_key or len(idempotency_key) > 160
        ):
            raise ValueError("INVALID_IDEMPOTENCY_KEY")
        key = (
            canonical_sha256(["external", "public", idempotency_key])
            if idempotency_key
            else canonical_sha256(
                [event_type, evidence_ids, target_record_ids, self.policy["policy_id"]]
            )
        )
        return self.store.create_event(
            event_type, evidence_ids, target_record_ids, "public", key, reason
        )

    def import_inventory(self, inventory, batch_id, *, max_documents=None):
        ids, offset, eligible_count = [], 0, 0
        if max_documents is not None and (type(max_documents) is not int or max_documents < 1):
            raise ValueError("INVALID_DOCUMENT_LIMIT")
        while True:
            page = inventory.documents(batch_id, offset, 100)
            for item in page["items"]:
                document = inventory.read_document(item["digest"])
                scope = document["scope_assessment"]
                eligible_count += 1
                if max_documents is not None and len(ids) >= max_documents:
                    continue
                # Collection text is a registered historical text snapshot, not an official PDF.
                raw_sha = document["raw_versions"][0]["raw_sha256"]
                raw = inventory._read_blob(raw_sha)
                source = self.corpus.register_evidence(
                    raw,
                    origin_id="collection:" + document["digest"],
                    text=document["text"],
                    extractor_version="collection-utf8-sig/1.0",
                )
                event = self.create_event(
                    "new_source",
                    [source["evidence_id"]],
                    [],
                    "隔离采集原件；待独立核查重复关系与业务范围。"
                    + f"范围建议：{scope['category']}；路由：{scope['route']}；"
                    + "依据：" + "；".join(e["reason"] for e in scope["evidence"])
                    + "；待核：" + "；".join(scope["unknowns"]),
                )
                ids.append(event["id"])
            offset += len(page["items"])
            if offset >= page["total"]:
                break
        return {
            "event_ids": ids,
            "count": len(ids),
            "remaining_documents": max(0, eligible_count - len(ids)),
            "publication_allowed": False,
        }

    def bind_snapshot(self, snapshot_root, expected_manifest_sha256):
        """Host-only, pinned legacy import; it never upgrades old review status."""
        from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash

        source_root = Path(snapshot_root).resolve()
        manifest = source_root / "manifest.json"
        if file_hash(manifest) != expected_manifest_sha256:
            raise ValueError("BASE_MANIFEST_HASH_MISMATCH")
        marker = self.root / "baseline-bound.json"
        with update_lock(self.root / "baseline-binding"):
            if marker.exists():
                prior = json.loads(marker.read_text(encoding="utf-8"))
                if prior["baseline_manifest_sha256"] != expected_manifest_sha256:
                    raise ValueError("BASELINE_IMMUTABLE")
                self.publisher.pin_legacy_seed(
                    prior["knowledge_commit"],
                    base_manifest_path=manifest,
                    expected_sha256=expected_manifest_sha256,
                )
                self.base_commit = self.publisher.head()
                return {**prior, "knowledge_commit": self.base_commit}
            snapshot = V2KnowledgeSnapshot(source_root)
            records, evidence_by_file = [], {}
            for sid, source in snapshot.sources.items():
                ev = self.corpus.register_evidence(
                    source["full_text"].encode("utf-8"),
                    origin_id="legacy-v2:" + sid,
                    extractor_version="legacy-v2-text-snapshot/1.0",
                )
                evidence_by_file[source["official_file_id"]] = ev["evidence_id"]
                records.append(
                    {
                        "record_id": sid,
                        "kind": "case",
                        "body": {
                            "original_id": sid,
                            "file_id": source["official_file_id"],
                            "title": source["title"],
                            "full_text": source["full_text"],
                            "text_sha256": source["text_sha256"],
                            "legacy_source_kind": "registered_text_snapshot",
                        },
                        "record_version": 1,
                        "evidence_ids": [ev["evidence_id"]],
                        "scope_id": "public",
                        "classification": "public",
                    }
                )
            for unit in snapshot.units:
                records.append(
                    {
                        "record_id": unit["unit_id"],
                        "kind": "behavior",
                        "body": {
                            "original_id": unit["unit_id"],
                            "file_id": unit["file_id"],
                            "text": unit["text"],
                            "source_sha256": unit["source_sha256"],
                        },
                        "record_version": 1,
                        "evidence_ids": [evidence_by_file[unit["file_id"]]],
                        "scope_id": "public",
                        "classification": "public",
                    }
                )
            result = self.publisher.seed_legacy_records(
                records, base_manifest_path=manifest, expected_sha256=expected_manifest_sha256
            )
            self.base_commit = result["knowledge_commit"]
            bound = {
                "baseline_manifest_sha256": expected_manifest_sha256,
                "source_count": len(snapshot.sources),
                "record_count": len(records),
                "knowledge_commit": self.base_commit,
                "admission_status": "legacy_candidate",
            }
            write_json(marker, bound)
            return bound

    def _bindings(self):
        def prompt(role):
            return hashlib.sha256(
                (PROJECT_ROOT / f"configs/knowledge_maintenance/{role}.txt").read_bytes()
            ).hexdigest()

        return {
            "dictionary_sha256": canonical_sha256(
                load_contract("model-output.schema.json")["$defs"]["tag"]["enum"]
            ),
            "business_schema_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tool_contract_sha256": canonical_sha256(
                __import__("agent_rag.maintenance_tools", fromlist=["PARAMETERS"]).PARAMETERS
            ),
            "proposer_prompt_sha256": prompt("proposer"),
            "reviewer_prompt_sha256": prompt("reviewer"),
            "policy_sha256": canonical_sha256(self.policy),
        }

    def _maintenance_contract(self):
        definitions = json.loads(
            (PROJECT_ROOT / "contracts/knowledge_maintenance/protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )["$defs"]
        names = ("Operation", "Claim", "Citation", "SourceCoverage", "ClaimCheck", "Issue")
        return {
            "$defs": {key: definitions[key] for key in names if key in definitions},
            "required_fields": {key: sorted(value) for key, value in _REQUIRED_FIELDS.items()},
            "official_risk_tags": load_contract("model-output.schema.json")["$defs"]["tag"]["enum"],
            "coordinate_system": "unicode_codepoints_half_open",
            "proposal_keys": ["operations", "claims", "source_coverage"],
            "review_keys": ["decision", "claim_checks", "omission_check_completed", "issues"],
        }

    def _role_run(
        self,
        event,
        revision,
        role,
        stage,
        ledger,
        script,
        *,
        changeset=None,
        feedback=None,
        deadline=None,
    ):
        remaining = (
            (deadline - time.monotonic()) if deadline else self.policy["event_deadline_seconds"]
        )
        if remaining <= 0 or not self._role_slots[role].acquire(timeout=max(0, remaining)):
            raise ValueError("AGENT_DEADLINE_EXCEEDED")
        try:
            return self._role_run_admitted(
                event,
                revision,
                role,
                stage,
                ledger,
                script,
                changeset=changeset,
                feedback=feedback,
                deadline=deadline,
            )
        finally:
            self._role_slots[role].release()

    def _role_run_admitted(
        self,
        event,
        revision,
        role,
        stage,
        ledger,
        script,
        *,
        changeset=None,
        feedback=None,
        deadline=None,
    ):
        if self._cancel.is_set():
            raise ValueError("CANCELLED")
        tools = self.corpus.tools(role, self.base_commit, self._scope, changeset=changeset)
        run_id = uuid4().hex
        limits = self.policy["stages"][stage]
        task = {
            "run_id": run_id,
            "role": role,
            "revision": revision,
            "execution_mode": "fixture",
            "knowledge_commit": self.base_commit,
            "scope": self._scope,
            "system_prompt": (PROJECT_ROOT / f"configs/knowledge_maintenance/{role}.txt").read_text(
                encoding="utf-8"
            ),
            "input": {
                "event": {k: v for k, v in event.items() if k not in {"created_at"}},
                "evidence_refs": [self._evidence_key(eid) for eid in event["evidence_ids"]],
                "changeset": changeset,
                "review_feedback": feedback,
                "maintenance_contract": self._maintenance_contract(),
            },
            "max_model_requests": limits["model_requests"],
            "max_tool_actions": limits["tool_actions"],
            "task_token_budget": self.policy["event_token_budget"],
            "event_token_budget": self.policy["event_token_budget"],
            "max_output_tokens": 2048,
            "model_request_timeout_seconds": self.policy["model_request_timeout_seconds"],
            "deadline_seconds": max(
                0.01,
                min(
                    self.policy["event_deadline_seconds"],
                    (deadline - time.monotonic())
                    if deadline
                    else self.policy["event_deadline_seconds"],
                ),
            ),
            "tools": tools.declarations(),
        }
        output = self.runner.run(
            task,
            tools,
            ledger,
            event_id=event["id"],
            stage=event["id"] + ":" + stage,
            script=script,
            cancelled=self._cancel.is_set,
        )
        write_json(
            self.root / "traces" / f"{run_id}.json",
            {"task": task, "result": output, "tools": tools.trace},
        )
        if output["stop_reason"] != "completed" or output.get("execution_mode") != "fixture":
            raise ValueError("AGENT_" + output["stop_reason"].upper())
        reads = tools.read_coverage()
        required = (
            {ref["evidence_id"] for ref in changeset["evidence_refs"]}
            if changeset
            else set(event["evidence_ids"])
        )
        if not required.issubset({r["evidence_id"] for r in reads if r["complete"]}):
            raise ValueError("FULL_SOURCE_READ_REQUIRED")
        result = tools.submission or output["output"]
        return (
            result,
            tools,
            run_id,
            hashlib.sha256(
                json.dumps(
                    {"events": output["events"], "tools": tools.trace},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        )

    def _evidence_key(self, eid):
        source = self.corpus.resolve_evidence(eid)
        return {key: source[key] for key in ("evidence_id", "raw_sha256", "text_sha256")}

    def _run_record(
        self, event, revision, role, run_id, trace_sha, ledger, bindings, evidence_manifest
    ):
        return {
            "run_id": run_id,
            "role": role,
            "provider": "fixture-no-provider",
            "model_id": "maintenance-fixture-v1",
            "model_family": "fixture",
            "execution_mode": "fixture",
            "adapter_config_sha256": canonical_sha256({"fixture": True}),
            "prompt_sha256": bindings[role + "_prompt_sha256"],
            "tool_contract_sha256": bindings["tool_contract_sha256"],
            "trace_sha256": trace_sha,
            "stop_reason": "completed",
            "budget_state": "settled",
            "budget_policy_id": self.policy["policy_id"],
            "budget_ledger_sha256": canonical_sha256(ledger.summary()),
            "changeset_id": event["id"],
            "revision": revision,
            "base_knowledge_commit": self.base_commit,
            "evidence_manifest_sha256": evidence_manifest,
            "bindings": bindings,
            "scope": self._scope,
        }

    def _prepare(self, event, ledger, scripts):
        job = self.store.create_job(event["id"])
        if job["status"] in {
            "engineering_complete",
            "approved",
            "rejected",
            "needs_evidence",
            "needs_review",
            "failed",
            "cancelled",
        }:
            return {"cached": True, "result": job["details"]}
        lease = self.store.acquire_lease(
            job["id"], uuid4().hex, self.policy["event_deadline_seconds"] + 30
        )
        fence = {
            "job_id": job["id"],
            "owner": lease["owner"],
            "fencing_token": lease["fencing_token"],
        }
        self.store.transition_job(
            job["id"], job["status"], "proposing", lease["owner"], lease["fencing_token"]
        )
        self.store.set_event_status(event["id"], event["status"], "proposing", **fence)
        return {"event": event, "job": job, "fence": fence, "scripts": scripts}

    def _process(self, context, ledger):
        event, fence, job = context["event"], context["fence"], context["job"]
        scripts = context["scripts"]
        status, error, review = "failed", None, None
        deadline = context.get("deadline", time.monotonic() + self.policy["event_deadline_seconds"])
        try:
            if not scripts:
                raise ValueError("FIXTURE_SCRIPT_REQUIRED")
            for revision in (1, 2):
                stage = "propose" if revision == 1 else "repair"
                if revision == 1 and "proposal_error" in context:
                    raise context["proposal_error"]
                proposal, p_tools, p_id, p_trace = (
                    context["proposal_result"]
                    if revision == 1 and "proposal_result" in context
                    else self._role_run(
                        event,
                        revision,
                        "proposer",
                        stage,
                        ledger,
                        scripts.get(stage),
                        feedback=review,
                        deadline=deadline,
                    )
                )
                if not isinstance(proposal, dict) or set(proposal) != {
                    "operations",
                    "claims",
                    "source_coverage",
                }:
                    raise ValueError("PROPOSAL_FIELDS")
                cited_ids = set(event["evidence_ids"]) | {
                    c["evidence_id"] for claim in proposal["claims"] for c in claim["citations"]
                }
                if len(cited_ids) > 3:
                    raise ValueError("EVIDENCE_SCOPE_TOO_LARGE")
                if not cited_ids.issubset(
                    {r["evidence_id"] for r in p_tools.read_coverage() if r["complete"]}
                ):
                    raise ValueError("FULL_SOURCE_READ_REQUIRED")
                bindings, refs = (
                    self._bindings(),
                    [self._evidence_key(eid) for eid in sorted(cited_ids)],
                )
                evidence_manifest = canonical_sha256(refs)
                p_run = self._run_record(
                    event, revision, "proposer", p_id, p_trace, ledger, bindings, evidence_manifest
                )
                changeset = {
                    "schema": "finrisk-changeset/1.0",
                    "changeset_id": event["id"],
                    "revision": revision,
                    "event": {
                        "event_id": event["id"],
                        "event_type": event["event_type"],
                        "idempotency_key": event["idempotency_key"],
                        "reason": event["reason"],
                    },
                    "base_knowledge_commit": self.base_commit,
                    "scope": self._scope,
                    "bindings": bindings,
                    "proposer_run": {
                        k: p_run[k]
                        for k in (
                            "run_id",
                            "provider",
                            "model_id",
                            "model_family",
                            "execution_mode",
                            "adapter_config_sha256",
                            "prompt_sha256",
                            "tool_contract_sha256",
                            "trace_sha256",
                        )
                    },
                    "proposer_attestation_sha256": attestation_sha256(p_run),
                    "evidence_manifest_sha256": evidence_manifest,
                    "evidence_refs": refs,
                    **proposal,
                    "affected_record_ids": sorted(
                        {op["target_id"] for op in proposal["operations"]}
                    ),
                    "dependency_manifest_sha256": canonical_sha256(
                        self.corpus.records(self.base_commit)
                    ),
                }
                changeset["changeset_sha256"] = changeset_sha256(changeset)
                validate_changeset(
                    changeset,
                    business_validator=business_validator,
                    evidence_resolver=self.corpus.resolve_evidence,
                )
                p_run["output_sha256"] = changeset["changeset_sha256"]
                self.store.save_changeset(event["id"], changeset, revision, **fence)
                self.store.save_review_run(event["id"], revision, p_run, run_id=p_id, **fence)
                stage = "review" if revision == 1 else "rereview"
                review, r_tools, r_id, r_trace = self._role_run(
                    event,
                    revision,
                    "reviewer",
                    stage,
                    ledger,
                    scripts.get(stage),
                    changeset=changeset,
                    deadline=deadline,
                )
                r_run = self._run_record(
                    event, revision, "reviewer", r_id, r_trace, ledger, bindings, evidence_manifest
                )
                r_run.update(
                    output_sha256=canonical_sha256(review),
                    source_reads=[
                        r for r in r_tools.read_coverage() if r["evidence_id"] in cited_ids
                    ],
                    independent_search_trace_ids=r_tools.search_trace_ids,
                    review_advice=review,
                )
                trusted = {p_id: p_run, r_id: r_run}
                if review.get("decision") == "approve":
                    if self._cancel.is_set():
                        raise ValueError("CANCELLED")
                    receipt = issue_receipt(
                        changeset,
                        review,
                        reviewer_run_id=r_id,
                        trusted_runs=trusted,
                        policy=self.policy,
                        key_id="maintenance-local-v1",
                        key=self._key,
                        expected_base=self.base_commit,
                        expected_bindings=bindings,
                        evidence_resolver=self.corpus.resolve_evidence,
                        business_validator=business_validator,
                        mock_mode=True,
                    )
                    r_run.update(
                        receipt_id=receipt["receipt_id"], receipt_sha256=canonical_sha256(receipt)
                    )
                    self.store.save_review_run(event["id"], revision, r_run, run_id=r_id, **fence)
                    self.store.save_receipt(event["id"], revision, receipt, **fence)
                    status = "engineering_complete"
                    break
                self.store.save_review_run(event["id"], revision, r_run, run_id=r_id, **fence)
                if review.get("decision") == "request_changes" and revision == 1:
                    continue
                status = (
                    review.get("decision")
                    if review.get("decision") in {"needs_evidence", "reject"}
                    else "needs_review"
                )
                status = "rejected" if status == "reject" else status
                break
        except Exception as exc:
            text = getattr(exc, "code", str(exc).split(":", 1)[0])
            error = (
                text
                if isinstance(text, str) and text.replace("_", "").isalnum() and text.isascii()
                else "MAINTENANCE_FAILED"
            )
            if error in {"CANCELLED", "AGENT_CANCELLED"}:
                status = "cancelled"
        result = {
            "event_id": event["id"],
            "status": status,
            "error": error,
            "review_decision": review.get("decision") if isinstance(review, dict) else None,
            "publication_allowed": False,
            "execution_mode": "fixture",
        }
        self.store.transition_job(
            job["id"], "proposing", status, fence["owner"], fence["fencing_token"], details=result
        )
        self.store.set_event_status(event["id"], "proposing", status, **fence)
        self.store.release_lease(job["id"], fence["owner"], fence["fencing_token"])
        return result

    def run_events(self, event_ids, *, execution_mode="fixture", scripts=None):
        if execution_mode != "fixture":
            raise ValueError("LIVE_ADAPTERS_NOT_READY")
        if (
            not isinstance(event_ids, list)
            or not 1 <= len(event_ids) <= 20
            or len(event_ids) != len(set(event_ids))
        ):
            raise ValueError("INVALID_EVENT_BATCH")
        events = [self.store.get_event(eid) for eid in event_ids]
        batch = canonical_sha256([sorted(event_ids), self.policy["policy_id"], "fixture"])
        with update_lock(self.root / "operations"):
            bindings = [self.store.get_artifact("budget:" + eid) for eid in event_ids]
            old_batches = {v["payload"]["batch_id"] for v in bindings if v}
            if len(old_batches) > 1:
                raise ValueError("BATCH_REPLAY_SCOPE_MISMATCH")
            if old_batches:
                batch = next(iter(old_batches))
            batch_artifact = self.store.get_artifact("batch:" + batch)
            if old_batches and (
                batch_artifact is None
                or not set(event_ids).issubset(batch_artifact["payload"]["event_ids"])
            ):
                raise ValueError("BATCH_REPLAY_SCOPE_MISMATCH")
            full_ids = (
                batch_artifact["payload"]["event_ids"] if batch_artifact else sorted(event_ids)
            )
            if not batch_artifact:
                self.store.save_artifact("batch:" + batch, {"event_ids": full_ids})
            for eid in event_ids:
                self.store.save_artifact("budget:" + eid, {"batch_id": batch})
            limits = {
                "event": {"*": {"max_requests": self.policy["max_event_model_requests"]}},
                "stage": {
                    eid + ":" + stage: {"max_requests": v["model_requests"]}
                    for eid in full_ids
                    for stage, v in self.policy["stages"].items()
                },
            }
            ledger = BudgetLedger(
                self.root / "budgets" / f"{batch}.sqlite3",
                self.policy["policy_id"],
                0,
                28 * len(full_ids),
                limits=limits,
            )
            items = []

            def propose(event):
                context = self._prepare(event, ledger, (scripts or {}).get(event["id"]))
                context["deadline"] = time.monotonic() + self.policy["event_deadline_seconds"]
                if not context.get("cached"):
                    try:
                        if not context["scripts"]:
                            raise ValueError("FIXTURE_SCRIPT_REQUIRED")
                        context["proposal_result"] = self._role_run(
                            event,
                            1,
                            "proposer",
                            "propose",
                            ledger,
                            context["scripts"].get("propose"),
                            deadline=context["deadline"],
                        )
                    except Exception as error:
                        context["proposal_error"] = error
                return context

            # Up to two proposer contexts; one reviewer/merge lane, all one ledger.
            with ThreadPoolExecutor(max_workers=self.policy["proposer_concurrency"]) as pool:
                futures = [pool.submit(propose, event) for event in events]
                for future in futures:
                    context = future.result()
                    items.append(
                        context["result"]
                        if context.get("cached")
                        else self._process(context, ledger)
                    )
            summary = ledger.summary()
            token_budget = NodeMaintenanceRunner.token_summary(ledger)
            result = {
                "batch_id": batch,
                "items": items,
                "publication_allowed": False,
                "execution_mode": "fixture",
                "budget": {
                    **summary,
                    "requests": summary["request_count"],
                    "settled_micro": summary["spent_micro"],
                    "tokens": token_budget,
                },
            }
            write_json(self.root / "reports" / f"{batch}.json", result)
            return result

    def cancel(self):
        self._cancel.set()

    def submit_events(self, event_ids, *, execution_mode="fixture", scripts=None):
        if execution_mode != "fixture":
            raise ValueError("LIVE_ADAPTERS_NOT_READY")
        if (
            not isinstance(event_ids, list)
            or not 1 <= len(event_ids) <= 20
            or len(set(event_ids)) != len(event_ids)
        ):
            raise ValueError("INVALID_EVENT_BATCH")
        for eid in event_ids:
            self.store.get_event(eid)
        with self._api_lock, update_lock(self.root / "apioperations"):
            if self._active_run_id and any(t.is_alive() for t in self._threads):
                raise ValueError("KNOWLEDGE_JOB_BUSY")
            run_id = uuid4().hex
            # A new token is installed before enqueue; workers never clear it.
            self._cancel = threading.Event()
            row = {
                "run_id": run_id,
                "status": "queued",
                "event_ids": event_ids,
                "execution_mode": execution_mode,
                "publication_allowed": False,
                "budget": None,
                "result": None,
                "error": None,
            }
            write_json(self.root / "api_runs" / f"{run_id}.json", row)
            self._active_run_id = run_id

            def work():
                try:
                    current = self.get_run(run_id)
                    if current.get("cancel_requested"):
                        raise ValueError("CANCELLED")
                    row.update(status="running")
                    write_json(self.root / "api_runs" / f"{run_id}.json", row)
                    result = self.run_events(
                        event_ids, execution_mode=execution_mode, scripts=scripts
                    )
                    ok = all(item["status"] == "engineering_complete" for item in result["items"])
                    row.update(
                        status="cancelled"
                        if self._cancel.is_set()
                        else "engineering_complete"
                        if ok
                        else "failed",
                        result=result,
                        budget=result["budget"],
                    )
                except Exception as exc:
                    code = getattr(exc, "code", str(exc).split(":", 1)[0])
                    row.update(
                        status="cancelled" if code == "CANCELLED" else "failed",
                        error=code
                        if isinstance(code, str)
                        and code.isascii()
                        and code.replace("_", "").isalnum()
                        else "MAINTENANCE_FAILED",
                    )
                finally:
                    if self.get_run(run_id).get("cancel_requested"):
                        row["cancel_requested"] = True
                    write_json(self.root / "api_runs" / f"{run_id}.json", row)

            thread = threading.Thread(target=work, daemon=True)
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
            thread.start()
            return dict(row)

    def list_runs(self):
        paths = sorted(
            (self.root / "api_runs").glob("*.json"),
            key=lambda path: (path.lstat().st_mtime_ns, path.name),
            reverse=True,
        )
        result = []
        for path in paths[:50]:
            try:
                result.append(self.get_run(path.stem))
            except (OSError, ValueError):
                # Corrupt or invalid host files are not public run records.
                continue
        return result

    def get_run(self, run_id):
        import re

        if not isinstance(run_id, str) or not re.fullmatch("[0-9a-f]{32}", run_id):
            raise ValueError("INVALID_RUN_ID")
        directory = self.root / "api_runs"
        path = directory / f"{run_id}.json"
        if not path.resolve().is_relative_to(directory):
            raise ValueError("INVALID_RUN_PATH")
        if not path.is_file():
            raise ValueError("UNKNOWN_RUN")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("run_id") != run_id
            or not isinstance(value.get("status"), str)
            or not value["status"]
            or not isinstance(value.get("event_ids"), list)
            or not value["event_ids"]
            or any(
                not isinstance(eid, str) or not re.fullmatch("[0-9a-f]{32}", eid)
                for eid in value["event_ids"]
            )
        ):
            raise ValueError("INVALID_RUN_RECORD")
        return value

    def cancel_run(self, run_id):
        row = self.get_run(run_id)
        if row["status"] in {"queued", "running"}:
            row["cancel_requested"] = True
            write_json(self.root / "api_runs" / f"{run_id}.json", row)
            if self._active_run_id == run_id:
                self.cancel()
        return row
