"""Knowledge maintenance protocol and fail-closed, host-issued review receipts.

``trusted_runs`` must come from the host's immutable run registry, never from a
model result or an uploaded JSON bundle. Evidence resolvers likewise resolve
registered evidence IDs; passing a model's copy of its source is not trust.
Business validators validate one operation and return the nonempty collection
of JSON pointers that require claims. This binds business semantics to claims.
Each claim's statement is its exact pointed-to string value, or compact
canonical JSON for a structured/scalar non-string value; free paraphrases
cannot stand in for a different business value.

Run records contain role/run_id, actual provider/model_id/model_family,
execution_mode, adapter_config_sha256, prompt_sha256, tool_contract_sha256,
trace_sha256, stop_reason, budget_state, budget_policy_id,
budget_ledger_sha256, changeset_id/revision/base_knowledge_commit,
evidence_manifest_sha256, bindings, scope and output_sha256. Reviewers also
contain source_reads and independent_search_trace_ids. The host registers the
issued receipt_id and canonical receipt_sha256 on the reviewer record before
production use. Proposer output_sha256 is the changeset's self-excluding hash;
reviewer output_sha256 is the canonical hash of its four-field review result.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker

MAX_SAFE_INTEGER = 9007199254740991
_ROOT = Path(__file__).resolve().parents[2]
_IDENTITY = (
    "provider",
    "model_id",
    "model_family",
    "execution_mode",
    "adapter_config_sha256",
    "prompt_sha256",
    "tool_contract_sha256",
)
_REVIEW_FIELDS = {"decision", "claim_checks", "omission_check_completed", "issues"}
_RUN_BINDINGS = (
    "changeset_id",
    "revision",
    "base_knowledge_commit",
    "evidence_manifest_sha256",
    "bindings",
    "scope",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class MaintenanceContractError(ValueError):
    """Stable machine-readable error code, with no source text or secrets."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(code + (": " + detail if detail else ""))


def _need(condition: Any, code: str, detail: str = "") -> None:
    if not condition:
        raise MaintenanceContractError(code, detail)


def canonical_bytes(value: Any) -> bytes:
    """UTF-8, sorted ASCII keys, compact JSON, no floats or implicit Unicode edits."""

    def check(item: Any) -> None:
        kind = type(item)
        if item is None or kind is bool:
            return
        if kind is int:
            _need(abs(item) <= MAX_SAFE_INTEGER, "CANONICAL_INTEGER")
        elif kind is str:
            try:
                item.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise MaintenanceContractError("CANONICAL_UNICODE") from exc
        elif kind is list:
            for child in item:
                check(child)
        elif kind is dict:
            for key, child in item.items():
                _need(type(key) is str and key.isascii(), "CANONICAL_KEY")
                check(child)
        else:
            raise MaintenanceContractError("CANONICAL_TYPE")

    check(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def changeset_sha256(value: dict) -> str:
    """Hash a changeset without its own hash (all other fields remain bound)."""
    return canonical_sha256({k: v for k, v in value.items() if k != "changeset_sha256"})


def attestation_sha256(run: Mapping) -> str:
    """Digest a trusted proposer record without circular output/receipt hashes."""
    return canonical_sha256(
        {k: v for k, v in run.items() if k not in {"output_sha256", "receipt_id", "receipt_sha256"}}
    )


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads(
        (_ROOT / "contracts/knowledge_maintenance/protocol.schema.json").read_text(
            encoding="utf-8-sig"
        )
    )


def _validate_shape(value: Any, definition: str) -> None:
    canonical_bytes(value)
    schema = _schema()
    validator = Draft202012Validator(
        {"$ref": "#/$defs/" + definition, "$defs": schema["$defs"]}, format_checker=FormatChecker()
    )
    error = next(validator.iter_errors(value), None)
    _need(error is None, "PROTOCOL_SCHEMA", definition)


def _unique(items: list, key: str) -> dict:
    result = {}
    for item in items:
        identity = item[key]
        _need(identity not in result, "DUPLICATE_ID", key)
        result[identity] = item
    return result


def _pointer(value: Any, pointer: str) -> Any:
    _need(type(pointer) is str and pointer.startswith("/"), "CLAIM_POINTER")
    current = value
    try:
        for part in pointer[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if type(current) is list:
                _need(part.isdigit() and (part == "0" or not part.startswith("0")), "CLAIM_POINTER")
                current = current[int(part)]
            else:
                current = current[part]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise MaintenanceContractError("CLAIM_POINTER") from exc
    return current


def _full_ranges(ranges: list, length: int, code: str) -> None:
    cursor = 0
    for span in sorted(ranges, key=lambda item: (item["start"], item["end"])):
        start, end = span["start"], span["end"]
        _need(
            type(start) is int
            and type(end) is int
            and 0 <= start < end <= length
            and start <= cursor,
            code,
        )
        cursor = max(cursor, end)
    _need(cursor == length, code)


def _resolve_evidence(changeset: dict, resolver: Callable | None) -> dict:
    if resolver is None:
        return {}
    resolved = {}
    for ref in changeset["evidence_refs"]:
        try:
            source = resolver(ref["evidence_id"])
        except Exception as exc:
            raise MaintenanceContractError("EVIDENCE_UNAVAILABLE") from exc
        _need(isinstance(source, Mapping), "EVIDENCE_UNAVAILABLE")
        _need(source.get("evidence_id") == ref["evidence_id"], "EVIDENCE_ID")
        raw = source.get("raw_bytes", source.get("raw"))
        text = source.get("text")
        _need(type(raw) is bytes and type(text) is str and text, "EVIDENCE_CONTENT")
        _need(hashlib.sha256(raw).hexdigest() == ref["raw_sha256"], "EVIDENCE_RAW_HASH")
        _need(
            hashlib.sha256(text.encode("utf-8")).hexdigest() == ref["text_sha256"],
            "EVIDENCE_TEXT_HASH",
        )
        _need(
            source.get("raw_sha256", ref["raw_sha256"]) == ref["raw_sha256"]
            and source.get("text_sha256", ref["text_sha256"]) == ref["text_sha256"],
            "EVIDENCE_METADATA_HASH",
        )
        from agent_rag.maintenance_tools import authorized

        _need(authorized(source, changeset["scope"]), "EVIDENCE_SCOPE")
        _need(source.get("extractor_version"), "EVIDENCE_EXTRACTOR")
        resolved[ref["evidence_id"]] = source
    return resolved


def _citation(citation: dict, refs: dict, sources: dict) -> None:
    ref = refs.get(citation["evidence_id"])
    _need(ref is not None, "CITATION_EVIDENCE")
    _need(all(citation[key] == ref[key] for key in ("raw_sha256", "text_sha256")), "CITATION_HASH")
    start, end = citation["start"], citation["end"]
    _need(start < end and end - start == len(citation["quote"]), "CITATION_QUOTE")
    if sources:
        text = sources[ref["evidence_id"]]["text"]
        _need(end <= len(text) and text[start:end] == citation["quote"], "CITATION_QUOTE")


def validate_changeset(
    value: dict,
    *,
    business_validator: Callable | None = None,
    evidence_resolver: Callable | None = None,
) -> dict:
    """Validate outer structure/references; injected validators add trusted semantics.

    This function alone is never a production approval. It may be used for
    incomplete/offline candidate inspection, without trusted adapters.
    """
    _validate_shape(value, "Changeset")
    _need(changeset_sha256(value) == value["changeset_sha256"], "CHANGESET_HASH")
    _need(
        canonical_sha256(value["evidence_refs"]) == value["evidence_manifest_sha256"],
        "EVIDENCE_MANIFEST_HASH",
    )
    operations = _unique(value["operations"], "operation_id")
    _unique(value["operations"], "target_id")
    claims = _unique(value["claims"], "claim_id")
    refs = _unique(value["evidence_refs"], "evidence_id")
    _need(
        len(value["affected_record_ids"]) == len(set(value["affected_record_ids"]))
        and set(value["affected_record_ids"]) == {op["target_id"] for op in operations.values()},
        "AFFECTED_RECORDS",
    )
    sources = _resolve_evidence(value, evidence_resolver)
    for claim in claims.values():
        operation = operations.get(claim["operation_id"])
        _need(operation is not None, "CLAIM_OPERATION")
        business_value = _pointer(operation["after"], claim["pointer"])
        expected_statement = (
            business_value
            if type(business_value) is str
            else canonical_bytes(business_value).decode("utf-8")
        )
        _need(claim["statement"] == expected_statement, "CLAIM_VALUE")
        if claim["claim_type"] == "retrieval_hypothesis":
            _need(claim["uncertainty"], "HYPOTHESIS_UNCERTAINTY")
        for citation in claim["citations"]:
            _citation(citation, refs, sources)
    for operation in operations.values():
        op_claims = [c for c in claims.values() if c["operation_id"] == operation["operation_id"]]
        _need(op_claims, "CLAIM_COVERAGE")
        if business_validator is not None:
            try:
                required = business_validator(copy.deepcopy(operation))
            except Exception as exc:
                raise MaintenanceContractError("BUSINESS_SCHEMA") from exc
            _need(
                isinstance(required, (list, tuple, set, frozenset)) and required,
                "BUSINESS_COVERAGE_REQUIRED",
            )
            for pointer in required:
                _need(
                    type(pointer) is str
                    and any(
                        pointer == c["pointer"] or pointer.startswith(c["pointer"] + "/")
                        for c in op_claims
                    ),
                    "CLAIM_COVERAGE",
                )
                _pointer(operation["after"], pointer)
    coverages = {identity: [] for identity in refs}
    for span in value["source_coverage"]:
        identity = span["evidence_id"]
        _need(
            identity in refs and span["text_sha256"] == refs[identity]["text_sha256"],
            "SOURCE_COVERAGE",
        )
        _need(span["start"] < span["end"], "SOURCE_COVERAGE")
        _need(
            all(cid in claims for cid in span["claim_ids"])
            and len(span["claim_ids"]) == len(set(span["claim_ids"])),
            "SOURCE_COVERAGE",
        )
        if span["disposition"] == "mapped":
            _need(span["claim_ids"], "SOURCE_COVERAGE")
            for cid in span["claim_ids"]:
                _need(
                    any(
                        c["evidence_id"] == identity
                        and c["start"] < span["end"]
                        and span["start"] < c["end"]
                        for c in claims[cid]["citations"]
                    ),
                    "SOURCE_COVERAGE",
                )
        else:
            _need(not span["claim_ids"], "SOURCE_COVERAGE")
        coverages[identity].append(span)
    if sources:
        for identity, source in sources.items():
            _full_ranges(coverages[identity], len(source["text"]), "SOURCE_COVERAGE")
    return copy.deepcopy(value)


def _trusted_record(trusted_runs: Mapping, run_id: str, role: str) -> dict:
    _need(isinstance(trusted_runs, Mapping), "TRUSTED_RUN_REQUIRED")
    run = trusted_runs.get(run_id)
    _need(
        isinstance(run, Mapping) and run.get("run_id") == run_id and run.get("role") == role,
        "TRUSTED_RUN_REQUIRED",
    )
    return dict(run)


def _run_checks(run: dict, changeset: dict, policy: dict, *, mock_mode: bool) -> None:
    canonical_bytes(run)
    _need(run.get("stop_reason") == "completed", "RUN_INCOMPLETE")
    _need(run.get("budget_state") == "settled", "BUDGET_UNCERTAIN")
    _need(
        run.get("execution_mode") in ({"fixture"} if mock_mode else {"session_assisted", "live"}),
        "FIXTURE_FORBIDDEN",
    )
    if run["execution_mode"] == "live":
        _need(policy["live"] is True, "LIVE_DISABLED")
    for field in _IDENTITY:
        _need(type(run.get(field)) is str and bool(run[field].strip()), "RUN_IDENTITY")
    if not mock_mode:
        unknown = {"unknown", "unverified", "none", "null", "fixture"}
        _need(
            all(
                run[field].strip().casefold() not in unknown
                and not run[field].strip().casefold().startswith("fixture-")
                for field in ("provider", "model_id", "model_family")
            ),
            "RUN_IDENTITY",
        )
    _need(
        all(
            type(run.get(key)) is str and _SHA.fullmatch(run[key])
            for key in (
                "adapter_config_sha256",
                "prompt_sha256",
                "tool_contract_sha256",
                "trace_sha256",
                "budget_ledger_sha256",
                "output_sha256",
            )
        ),
        "RUN_HASH",
    )
    _need(run.get("budget_policy_id") == policy["policy_id"], "RUN_POLICY")
    for field in _RUN_BINDINGS:
        _need(run.get(field) == changeset[field], "RUN_BINDING", field)
    role = run["role"]
    _need(
        run["prompt_sha256"] == changeset["bindings"][role + "_prompt_sha256"]
        and run["tool_contract_sha256"] == changeset["bindings"]["tool_contract_sha256"],
        "RUN_BINDING",
        "prompt/tool",
    )


def _read_checks(reads: list, sources: dict) -> None:
    _need(type(reads) is list and reads, "READ_COVERAGE")
    for read in reads:
        _validate_shape(read, "ReadCoverage")
    by_id = _unique(reads, "evidence_id")
    _need(set(by_id) == set(sources), "READ_COVERAGE")
    for identity, source in sources.items():
        read = by_id[identity]
        _need(
            read["complete"] is True
            and read["text_sha256"] == source["text_sha256"]
            and read["text_length_codepoints"] == len(source["text"]),
            "READ_COVERAGE",
        )
        _full_ranges(read["ranges"], len(source["text"]), "READ_COVERAGE")


def _review_checks(review: dict, changeset: dict, sources: dict) -> None:
    _need(type(review) is dict and set(review) == _REVIEW_FIELDS, "REVIEW_SCHEMA")
    canonical_bytes(review)
    _need(review["decision"] == "approve", "REVIEW_NOT_APPROVED")
    _need(review["omission_check_completed"] is True, "OMISSION_INCOMPLETE")
    _need(type(review["claim_checks"]) is list, "CLAIM_COVERAGE")
    for check in review["claim_checks"]:
        _validate_shape(check, "ClaimCheck")
    checks = _unique(review["claim_checks"], "claim_id")
    claims = {claim["claim_id"]: claim for claim in changeset["claims"]}
    _need(set(checks) == set(claims), "CLAIM_COVERAGE")
    refs = {ref["evidence_id"]: ref for ref in changeset["evidence_refs"]}
    for identity, claim in claims.items():
        check = checks[identity]
        expected = (
            "supported_as_hypothesis"
            if claim["claim_type"] == "retrieval_hypothesis"
            else "supported"
        )
        _need(check["verdict"] == expected, "CLAIM_UNSUPPORTED")
        _need(check["citations"], "CLAIM_COVERAGE")
        for citation in check["citations"]:
            _citation(citation, refs, sources)
    _need(type(review["issues"]) is list, "REVIEW_SCHEMA")
    for issue in review["issues"]:
        _validate_shape(issue, "Issue")
        _need(issue["severity"] != "blocking", "BLOCKING_ISSUE")
        _need(issue["claim_id"] is None or issue["claim_id"] in claims, "ISSUE_REFERENCE")
        operations = {op["operation_id"]: op for op in changeset["operations"]}
        _need(
            issue["operation_id"] is None or issue["operation_id"] in operations, "ISSUE_REFERENCE"
        )
    _unique(review["issues"], "issue_id")


def _approval_inputs(
    changeset: dict,
    *,
    reviewer_run: dict,
    review: dict,
    trusted_runs: Mapping,
    policy: dict,
    expected_base: str,
    expected_bindings: dict,
    evidence_resolver: Callable | None,
    business_validator: Callable | None,
    mock_mode: bool = False,
) -> tuple[dict, dict]:
    _need(callable(business_validator), "BUSINESS_VALIDATOR_REQUIRED")
    _need(callable(evidence_resolver), "EVIDENCE_RESOLVER_REQUIRED")
    validate_changeset(
        changeset, business_validator=business_validator, evidence_resolver=evidence_resolver
    )
    _validate_shape(policy, "Policy")
    _need(changeset["base_knowledge_commit"] == expected_base, "BASE_MISMATCH")
    _need(changeset["bindings"] == expected_bindings, "BINDINGS_MISMATCH")
    _need(changeset["bindings"]["policy_sha256"] == canonical_sha256(policy), "POLICY_HASH")
    _need(
        all(span["disposition"] != "needs_clarification" for span in changeset["source_coverage"]),
        "SOURCE_UNRESOLVED",
    )
    proposer = _trusted_record(trusted_runs, changeset["proposer_run"]["run_id"], "proposer")
    _run_checks(proposer, changeset, policy, mock_mode=mock_mode)
    _run_checks(reviewer_run, changeset, policy, mock_mode=mock_mode)
    _need(proposer["run_id"] != reviewer_run["run_id"], "MODEL_INDEPENDENCE")
    if not mock_mode:
        _need(
            proposer["model_family"].strip().casefold()
            != reviewer_run["model_family"].strip().casefold(),
            "MODEL_INDEPENDENCE",
        )
    _need(
        changeset["proposer_run"]
        == {key: proposer[key] for key in ("run_id", *_IDENTITY, "trace_sha256")},
        "PROPOSER_IDENTITY",
    )
    _need(
        changeset["proposer_attestation_sha256"] == attestation_sha256(proposer),
        "PROPOSER_ATTESTATION",
    )
    _need(proposer["output_sha256"] == changeset["changeset_sha256"], "PROPOSER_OUTPUT")
    _need(reviewer_run["output_sha256"] == canonical_sha256(review), "REVIEW_OUTPUT")
    sources = _resolve_evidence(changeset, evidence_resolver)
    if not mock_mode:
        _need(
            not any(source.get("example_only") is True for source in sources.values()),
            "FIXTURE_FORBIDDEN",
        )
    _read_checks(reviewer_run.get("source_reads"), sources)
    searches = reviewer_run.get("independent_search_trace_ids")
    _need(
        type(searches) is list
        and searches
        and all(type(v) is str and v for v in searches)
        and len(set(searches)) == len(searches),
        "INDEPENDENT_SEARCH_REQUIRED",
    )
    _review_checks(review, changeset, sources)
    return proposer, sources


def _signature(receipt: dict, key: bytes) -> str:
    _need(type(key) is bytes and len(key) >= 32, "RECEIPT_KEY")
    unsigned = {k: v for k, v in receipt.items() if k != "authentication"}
    # The key identifier and algorithm are authenticated as well as the payload.
    auth = receipt["authentication"]
    unsigned["authentication"] = {"algorithm": auth["algorithm"], "key_id": auth["key_id"]}
    return hmac.new(key, canonical_bytes(unsigned), hashlib.sha256).hexdigest()


def verify_receipt(receipt: dict, *, key_resolver: Mapping | Callable) -> dict:
    """Verify shape/HMAC only. Production callers must use require_approved."""
    _validate_shape(receipt, "Receipt")
    auth = receipt["authentication"]
    try:
        key = (
            key_resolver(auth["key_id"])
            if callable(key_resolver)
            else key_resolver.get(auth["key_id"])
        )
    except Exception as exc:
        raise MaintenanceContractError("RECEIPT_AUTH") from exc
    _need(type(key) is bytes and len(key) >= 32, "RECEIPT_AUTH")
    _need(hmac.compare_digest(auth["signature"], _signature(receipt, key)), "RECEIPT_AUTH")
    return copy.deepcopy(receipt)


def issue_receipt(
    changeset: dict,
    review: dict,
    *,
    reviewer_run_id: str,
    trusted_runs: Mapping,
    policy: dict,
    key_id: str,
    key: bytes,
    expected_base: str,
    expected_bindings: dict,
    evidence_resolver: Callable,
    business_validator: Callable,
    mock_mode: bool = False,
) -> dict:
    """Host-only signing API. Model review output cannot supply identity/trace.

    mock_mode permits fixture signing for engineering replay only; such receipts
    always fail require_approved, which has no mock switch. Only approve advice
    can be issued as an approval receipt; other advice remains a candidate log.
    """
    reviewer = _trusted_record(trusted_runs, reviewer_run_id, "reviewer")
    _approval_inputs(
        changeset,
        reviewer_run=reviewer,
        review=review,
        trusted_runs=trusted_runs,
        policy=policy,
        expected_base=expected_base,
        expected_bindings=expected_bindings,
        evidence_resolver=evidence_resolver,
        business_validator=business_validator,
        mock_mode=mock_mode,
    )
    receipt = {
        "schema": "finrisk-review-receipt/1.0",
        "receipt_id": str(uuid4()),
        **{
            field: changeset[field]
            for field in (
                "changeset_id",
                "revision",
                "changeset_sha256",
                "base_knowledge_commit",
                "proposer_attestation_sha256",
                "evidence_manifest_sha256",
            )
        },
        "reviewer": {
            **{field: reviewer[field] for field in _IDENTITY},
            "policy_sha256": changeset["bindings"]["policy_sha256"],
            "independence": "fixture" if mock_mode else "heterogeneous",
        },
        **copy.deepcopy(review),
        "source_reads": copy.deepcopy(reviewer["source_reads"]),
        "independent_search_trace_ids": list(reviewer["independent_search_trace_ids"]),
        "trace_sha256": reviewer["trace_sha256"],
        "budget_policy_id": reviewer["budget_policy_id"],
        "budget_ledger_sha256": reviewer["budget_ledger_sha256"],
        "stop_reason": reviewer["stop_reason"],
        "issued_at": datetime.now(UTC).isoformat(),
        "authentication": {"algorithm": "HMAC-SHA256", "key_id": key_id, "signature": "0" * 64},
    }
    receipt["authentication"]["signature"] = _signature(receipt, key)
    _validate_shape(receipt, "Receipt")
    return receipt


def require_approved(
    changeset: dict,
    receipt: dict,
    *,
    trusted_runs: Mapping,
    key_resolver: Mapping | Callable,
    policy: dict,
    expected_base: str,
    expected_bindings: dict,
    evidence_resolver: Callable,
    business_validator: Callable,
) -> dict:
    """Production gate: no fixtures, legacy declarations, or self-attested runs.

    The fixed historical baseline is exempted only by a separately pinned host
    integration. This API never accepts a legacy flag or an arbitrary path.
    """
    verify_receipt(receipt, key_resolver=key_resolver)
    _need(isinstance(trusted_runs, Mapping), "TRUSTED_RUN_REQUIRED")
    matches = [
        run
        for run in trusted_runs.values()
        if isinstance(run, Mapping)
        and run.get("role") == "reviewer"
        and run.get("receipt_id") == receipt["receipt_id"]
    ]
    _need(len(matches) == 1, "TRUSTED_RECEIPT_REQUIRED")
    reviewer = _trusted_record(trusted_runs, matches[0].get("run_id"), "reviewer")
    # Check actual completion/identity before comparing recorded receipt bytes.
    _approval_inputs(
        changeset,
        reviewer_run=reviewer,
        review={key: receipt[key] for key in _REVIEW_FIELDS},
        trusted_runs=trusted_runs,
        policy=policy,
        expected_base=expected_base,
        expected_bindings=expected_bindings,
        evidence_resolver=evidence_resolver,
        business_validator=business_validator,
    )
    _need(reviewer.get("receipt_sha256") == canonical_sha256(receipt), "TRUSTED_RECEIPT_HASH")
    _need(receipt["reviewer"]["independence"] == "heterogeneous", "MODEL_INDEPENDENCE")
    for field in (
        "changeset_id",
        "revision",
        "changeset_sha256",
        "base_knowledge_commit",
        "proposer_attestation_sha256",
        "evidence_manifest_sha256",
    ):
        _need(receipt[field] == changeset[field], "RECEIPT_BINDING", field)
    expected_reviewer = {
        **{field: reviewer[field] for field in _IDENTITY},
        "policy_sha256": changeset["bindings"]["policy_sha256"],
        "independence": "heterogeneous",
    }
    _need(receipt["reviewer"] == expected_reviewer, "REVIEWER_IDENTITY")
    for field in (
        "source_reads",
        "independent_search_trace_ids",
        "trace_sha256",
        "budget_policy_id",
        "budget_ledger_sha256",
        "stop_reason",
    ):
        _need(receipt[field] == reviewer[field], "RECEIPT_BINDING", field)
    return copy.deepcopy(changeset)
