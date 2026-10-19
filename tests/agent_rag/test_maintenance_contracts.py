import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module():
    return importlib.import_module("agent_rag.maintenance_contracts")


@pytest.mark.parametrize(
    "value", [1.0, float("nan"), 9007199254740992, {"中文": 1}, {1: "x"}, (1,), "\ud800"]
)
def test_canonical_rejects_ambiguous_json(value):
    with pytest.raises(ValueError, match="CANONICAL_"):
        module().canonical_bytes(value)


def test_canonical_utf8_fixed_vector():
    assert (
        module().canonical_bytes({"z": "保😀", "a": [True, None, -1]})
        == '{"a":[true,null,-1],"z":"保😀"}'.encode()
    )


def setup_case():
    m = module()
    policy = json.loads(
        (ROOT / "configs/knowledge_maintenance/policy.json").read_text(encoding="utf-8")
    )
    text = "甲😀未承诺收益。"
    raw = b"original:" + text.encode("utf-8")

    def digest(v):
        return hashlib.sha256(v).hexdigest()

    ref = {
        "evidence_id": "E1",
        "raw_sha256": digest(raw),
        "text_sha256": digest(text.encode("utf-8")),
    }
    evidence = {
        **ref,
        "raw_bytes": raw,
        "text": text,
        "scope_id": "public",
        "classification": "public",
        "extractor_version": "test/1",
    }
    bindings = {
        k: "1" * 64
        for k in (
            "dictionary_sha256",
            "business_schema_sha256",
            "tool_contract_sha256",
            "proposer_prompt_sha256",
            "reviewer_prompt_sha256",
            "policy_sha256",
        )
    }
    bindings["policy_sha256"] = m.canonical_sha256(policy)
    citation = {**ref, "start": 0, "end": len(text), "quote": text}
    changeset = {
        "schema": "finrisk-changeset/1.0",
        "changeset_id": "C1",
        "revision": 1,
        "event": {
            "event_id": "event1",
            "event_type": "new_source",
            "idempotency_key": "2" * 64,
            "reason": "新增证据",
        },
        "base_knowledge_commit": "3" * 40,
        "scope": {"scope_id": "public", "classification": "public"},
        "bindings": bindings,
        "proposer_run": {},
        "proposer_attestation_sha256": "0" * 64,
        "evidence_refs": [ref],
        "evidence_manifest_sha256": m.canonical_sha256([ref]),
        "operations": [
            {
                "operation_id": "OP1",
                "kind": "add",
                "record_kind": "behavior",
                "target_id": "B1",
                "expected_record_version": None,
                "before_sha256": None,
                "after": {"summary": text},
                "reason": "新增",
            }
        ],
        "claims": [
            {
                "claim_id": "CL1",
                "operation_id": "OP1",
                "pointer": "/summary",
                "claim_type": "fact",
                "statement": text,
                "citations": [citation],
                "uncertainty": None,
            }
        ],
        "source_coverage": [
            {
                "evidence_id": "E1",
                "text_sha256": ref["text_sha256"],
                "start": 0,
                "end": len(text),
                "disposition": "mapped",
                "claim_ids": ["CL1"],
                "reason": "覆盖全文",
            }
        ],
        "affected_record_ids": ["B1"],
        "dependency_manifest_sha256": "4" * 64,
        "changeset_sha256": "0" * 64,
    }
    read = {
        "evidence_id": "E1",
        "text_sha256": ref["text_sha256"],
        "text_length_codepoints": len(text),
        "ranges": [{"start": 0, "end": 2}, {"start": 2, "end": len(text)}],
        "complete": True,
    }
    common = {
        "provider": "local-test",
        "model_id": "model-a",
        "model_family": "family-a",
        "execution_mode": "session_assisted",
        "adapter_config_sha256": "5" * 64,
        "prompt_sha256": "1" * 64,
        "tool_contract_sha256": "1" * 64,
        "trace_sha256": "6" * 64,
        "stop_reason": "completed",
        "budget_state": "settled",
        "budget_policy_id": policy["policy_id"],
        "budget_ledger_sha256": "7" * 64,
        "changeset_id": "C1",
        "revision": 1,
        "base_knowledge_commit": changeset["base_knowledge_commit"],
        "evidence_manifest_sha256": changeset["evidence_manifest_sha256"],
        "bindings": copy.deepcopy(bindings),
        "scope": copy.deepcopy(changeset["scope"]),
    }
    proposer = {**common, "role": "proposer", "run_id": "P1"}
    changeset["proposer_run"] = {
        k: proposer[k]
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
    }
    changeset["proposer_attestation_sha256"] = m.attestation_sha256(proposer)
    rehash(changeset)
    proposer["output_sha256"] = changeset["changeset_sha256"]
    review = {
        "decision": "approve",
        "claim_checks": [
            {
                "claim_id": "CL1",
                "verdict": "supported",
                "citations": [citation],
                "comment": "核验通过",
            }
        ],
        "omission_check_completed": True,
        "issues": [],
    }
    reviewer = {
        **common,
        "role": "reviewer",
        "run_id": "R1",
        "model_id": "model-b",
        "model_family": "family-b",
        "source_reads": [read],
        "independent_search_trace_ids": ["search1"],
        "output_sha256": m.canonical_sha256(review),
    }
    runs = {"P1": proposer, "R1": reviewer}
    context = dict(
        trusted_runs=runs,
        policy=policy,
        expected_base=changeset["base_knowledge_commit"],
        expected_bindings=copy.deepcopy(bindings),
        evidence_resolver=lambda eid: evidence if eid == "E1" else None,
        business_validator=lambda op: {"/summary"},
    )
    return changeset, review, runs, context, evidence


def rehash(changeset):
    changeset["changeset_sha256"] = module().canonical_sha256(
        {k: v for k, v in changeset.items() if k != "changeset_sha256"}
    )


def test_private_changeset_can_reference_public_evidence_without_scope_escalation():
    changeset, _, _, context, evidence = setup_case()
    changeset["scope"] = {"scope_id": "team-a", "classification": "team_private"}
    rehash(changeset)
    module().validate_changeset(changeset, evidence_resolver=context["evidence_resolver"])
    evidence.update(scope_id="team-a", classification="business_sensitive")
    with pytest.raises(ValueError, match="EVIDENCE_SCOPE"):
        module().validate_changeset(changeset, evidence_resolver=context["evidence_resolver"])


def signed_case():
    changeset, review, runs, context, evidence = setup_case()
    receipt = module().issue_receipt(
        changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
    )
    runs["R1"].update(
        receipt_id=receipt["receipt_id"], receipt_sha256=module().canonical_sha256(receipt)
    )
    return changeset, receipt, runs, context, evidence


def approve(changeset, receipt, context):
    return module().require_approved(
        changeset, receipt, key_resolver={"test": b"s" * 32}, **context
    )


def test_signed_trusted_runs_approve():
    changeset, receipt, _, context, _ = signed_case()
    assert approve(changeset, receipt, context)["changeset_sha256"] == changeset["changeset_sha256"]


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("stop_reason", "interrupted", "RUN_INCOMPLETE"),
        ("budget_state", "unknown", "BUDGET_UNCERTAIN"),
        ("execution_mode", "fixture", "FIXTURE_FORBIDDEN"),
        ("model_family", "family-a", "MODEL_INDEPENDENCE"),
    ],
)
def test_rejects_actual_run_failures_even_when_receipt_claims_approve(field, value, code):
    changeset, receipt, runs, context, _ = signed_case()
    runs["R1"][field] = value
    with pytest.raises(ValueError, match=code):
        approve(changeset, receipt, context)


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda c, r, ctx, e: c["operations"][0]["after"].update(summary="篡改"), "CHANGESET_HASH"),
        (lambda c, r, ctx, e: r.update(decision="reject"), "RECEIPT_AUTH"),
        (lambda c, r, ctx, e: ctx.update(expected_base="9" * 40), "BASE_MISMATCH"),
        (lambda c, r, ctx, e: ctx.update(business_validator=None), "BUSINESS_VALIDATOR_REQUIRED"),
        (lambda c, r, ctx, e: ctx.update(evidence_resolver=None), "EVIDENCE_RESOLVER_REQUIRED"),
        (lambda c, r, ctx, e: e.update(raw_bytes=b"tampered"), "EVIDENCE_RAW_HASH"),
        (
            lambda c, r, ctx, e: ctx.update(business_validator=lambda op: {"/summary", "/missing"}),
            "CLAIM_COVERAGE",
        ),
    ],
)
def test_production_gate_rejects_tampering_and_missing_trust(mutate, code):
    changeset, receipt, _, context, evidence = signed_case()
    mutate(changeset, receipt, context, evidence)
    with pytest.raises(ValueError, match=code):
        approve(changeset, receipt, context)


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda c, r: c["claims"][0]["citations"][0].update(start=1), "CITATION_QUOTE"),
        (lambda c, r: c["claims"].append(copy.deepcopy(c["claims"][0])), "DUPLICATE_ID"),
        (lambda c, r: c["source_coverage"][0].update(end=2), "SOURCE_COVERAGE"),
        (lambda c, r: r.update(omission_check_completed=False), "OMISSION_INCOMPLETE"),
        (lambda c, r: r["claim_checks"][0].update(verdict="unsupported"), "CLAIM_UNSUPPORTED"),
        (lambda c, r: r.update(claim_checks=[]), "CLAIM_COVERAGE"),
    ],
)
def test_signer_refuses_invalid_semantics(change, code):
    changeset, review, runs, context, _ = setup_case()
    change(changeset, review)
    rehash(changeset)
    runs["P1"]["output_sha256"] = changeset["changeset_sha256"]
    runs["R1"]["output_sha256"] = module().canonical_sha256(review)
    with pytest.raises(ValueError, match=code):
        module().issue_receipt(
            changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
        )


def test_actual_read_gaps_prevent_signing():
    changeset, review, runs, context, _ = setup_case()
    runs["R1"]["source_reads"][0]["ranges"] = [{"start": 0, "end": 2}, {"start": 3, "end": 8}]
    with pytest.raises(ValueError, match="READ_COVERAGE"):
        module().issue_receipt(
            changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
        )


def test_default_policy_disables_live():
    _, _, _, context, _ = setup_case()
    assert context["policy"]["live"] is False


def test_unknown_registered_model_identity_cannot_approve():
    changeset, review, runs, context, _ = setup_case()
    runs["R1"]["model_family"] = "unknown"
    with pytest.raises(ValueError, match="RUN_IDENTITY"):
        module().issue_receipt(
            changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
        )


def test_fixture_receipt_is_verifiable_but_never_production_approved():
    changeset, review, runs, context, evidence = setup_case()
    evidence["example_only"] = True
    for run in runs.values():
        run["execution_mode"] = "fixture"
    changeset["proposer_run"]["execution_mode"] = "fixture"
    changeset["proposer_attestation_sha256"] = module().attestation_sha256(runs["P1"])
    rehash(changeset)
    runs["P1"]["output_sha256"] = changeset["changeset_sha256"]
    receipt = module().issue_receipt(
        changeset,
        review,
        reviewer_run_id="R1",
        key_id="test",
        key=b"s" * 32,
        mock_mode=True,
        **context,
    )
    runs["R1"].update(
        receipt_id=receipt["receipt_id"], receipt_sha256=module().canonical_sha256(receipt)
    )
    assert module().verify_receipt(receipt, key_resolver={"test": b"s" * 32}) == receipt
    with pytest.raises(ValueError, match="FIXTURE_FORBIDDEN"):
        approve(changeset, receipt, context)


@pytest.mark.parametrize(
    "change,code",
    [
        (
            lambda c, r, runs, ctx, e: runs["R1"].update(receipt_sha256="0" * 64),
            "TRUSTED_RECEIPT_HASH",
        ),
        (lambda c, r, runs, ctx, e: runs["R1"].pop("receipt_id"), "TRUSTED_RECEIPT_REQUIRED"),
        (lambda c, r, runs, ctx, e: runs["P1"].update(model_id="fake-model"), "PROPOSER_IDENTITY"),
        (lambda c, r, runs, ctx, e: e.update(text="甲😀承诺收益。"), "EVIDENCE_TEXT_HASH"),
        (lambda c, r, runs, ctx, e: e.update(example_only=True), "FIXTURE_FORBIDDEN"),
        (
            lambda c, r, runs, ctx, e: ctx.update(business_validator=lambda op: None),
            "BUSINESS_COVERAGE_REQUIRED",
        ),
        (
            lambda c, r, runs, ctx, e: ctx["expected_bindings"].update(
                proposer_prompt_sha256="9" * 64
            ),
            "BINDINGS_MISMATCH",
        ),
        (lambda c, r, runs, ctx, e: r["authentication"].update(key_id="renamed"), "RECEIPT_AUTH"),
    ],
)
def test_trusted_registry_and_source_must_match(change, code):
    changeset, receipt, runs, context, evidence = signed_case()
    change(changeset, receipt, runs, context, evidence)
    with pytest.raises(ValueError, match=code):
        approve(changeset, receipt, context)


def test_blocking_issue_prevents_signing():
    changeset, review, runs, context, _ = setup_case()
    review["issues"] = [
        {
            "issue_id": "I1",
            "severity": "blocking",
            "category": "omission",
            "operation_id": "OP1",
            "claim_id": "CL1",
            "pointer": "/summary",
            "diff_line": None,
            "citations": [],
            "change_request": "保留否定",
        }
    ]
    runs["R1"]["output_sha256"] = module().canonical_sha256(review)
    with pytest.raises(ValueError, match="BLOCKING_ISSUE"):
        module().issue_receipt(
            changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
        )


def test_proposer_attestation_is_from_registered_record():
    changeset, review, runs, context, _ = setup_case()
    runs["P1"]["trace_sha256"] = "8" * 64
    changeset["proposer_run"]["trace_sha256"] = "8" * 64
    rehash(changeset)
    runs["P1"]["output_sha256"] = changeset["changeset_sha256"]
    with pytest.raises(ValueError, match="PROPOSER_ATTESTATION"):
        module().issue_receipt(
            changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
        )


def test_review_identity_is_derived_from_registry():
    changeset, review, runs, context, _ = setup_case()
    review["reviewer"] = {"model_family": "invented"}
    runs["R1"]["output_sha256"] = module().canonical_sha256(review)
    with pytest.raises(ValueError, match="REVIEW_SCHEMA"):
        module().issue_receipt(
            changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
        )


def test_live_run_cannot_be_enabled_by_model():
    changeset, review, runs, context, _ = setup_case()
    runs["R1"]["execution_mode"] = "live"
    with pytest.raises(ValueError, match="LIVE_DISABLED"):
        module().issue_receipt(
            changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
        )


def test_citation_offsets_use_codepoints():
    changeset, _, _, context, _ = setup_case()
    citation = changeset["claims"][0]["citations"][0]
    citation.update(start=1, end=2, quote="😀")
    rehash(changeset)
    assert module().validate_changeset(changeset, evidence_resolver=context["evidence_resolver"])
    citation.update(end=3)
    rehash(changeset)
    with pytest.raises(ValueError, match="CITATION_QUOTE"):
        module().validate_changeset(changeset, evidence_resolver=context["evidence_resolver"])


def test_claim_must_describe_the_actual_after_value():
    changeset, _, _, context, _ = setup_case()
    changeset["operations"][0]["after"]["summary"] = "甲承诺收益。"
    rehash(changeset)
    with pytest.raises(ValueError, match="CLAIM_VALUE"):
        module().validate_changeset(changeset, evidence_resolver=context["evidence_resolver"])


def test_structured_claim_must_bind_canonical_business_value():
    changeset, _, _, context, _ = setup_case()
    changeset["operations"][0]["after"]["summary"] = {"text": "甲承诺收益。"}
    rehash(changeset)
    with pytest.raises(ValueError, match="CLAIM_VALUE"):
        module().validate_changeset(changeset, evidence_resolver=context["evidence_resolver"])


def test_invalid_run_hash_has_stable_failure_code():
    changeset, review, runs, context, _ = setup_case()
    runs["R1"]["budget_ledger_sha256"] = 123
    with pytest.raises(ValueError, match="RUN_HASH"):
        module().issue_receipt(
            changeset, review, reviewer_run_id="R1", key_id="test", key=b"s" * 32, **context
        )
