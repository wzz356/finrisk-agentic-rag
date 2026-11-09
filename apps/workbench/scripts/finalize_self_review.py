"""Publish Codex's direct review and expand a source-anchored mechanism matrix offline.

This utility does not import or call any model client. Review decisions are authored
in the review ledger; this file only validates, versions and materializes them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from build_competition_dataset import DATA, ROOT, read_rows, save, write_rows
from self_review_dataset import LABELS, OUT

VERSION = "competition500-codex-reviewed-v2-20261019"
PROTOCOL = "direct-source-review-mechanisms-v3"

# These aliases reconcile spelling and the same concrete act; they are scorer
# metadata only and are never imported by production inference or retrieval.
ALIASES = {
    "agent_unauthorized_subdelegation": "agent_redelegation",
    "agent_registration_timing": "agent_unregistered",
    "branch_establishment_unreported": "branch_unreported",
    "branch_address_change": "branch_address_unapproved",
    "callback_guide_false": "callback_answers_coached",
    "callback_confirmation_false": "callback_answers_forged",
    "callback_not_completed": "callback_noncompliance",
    "client_health_false": "health_disclosure_false",
    "client_confirmation_false": "confirmation_forged",
    "client_risk_confirmation_forged": "confirmation_forged",
    "callback_answers_forged": "confirmation_forged",
    "commission_expense_misclassification": "expense_misclassification",
    "criminal_case_unreported": "crime_reporting_missing",
    "guarantee_deposit_missing": "professional_insurance_missing",
    "yield_as_guaranteed": "guaranteed_return",
    "phone_yield_promise": "guaranteed_return",
    "return_promise": "guaranteed_return",
    "license_not_displayed": "license_display_missing",
    "meeting_gift_purchase": "gift_purchase",
    "meeting_records_fabricated": "meeting_records_forged",
    "omit_exclusion": "omit_exclusions",
    "premium_misappropriation": "premium_embezzlement",
    "premium_personal_account": "personal_premium_account",
    "premium_records_false": "policy_copies_inconsistent",
    "related_party_records_false": "related_party_data_false",
    "qualification_exam_tampering": "agent_exam_forged",
    "risk_warning_copy": "warning_copy_forged",
    "sales_trace_fabrication": "phone_recording_forged",
    "tax_evasion_promo": "tax_debt_avoidance",
    "yield_data_fabricated": "yield_rate_fabricated",
}

# Carrier, vague category, missing evidence and normal procedures alone do not
# establish a strong relevant case. In particular, shared risk tags are not used.
WEAK = {
    "deception_unspecified",
    "sales_mislead_unspecified",
    "false_promo_unspecified",
    "phone_mislead_unspecified",
    "meeting_mislead_unspecified",
    "phone_mislead",
    "meeting_yield_mislead",
    "meeting_coverage_mislead",
    "meeting_false_promo",
    "training_unsafe_instruction",
    "training_yield_mislead",
    "meeting_rate_mislead",
    "agent_supervision",
    "internal_control_failure",
    "insurance_sales_violation_unspecified",
    "false_promo_scope",
    "online_false_promo",
    "coverage_or_yield_exaggeration",
}


def canonical(atoms):
    return {ALIASES.get(atom, atom) for atom in atoms}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    queries = {r["query_id"]: r for r in read_rows(DATA / "model_inputs.jsonl")}
    with ZipFile(OUT / "before_review.zip") as archive:
        old = {
            r["query_id"]: r
            for r in (
                json.loads(line)
                for line in archive.read(
                    str((DATA / "reference/labels.jsonl").relative_to(ROOT)).replace("\\", "/")
                )
                .decode("utf-8-sig")
                .splitlines()
            )
        }
        original_cases = [
            json.loads(line)
            for line in archive.read(
                str((DATA / "cases.jsonl").relative_to(ROOT)).replace("\\", "/")
            )
            .decode("utf-8-sig")
            .splitlines()
        ]
    reviewed = {r["id"]: r for r in read_rows(OUT / "query_reviews.jsonl")}
    case_review = read_rows(OUT / "case_reviews.jsonl")
    cases = original_cases
    assert len(reviewed) == 600 and set(reviewed) == set(queries)
    assert len(case_review) == 500 and {r["id"] for r in case_review} == {
        f"C{i:03}" for i in range(500)
    }
    changes = []
    labels = []
    for qid, row in queries.items():
        review = reviewed[qid]
        assert row["text"] == review["source_text"]
        assert review["state"] != "risk" or review["labels"]
        assert all(quote in row["text"] for quote in review["evidence"])
        assert review["state"] != "risk" or review["evidence"]
        label = {
            "query_id": qid,
            "scene": review["scene"],
            "decision_state": review["state"],
            "risk_labels": sorted(set(review["labels"]), key=LABELS.index),
            "evidence": review["evidence"],
            "reason": review["note"],
            "nonliteral_evidence_count": 0,
            "reviewer": review["reviewer"],
            "annotation_status": "codex_direct_source_review_not_human_expert_gold",
        }
        fields = [
            key
            for key in ("scene", "decision_state", "risk_labels")
            if (
                set(old[qid][key]) != set(label[key])
                if key == "risk_labels"
                else old[qid][key] != label[key]
            )
        ]
        if fields:
            changes.append(
                {
                    "query_id": qid,
                    "changed_fields": fields,
                    "before": {key: old[qid][key] for key in fields},
                    "after": {key: label[key] for key in fields},
                    "reason": review["note"],
                }
            )
        labels.append(label)
    # Keep text and case IDs stable. A chronology typo in DEV397 is documented in
    # its review note, without changing the inference inputs for a metric run.
    write_rows(DATA / "reference/labels.jsonl", labels)
    by_case = {r["id"]: r for r in case_review}
    case_changes = []
    for i, case in enumerate(cases):
        review = by_case[f"C{i:03}"]
        corrected = review.get("corrected_labels")
        match = re.search(r"tags=([\d,]+)", review["note"])
        if corrected is None and match:
            corrected = [LABELS[int(n)] for n in match[1].split(",")]
        before = {"risk_tags": case["risk_tags"], "fact_text": case["fact_text"]}
        if corrected is not None:
            case["risk_tags"] = corrected
        if review.get("context_quotes"):
            assert review["context_source_sha256"] == case["source_sha256"]
            case["fact_text"] = "；".join(review["context_quotes"])
        case["source_quotes"] = [
            quote.strip() for quote in case["fact_text"].split("；") if quote.strip()
        ]
        if before != {key: case[key] for key in before}:
            case_changes.append(
                {
                    "case_id": case["case_id"],
                    "catalog_id": review["id"],
                    "before": before,
                    "after": {key: case[key] for key in before},
                    "reason": review["note"],
                }
            )
        case["review_status"] = "codex_direct_source_review"
    write_rows(DATA / "cases.jsonl", cases)
    compact_queries = [
        {key: value for key, value in r.items() if key != "source_text"} for r in reviewed.values()
    ]
    write_rows(DATA / "reference/query_reviews.jsonl", compact_queries)
    write_rows(DATA / "reference/case_reviews.jsonl", case_review)
    save(
        DATA / "reference/review_changes.json",
        {"query_changes": changes, "case_changes": case_changes},
    )
    save(
        DATA / "reference/relevance_protocol.json",
        {
            "version": PROTOCOL,
            "reviewer": "Codex direct source review",
            "external_model_review_calls": 0,
            "query_reviews": 600,
            "case_source_reviews": 500,
            "pair_expansion": (
                "Deterministic expansion of individually reviewed action catalogs; "
                "not 300000 independent human judgments."
            ),
            "grade_0": "No shared reviewed action",
            "grade_1": "Shared carrier or broad category only",
            "grade_2": "At least one identical concrete act with literal query and source context",
            "grade_3": "At least two shared concrete acts, with source and query evidence",
            "normal_and_uncertain_queries": (
                "Cases may explain an explicitly mentioned prohibited or unverified act; "
                "relevance is distinct from current-query risk."
            ),
            "no_positive_queries": (
                "Retained and reported; never force the generating seed to be relevant."
            ),
            "aliases": ALIASES,
            "weak_atoms": sorted(WEAK),
        },
    )
    cases_by_index = [
        (case, canonical(by_case[f"C{i:03}"]["atoms"])) for i, case in enumerate(cases)
    ]
    qrel_report = {}
    for split in ("development", "test"):
        rows = []
        for sample in read_rows(DATA / f"{split}.jsonl"):
            qid = sample["query_id"]
            actions = canonical(reviewed[qid]["atoms"])
            grades, evidence = {}, {}
            for case, case_actions in cases_by_index:
                shared = actions & case_actions
                concrete = shared - WEAK
                grade = 3 if len(concrete) >= 2 else 2 if concrete else 1 if shared else 0
                grades[case["case_id"]] = grade
                if grade >= 2:
                    evidence[case["case_id"]] = {
                        "grade": grade,
                        "mechanisms": sorted(concrete),
                        "query_quote": sample["text"],
                        "case_quote": case["fact_text"],
                        "file_id": case["file_id"],
                        "source_sha256": case["source_sha256"],
                        "reason": "Same directly reviewed action: " + ", ".join(sorted(concrete)),
                    }
            rows.append(
                {
                    "query_id": qid,
                    "grades": grades,
                    "positive_evidence": evidence,
                    "protocol": PROTOCOL,
                }
            )
        write_rows(DATA / f"reference/qrels_{split}.jsonl", rows)
        counts = [len(r["positive_evidence"]) for r in rows]
        qrel_report[split] = {
            "queries": len(rows),
            "explicit_pairs": len(rows) * 500,
            "strong_pairs": sum(counts),
            "queries_without_strong_case": counts.count(0),
        }
    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    manifest.update(
        version=VERSION,
        label_judge="Codex direct source review",
        annotation_status="codex_direct_source_review_not_human_expert_gold",
        review_api_calls=0,
        relevance_protocol=PROTOCOL,
        qrels=qrel_report,
        state_distribution=dict(Counter(r["decision_state"] for r in labels)),
        label_distribution=dict(Counter(tag for r in labels for tag in r["risk_labels"])),
        review_summary={
            "query_rows_changed": len(changes),
            "case_rows_corrected": len(case_changes),
            "context_sources_rechecked": sum("context_quotes" in r for r in case_review),
            "test_predictions_used_for_gold": False,
        },
    )
    manifest["hashes"] = {
        str(p.relative_to(DATA)): digest(p)
        for p in DATA.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    save(DATA / "manifest.json", manifest)
    save(
        OUT / "finalization.json",
        {"version": VERSION, **manifest["review_summary"], "qrels": qrel_report},
    )
    print(
        json.dumps(
            {"version": VERSION, **manifest["review_summary"], "qrels": qrel_report},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
