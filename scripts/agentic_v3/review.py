"""Convert the ten genuinely approved proposals without changing source bytes."""

import argparse
import hashlib
import json
from pathlib import Path

PROPOSAL_SHA = "33569faa3eb107f48a1415c07fe32907bd5c40a3c03c46e969cfb37451ecd4d8"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def obj_sha(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_review_bundle(proposal_path, approval_path, *, expected_sha=PROPOSAL_SHA):
    proposal_path, approval_path = Path(proposal_path).resolve(), Path(approval_path).resolve()
    proposal = json.loads(proposal_path.read_text(encoding="utf-8-sig"))
    approval = json.loads(approval_path.read_text(encoding="utf-8-sig"))
    ids = [f"N{i:02}" for i in range(1, 11)]
    samples = proposal["samples"]
    if (
        file_sha(proposal_path) != expected_sha
        or approval.get("authority") != "human_user_in_current_conversation"
        or approval.get("decision") != "accept_all_10_proposals"
        or approval.get("statement") != "认可N01–N10全部参照"
        or approval.get("source_proposal_sha256") != expected_sha
        or approval.get("review_ids") != ids
        or [s["review_id"] for s in samples] != ids
    ):
        raise ValueError("INDEPENDENT_APPROVAL_BINDING_MISMATCH")
    queries, references, mapping = [], [], []
    for s in samples:
        sample_id = "LOOP-INDEPENDENT-20261018-" + s["review_id"]
        q = {"sample_id": sample_id, "text": s["text"], "user_context": s["user_context"]}
        reference = s["proposed_reference"]
        queries.append(q)
        references.append(
            {
                "sample_id": sample_id,
                "review_id": s["review_id"],
                "family_id": s["family_id"],
                "difficulty": s["difficulty"],
                "input_sha256": obj_sha({k: q[k] for k in ("text", "user_context")}),
                "human_approved": True,
                "reference": reference,
            }
        )
        mapping.append({"review_id": s["review_id"], "sample_id": sample_id})
    result = {
        "schema_version": "finrisk-loop-independent-review/1.0",
        "authority": approval["authority"],
        "decision": approval["decision"],
        "source_proposal": str(proposal_path),
        "source_proposal_sha256": file_sha(proposal_path),
        "source_approval": str(approval_path),
        "source_approval_sha256": file_sha(approval_path),
        "sample_id_mapping": mapping,
        "queries": queries,
        "references": references,
        "development_gate_required": True,
    }
    return result | {"bundle_sha256": obj_sha(result)}


def verify_review_bundle(bundle):
    if bundle != make_review_bundle(bundle["source_proposal"], bundle["source_approval"]):
        raise ValueError("INDEPENDENT_REVIEW_BUNDLE_CHANGED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_review_bundle(args.proposal, args.approval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        stream.write(canonical(result) + "\n")
    print(
        json.dumps({"output": str(args.output.resolve()), "bundle_sha256": result["bundle_sha256"]})
    )


if __name__ == "__main__":
    main()
