"""Audit authored query examples without reading test answers or changing annotations."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/eval/query_boundaries_20261024"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8-sig").splitlines() if line]


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def grams(text: str) -> set[str]:
    value = normalize(text)
    return {value[i : i + 3] for i in range(len(value) - 2)}


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    splits = {name: read_rows(DATA / f"{name}.jsonl") for name in ("train", "validation")}
    groups = {name: {r["group_id"] for r in rows} for name, rows in splits.items()}
    assert not groups["train"] & groups["validation"]
    all_rows = [row for rows in splits.values() for row in rows]
    assert len({r["query_id"] for r in all_rows}) == len(all_rows)
    for name, rows in splits.items():
        assert (
            hashlib.sha256((DATA / f"{name}.jsonl").read_bytes()).hexdigest()
            == (manifest["files"][f"{name}.jsonl"])
        )
        for row in rows:
            assert row["split"] == name
            assert hashlib.sha256(row["text"].encode()).hexdigest() == row["source_text_sha256"]
            assert (row["decision_state"] == "risk") == bool(row["risk_labels"])
            if row["decision_state"] == "insufficient_information":
                assert row["missing_facts"]
            assert row["human_expert_gold"] is False
            for evidence in row["evidence"]:
                assert row["text"][evidence["start"] : evidence["end"]] == evidence["quote"]
        for group in groups[name]:
            assert {r["decision_state"] for r in rows if r["group_id"] == group} == {
                "risk",
                "no_clear_risk",
                "insufficient_information",
            }
    comparisons = {}
    for name, reference in [
        ("new_validation", splits["validation"]),
        ("original_development", read_rows(ROOT / "data/eval/competition500_v1/development.jsonl")),
        ("original_test_inputs", read_rows(ROOT / "data/eval/competition500_v1/test.jsonl")),
    ]:
        other = {normalize(r["text"]) for r in reference}
        exact = sum(normalize(r["text"]) in other for r in splits["train"])
        assert exact == 0
        reference_grams = [grams(r["text"]) for r in reference]
        maximum = max(
            len(a & b) / max(1, len(a | b))
            for a in (grams(r["text"]) for r in splits["train"])
            for b in reference_grams
        )
        comparisons[name] = {
            "normalized_exact_duplicates": exact,
            "max_character_trigram_jaccard": round(maximum, 6),
        }
    report = {
        "checks_passed": True,
        "annotation_status": "Codex synthetic self-review; not expert gold",
        "group_overlap": 0,
        "splits": {
            name: {
                "queries": len(rows),
                "groups": len(groups[name]),
                "states": dict(Counter(r["decision_state"] for r in rows)),
                "label_count": len({label for r in rows for label in r["risk_labels"]}),
            }
            for name, rows in splits.items()
        },
        "train_overlap_audit": comparisons,
        "test_answers_read_by_audit": False,
    }
    destination = ROOT / "reports/query_optimization_20261024/dataset_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
