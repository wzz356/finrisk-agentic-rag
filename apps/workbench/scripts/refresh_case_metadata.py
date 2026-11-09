"""Refresh runtime metadata only; original evidence and vector fingerprints stay unchanged."""

import hashlib
import json
import runpy

from build_competition_dataset import ROOT, read_rows, save

from offline.knowledge_projection import project_metadata


def main():
    m = runpy.run_path(str(ROOT / "scripts/import_competition_kb.py"))
    base = m["DEST"]
    cases = m["table"]("cases", "case_id")
    docs = m["table"]("documents", "official_file_id")
    path = base / "data/cases.jsonl"
    records = read_rows(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for record in records:
        case = cases.get(record["case_id"])
        if case:
            record.update(project_metadata(case, docs[case["official_file_id"]]))
    temp = path.with_suffix(".jsonl.tmp")
    temp.write_text(
        "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records),
        "utf-8",
    )
    temp.replace(path)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["metadata_version"] = "case-fields-20261019"
    dates = [r["penalty_date"] for r in records if r.get("penalty_date")]
    manifest["latest_penalty_date"] = max(dates) if dates else None
    save(manifest_path, manifest)
    save(
        ROOT / "outputs/competition_upgrade_20261019/metadata.json",
        {
            "cases": len(records),
            "before_sha256": before,
            "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "nonempty": {
                k: sum(bool(r.get(k)) for r in records)
                for k in [
                    "institution_type",
                    "penalty_date",
                    "publish_date",
                    "penalty_amounts",
                    "field_sources",
                    "entity_relations",
                ]
            },
        },
    )
    print("Metadata refreshed; original chunks and Qwen vectors unchanged.")


if __name__ == "__main__":
    main()
