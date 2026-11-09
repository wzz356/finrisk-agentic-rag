"""Build local application views from the pinned V2.1 assets without altering the source bundle."""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offline.knowledge_projection import project_metadata  # noqa: E402

DEST = ROOT / "data/processed/knowledge/finrisk_v2_1_20261018"
BUNDLE = DEST / "source_bundle"
NORMAL = BUNDLE / "versions/v2-20261012/normalized_final/data"
CLOSEOUT = BUNDLE / "releases/2026-10-17/outputs/FinRisk_知识库V2.1轻量收尾_20261015"
HANDOFF = BUNDLE / "releases/2026-10-18/data/handoff_scope"
VERSION = "finrisk-v2.1-local-20261018"


def rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_dir():
        for part in sorted(path.glob("part-*.jsonl.gz")):
            yield from rows(part)
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, values: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def table(name: str, key: str) -> dict[str, dict[str, Any]]:
    base = {str(row[key]): row for row in rows(NORMAL / f"{name}.jsonl")}
    patch = list(rows(CLOSEOUT / f"patch_normalized/data/{name}.jsonl"))
    changed = {str(row["official_file_id"]) for row in patch}
    base = {k: v for k, v in base.items() if v.get("official_file_id") not in changed}
    for row in patch:
        if str(row[key]) in base:
            raise ValueError("Duplicate canonical ID")
        base[str(row[key])] = row
    return base


def mechanism_ids(text: str) -> list[str]:
    patterns = {
        "M01": r"(?:指导|诱导|虚假|不实).{0,25}回访|回访.{0,30}(?:指导|诱导|虚假|不实)",
        "M02": r"冒.{0,15}回访|代接.{0,15}回访|回访.{0,15}(?:冒名|代答)",
        "M03": r"避债|避税|免税|逃避债务",
        "M04": r"合同外利益|合同约定以外|返佣|返还保费|回扣|赠送",
        "M05": r"保证收益|承诺收益|保证.{0,10}(?:分红|收益)|(?:分红|收益).{0,10}(?:保证|承诺)",
        "M06": r"存款|储蓄|混淆.{0,8}产品性质",
        "M07": r"退保|现金价值",
        "M08": r"免责|等待期|夸大.{0,12}保障|隐瞒.{0,12}保障",
        "M09": r"停售|停售|虚构.{0,8}限量",
        "M10": r"无资质|无资格|未取得.{0,10}(?:资格|许可)|未.{0,5}合法资格",
        "M11": r"代签|代.{0,8}签名|代.{0,8}确认|冒.{0,8}签",
        "M12": r"拒绝.{0,15}(?:电话|外呼)|(?:电话|外呼).{0,15}拒绝|禁呼",
    }
    return [key for key, pattern in patterns.items() if re.search(pattern, text)]


def main() -> None:
    receipt = json.loads((BUNDLE / "local_download_manifest.json").read_text(encoding="utf-8"))
    if receipt["revision"] != "55402453118f4dee9dd06f6b8ec39d7b21ae1e08":
        raise ValueError("Unreviewed knowledge revision")
    for file in receipt["files"]:
        path = BUNDLE / file["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != file["sha256"]:
            raise ValueError("Downloaded asset changed: " + file["path"])
    print("Loading canonical current tables", flush=True)
    documents = table("documents", "official_file_id")
    cases = table("cases", "case_id")
    behaviors = table("behaviors", "behavior_id")
    evidence = table("evidence", "evidence_id")
    if (len(documents), len(cases), len(behaviors)) != (9639, 15620, 25186):
        raise ValueError("Unexpected canonical counts")
    current_shas = {
        r["file_id"]: r["effective_draft_json_sha256"]
        for r in rows(CLOSEOUT / "effective_candidates.jsonl.gz")
    }
    if len(current_shas) != len(documents):
        raise ValueError("Current candidate inventory mismatch")
    for fid, doc in documents.items():
        if doc["effective_draft_json_sha256"] != current_shas[fid]:
            raise ValueError("Stale canonical current draft: " + fid)
        if hashlib.sha256(doc["full_text"].encode()).hexdigest() != doc["source_sha256"]:
            raise ValueError("Source hash mismatch: " + fid)

    out = DEST / "data"
    out.mkdir(parents=True, exist_ok=True)
    database = DEST / "evidence_registry.building.sqlite3"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE sources (file_id TEXT PRIMARY KEY, text TEXT NOT NULL, sha256 TEXT NOT NULL);
        CREATE TABLE evidence (id TEXT PRIMARY KEY, file_id TEXT NOT NULL, data TEXT NOT NULL);
        CREATE TABLE units (id TEXT PRIMARY KEY, file_id TEXT NOT NULL, data TEXT NOT NULL);
        CREATE TABLE behaviors (id TEXT PRIMARY KEY, case_id TEXT NOT NULL, data TEXT NOT NULL);
        CREATE INDEX behavior_case ON behaviors(case_id);
    """)
    connection.executemany(
        "INSERT INTO sources VALUES (?,?,?)",
        [(fid, d["full_text"], d["source_sha256"]) for fid, d in documents.items()],
    )
    literal_errors = 0
    for eid, item in evidence.items():
        fid = item["official_file_id"]
        source = documents[fid]["full_text"]
        start, end = item.get("start_char"), item.get("end_char")
        valid = (
            isinstance(start, int)
            and isinstance(end, int)
            and source[start:end] == item.get("verbatim")
        )
        if not valid:
            literal_errors += 1
        item["local_literal_verified"] = valid
        connection.execute(
            "INSERT INTO evidence VALUES (?,?,?)", (eid, fid, json.dumps(item, ensure_ascii=False))
        )
    if literal_errors:
        raise ValueError(f"Canonical evidence intervals failed: {literal_errors}")
    by_case: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    id_map: dict[str, str] = {}
    for cid, case in cases.items():
        id_map[f"{case['official_file_id']}:{case['proposal_local_key']}"] = cid
    for bid, behavior in behaviors.items():
        by_case[behavior["case_id"]].append(behavior)
        id_map[f"{behavior['official_file_id']}:{behavior['proposal_local_key']}"] = bid
        connection.execute(
            "INSERT INTO behaviors VALUES (?,?,?)",
            (bid, behavior["case_id"], json.dumps(behavior, ensure_ascii=False)),
        )
    for row in rows(HANDOFF / "projection_patch_compatibility.jsonl"):
        for mapping in row["legacy_to_stable_id_map"]:
            id_map[mapping["patch_local_id"]] = mapping["stable_V2_id"]

    records = []
    for cid, case in cases.items():
        bs = by_case[cid]
        tags = list(dict.fromkeys(tag for b in bs for tag in b.get("risk_tags", [])))
        parties = case.get("parties", [])
        party = parties[0].get("name") if parties else None
        date = case.get("penalty_date")
        if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            date = None
        fid = case["official_file_id"]
        records.append(
            {
                "schema_version": "case_kb_v1",
                "case_id": cid,
                "file_id": fid,
                "case_title": f"{party or fid} {case.get('penalty_doc_no') or ''}".strip(),
                "party_name": party,
                "penalty_doc_no": case.get("penalty_doc_no"),
                "violation_behavior": "；".join(
                    dict.fromkeys(
                        b.get("regulatory_finding")
                        or b.get("behavior_summary")
                        or b.get("action")
                        or ""
                        for b in bs
                    )
                )
                or case.get("case_summary")
                or "原文事实待核",
                "case_summary": case.get("case_summary"),
                "penalty_content": "；".join(
                    str(p.get("type") or "") for p in case.get("penalties", [])
                ),
                "penalty_date": date,
                "regulator": case.get("regulator"),
                "institution_type": None,
                "source_url": None,
                "source_file": fid,
                "source_sha256": documents[fid]["source_sha256"],
                "risk_tags": tags,
                "risk_domains": [],
                "risk_tag_basis": [v for b in bs for v in b.get("risk_tag_basis", [])],
                "case_origin": VERSION,
                "retrieval_status": "active",
                "evidence_status": "limited",
                "review_status": "model_candidate_pending_human",
                "data_version": VERSION,
                "quality_flags": [
                    "candidate",
                    "exported_text_verified_only",
                    "human_approval_not_granted",
                ],
                "legal_basis_refs": case.get("legal_refs", []),
                "related_parties": parties,
                "behavior_ids": case.get("behavior_ids", []),
                "candidate_sha256": current_shas[fid],
                **project_metadata(case, documents[fid]),
            }
        )
    write_rows(out / "cases.jsonl", records)
    records_by_id = {r["case_id"]: r for r in records}
    bindings = {r["unit_id"]: r for r in rows(HANDOFF / "projection_candidate_rebinds.jsonl")}
    used_bindings = set()
    seen_cases = set()
    indexed = 0
    all_units = 0
    chunk_path = out / "case_chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8", newline="\n") as stream:
        for corpus in ["structured", "raw"]:
            projection = BUNDLE / f"versions/v2-20261012/indexes/all9639_final/{corpus}.jsonl"
            for unit in rows(projection):
                all_units += 1
                fid = unit["file_id"]
                binding = bindings.get(unit["unit_id"])
                if binding:
                    if unit["candidate_sha256"] != binding["expected_baseline_draft_sha256"]:
                        raise ValueError("Unexpected rebind baseline")
                    unit["candidate_sha256"] = binding["current_effective_draft_sha256"]
                    used_bindings.add(unit["unit_id"])
                if unit["candidate_sha256"] != current_shas[fid]:
                    raise ValueError("Stale projection: " + unit["unit_id"])
                spans = unit.get("source_spans", [])
                source = documents[fid]["full_text"]
                if not spans or not all(
                    source[s["start"] : s["end"]] == unit["source_text"] for s in spans
                ):
                    raise ValueError("Non-literal projection: " + unit["unit_id"])
                raw_ids = [*unit.get("case_ids", []), unit.get("case_id")]
                case_ids = list(dict.fromkeys(id_map.get(str(x), str(x)) for x in raw_ids if x))
                case_ids = [x for x in case_ids if x in cases]
                for bid in [*unit.get("behavior_ids", []), unit.get("behavior_id")]:
                    behavior = behaviors.get(id_map.get(str(bid), str(bid)))
                    if behavior and behavior["case_id"] not in case_ids:
                        case_ids.append(behavior["case_id"])
                unit["application_case_ids"] = case_ids
                connection.execute(
                    "INSERT INTO units VALUES (?,?,?)",
                    (unit["unit_id"], fid, json.dumps(unit, ensure_ascii=False)),
                )
                # Source-only headings and unattributed context stay in the registry,
                # not attributed to arbitrary penalized parties in the case search view.
                if corpus != "structured" or not case_ids:
                    continue
                tags = list(
                    dict.fromkeys(t for cid in case_ids for t in records_by_id[cid]["risk_tags"])
                )
                chunk = {
                    "chunk_id": unit["unit_id"],
                    "case_id": case_ids[0],
                    "case_ids": case_ids,
                    "file_id": fid,
                    "chunk_type": "evidence",
                    "chunk_text": unit["source_text"],
                    "is_verbatim": True,
                    "supports_risk_tags": tags,
                    "retrieval_enabled": True,
                    "retrieval_weight": 1.0,
                    "source_span": {
                        "spans": spans,
                        "source_sha256": unit["source_sha256"],
                        "unit_id": unit["unit_id"],
                        "scope_review": "base_candidate_only",
                    },
                }
                stream.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
                indexed += 1
                seen_cases.update(case_ids)
        if used_bindings != set(bindings):
            raise ValueError("Not all 33 projection rebindings were applied")
        # Explicit original evidence is a safe fallback for cases with no linked retrieval unit.
        for cid in sorted(set(cases) - seen_cases):
            for b in by_case[cid]:
                for eid in b.get("fact_evidence_ids", []):
                    e = evidence.get(eid)
                    if not e:
                        continue
                    chunk = {
                        "chunk_id": f"application:{cid}:{eid}",
                        "case_id": cid,
                        "file_id": e["official_file_id"],
                        "chunk_type": "evidence",
                        "chunk_text": e["verbatim"],
                        "is_verbatim": True,
                        "supports_risk_tags": records_by_id[cid]["risk_tags"],
                        "retrieval_enabled": True,
                        "retrieval_weight": 1.0,
                        "source_span": {
                            "evidence_id": eid,
                            "start": e["start_char"],
                            "end": e["end_char"],
                            "source_sha256": e["source_sha256"],
                            "scope_review": "base_candidate_only",
                        },
                    }
                    stream.write(
                        json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    indexed += 1
                    seen_cases.add(cid)

    lexicon = json.loads(
        (BUNDLE / "releases/2026-10-18/data/auxiliary_lexicon.json").read_text(encoding="utf-8")
    )
    lex_by_id = {r["mechanism_id"]: r for r in lexicon}
    expansions = []
    for bid, behavior in behaviors.items():
        grounding = [eid for eid in behavior.get("fact_evidence_ids", []) if eid in evidence]
        if not grounding:
            continue
        text = " ".join(
            str(behavior.get(k) or "") for k in ["action", "object", "behavior_summary"]
        )
        mechanisms = mechanism_ids(text)
        for mid in mechanisms:
            for number, expression in enumerate(lex_by_id[mid]["alias_expressions"][:2]):
                expansions.append(
                    {
                        "expansion_id": f"APP-EXP:{bid}:{mid}:{number}",
                        "expansion_text": expression,
                        "expansion_type": "grounded_mechanism_alias",
                        "case_links": [
                            {
                                "case_id": behavior["case_id"],
                                "behavior_id": bid,
                                "grounding_chunk_ids": grounding,
                            }
                        ],
                        "risk_tags": behavior.get("risk_tags", []),
                        "retrieval_enabled": True,
                        "retrieval_weight": 0.55,
                        "text_origin": "static_generic_retrieval_aid",
                        "allowed_usage": "retrieval_only",
                        "not_case_fact": True,
                        "human_approved": False,
                        "data_version": VERSION,
                    }
                )
    write_rows(out / "case_expansions.jsonl", expansions)
    connection.commit()
    connection.close()
    database.replace(DEST / "evidence_registry.sqlite3")
    (DEST / "dictionary").mkdir(exist_ok=True)
    shutil.copy2(
        BUNDLE / "dictionary/risk_type_dictionary.csv", DEST / "dictionary/risk_type_dictionary.csv"
    )
    # The new corpus keeps the same 27-label contract as the local ingestion workflow.
    domain_mapping = ROOT / "configs/risk_domain_mapping.json"
    if not domain_mapping.is_file():
        raise FileNotFoundError("Local ingestion domain mapping is required")
    shutil.copy2(domain_mapping, DEST / "dictionary/risk_domain_mapping.json")
    versions = json.loads((HANDOFF / "version_lock.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "case_kb_v1",
        "data_version": VERSION,
        "repository_revision": receipt["revision"],
        "knowledge_versions": versions,
        "case_count": len(cases),
        "document_count": len(documents),
        "behavior_count": len(behaviors),
        "evidence_count": len(evidence),
        "retrieval_unit_count": all_units,
        "indexed_case_units": indexed,
        "cases_with_retrieval_units": len(seen_cases),
        "expansion_count": len(expansions),
        "applied_projection_rebindings": len(used_bindings),
        "literal_interval_errors": literal_errors,
        "human_approved": False,
        "scope_overlay_reviewed_units": 19,
        "unattributed_source_units_not_assigned_to_parties": True,
        "evaluation_data_indexed": False,
        "uploaded": False,
    }
    (DEST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
