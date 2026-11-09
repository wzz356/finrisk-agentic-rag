"""Build versioned application partitions from SHA-pinned data, retaining legacy sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_source_pdf_index import inspect_pdf, normalized_text_hash  # noqa: E402

from offline.insurance_relevance import evaluate_insurance_scope  # noqa: E402
from offline.knowledge_projection import normalize_date, project_metadata  # noqa: E402
from offline.source_identity import source_identity  # noqa: E402

REVISION = "52f721db0bd8bc19a5153eff05e015a36189d8f2"
VERSION = "finrisk-full-20261024"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows_file(path):
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def zip_rows(archive, name):
    with archive.open(name) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify_archive(path, expected):
    if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
        raise ValueError("Archive does not match the pinned release: " + path.name)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("bundle-manifest.json"))
        for name, metadata in manifest["files"].items():
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as stream:
                while data := stream.read(1024 * 1024):
                    digest.update(data)
                    size += len(data)
            if size != metadata["bytes"] or digest.hexdigest() != metadata["sha256"]:
                raise ValueError("Bundle member changed: " + name)
    print("Verified archive and members:", path.name, flush=True)


def build_partition(normalized, runtime, output, *, original, raw_by_sha):
    with zipfile.ZipFile(normalized) as norm, zipfile.ZipFile(runtime) as run:
        documents = {r["official_file_id"]: r for r in zip_rows(norm, "data/documents.jsonl")}
        cases = list(zip_rows(norm, "data/cases.jsonl"))
        behaviors = list(zip_rows(norm, "data/behaviors.jsonl"))
        by_case = defaultdict(list)
        local_ids = {}
        for case in cases:
            local_ids[f"{case['official_file_id']}:{case['proposal_local_key']}"] = case["case_id"]
        for behavior in behaviors:
            by_case[behavior["case_id"]].append(behavior)
        data = output / "data" if original else output
        data.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(output / "evidence_registry.sqlite3")
        db.executescript("""
            CREATE TABLE sources(file_id TEXT PRIMARY KEY,text TEXT NOT NULL,sha256 TEXT NOT NULL);
            CREATE TABLE evidence(id TEXT PRIMARY KEY,file_id TEXT NOT NULL,data TEXT NOT NULL);
            CREATE TABLE units(id TEXT PRIMARY KEY,file_id TEXT NOT NULL,data TEXT NOT NULL);
            CREATE TABLE behaviors(id TEXT PRIMARY KEY,case_id TEXT NOT NULL,data TEXT NOT NULL);
            CREATE INDEX behavior_case ON behaviors(case_id);
        """)
        source_records = {}
        for fid, doc in documents.items():
            text = doc["full_text"]
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest != doc["text_sha256"]:
                raise ValueError("Source SHA mismatch: " + fid)
            db.execute("INSERT INTO sources VALUES(?,?,?)", (fid, text, digest))
            scope = evaluate_insurance_scope(text)
            kind = doc.get("document_kind") or "penalty"
            relevance = doc.get("insurance_relevance")
            usable = original or (relevance is True and kind == "penalty")
            raw_file = None if original else raw_by_sha.get(digest)
            if not original and raw_file is None:
                raise ValueError(
                    "Released web document does not bind to an attachment original: " + fid
                )
            source_records[fid] = {
                "case_id": fid + ":SOURCE",
                "file_id": fid,
                "source_file": fid,
                "source_sha256": digest,
                "source_content_sha256": hashlib.sha256("".join(text.split()).encode()).hexdigest(),
                "case_title": doc.get("title") or doc.get("original_filename") or fid,
                "violation_behavior": "完整来源已归档；具体主体、行为和责任请核对原文",
                "case_origin": "source_document_only" if original else "official_crawl",
                "source_group": "original" if original else "official_crawl",
                "risk_tags": [],
                "risk_domains": [],
                "retrieval_status": "active" if usable else "inactive",
                "retrieval_disposition": "active"
                if usable
                else "archive_only"
                if kind != "penalty"
                else "scope_review",
                "document_kind": kind,
                "decision_status": doc.get("decision_status"),
                "insurance_relevance": relevance,
                "scope_assessment": scope,
                "review_status": "model_candidate_pending_human",
                "human_approved": False,
                "evidence_status": "source_only_attribution_unknown",
                "data_version": VERSION,
                "raw_source_file": raw_file,
                "source_url": doc.get("source_url"),
                "publish_date": normalize_date(str(doc.get("collection_publish_date") or "")[:10]),
            }
        records = []
        for case in cases:
            fid = case["official_file_id"]
            bs = by_case[case["case_id"]]
            base = source_records[fid]
            metadata = project_metadata(case, documents[fid])
            parties = case.get("parties") or []
            party = parties[0].get("name") if parties else None
            penalties = list(
                dict.fromkeys(
                    str(p.get("quote") or p.get("amount_quote") or p.get("type") or "")
                    for p in case.get("penalties", [])
                )
            )
            records.append(
                {
                    **base,
                    **metadata,
                    "case_id": case["case_id"],
                    "case_origin": VERSION if original else "official_crawl",
                    "case_title": f"{party or fid} {case.get('penalty_doc_no') or ''}".strip(),
                    "party_name": party,
                    "penalty_doc_no": case.get("penalty_doc_no"),
                    "regulator": case.get("regulator"),
                    "case_summary": case.get("case_summary"),
                    "violation_behavior": "；".join(
                        dict.fromkeys(
                            b.get("regulatory_finding")
                            or b.get("action")
                            or b.get("behavior_summary")
                            or ""
                            for b in bs
                        )
                    )
                    or case.get("case_summary")
                    or "原文事实待核",
                    "risk_tags": list(dict.fromkeys(t for b in bs for t in b.get("risk_tags", []))),
                    "risk_tag_basis": [v for b in bs for v in b.get("risk_tag_basis", [])],
                    "penalty_content": "；".join(penalties),
                    "is_insurance_related": base["insurance_relevance"],
                    "behavior_ids": case.get("behavior_ids", []),
                    "quality_flags": [
                        "candidate",
                        "human_approval_not_granted",
                        "literal_evidence_checked",
                    ],
                }
            )
        write_rows(data / "cases.jsonl", records)
        write_rows(data / "source_records.jsonl", source_records.values())
        write_rows(data / "case_expansions.jsonl", [])
        evidence_count = 0
        for evidence in zip_rows(norm, "data/evidence.jsonl"):
            fid = evidence["official_file_id"]
            text = documents[fid]["full_text"]
            start, end = evidence["start_char"], evidence["end_char"]
            if not 0 <= start < end <= len(text) or text[start:end] != evidence["verbatim"]:
                raise ValueError("Evidence interval mismatch: " + evidence["evidence_id"])
            evidence["source_sha256"] = documents[fid]["text_sha256"]
            db.execute(
                "INSERT INTO evidence VALUES(?,?,?)",
                (evidence["evidence_id"], fid, json.dumps(evidence, ensure_ascii=False)),
            )
            evidence_count += 1
        for behavior in behaviors:
            db.execute(
                "INSERT INTO behaviors VALUES(?,?,?)",
                (
                    behavior["behavior_id"],
                    behavior["case_id"],
                    json.dumps(behavior, ensure_ascii=False),
                ),
            )
        chunks = []
        unit_count = 0
        valid_ids = {c["case_id"] for c in cases}
        for unit in zip_rows(run, "structured.jsonl"):
            fid = unit["file_id"]
            doc = documents[fid]
            if unit["source_sha256"] != doc["text_sha256"]:
                raise ValueError("Projection source SHA mismatch")
            spans = unit["source_spans"]
            if not spans or any(
                not 0 <= s["start"] < s["end"] <= len(doc["full_text"])
                or doc["full_text"][s["start"] : s["end"]] != unit["source_text"]
                for s in spans
            ):
                raise ValueError("Projection quote mismatch: " + unit["unit_id"])
            for span in unit.get("context_spans", []):
                if not 0 <= span["start"] < span["end"] <= len(doc["full_text"]):
                    raise ValueError("Projection context out of range")
            linked = list(dict.fromkeys(local_ids.get(c, c) for c in unit.get("case_ids", [])))
            if any(c not in valid_ids for c in linked):
                raise ValueError("Unknown case association: " + unit["unit_id"])
            unit["application_case_ids"] = linked
            db.execute(
                "INSERT INTO units VALUES(?,?,?)",
                (unit["unit_id"], fid, json.dumps(unit, ensure_ascii=False)),
            )
            chunks.append(
                {
                    "chunk_id": unit["unit_id"],
                    "case_id": linked[0] if linked else fid + ":SOURCE",
                    "case_ids": linked or [fid + ":SOURCE"],
                    "file_id": fid,
                    "chunk_type": "evidence",
                    "chunk_text": unit["source_text"],
                    "is_verbatim": True,
                    "retrieval_enabled": True,
                    "retrieval_weight": 1.0 if linked else 0.65,
                    "source_span": {
                        "unit_id": unit["unit_id"],
                        "spans": spans,
                        "source_sha256": unit["source_sha256"],
                        "attribution": "registered_case"
                        if linked
                        else "source_only_no_actor_inference",
                    },
                }
            )
            unit_count += 1
        write_rows(data / "case_chunks.jsonl", chunks)
        if original:
            shutil.copy2(data / "case_chunks.jsonl", data / "case_chunks_full.jsonl")
        db.commit()
        db.close()
        (output / "dictionary").mkdir(exist_ok=True)
        (output / "dictionary/risk_type_dictionary.csv").write_bytes(
            run.read("risk_type_dictionary.csv")
        )
        manifest = {
            "data_version": VERSION,
            "repository_revision": REVISION,
            "document_count": len(documents),
            "case_count": len(cases),
            "behavior_count": len(behaviors),
            "evidence_count": evidence_count,
            "retrieval_unit_count": unit_count,
            "human_approved": False,
            "latest_penalty_date": max(
                (r["penalty_date"] for r in records if r.get("penalty_date")), default=None
            ),
            "insurance_relevance": dict(
                Counter(str(d.get("insurance_relevance")) for d in documents.values())
            ),
            "document_kinds": dict(
                Counter(d.get("document_kind", "penalty") for d in documents.values())
            ),
        }
        json_write(output / "manifest.json", manifest)
        print("Built", "original" if original else "official", manifest, flush=True)
        return documents, records, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--web-archive", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/knowledge/finrisk_full_20261024"
    )
    parser.add_argument(
        "--legacy-root", type=Path, default=ROOT / "data/processed/knowledge/finrisk_v2_1_20261018"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / "configs/knowledge_full_20261024.json").read_text("utf-8"))
    for asset in manifest["assets"]:
        verify_archive(args.assets / asset["file"], asset)
    raw_by_sha = {}
    raw_root = ROOT / "data/raw/cases/official_crawl/imports/collection-857-20261024"
    with zipfile.ZipFile(args.web_archive) as archive:
        for entry in zip_rows(archive, "index.jsonl"):
            member = entry["relative_path"]
            path = (raw_root / member).resolve()
            if not path.is_relative_to(raw_root.resolve()):
                raise ValueError("Attachment path escapes import root")
            raw = archive.read(member)
            if hashlib.sha256(raw).hexdigest() != entry["raw_sha256"]:
                raise ValueError("Attachment original SHA mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("Import would overwrite different original")
            path.write_bytes(raw)
            raw_by_sha[entry["text_sha256"]] = path.relative_to(ROOT).as_posix()
    if len(raw_by_sha) != 857:
        raise ValueError("Expected 857 attachment originals")
    all_docs = {}
    all_records = []
    parts = {}
    for name in ("original_dataset", "official_crawl"):
        target = output if name == "original_dataset" else output / "partitions/official_crawl"
        target.mkdir(parents=True, exist_ok=True)
        docs, records, part_manifest = build_partition(
            args.assets / f"finrisk-{name}-normalized-20261024.zip",
            args.assets / f"finrisk-{name}-runtime-20261024.zip",
            target,
            original=name == "original_dataset",
            raw_by_sha=raw_by_sha,
        )
        all_docs.update(docs)
        all_records.extend(records)
        parts[name] = part_manifest
    if (parts["original_dataset"]["document_count"], parts["official_crawl"]["document_count"]) != (
        10000,
        850,
    ):
        raise ValueError("Unexpected release source counts")
    audit = args.assets / "finrisk-source-audit-20261024.zip"
    with zipfile.ZipFile(audit) as archive:
        reviewed_aliases = list(zip_rows(archive, "official_crawl/source_aliases.jsonl"))
    by_identity = {source_identity(d["full_text"]): fid for fid, d in all_docs.items()}
    by_identity.update(
        {
            source_identity(
                Path(ROOT / raw_by_sha[a["source_text_sha256"]]).read_bytes().decode("utf-8")
            ): a["canonical_file_id"]
            for a in reviewed_aliases
        }
    )
    by_case_fid = defaultdict(list)
    for record in all_records:
        by_case_fid[record["file_id"]].append(record)
    existing_root = ROOT / "data/processed/case_sources/official_crawl"
    old_cases = list(rows_file(existing_root / "cases.jsonl"))
    old_documents = {}
    database = existing_root / "evidence_registry.sqlite3"
    if database.is_file():
        with sqlite3.connect(database) as db:
            old_documents = dict(db.execute("SELECT file_id,text FROM sources"))
    suppressed = {}
    case_aliases = {}
    for fid, text in old_documents.items():
        canonical = by_identity.get(source_identity(text))
        if canonical:
            suppressed[fid] = canonical
            for old in (c for c in old_cases if c["file_id"] == fid):
                matches = [
                    r
                    for r in by_case_fid[canonical]
                    if r.get("party_name") == old.get("party_name")
                ]
                case_aliases[old["case_id"]] = (
                    matches[0]["case_id"] if len(matches) == 1 else canonical + ":SOURCE"
                )
    alias_data = {
        "suppressed_file_ids": sorted(suppressed),
        "file_ids": suppressed,
        "case_ids": case_aliases,
        "reviewed_web_to_original_aliases": reviewed_aliases,
        "original_files_preserved": True,
    }
    json_write(output / "source_aliases.json", alias_data)
    print(
        "Deduplicated old web sources:",
        len(suppressed),
        "retained:",
        len(old_documents) - len(suppressed),
        flush=True,
    )
    # Preserve the frozen legal/source overlays under their own five-component lock.
    for name in [
        "p0_release_20261018",
        "source_bundle/releases/2026-10-18/data",
        "source_bundle/releases/2026-10-17/outputs/FinRisk_法规库V1_20261017",
    ]:
        source = args.legacy_root / name
        if source.exists():
            shutil.copytree(source, output / name)
    shutil.copy2(
        ROOT / "configs/risk_domain_mapping.json", output / "dictionary/risk_domain_mapping.json"
    )
    print("Binding all raw PDFs by complete extracted text", flush=True)
    by_pdf_text = defaultdict(list)
    pdf_inventory = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(
            pool.map(inspect_pdf, sorted((ROOT / "raw_pdfs").glob("*.pdf")), chunksize=32), 1
        ):
            if result is None:
                raise ValueError("Unreadable raw PDF")
            path, text_hash, pdf_hash = result
            by_pdf_text[text_hash].append((path, pdf_hash))
            pdf_inventory.append(
                {
                    "path": Path(path).relative_to(ROOT).as_posix(),
                    "pdf_sha256": pdf_hash,
                    "text_sha256": text_hash,
                }
            )
            if i % 2000 == 0:
                print("Scanned", i, "PDFs", flush=True)
    bindings = {}
    for fid, doc in all_docs.items():
        if fid.startswith("WEB-"):
            continue
        matches = by_pdf_text.get(normalized_text_hash(doc["full_text"]), [])
        if not matches:
            raise ValueError("No complete-text PDF match: " + fid)
        # Keep an explicit mapping for every physical original, even identical PDFs.
        used = {v["path"] for v in bindings.values()}
        path, digest = next(
            ((p, h) for p, h in matches if Path(p).relative_to(ROOT).as_posix() not in used),
            matches[0],
        )
        bindings[fid] = {
            "path": Path(path).relative_to(ROOT).as_posix(),
            "pdf_sha256": digest,
            "source_sha256": doc["text_sha256"],
        }
    if len(bindings) != 10000 or len(pdf_inventory) != 10000:
        raise ValueError("Raw PDF coverage is incomplete")
    json_write(
        output / "source_pdf_index.json",
        {
            "version": 1,
            "match": "complete_text_nfkc_without_whitespace_and_site_footer",
            "files": bindings,
        },
    )
    json_write(output / "raw_pdf_inventory.json", pdf_inventory)
    current = json.loads((output / "manifest.json").read_text("utf-8"))
    legacy = json.loads((args.legacy_root / "manifest.json").read_text("utf-8"))
    versions = {
        **legacy["knowledge_versions"],
        "content": VERSION + ":" + sha(output / "data/cases.jsonl"),
        "projection": VERSION + ":" + sha(output / "data/case_chunks_full.jsonl"),
    }
    files = {
        p.relative_to(output).as_posix(): sha(p)
        for p in [
            output / "data/cases.jsonl",
            output / "data/case_chunks_full.jsonl",
            output / "evidence_registry.sqlite3",
            output / "partitions/official_crawl/cases.jsonl",
            output / "partitions/official_crawl/case_chunks.jsonl",
            output / "partitions/official_crawl/evidence_registry.sqlite3",
            output / "source_aliases.json",
            output / "source_pdf_index.json",
        ]
    }
    json_write(
        output / "release_lock.json",
        {"upstream_revision": REVISION, "knowledge_versions": versions, "files": files},
    )
    current.update(
        knowledge_versions=versions,
        base_knowledge_versions=legacy["knowledge_versions"],
        release_lock_sha256=sha(output / "release_lock.json"),
        upstream_revision=REVISION,
        full_projection={
            "units": current["retrieval_unit_count"],
            "documents": 10000,
            "sha256": sha(output / "data/case_chunks_full.jsonl"),
        },
        partitions=parts,
        raw_pdf_count=len(pdf_inventory),
    )
    json_write(output / "manifest.json", current)
    receipt = {
        "upstream_revision": REVISION,
        "attachment_sha256": sha(args.web_archive),
        "attachment_documents": 857,
        "legacy_web_documents": len(old_documents),
        "legacy_matches_new_web": sum(fid.startswith("WEB-") for fid in suppressed.values()),
        "legacy_matches_original": sum(not fid.startswith("WEB-") for fid in suppressed.values()),
        "retained_legacy_documents": len(old_documents) - len(suppressed),
        "reviewed_web_original_aliases": len(reviewed_aliases),
        "new_web_documents": 850,
        "raw_pdf_count": len(pdf_inventory),
        "bound_documents": len(bindings),
        "distinct_bound_pdf_paths": len({v["path"] for v in bindings.values()}),
        "partitions": parts,
        "human_approved": False,
    }
    json_write(output / "build_receipt.json", receipt)
    json_write(args.assets / "build_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
