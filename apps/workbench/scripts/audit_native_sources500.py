"""Read original PDFs without modifying them; bind exact normalized bodies for the 500-case pool."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf
from build_competition_dataset import DATA, ROOT, read_rows, save, write_rows

from settings import AppSettings

ANCHORS = set()


def normalize(text):
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).casefold()


def initialize(anchors):
    global ANCHORS
    ANCHORS = set(anchors)


def read_pdf(path):
    try:
        with pymupdf.open(path) as document:
            text = "\n".join(page.get_text() for page in document)
        normalized = normalize(text)
        hits = {
            normalized[i : i + 18]
            for i in range(max(0, len(normalized) - 17))
            if normalized[i : i + 18] in ANCHORS
        }
        return str(path), normalized, list(hits), None
    except Exception as error:
        return str(path), "", [], type(error).__name__


def main():
    settings = AppSettings.from_env()
    pool = read_rows(DATA / "cases.jsonl")
    wanted = {r["file_id"] for r in pool}
    with sqlite3.connect(
        f"file:{(settings.case_kb_root / 'evidence_registry.sqlite3').as_posix()}?mode=ro", uri=True
    ) as con:
        sources = {
            fid: normalize(
                con.execute("SELECT text FROM sources WHERE file_id=?", (fid,)).fetchone()[0]
            )
            for fid in wanted
        }
    anchors = defaultdict(set)
    for fid, text in sources.items():
        for start in range(0, max(0, len(text) - 17), 32):
            anchors[text[start : start + 18]].add(fid)
    paths = sorted((ROOT / "raw_pdfs").glob("*.pdf"))
    out = ROOT / "outputs/competition_upgrade_20261019"
    matches = defaultdict(list)
    scanned = empty = errors = 0
    with ProcessPoolExecutor(
        max_workers=2, initializer=initialize, initargs=(list(anchors),)
    ) as executor:
        for path, text, hits, error in executor.map(read_pdf, paths, chunksize=20):
            scanned += 1
            empty += not text
            errors += bool(error)
            candidates = {fid for anchor in hits for fid in anchors[anchor]}
            for fid in candidates:
                original = sources[fid]
                if len(original) >= 80 and (
                    original == text
                    or (original in text and len(original) / max(1, len(text)) >= 0.8)
                ):
                    matches[fid].append(str(Path(path).relative_to(ROOT)))
            if scanned % 1000 == 0:
                print("native PDFs", scanned, "bound pool files", len(matches), flush=True)
    bindings = [
        {
            "file_id": fid,
            "raw_source_file": values[0],
            "matching_files": values,
            "verification": "normalized_native_text_contains_complete_exported_source",
        }
        for fid, values in sorted(matches.items())
    ]
    write_rows(out / "native_pdf_bindings500.jsonl", bindings)
    save(
        out / "native_pdf_audit500.json",
        {
            "pdfs_read": scanned,
            "empty_native_text": empty,
            "read_errors": errors,
            "pool_source_files": 500,
            "body_bound_files": len(matches),
            "ocr_reprocessed": False,
            "raw_files_changed": False,
        },
    )
    records = read_rows(settings.cases_path)
    for r in records:
        if r.get("file_id") in matches:
            r["raw_source_file"] = matches[r["file_id"]][0]
            r["raw_source_binding"] = "normalized_native_body_verified"
    write_rows(settings.cases_path.with_suffix(".jsonl.tmp"), records)
    settings.cases_path.with_suffix(".jsonl.tmp").replace(settings.cases_path)
    print("native source binding complete", len(matches), flush=True)


if __name__ == "__main__":
    main()
