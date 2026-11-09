"""Bind legacy sources to local PDFs only when their complete normalized text matches."""

import json
import re
import sqlite3
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import pymupdf

from settings import AppSettings


def normalized_text_hash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\\n", "\n"))
    compact = re.sub(r"\s+", "", text)
    compact = re.sub(r"手机扫一扫打开此页$", "", compact)
    return sha256(compact.encode("utf-8")).hexdigest()


def inspect_pdf(path: Path) -> tuple[str, str, str] | None:
    try:
        with pymupdf.open(path) as document:
            text = "\n".join(page.get_text() for page in document)
        if not text.strip():
            return None
        return str(path), normalized_text_hash(text), sha256(path.read_bytes()).hexdigest()
    except (OSError, RuntimeError, ValueError):
        return None


def main() -> None:
    settings = AppSettings.from_env()
    by_text: dict[str, list[tuple[str, str]]] = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(
            pool.map(
                inspect_pdf,
                sorted((settings.project_root / "raw_pdfs").glob("*.pdf")),
                chunksize=32,
            ),
            1,
        ):
            if result is not None:
                path, text_hash, pdf_hash = result
                by_text.setdefault(text_hash, []).append((path, pdf_hash))
            if index % 1000 == 0:
                print(f"Scanned {index} PDFs", flush=True)
    files = {}
    with sqlite3.connect(settings.case_kb_root / "evidence_registry.sqlite3") as connection:
        for file_id, text, source_hash in connection.execute(
            "SELECT file_id,text,sha256 FROM sources"
        ):
            matches = by_text.get(normalized_text_hash(text), [])
            if not matches:
                continue
            path, pdf_hash = matches[0]
            files[file_id] = {
                "path": Path(path).relative_to(settings.project_root).as_posix(),
                "source_sha256": source_hash,
                "pdf_sha256": pdf_hash,
            }
    output = settings.case_kb_root / "source_pdf_index.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": 1,
                "match": "complete_text_nfkc_without_whitespace_and_site_footer",
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Bound {len(files)} documents by complete text; saved {output}", flush=True)


if __name__ == "__main__":
    main()
