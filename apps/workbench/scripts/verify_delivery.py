"""Validate the files required for a runnable FinRisk delivery package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    PROJECT_ROOT / ".env.example",
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "requirements.lock.txt",
    PROJECT_ROOT / "src" / "api" / "app.py",
    PROJECT_ROOT / "src" / "web" / "static" / "index.html",
    PROJECT_ROOT / "src" / "web" / "static" / "source-map.js",
    PROJECT_ROOT / "src" / "web" / "static" / "fonts" / "InterVariable.woff2",
    PROJECT_ROOT / "src" / "web" / "static" / "fonts" / "OFL.txt",
    PROJECT_ROOT / "agent_runtime" / "package.json",
    PROJECT_ROOT / "agent_runtime" / "package-lock.json",
    PROJECT_ROOT / "agent_runtime" / "worker.mjs",
    PROJECT_ROOT / "agent_runtime" / "runtime.mjs",
    PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for _ in source)


def main() -> None:
    """Check required assets, knowledge-base integrity, and the app import."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-knowledge", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from settings import AppSettings

    load_dotenv(PROJECT_ROOT / ".env")
    settings = AppSettings.from_env()
    knowledge_root = settings.case_kb_root
    manifest_path = settings.manifest_path
    if args.allow_missing_knowledge and not manifest_path.is_file():
        missing_source = [str(path) for path in REQUIRED_PATHS if not path.is_file()]
        if missing_source:
            raise SystemExit(f"Missing source files: {', '.join(missing_source)}")
        print(
            "Dependencies installed. Restore the knowledge base as described in README, "
            "then verify."
        )
        return
    required = (
        *REQUIRED_PATHS,
        manifest_path,
        settings.cases_path,
        settings.chunks_path,
        settings.expansions_path,
        settings.risk_dictionary_path,
        knowledge_root / "dictionary/risk_domain_mapping.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing delivery files: {', '.join(missing)}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("valid") is False:
        raise SystemExit("Knowledge-base manifest is marked invalid.")

    for name, metadata in manifest.get("files", {}).items():
        path = knowledge_root / metadata["path"]
        if not path.is_file():
            raise SystemExit(f"Missing knowledge-base file: {path.relative_to(PROJECT_ROOT)}")
        actual_bytes = path.stat().st_size
        if actual_bytes != metadata["bytes"]:
            raise SystemExit(f"Byte-size mismatch for {name}: {actual_bytes}")
        actual_hash = _sha256(path)
        if actual_hash != metadata["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {name}: {actual_hash}")
        actual_rows = _line_count(path)
        if actual_rows != metadata["rows"]:
            raise SystemExit(f"Row-count mismatch for {name}: {actual_rows}")

    cases = _line_count(settings.cases_path)
    expected_cases = manifest.get("counts", {}).get("cases", manifest.get("case_count"))
    if expected_cases is not None and cases != expected_cases:
        raise SystemExit(f"Case count mismatch: {cases} != {expected_cases}")
    full = manifest.get("full_projection")
    if full:
        projection = knowledge_root / "data/case_chunks_full.jsonl"
        if _line_count(projection) != full["units"] or _sha256(projection) != full["sha256"]:
            raise SystemExit("Full projection integrity mismatch")
    from api.app import app  # noqa: PLC0415

    if app.version != "1.0.0":
        raise SystemExit(f"Unexpected application version: {app.version}")

    print(
        "FinRisk delivery integrity check passed: "
        f"{manifest.get('document_count', cases)} documents, {cases} subject cases, "
        f"app version {app.version}. No external API called."
    )


if __name__ == "__main__":
    main()
