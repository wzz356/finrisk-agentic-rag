"""Download pinned public FinRisk assets for local integration; never executes remote code."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

REVISION = "55402453118f4dee9dd06f6b8ec39d7b21ae1e08"
REPOSITORY = "wzz356/finrisk-knowledge-base"
ROOT = Path(__file__).resolve().parents[1]
BASE = "releases/2026-10-17/outputs/FinRisk_知识库V2.1轻量收尾_20261015/"
LAW = "releases/2026-10-17/outputs/FinRisk_法规库V1_20261017/"


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "FinRisk-local-integration"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def selected(name: str) -> bool:
    if name.startswith(("versions/v2-20261012/normalized_final/data/", BASE + "patch_normalized/")):
        return True
    if name.startswith("versions/v2-20261012/indexes/all9639_final/"):
        return "structured.jsonl" in name or "raw.jsonl" in name or name.endswith("manifest.json")
    if name.startswith("releases/2026-10-18/data/handoff_scope/"):
        return name.endswith((".json", ".jsonl", ".md"))
    if name.startswith(LAW):
        return name[len(LAW) :] in {
            "articles.jsonl",
            "catalog.jsonl",
            "rules.jsonl",
            "source_registry.jsonl",
            "article_dependencies.json",
            "article_model_views.jsonl",
            "sources/texts.jsonl",
            "mechanism_rule_links.jsonl",
            "tag_rule_links.jsonl",
            "indexes/metadata.json",
        }
    return name in {
        "dictionary/risk_type_dictionary.csv",
        BASE + "effective_candidates.jsonl.gz",
        BASE + "file_index.jsonl",
        BASE + "version.json",
        BASE + "corrected_projection_units.jsonl",
        "releases/2026-10-18/data/auxiliary_lexicon.json",
        "releases/2026-10-18/data/boundary_evidence_contract.json",
        "releases/2026-10-18/data/delivery_articles43.jsonl",
        "releases/2026-10-18/data/law_review/fact_and_scope_policy.json",
        "releases/2026-10-18/data/接入说明.md",
        "releases/2026-10-18/README.md",
    }


def restore_p0(destination: Path) -> None:
    source = ROOT / "data/runtime_knowledge/p0_20261018"
    manifest = json.loads((source / "runtime_assets.json").read_text("utf-8"))
    for name, meta in manifest["files"].items():
        path = (source / name).resolve()
        target = (destination / name).resolve()
        if not path.is_relative_to(source.resolve()) or not target.is_relative_to(destination):
            raise ValueError("Unsafe runtime asset path")
        raw = path.read_bytes()
        if len(raw) != meta["bytes"] or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
            raise ValueError("Runtime asset fingerprint mismatch: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(
        f"Restored {len(manifest['files'])} pinned runtime data files; no old evaluation or Python."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--p0", action="store_true", help="Restore the bundled, fingerprinted P0 runtime data"
    )
    args = parser.parse_args()
    revision = REVISION
    default_name = "p0_release_20261018" if args.p0 else "source_bundle"
    destination = (
        args.destination or ROOT / "data/processed/knowledge/finrisk_v2_1_20261018" / default_name
    ).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Asset destination must remain inside this workspace")
    destination.mkdir(parents=True, exist_ok=True)
    if args.p0:
        restore_p0(destination)
        return
    tree = json.loads(
        fetch(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{revision}?recursive=1")
    )
    if tree.get("truncated"):
        raise ValueError("Truncated source inventory")
    assets = [item for item in tree["tree"] if item["type"] == "blob" and selected(item["path"])]
    print(
        f"Pinned {revision}: {len(assets)} files, "
        f"{sum(x.get('size', 0) for x in assets) / 1024**2:.1f} MiB",
        flush=True,
    )

    def download(item: dict[str, object]) -> dict[str, object]:
        source_name = name = str(item["path"])
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Unsafe public asset path")
        raw = target.read_bytes() if target.is_file() else b""

        def git_hash(content: bytes) -> str:
            return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()

        if not target.is_file() or git_hash(raw) != item["sha"]:
            raw = fetch(
                f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/"
                + urllib.parse.quote(source_name)
            )
            if git_hash(raw) != item["sha"]:
                raise ValueError("Source Git blob mismatch: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".download")
            temporary.write_bytes(raw)
            temporary.replace(target)
        return {
            "path": name,
            "bytes": len(raw),
            "git_blob": item["sha"],
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for number, receipt in enumerate(pool.map(download, assets), 1):
            receipts.append(receipt)
            if number % 20 == 0 or number == len(assets):
                print(f"Verified {number}/{len(assets)} assets", flush=True)
    (destination / "local_download_manifest.json").write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "revision": revision,
                "files": receipts,
                "remote_code_executed": False,
                "uploaded": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Local asset download and fingerprint verification complete", flush=True)


if __name__ == "__main__":
    main()
