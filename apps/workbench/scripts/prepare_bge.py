"""Fetch the official pinned BGE-small weights into this workspace; no remote code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fetch_competition_assets import fetch

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7999e1d3359715c523056ef9478215996d62a620"


def main() -> None:
    destination = ROOT / "models/bge-small-zh-v1.5"
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "model.safetensors",
    ]
    receipts = []
    for name in files:
        target = destination / name
        if not target.is_file():
            raw = fetch(f"https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/{REVISION}/{name}")
            temporary = target.with_suffix(target.suffix + ".download")
            temporary.write_bytes(raw)
            temporary.replace(target)
        raw = target.read_bytes()
        receipts.append(
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
        print(f"BGE ready: {name} ({len(raw)} bytes)", flush=True)
    (destination / "finrisk_model_manifest.json").write_text(
        json.dumps(
            {
                "model": "BAAI/bge-small-zh-v1.5",
                "revision": REVISION,
                "files": receipts,
                "pooling": "CLS",
                "normalization": "L2",
                "dimension": 512,
                "query_instruction": "为这个句子生成表示以用于检索相关文章：",
                "max_length": 512,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
