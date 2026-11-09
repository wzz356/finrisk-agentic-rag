"""冻结官方TXT字节/文本和字典，读取时重新验哈希；不写原库。"""

import csv
import gzip
import hashlib
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from agent_rag.contracts import canonical_json, load_contract
from agent_rag.input_registry import text_hash
from agent_rag.retrieval import PROFILE, TOKENIZER_VERSION, BM25Index

CHUNKER_VERSION = "paragraph-window1000-overlap200/1.0"


def source_windows(text: str):
    start = 0
    while start < len(text):
        end = min(len(text), start + 1000)
        if end < len(text):
            paragraph_boundary = text.rfind("\n", start + 600, end)
            if paragraph_boundary >= 0:
                end = paragraph_boundary + 1
        if text[start:end].strip():
            yield start, end
        if end == len(text):
            break
        overlap_boundary = text.rfind("\n", start + 1, max(start + 1, end - 200))
        next_start = (
            overlap_boundary + 1 if overlap_boundary >= start + 400 else max(start + 1, end - 200)
        )
        start = next_start


def raw_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_evidence(source: dict[str, Any], start: int, end: int, text: str) -> dict[str, Any]:
    key = f"{source['source_id']}:{source['text_sha256']}:{start}:{end}"
    return {
        "evidence_id": f"EV-{text_hash(key)[:24]}",
        "source_id": source["source_id"],
        "source_kind": "penalty",
        "text_version": source["text_version"],
        "text_sha256": source["text_sha256"],
        "start": start,
        "end": end,
        "quote": text[start:end],
        "source_verified": True,
        "semantic_review_status": "not_assessed",
        "text_quality": ("synthetic_demo_not_real_penalty" if source.get("synthetic")
                         else "official_txt_pdf_not_verified"),
        "context_has_more": start > 0 or end < len(text),
    }


class KnowledgeSnapshot:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_data = (self.root / "manifest.json").read_bytes()
        self.manifest = json.loads(manifest_data)
        dictionary_bytes = (self.root / "risk_type_dictionary.csv").read_bytes()
        if raw_hash(dictionary_bytes) != self.manifest["dictionary_sha256"]:
            raise ValueError("DICTIONARY_HASH_MISMATCH")
        rows = list(csv.DictReader(io.StringIO(dictionary_bytes.decode("utf-8-sig"))))
        expected = set(load_contract("model-output.schema.json")["$defs"]["tag"]["enum"])
        if len(rows) != 27 or {r["risk_tag"] for r in rows} != expected:
            raise ValueError("INVALID_DICTIONARY")
        self.definitions = {
            row["risk_tag"]: {**row, "dictionary_sha256": self.manifest["dictionary_sha256"]}
            for row in rows
        }
        with gzip.open(self.root / "index.json.gz", "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        if text_hash(canonical_json(data)) != self.manifest["index_sha256"]:
            raise ValueError("INDEX_HASH_MISMATCH")
        self.sources = {
            row["source_id"]: {**row, "path": str(self.root / row["relative_path"])}
            for row in data["sources"]
        }
        self.index = BM25Index(data["chunks"])
        self.cache: OrderedDict[str, list[tuple[dict[str, Any], float]]] = OrderedDict()

    @classmethod
    def build(
        cls, source_root: Path, dictionary_path: Path, destination: Path
    ) -> "KnowledgeSnapshot":
        source_root, destination = source_root.resolve(), destination.resolve()
        if destination.exists() and (destination / "manifest.json").exists():
            raise ValueError("SNAPSHOT_ALREADY_EXISTS")
        if destination == source_root or source_root in destination.parents:
            raise ValueError("SNAPSHOT_INSIDE_SOURCE_ROOT")
        (destination / "texts").mkdir(parents=True, exist_ok=True)
        sources, chunks, excluded = [], [], []
        for path in sorted(source_root.glob("*.txt"), key=lambda p: p.name):
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                excluded.append(
                    {
                        "filename": path.name,
                        "reason": "decode_failure",
                        "raw_bytes_sha256": raw_hash(raw),
                    }
                )
                continue
            source_id = f"OFFTXT-{text_hash(path.name)[:20]}"
            relative_path = f"texts/{source_id}.txt"
            (destination / relative_path).write_bytes(raw)
            source = {
                "source_id": source_id,
                "title": path.stem,
                "relative_path": relative_path,
                "original_filename": path.name,
                "text_version": "official-txt-utf8/1.0",
                "raw_bytes_sha256": raw_hash(raw),
                "text_sha256": text_hash(text),
                "text_length": len(text),
                "case_id": None,
                "behavior_id": None,
                "annotation_status": "source_only",
                "pdf_verified": False,
            }
            sources.append(source)
            for start, end in source_windows(text):
                chunks.append(make_evidence(source, start, end, text))
        data = {"sources": sources, "chunks": chunks}
        with gzip.open(destination / "index.json.gz", "wt", encoding="utf-8") as handle:
            handle.write(canonical_json(data))
        dictionary = dictionary_path.read_bytes()
        (destination / "risk_type_dictionary.csv").write_bytes(dictionary)
        identity = {
            "source_hashes": [(r["source_id"], r["raw_bytes_sha256"]) for r in sources],
            "dictionary": raw_hash(dictionary),
            "tokenizer": TOKENIZER_VERSION,
            "chunker": CHUNKER_VERSION,
        }
        manifest = {
            "snapshot_id": f"official-txt-{text_hash(canonical_json(identity))[:24]}",
            "schema_version": "finrisk-snapshot/1.0",
            "source_count": len(sources),
            "chunk_count": len(chunks),
            "dictionary_sha256": raw_hash(dictionary),
            "index_sha256": text_hash(canonical_json(data)),
            "retrieval_profile": PROFILE,
            "tokenizer_version": TOKENIZER_VERSION,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "chunker_version": CHUNKER_VERSION,
            "excluded_sources": excluded,
            "semantic_coverage_verified": False,
        }
        (destination / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
        return cls(destination)

    def read(self, source_id: str) -> tuple[dict[str, Any], str]:
        if source_id not in self.sources:
            raise ValueError("SOURCE_NOT_ALLOWED")
        source = self.sources[source_id]
        path = Path(source["path"]).resolve()
        if not path.is_relative_to(self.root / "texts"):
            raise ValueError("SOURCE_NOT_ALLOWED")
        raw = path.read_bytes()
        if raw_hash(raw) != source["raw_bytes_sha256"]:
            raise ValueError("SOURCE_HASH_MISMATCH")
        text = raw.decode("utf-8-sig")
        if raw_hash(raw) != source["raw_bytes_sha256"] or text_hash(text) != source["text_sha256"]:
            raise ValueError("SOURCE_HASH_MISMATCH")
        return source, text

    def search(self, text: str, top_k: int) -> tuple[list[tuple[dict[str, Any], float]], bool]:
        key = canonical_json(
            {
                "text": " ".join(text.split()),
                "top_k": top_k,
                "snapshot": self.manifest["snapshot_id"],
                "profile": PROFILE,
            }
        )
        hit = key in self.cache
        if hit:
            self.cache.move_to_end(key)
            results = self.cache[key]
        else:
            results = self.index.search(text, top_k)
            self.cache[key] = results
            if len(self.cache) > 512:
                self.cache.popitem(last=False)
        # 即使缓存命中，每次回传前仍核查真实冻结文本。
        for evidence, _ in results:
            self.read(evidence["source_id"])
        return results, hit
