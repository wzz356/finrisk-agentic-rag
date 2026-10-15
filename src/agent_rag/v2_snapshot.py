"""冻结V2原文与投影的只读适配；组合证据始终保留多个真实连续区间。"""

import csv
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from agent_rag.contracts import canonical_json, load_contract
from agent_rag.input_registry import text_hash
from agent_rag.knowledge_snapshot import make_evidence

PROFILE = "v2-hybrid-flash-v1"


def file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class V2KnowledgeSnapshot:
    def __init__(self, root: Path, backend: Any = None) -> None:
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "finrisk-v2-snapshot/1.0":
            raise ValueError("INVALID_V2_SNAPSHOT")
        self.assets = self.manifest["assets"]
        for name, asset in self.assets.items():
            if file_hash(Path(asset["path"])) != asset["sha256"]:
                raise ValueError(
                    "SOURCE_HASH_MISMATCH" if name == "sources" else "INDEX_HASH_MISMATCH"
                )
        dictionary = Path(self.assets["dictionary"]["path"]).read_bytes()
        definitions = list(csv.DictReader(io.StringIO(dictionary.decode("utf-8-sig"))))
        expected = set(load_contract("model-output.schema.json")["$defs"]["tag"]["enum"])
        if len(definitions) != 27 or {row["risk_tag"] for row in definitions} != expected:
            raise ValueError("INVALID_DICTIONARY")
        self.definitions = {
            row["risk_tag"]: {**row, "dictionary_sha256": self.manifest["dictionary_sha256"]}
            for row in definitions
        }
        self.sources: dict[str, dict[str, Any]] = {}
        self.by_file: dict[str, dict[str, Any]] = {}
        for row in rows(Path(self.assets["sources"]["path"])):
            sid, fid, text = row["source_id"], row["official_file_id"], row["full_text"]
            if sid in self.sources or fid in self.by_file or text_hash(text) != row["text_sha256"]:
                raise ValueError("INVALID_V2_SOURCE")
            source = {
                **row,
                "title": row.get("title") or fid,
                "text_version": str(row["text_version"]),
            }
            self.sources[sid] = source
            self.by_file[fid] = source
        self.units = list(rows(Path(self.assets["index"]["path"])))
        if len({u["unit_id"] for u in self.units}) != len(self.units):
            raise ValueError("DUPLICATE_UNIT_ID")
        if (
            len(self.sources) != self.manifest["source_count"]
            or len(self.units) != self.manifest["chunk_count"]
        ):
            raise ValueError("SNAPSHOT_COUNT_MISMATCH")
        self.backend = backend
        self.cache: OrderedDict = OrderedDict()

    @classmethod
    def create(
        cls,
        source_path: Path,
        index_path: Path,
        dictionary_path: Path,
        destination: Path,
        backend: Any = None,
    ) -> "V2KnowledgeSnapshot":
        destination = destination.resolve()
        if (destination / "manifest.json").exists():
            raise ValueError("SNAPSHOT_ALREADY_EXISTS")
        destination.mkdir(parents=True, exist_ok=True)
        assets = {
            name: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for name, path in [
                ("sources", source_path),
                ("index", index_path),
                ("dictionary", dictionary_path),
            ]
        }
        manifest = {
            "schema_version": "finrisk-v2-snapshot/1.0",
            "snapshot_id": "v2-"
            + text_hash(canonical_json({k: v["sha256"] for k, v in assets.items()}))[:24],
            "knowledge_version": "finrisk-v2-20261012",
            "retrieval_profile": PROFILE,
            "dictionary_sha256": assets["dictionary"]["sha256"],
            "index_sha256": assets["index"]["sha256"],
            "source_count": sum(1 for _ in rows(source_path)),
            "chunk_count": sum(1 for _ in rows(index_path)),
            "assets": assets,
            "human_approved": False,
            "semantic_coverage_verified": False,
        }
        (destination / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
        return cls(destination, backend)

    def read(self, source_id: str) -> tuple[dict[str, Any], str]:
        if source_id not in self.sources:
            raise ValueError("SOURCE_NOT_ALLOWED")
        if file_hash(Path(self.assets["sources"]["path"])) != self.assets["sources"]["sha256"]:
            raise ValueError("SOURCE_HASH_MISMATCH")
        source = self.sources[source_id]
        if text_hash(source["full_text"]) != source["text_sha256"]:
            raise ValueError("SOURCE_HASH_MISMATCH")
        return source, source["full_text"]

    def candidate(self, unit: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
        source = self.by_file.get(unit["file_id"])
        if not source:
            raise ValueError("SOURCE_NOT_ALLOWED")
        if unit.get("source_sha256") != source["text_sha256"]:
            raise ValueError("SOURCE_HASH_MISMATCH")
        text = source["full_text"]
        evidence: dict[str, dict[str, Any]] = {}

        def validate_spans(spans, quote, segments=None):
            if not isinstance(spans, list):
                raise ValueError("INVALID_SOURCE_SPAN")
            for span in spans:
                if not isinstance(span, dict):
                    raise ValueError("INVALID_SOURCE_SPAN")
                start, end = span.get("start"), span.get("end")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(text)
                ):
                    raise ValueError("INVALID_SOURCE_SPAN")
            if not spans:
                return
            if segments is not None:
                if not isinstance(segments, list) or len(segments) != len(spans):
                    raise ValueError("SOURCE_QUOTE_MISMATCH")
                for span, segment in zip(spans, segments):
                    if (
                        not isinstance(segment, dict)
                        or segment.get("span") != span
                        or not isinstance(segment.get("text"), str)
                        or text[span["start"]:span["end"]] != segment["text"]
                    ):
                        raise ValueError("SOURCE_QUOTE_MISMATCH")
                if quote is not None and quote != "".join(s["text"] for s in segments):
                    raise ValueError("SOURCE_QUOTE_MISMATCH")
            else:
                if not isinstance(quote, str) or not quote:
                    raise ValueError("MISSING_SOURCE_QUOTE")
                # Index spans are repeated occurrences of the same quote,
                # unlike returned context segments, which are distinct pieces.
                if any(text[span["start"]:span["end"]] != quote for span in spans):
                    raise ValueError("SOURCE_QUOTE_MISMATCH")

        def add(spans, role, quote, segments=None):
            validate_spans(spans, quote, segments)
            ids = []
            for span in spans:
                start, end = span["start"], span["end"]
                item = make_evidence(source, start, end, text)
                item["evidence_roles"] = sorted(
                    set(evidence.get(item["evidence_id"], {}).get("evidence_roles", []) + [role])
                )
                evidence[item["evidence_id"]] = item
                ids.append(item["evidence_id"])
            return ids

        fact_ids = add(
            unit["source_spans"], "fact", unit.get("source_text"), unit.get("source_segments")
        )
        if not fact_ids:
            raise ValueError("MISSING_FACT_SOURCE_SPAN")
        associations = []
        for association in unit.get("associations", []):
            if "quote_source_spans" in association:
                validate_spans(association["quote_source_spans"], association.get("quote"))
            party = add(
                association.get("party_source_spans", []), "party", association.get("party_quote")
            )
            qualifier = []
            for link in association.get("qualifier_source_links", []):
                qualifier += add(
                    link.get("source_spans", [link["span"]] if "span" in link else []),
                    link.get("field", link.get("kind", "condition")),
                    link.get("quote"),
                )
            associations.append(
                {
                    key: association.get(key)
                    for key in [
                        "case_id",
                        "behavior_id",
                        "actor_role",
                        "attribution_status",
                        "party_position_ambiguous",
                    ]
                }
                | {
                    "fact_evidence_ids": fact_ids,
                    "party_evidence_ids": party,
                    "qualifier_evidence_ids": qualifier,
                }
            )
        for link in unit.get("qualifier_source_links", []):
            add(
                link.get("source_spans", [link["span"]] if "span" in link else []),
                link.get("field", "condition"),
                link.get("quote"),
            )
        return {
            "file_id": unit["file_id"],
            "unit_id": unit["unit_id"],
            "source_id": source["source_id"],
            "title": source["title"],
            "case_id": unit.get("case_id"),
            "behavior_id": unit.get("behavior_id"),
            "case_ids": unit.get("case_ids", []),
            "behavior_ids": unit.get("behavior_ids", []),
            "associations": associations,
            "required_evidence_ids": list(evidence),
            "evidence": list(evidence.values()),
            "rank": rank,
            "score": score,
            "score_type": "deepseek_relevance_0_100_not_risk_probability",
            "algorithm": PROFILE,
            "annotation_status": "candidate_pending_human_acceptance",
            "context_ambiguous": bool(
                unit.get("context_ambiguous") or unit.get("evidence_ambiguous")
            ),
            "duplicate_hint": None,
        }

    def search(self, text: str, top_k: int, control: Any = None) -> dict[str, Any]:
        if self.backend is None:
            raise ValueError("V2_BACKEND_NOT_CONFIGURED")
        return self.backend.search(text, top_k, control)
