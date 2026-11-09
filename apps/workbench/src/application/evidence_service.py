"""Version-bound, read-only evidence and regulation access for the Pi tools."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from retrieval.query_planner import RetrievalQueryPlanner
from retrieval.sparse import RetrievalDocument, SparseRetrievalIndex


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


class KnowledgeEvidenceService:
    def __init__(self, root: Path, additional_databases: tuple[Path, ...] = ()) -> None:
        self.root = root
        self.additional_databases = additional_databases
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.versions = manifest.get("knowledge_versions", {})
        bundle = root / "source_bundle"
        handoff = bundle / "releases/2026-10-18/data/handoff_scope"
        expected = json.loads((handoff / "version_lock.json").read_text(encoding="utf-8"))
        base_versions = manifest.get("base_knowledge_versions", self.versions)
        if base_versions != expected or set(expected) != {
            "content",
            "projection",
            "articles",
            "rules",
            "scope",
        }:
            raise ValueError("知识库五类版本锁不一致")
        if "base_knowledge_versions" in manifest:
            lock_path = root / "release_lock.json"
            if (
                hashlib.sha256(lock_path.read_bytes()).hexdigest()
                != manifest["release_lock_sha256"]
            ):
                raise ValueError("知识库发布版本锁指纹不一致")
            lock = json.loads(lock_path.read_text("utf-8"))
            if lock["knowledge_versions"] != self.versions:
                raise ValueError("知识库发布版本不一致")
            for name, digest in lock["files"].items():
                path = (root / name).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("知识库发布路径越界")
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        raise ValueError("知识库发布资产指纹不一致: " + name)
        self.database = root / "evidence_registry.sqlite3"
        self.scope_units = {r["unit_id"]: r for r in _rows(handoff / "scope_overlays.jsonl")}
        self.scope_spans = {r["span_id"]: r for r in _rows(handoff / "spans.jsonl")}
        self.scope_sources = {r["source_id"]: r for r in _rows(handoff / "sources.jsonl")}
        law = bundle / "releases/2026-10-17/outputs/FinRisk_法规库V1_20261017"
        self.articles = {r["article_id"]: r for r in _rows(law / "articles.jsonl")}
        self.rules = {r["rule_id"]: r for r in _rows(law / "rules.jsonl")}
        self.law_sources = {r["source_id"]: r for r in _rows(law / "sources/texts.jsonl")}
        self.law_registry = {r["source_id"]: r for r in _rows(law / "source_registry.jsonl")}
        self.planner = RetrievalQueryPlanner(
            bundle / "releases/2026-10-18/data/auxiliary_lexicon.json"
        )
        self.p0 = None
        p0_root = root / "p0_release_20261018"
        if p0_root.is_dir():
            from application.knowledge_release import P0KnowledgeRelease

            self.p0 = P0KnowledgeRelease(p0_root, base_versions)
            self.articles = self.p0.articles
            self.law_registry = self.p0.sources
        self._law_index = SparseRetrievalIndex(
            RetrievalDocument(
                unit_id=key,
                case_ids=(key,),
                text=article["text"],
                source="regulatory_reader_snapshot",
            )
            for key, article in self.articles.items()
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(":memory:", uri=True)
        try:
            databases = [self.database, *(p for p in self.additional_databases if p.is_file())]
            for index, path in enumerate(databases):
                connection.execute(
                    f"ATTACH DATABASE ? AS source_{index}", (path.resolve().as_uri() + "?mode=ro",)
                )
            for table in ("sources", "units", "evidence"):
                union = " UNION ALL ".join(
                    f"SELECT * FROM source_{i}.{table}" for i in range(len(databases))
                )
                connection.execute(f"CREATE TEMP VIEW {table} AS {union}")
            yield connection
        finally:
            connection.close()

    @lru_cache(maxsize=4096)
    def read(self, identifier: str, layer: str = "hit") -> dict[str, Any]:
        if layer not in {"hit", "supplement", "extended"}:
            raise ValueError("未知证据层级")
        if self.p0 is not None and identifier in self.p0.articles:
            if layer != "hit":
                raise ValueError("法规上下文随登记依赖整组返回")
            return {**self.p0.read_article(identifier), "versions": self.versions}
        if identifier in self.articles and identifier not in self.scope_units:
            if layer != "hit":
                raise ValueError("法规条文只有已登记的 hit 层；上下文须使用独立条文ID")
            article = self.articles[identifier]
            source = self.law_sources[article["source_id"]]
            text = article["text"]
            if (
                source["text"][article["start"] : article["end"]] != text
                or hashlib.sha256(text.encode()).hexdigest() != article["text_sha256"]
            ):
                raise ValueError("法规原文回源失败")
            return {
                "id": identifier,
                "kind": "regulation",
                "layer": "hit",
                "versions": self.versions,
                "segments": [{"text": text, "start": article["start"], "end": article["end"]}],
                "source_sha256": source["text_sha256"],
                "version_id": article["version_id"],
                "literal_verified": True,
                "source_representation": "official_reader_snapshot_not_original_artifact",
                "production_active": False,
                "legal_applicability": "requires_review",
                "human_approved": False,
            }
        if identifier in self.scope_units:
            unit = self.scope_units[identifier]
            source = self.scope_sources[unit["source_id"]]
            segments = [self.scope_spans[key] for key in unit["layers"].get(layer, [])]
            if hashlib.sha256(source["text"].encode()).hexdigest() != unit["source_sha256"]:
                raise ValueError("范围原文指纹失配")
            for span in segments:
                if source["text"][span["start"] : span["end"]] != span["text"]:
                    raise ValueError("范围原文区间失配")
            return {
                "id": identifier,
                "kind": "regulation" if unit.get("article_id") else "case",
                "layer": layer,
                "versions": self.versions,
                "file_id": unit.get("file_id"),
                "segments": segments,
                "source_sha256": unit["source_sha256"],
                "literal_verified": True,
                "scope_review": "registered_targeted_overlay",
                "source_representation": unit.get(
                    "source_representation", "exported_candidate_text"
                ),
                "source_id": unit["source_id"],
                "production_active": False,
                "legal_applicability": "requires_review",
                "human_approved": False,
                "attribution_limits": unit.get("attribution_limits"),
                "join_semantics": "separate_spans_no_shared_actor_inference",
            }
        with self._connection() as connection:
            unit_row = connection.execute(
                "SELECT file_id,data FROM units WHERE id=?", (identifier,)
            ).fetchone()
            evidence_row = (
                None
                if unit_row
                else connection.execute(
                    "SELECT file_id,data FROM evidence WHERE id=?", (identifier,)
                ).fetchone()
            )
            row = unit_row or evidence_row
            if row is None:
                raise ValueError("未登记的证据ID")
            fid, raw = row
            item = json.loads(raw)
            source_row = connection.execute(
                "SELECT text,sha256 FROM sources WHERE file_id=?", (fid,)
            ).fetchone()
        if source_row is None:
            raise ValueError("缺少原始来源")
        source, sha = source_row
        if hashlib.sha256(source.encode()).hexdigest() != sha:
            raise ValueError("来源指纹失配")
        if evidence_row:
            if layer != "hit":
                raise ValueError("独立证据没有登记补充或扩展层")
            if sha != item["source_sha256"]:
                raise ValueError("独立证据来源指纹失配")
            spans = [{"start": item["start_char"], "end": item["end_char"]}]
            quote = item["verbatim"]
        else:
            if layer == "supplement":
                raise ValueError("基础单元没有登记补充范围；请按extended读取并独立核对主体")
            spans = (
                item.get("source_spans", []) if layer == "hit" else item.get("context_spans", [])
            )
            quote = item["source_text"] if layer == "hit" else None
            if sha != item["source_sha256"]:
                raise ValueError("检索单元来源指纹失配")
        segments = []
        truncated = False
        for span in spans:
            start, end = span["start"], span["end"]
            if not 0 <= start < end <= len(source):
                raise ValueError("未登记原文区间")
            text = source[start:end]
            if quote is not None and text != quote:
                raise ValueError("证据不等于登记原文")
            if sum(len(s["text"]) for s in segments) + len(text) > 6000:
                truncated = True
                continue
            segments.append({"text": text, "start": start, "end": end})
        return {
            "id": identifier,
            "kind": "case",
            "layer": layer,
            "file_id": fid,
            "versions": self.versions,
            "source_sha256": sha,
            "segments": segments,
            "literal_verified": True,
            "scope_review": "base_candidate_only",
            "human_approved": False,
            "context_truncated": truncated,
            "case_ids": item.get("application_case_ids", [item.get("case_id")]),
            "join_semantics": "separate_spans_no_shared_actor_inference",
        }

    def rule_context(self, text: str, *, limit: int = 12) -> list[dict[str, Any]]:
        mechanisms = self.planner.mechanisms(text)
        if self.p0 is not None:
            return self.p0.guides(mechanisms[:limit])
        selected = [rule for rule in self.rules.values() if rule["mechanism_id"] in mechanisms]
        selected.sort(
            key=lambda rule: (
                rule.get("norm_type") in {"scope_guard", "interpretation_guard"},
                rule["rule_id"],
            )
        )
        return [
            {
                "rule_id": r["rule_id"],
                "mechanism_id": r["mechanism_id"],
                "title": r["title"],
                "norm_type": r["norm_type"],
                "conditions": r.get("conditions", []),
                "condition_logic": r.get("condition_logic"),
                "exceptions": r.get("exceptions"),
                "forbidden_inferences": r.get("forbidden_inferences", []),
                "interpretation": r["interpretation"],
                "article_refs": r["article_refs"],
                "origin": "model_authored_interpretation",
                "human_approved": False,
                "legal_effect_status": "pending_review",
            }
            for r in selected[:limit]
        ]

    def search_regulations(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not query.strip() or not 1 <= top_k <= 10:
            raise ValueError("法规查询参数不合法")
        hits = self._law_index.search(query, top_k=top_k)
        return {
            "articles": [self.read(hit.document.unit_id) for hit in hits],
            "rules": self.rule_context(query),
            "versions": self.versions,
            "current_facts_proved": False,
            "production_active": False,
        }
