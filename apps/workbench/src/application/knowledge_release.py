"""Read the pinned P0 legal/navigation assets without executing downloaded Python."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from pathlib import Path
from typing import Any

RUNTIME_PIN = "c178df130ac618d5ba6e5a7944648fb6d25d60fb3de2c095b6f833fc16cf96a8"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8-sig").splitlines() if line.strip()]


class P0KnowledgeRelease:
    def __init__(self, root: Path, versions: dict[str, str]) -> None:
        self.root = root.resolve()
        path = root / "contracts/runtime_manifest.json"
        if _sha(path) != RUNTIME_PIN:
            raise ValueError("P0 runtime manifest pin mismatch")
        manifest = json.loads(path.read_text("utf-8"))
        # Only the data used by this application is required. Upstream Python and
        # its retired regression datasets are not part of our runtime package.
        for name in ("contracts/asset_lock.json", "policy/article_scopes.jsonl"):
            self._verify(name, manifest["files"][name])
        if manifest["full_reader"]["versions"] != versions:
            raise ValueError("P0 base version mismatch")
        # Nested manifests bind every legal snapshot and every mechanism card, not just their IDs.
        lock = json.loads((root / "contracts/asset_lock.json").read_text("utf-8"))
        self.asset_lock = lock
        if lock["base_versions"] != versions or any(
            _sha(root / name) != lock[key]
            for name, key in [
                ("assets/legal/manifest.json", "legal_manifest_sha256"),
                ("assets/expanded/contracts/knowledge_manifest.json", "expanded_manifest_sha256"),
            ]
        ):
            raise ValueError("P0 nested asset manifest pin mismatch")
        expanded = json.loads(
            (root / "assets/expanded/contracts/knowledge_manifest.json").read_text("utf-8")
        )
        name = "expanded/knowledge/mechanisms.jsonl"
        self._verify("assets/" + name, expanded["files"][name])
        legal = root / "assets/legal"
        legal_manifest = json.loads((legal / "manifest.json").read_text("utf-8"))
        self.snapshot_shas = {meta["path"]: meta["sha256"] for meta in legal_manifest["files"]}
        legal_files = {meta["path"]: meta for meta in legal_manifest["files"]}
        for name in ("articles.jsonl", "dependencies.jsonl", "source_registry.jsonl"):
            self._verify("assets/legal/" + name, legal_files[name])
        self.articles = {r["article_id"]: r for r in _rows(legal / "articles.jsonl")}
        for name in {r["provenance"]["snapshot_file"] for r in self.articles.values()}:
            self._verify("assets/legal/" + name, legal_files[name])
        self.sources = {r["source_id"]: r for r in _rows(legal / "source_registry.jsonl")}
        self.cards = {
            r["mechanism_id"]: r for r in _rows(root / "assets/expanded/knowledge/mechanisms.jsonl")
        }
        self.scopes = {r["article_id"]: r for r in _rows(root / "policy/article_scopes.jsonl")}
        self.dependencies: dict[str, list[str]] = {}
        for edge in _rows(legal / "dependencies.jsonl"):
            self.dependencies.setdefault(edge["from_article_id"], []).append(edge["to_article_id"])
        for article in self.articles.values():
            if hashlib.sha256(article["text"].encode()).hexdigest() != article["text_sha256"]:
                raise ValueError("P0 article hash mismatch")
            provenance = article["provenance"]
            source = (legal / provenance["snapshot_file"]).read_text("utf-8")
            span = provenance["span"]
            actual = source[span["start_char"] : span["end_char"]]
            if provenance["transform"] == "none":
                valid = actual == article["text"]
            else:

                def normalized(text: str) -> str:
                    return "".join(re.sub(r"^L\d+:\s?", "", text, flags=re.MULTILINE).split())

                valid = normalized(actual) == normalized(article["text"])
            if not valid:
                raise ValueError("P0 legal source interval mismatch")

    def _verify(self, name: str, meta: dict[str, Any]) -> None:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path.stat().st_size != meta["bytes"]:
            raise ValueError("P0 payload path/size mismatch")
        if _sha(path) != meta["sha256"]:
            raise ValueError("P0 payload hash mismatch: " + name)

    def read_article(self, identifier: str) -> dict[str, Any]:
        if identifier not in self.articles:
            raise ValueError("未登记的法规ID")
        pending = deque([identifier])
        seen: list[str] = []
        while pending and len(seen) < 32:
            key = pending.popleft()
            if key in seen:
                continue
            if key not in self.articles:
                continue
            seen.append(key)
            pending.extend(self.dependencies.get(key, []))
        article = self.articles[identifier]
        span = article["provenance"]["span"]
        return {
            "id": identifier,
            "kind": "regulation",
            "layer": "hit",
            "segments": [
                {
                    "text": article["text"],
                    "start": article.get("start", span["start_char"]),
                    "end": article.get("end", span["end_char"]),
                }
            ],
            "article": article,
            "dependencies": [self.articles[k] for k in seen[1:]],
            "source_registry": {
                self.articles[k]["source_id"]: self.sources[self.articles[k]["source_id"]]
                for k in seen
            },
            "source_sha256": self.snapshot_shas[article["provenance"]["snapshot_file"]],
            "version_id": article["version_id"],
            "literal_verified": True,
            "source_representation": article.get("representation", span.get("span_representation")),
            "source_originals_complete": False,
            "production_active": False,
            "human_approved": False,
            "current_facts_proved": False,
            "coverage_status": "partial" if pending else "complete_registered_dependencies",
            "scope": self.scopes.get(identifier),
            "legal_applicability": "requires_review",
        }

    def guides(self, mechanisms: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "mechanism_id": mid,
                "title": card["title"],
                "interpretation": card["overview"],
                "conditions": [
                    {
                        "condition_id": p["predicate_id"],
                        "description": p["description"],
                        "required_for_risk": p.get("required_for_risk", False),
                    }
                    for p in card["predicates"]
                ],
                "article_refs": card["article_ids"],
                "scope_axes": card["scope_axes"],
                "candidate_labels": card.get("candidate_label_links", []),
                "origin": "model_authored_screening_guide_not_legal_text",
                "human_approved": False,
                "label_mapping_is_automatic": False,
                "current_fact_status": "not_assessed",
            }
            for mid in mechanisms
            if (card := self.cards.get(mid))
        ]
