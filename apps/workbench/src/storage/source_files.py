"""Resolve original files within approved roots, including verified legacy PDF bindings."""

import json
from hashlib import sha256
from pathlib import Path

from domain.knowledge_models import CaseDetail
from settings import AppSettings


class SourceFiles:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.index_path = settings.case_kb_root / "source_pdf_index.json"
        self._stamp: int | None = None
        self._entries: dict[str, dict[str, str]] = {}

    def resolve(self, item: CaseDetail) -> Path | None:
        if item.raw_source_file:
            path = self._allowed_path(item.raw_source_file)
            if path is not None:
                return path
        stamp = self.index_path.stat().st_mtime_ns if self.index_path.is_file() else None
        if stamp != self._stamp:
            self._entries = (
                json.loads(self.index_path.read_text(encoding="utf-8")).get("files", {})
                if stamp is not None
                else {}
            )
            self._stamp = stamp
        key = item.document_id or item.source_file or ""
        entry = self._entries.get(key)
        if not entry or entry.get("source_sha256") != item.source_sha256:
            return None
        path = self._allowed_path(entry.get("path", ""))
        if path is None or path.suffix.lower() != ".pdf":
            return None
        if sha256(path.read_bytes()).hexdigest() != entry.get("pdf_sha256"):
            return None
        return path

    def _allowed_path(self, value: str) -> Path | None:
        if not value:
            return None
        source = self.settings.project_root / value.replace("\\", "/")
        alias = source.with_name(sha256(source.name.encode("utf-8")).hexdigest() + source.suffix)
        path = (alias if alias.is_file() else source).resolve()
        roots = (
            self.settings.project_root / "raw_pdfs",
            self.settings.raw_case_upload_dir,
            self.settings.collection_root,
        )
        if not any(path.is_relative_to(root.resolve()) for root in roots) or not path.is_file():
            return None
        return path
