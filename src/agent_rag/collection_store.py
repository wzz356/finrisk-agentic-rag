"""Isolated original inventory; collecting never publishes retrieval knowledge."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL,
    status TEXT NOT NULL, metadata TEXT NOT NULL, details TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs (
    raw_sha256 TEXT PRIMARY KEY, size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    digest TEXT PRIMARY KEY, raw_sha256 TEXT NOT NULL REFERENCES blobs(raw_sha256),
    created_at TEXT NOT NULL, baseline_duplicate INTEGER NOT NULL,
    contains_insurance INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS document_blobs (
    digest TEXT NOT NULL REFERENCES documents(digest),
    raw_sha256 TEXT NOT NULL REFERENCES blobs(raw_sha256), created_at TEXT NOT NULL,
    PRIMARY KEY (digest, raw_sha256)
);
CREATE TABLE IF NOT EXISTS batch_documents (
    batch_id TEXT NOT NULL REFERENCES batches(id),
    digest TEXT NOT NULL REFERENCES documents(digest), disposition TEXT NOT NULL,
    PRIMARY KEY (batch_id, digest)
);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY, digest TEXT NOT NULL REFERENCES documents(digest),
    batch_id TEXT NOT NULL REFERENCES batches(id),
    raw_sha256 TEXT NOT NULL REFERENCES blobs(raw_sha256),
    metadata TEXT NOT NULL, created_at TEXT NOT NULL,
    UNIQUE (digest, batch_id, raw_sha256, metadata)
);
CREATE INDEX IF NOT EXISTS sources_digest ON sources(digest);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _public(value: Any) -> Any:
    """Keep original metadata in SQLite while omitting local paths from API views."""

    def local_path(item: Any) -> bool:
        return isinstance(item, str) and (
            PureWindowsPath(item).is_absolute() or PurePosixPath(item).is_absolute()
        )

    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if not local_path(item)}
    if isinstance(value, list):
        return [_public(item) for item in value if not local_path(item)]
    return value


def _official_url(value: Any) -> None:
    if not isinstance(value, str) or "\\" in value or any(char.isspace() for char in value):
        raise ValueError("UNTRUSTED_SOURCE_URL")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname in {"nfra.gov.cn", "www.nfra.gov.cn"}
            and parsed.port in {None, 443}
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("UNTRUSTED_SOURCE_URL")


class CollectionStore:
    """SQLite associations and immutable SHA-addressed original byte versions.

    Each operation owns a connection. Writers acquire SQLite's immediate transaction
    before touching blobs, so separate instances and threads share the same lock.
    A crash may leave an unregistered blob, which is never readable through this API.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._blobs = self.root / "blobs"
        self._blobs.mkdir(exist_ok=True)
        self._database = self.root / "collection.sqlite3"
        self._check_location(self._blobs)
        self._check_location(self._database)
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    def _check_location(self, path: Path) -> None:
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("INVENTORY_PATH_OUTSIDE_ROOT")

    @contextmanager
    def _connection(self, *, write: bool = False):
        self._check_location(self._database)
        connection = sqlite3.connect(self._database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _blob_path(self, raw_sha256: str) -> Path:
        if not isinstance(raw_sha256, str) or _HASH.fullmatch(raw_sha256) is None:
            raise ValueError("INVALID_RAW_HASH")
        path = self._blobs / f"{raw_sha256}.bin"
        self._check_location(path)
        return path

    def _read_blob(self, raw_sha256: str) -> bytes:
        try:
            raw = self._blob_path(raw_sha256).read_bytes()
        except OSError as error:
            raise ValueError("ORIGINAL_UNAVAILABLE") from error
        if hashlib.sha256(raw).hexdigest() != raw_sha256:
            raise ValueError("ORIGINAL_HASH_MISMATCH")
        return raw

    def _write_blob(self, raw_sha256: str, raw: bytes) -> None:
        destination = self._blob_path(raw_sha256)
        if destination.exists():
            self._read_blob(raw_sha256)
            return
        descriptor, name = tempfile.mkstemp(prefix=".original-", dir=self._blobs)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def create_batch(self, kind: str, metadata: dict | None = None) -> dict:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("INVALID_BATCH_KIND")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("INVALID_METADATA")
        batch = {
            "id": uuid4().hex,
            "kind": kind,
            "created_at": _now(),
            "status": "pending",
            "metadata": metadata or {},
        }
        encoded = _json(batch["metadata"])
        with self._connection(write=True) as connection:
            connection.execute(
                "INSERT INTO batches VALUES (?, ?, ?, ?, ?, ?)",
                (batch["id"], kind, batch["created_at"], "pending", encoded, "{}"),
            )
        return _public(batch)

    def ingest(
        self,
        batch_id: str,
        raw: bytes,
        metadata: dict,
        baseline_hashes: set[str] | None = None,
    ) -> dict:
        if not isinstance(raw, bytes) or not isinstance(metadata, dict):
            raise ValueError("INVALID_ORIGINAL")
        _official_url(metadata.get("source_url"))
        text = raw.decode("utf-8-sig")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        now = _now()
        with self._connection(write=True) as connection:
            batch = connection.execute(
                "SELECT kind FROM batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise ValueError("UNKNOWN_BATCH")
            stored_metadata = dict(metadata)
            stored_metadata.setdefault(
                "origin", "legacy_import" if batch["kind"] == "legacy_import" else "official_crawl"
            )
            if stored_metadata["origin"] not in {"legacy_import", "official_crawl"}:
                raise ValueError("INVALID_ORIGIN")
            encoded_metadata = _json(stored_metadata)
            existing = connection.execute(
                "SELECT * FROM documents WHERE digest = ?", (digest,)
            ).fetchone()
            baseline = digest in (baseline_hashes or set()) or bool(
                existing and existing["baseline_duplicate"]
            )
            disposition = "baseline_duplicate" if baseline else "duplicate" if existing else "new"
            self._write_blob(raw_sha256, raw)
            connection.execute("INSERT OR IGNORE INTO blobs VALUES (?, ?)", (raw_sha256, len(raw)))
            connection.execute(
                "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?)",
                (digest, raw_sha256, now, int(baseline), int("保险" in text)),
            )
            if baseline:
                connection.execute(
                    "UPDATE documents SET baseline_duplicate = 1 WHERE digest = ?", (digest,)
                )
            connection.execute(
                "INSERT OR IGNORE INTO document_blobs VALUES (?, ?, ?)", (digest, raw_sha256, now)
            )
            connection.execute(
                "INSERT OR IGNORE INTO batch_documents VALUES (?, ?, ?)",
                (batch_id, digest, disposition),
            )
            connection.execute(
                "INSERT OR IGNORE INTO sources "
                "(digest, batch_id, raw_sha256, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (digest, batch_id, raw_sha256, encoded_metadata, now),
            )
        return {"digest": digest, "disposition": disposition}

    def list_batches(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT b.*, COUNT(d.digest) AS documents,
                    COALESCE(SUM(d.disposition = 'new'), 0) AS new_documents,
                    COALESCE(SUM(d.disposition = 'duplicate'), 0) AS duplicates,
                    COALESCE(SUM(d.disposition = 'baseline_duplicate'), 0) AS baseline_duplicates
                FROM batches b LEFT JOIN batch_documents d ON d.batch_id = b.id
                GROUP BY b.id ORDER BY b.created_at DESC, b.rowid DESC"""
            ).fetchall()
        batches = []
        for row in rows:
            batch = dict(row)
            batch["metadata"] = json.loads(batch["metadata"])
            batch["details"] = json.loads(batch["details"])
            batches.append(_public(batch))
        return batches

    def set_batch_status(self, batch_id: str, status: str, details: dict | None = None) -> None:
        if not isinstance(status, str) or not status.strip():
            raise ValueError("INVALID_BATCH_STATUS")
        if details is not None and not isinstance(details, dict):
            raise ValueError("INVALID_METADATA")
        with self._connection(write=True) as connection:
            result = connection.execute(
                "UPDATE batches SET status = ?, details = ? WHERE id = ?",
                (status, _json(details or {}), batch_id),
            )
            if result.rowcount != 1:
                raise ValueError("UNKNOWN_BATCH")

    def _document(
        self, connection: sqlite3.Connection, row: sqlite3.Row, batch_id: str | None = None
    ) -> dict:
        sources = []
        for source in connection.execute(
            "SELECT batch_id, raw_sha256, metadata, created_at "
            "FROM sources WHERE digest = ? ORDER BY id",
            (row["digest"],),
        ):
            item = dict(source)
            item["metadata"] = json.loads(item["metadata"])
            item.update(
                {
                    key: item["metadata"].get(key, "")
                    for key in ("title", "source_url", "publish_date", "origin")
                }
            )
            sources.append(item)
        preferred = next(
            (source for source in sources if source["batch_id"] == batch_id),
            sources[0] if sources else {},
        )
        metadata = preferred.get("metadata", {})
        versions = [
            dict(version)
            for version in connection.execute(
                """SELECT db.raw_sha256, b.size, db.created_at FROM document_blobs db
            JOIN blobs b ON b.raw_sha256 = db.raw_sha256
            WHERE db.digest = ? ORDER BY db.created_at, db.rowid""",
                (row["digest"],),
            )
        ]
        return _public(
            {
                "digest": row["digest"],
                "created_at": row["created_at"],
                "status": "baseline_duplicate" if row["baseline_duplicate"] else "collected",
                "baseline_duplicate": bool(row["baseline_duplicate"]),
                "contains_insurance": bool(row["contains_insurance"]),
                "metadata": metadata,
                "sources": sources,
                "raw_versions": versions,
                **{
                    key: metadata.get(key, "")
                    for key in ("title", "source_url", "publish_date", "origin")
                },
            }
        )

    def documents(self, batch_id: str | None = None, offset: int = 0, limit: int = 20) -> dict:
        if type(offset) is not int or type(limit) is not int or offset < 0 or limit < 1:
            raise ValueError("INVALID_PAGINATION")
        where = (
            " WHERE digest IN (SELECT digest FROM batch_documents WHERE batch_id = ?)"
            if batch_id is not None
            else ""
        )
        parameters = (batch_id,) if batch_id is not None else ()
        with self._connection() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM documents" + where, parameters
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM documents"
                + where
                + " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
            items = [self._document(connection, row, batch_id) for row in rows]
        return {"items": items, "total": total}

    def read_document(self, digest: str) -> dict:
        if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
            raise ValueError("INVALID_DOCUMENT_DIGEST")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE digest = ?", (digest,)
            ).fetchone()
            if row is None:
                raise ValueError("UNKNOWN_DOCUMENT")
            document = self._document(connection, row)
            if row["raw_sha256"] not in {item["raw_sha256"] for item in document["raw_versions"]}:
                raise ValueError("UNREGISTERED_ORIGINAL")
            # Validate every retained byte version, including the canonical first one.
            hashes = {row["raw_sha256"], *(item["raw_sha256"] for item in document["raw_versions"])}
            text = None
            for raw_sha256 in hashes:
                candidate = self._read_blob(raw_sha256).decode("utf-8-sig")
                if hashlib.sha256(candidate.encode("utf-8")).hexdigest() != digest:
                    raise ValueError("DOCUMENT_HASH_MISMATCH")
                if raw_sha256 == row["raw_sha256"]:
                    text = candidate
            document["text"] = text
        return document

    def get_value(self, key: str, default: Any = None) -> Any:
        with self._connection() as connection:
            row = connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_value(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("INVALID_SETTING_KEY")
        with self._connection(write=True) as connection:
            connection.execute(
                "INSERT INTO settings VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, _json(value)),
            )

    def counts(self) -> dict:
        with self._connection() as connection:
            totals = {
                name: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for name, table in (
                    ("documents", "documents"),
                    ("batches", "batches"),
                    ("sources", "sources"),
                    ("raw_versions", "blobs"),
                )
            }
            totals["baseline_duplicates"] = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE baseline_duplicate = 1"
            ).fetchone()[0]
            totals["new_documents"] = totals["documents"] - totals["baseline_duplicates"]
            totals["duplicates"] = connection.execute(
                "SELECT COUNT(*) FROM batch_documents WHERE disposition = 'duplicate'"
            ).fetchone()[0]
        return totals
