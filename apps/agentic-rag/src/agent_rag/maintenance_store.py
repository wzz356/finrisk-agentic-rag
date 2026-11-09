"""Durable maintenance workflow state with immutable revisions and fenced leases.

Every mutation owns a SQLite BEGIN IMMEDIATE transaction. A leased job fences all
writes to its event; an expired/replaced worker cannot publish even an old receipt.
The store records publication receipts, while the publication adapter must use the
same revision idempotency key at its own atomic write boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4


class MaintenanceStoreError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MaintenanceStoreError("INVALID_IDENTIFIER")
    return value


def _instant(now: float | None) -> float:
    value = time.time() if now is None else now
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MaintenanceStoreError("INVALID_LEASE_TIME")
    return float(value)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
 id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL,
 payload_sha256 TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
 event_id TEXT NOT NULL REFERENCES events(id), revision INTEGER NOT NULL,
 payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL, created_at REAL NOT NULL,
 PRIMARY KEY(event_id, revision)
);
CREATE TABLE IF NOT EXISTS review_runs (
 id TEXT PRIMARY KEY, event_id TEXT NOT NULL, revision INTEGER NOT NULL,
 payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL, created_at REAL NOT NULL,
 FOREIGN KEY(event_id,revision) REFERENCES revisions(event_id,revision)
);
CREATE TABLE IF NOT EXISTS receipts (
 event_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
 payload_sha256 TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(event_id,revision),
 FOREIGN KEY(event_id,revision) REFERENCES revisions(event_id,revision)
);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(id), kind TEXT NOT NULL,
 idempotency_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
 details TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
 job_id TEXT PRIMARY KEY REFERENCES jobs(id), owner TEXT NOT NULL,
 fencing_token INTEGER NOT NULL, expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS policies (
 id TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
 id TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TRIGGER IF NOT EXISTS immutable_event_payload
BEFORE UPDATE OF payload,payload_sha256,idempotency_key ON events
 BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_EVENT'); END;
"""


class MaintenanceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "maintenance.sqlite3"
        with closing(self._connect()) as connection:
            connection.executescript(_SCHEMA)
            for table, code in [
                ("revisions", "REVISION"),
                ("review_runs", "REVIEW"),
                ("receipts", "RECEIPT"),
                ("policies", "POLICY"),
                ("artifacts", "ARTIFACT"),
            ]:
                for action in ("UPDATE", "DELETE"):
                    connection.execute(
                        f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action.lower()} "
                        f"BEFORE {action} ON {table} "
                        f"BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_{code}'); END"
                    )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _tx(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        for name in ("payload", "details"):
            if name in result:
                result[name] = json.loads(result[name])
        return result

    @classmethod
    def _event(cls, row) -> dict:
        result = cls._row(row)
        result.update(result.pop("payload"))
        return result

    @staticmethod
    def _require_event(connection, event_id):
        row = connection.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise MaintenanceStoreError("EVENT_NOT_FOUND")
        return row

    @staticmethod
    def _require_job(connection, job_id):
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise MaintenanceStoreError("JOB_NOT_FOUND")
        return row

    @staticmethod
    def _require_revision(connection, event_id, revision):
        if (
            connection.execute(
                "SELECT 1 FROM revisions WHERE event_id=? AND revision=?", (event_id, revision)
            ).fetchone()
            is None
        ):
            raise MaintenanceStoreError("REVISION_NOT_FOUND")

    def create_event(
        self, event_type, evidence_ids, target_record_ids, scope_id, idempotency_key, reason
    ) -> dict:
        _text(idempotency_key)
        if not isinstance(evidence_ids, (list, tuple)) or not isinstance(
            target_record_ids, (list, tuple)
        ):
            raise MaintenanceStoreError("INVALID_EVENT_REFERENCES")
        payload = _json(
            {
                "event_type": _text(event_type),
                "evidence_ids": [_text(x) for x in evidence_ids],
                "target_record_ids": [_text(x) for x in target_record_ids],
                "scope_id": _text(scope_id),
                "reason": _text(reason),
            }
        )
        with self._tx() as connection:
            existing = connection.execute(
                "SELECT * FROM events WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                if existing["payload"] != payload:
                    raise MaintenanceStoreError("IDEMPOTENCY_CONFLICT")
                return self._event(existing)
            event_id = uuid4().hex
            connection.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?)",
                (event_id, idempotency_key, payload, _sha(payload), "pending", time.time()),
            )
            return self._event(self._require_event(connection, event_id))

    def get_event(self, event_id) -> dict:
        with self._tx() as connection:
            return self._event(self._require_event(connection, event_id))

    def list_events(self, status=None) -> list[dict]:
        with self._tx() as connection:
            rows = connection.execute(
                "SELECT * FROM events"
                + (" WHERE status=?" if status is not None else "")
                + " ORDER BY created_at,id",
                (status,) if status is not None else (),
            ).fetchall()
            return [self._event(row) for row in rows]

    def _assert_lease(self, connection, job_id, owner, fencing_token, now=None) -> dict:
        if type(fencing_token) is not int or fencing_token < 1:
            raise MaintenanceStoreError("INVALID_FENCING_TOKEN")
        self._require_job(connection, job_id)
        row = connection.execute("SELECT * FROM leases WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise MaintenanceStoreError("LEASE_REQUIRED")
        if row["owner"] != owner or row["fencing_token"] != fencing_token:
            raise MaintenanceStoreError("STALE_LEASE")
        if row["expires_at"] <= _instant(now):
            raise MaintenanceStoreError("LEASE_EXPIRED")
        return dict(row)

    def assert_lease(self, job_id, owner, fencing_token, *, now=None) -> dict:
        with self._tx() as connection:
            return self._assert_lease(connection, job_id, owner, fencing_token, now)

    def _guard(self, connection, event_id, job_id=None, owner=None, fencing_token=None, now=None):
        self._require_event(connection, event_id)
        if job_id is None:
            if connection.execute("SELECT 1 FROM jobs WHERE event_id=?", (event_id,)).fetchone():
                raise MaintenanceStoreError("LEASE_REQUIRED")
            return
        job = self._require_job(connection, job_id)
        if job["event_id"] != event_id:
            raise MaintenanceStoreError("LEASE_EVENT_MISMATCH")
        self._assert_lease(connection, job_id, owner, fencing_token, now)

    def set_event_status(self, event_id, expected_status, new_status, **lease) -> dict:
        _text(new_status)
        with self._tx() as connection:
            self._guard(connection, event_id, **lease)
            cursor = connection.execute(
                "UPDATE events SET status=? WHERE id=? AND status=?",
                (new_status, event_id, expected_status),
            )
            if cursor.rowcount != 1:
                raise MaintenanceStoreError("EVENT_STATUS_CONFLICT")
            return self._event(self._require_event(connection, event_id))

    def save_changeset(self, event_id, changeset, revision=None, **lease) -> dict:
        payload = _json(changeset)
        with self._tx() as connection:
            self._guard(connection, event_id, **lease)
            if revision is None:
                revision = connection.execute(
                    "SELECT COALESCE(MAX(revision),0)+1 FROM revisions WHERE event_id=?",
                    (event_id,),
                ).fetchone()[0]
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise MaintenanceStoreError("INVALID_REVISION")
            row = connection.execute(
                "SELECT * FROM revisions WHERE event_id=? AND revision=?", (event_id, revision)
            ).fetchone()
            if row is not None:
                if row["payload"] != payload:
                    raise MaintenanceStoreError("IMMUTABLE_REVISION")
                return self._row(row)
            expected = connection.execute(
                "SELECT COALESCE(MAX(revision),0)+1 FROM revisions WHERE event_id=?", (event_id,)
            ).fetchone()[0]
            if revision != expected:
                raise MaintenanceStoreError("REVISION_SEQUENCE_CONFLICT")
            connection.execute(
                "INSERT INTO revisions VALUES(?,?,?,?,?)",
                (event_id, revision, payload, _sha(payload), time.time()),
            )
            return self._row(
                connection.execute(
                    "SELECT * FROM revisions WHERE event_id=? AND revision=?", (event_id, revision)
                ).fetchone()
            )

    def get_changeset(self, event_id, revision=None) -> dict:
        with self._tx() as connection:
            self._require_event(connection, event_id)
            row = connection.execute(
                "SELECT * FROM revisions WHERE event_id=?"
                + (" AND revision=?" if revision is not None else "")
                + " ORDER BY revision DESC LIMIT 1",
                (event_id, revision) if revision is not None else (event_id,),
            ).fetchone()
            if row is None:
                raise MaintenanceStoreError("REVISION_NOT_FOUND")
            return self._row(row)

    def save_review_run(self, event_id, revision, run, run_id=None, **lease) -> dict:
        payload, run_id = _json(run), _text(run_id) if run_id is not None else uuid4().hex
        with self._tx() as connection:
            self._guard(connection, event_id, **lease)
            self._require_revision(connection, event_id, revision)
            row = connection.execute("SELECT * FROM review_runs WHERE id=?", (run_id,)).fetchone()
            if row is not None:
                if (
                    row["payload"] != payload
                    or row["event_id"] != event_id
                    or row["revision"] != revision
                ):
                    raise MaintenanceStoreError("IMMUTABLE_REVIEW")
                return self._row(row)
            connection.execute(
                "INSERT INTO review_runs VALUES(?,?,?,?,?,?)",
                (run_id, event_id, revision, payload, _sha(payload), time.time()),
            )
            return self._row(
                connection.execute("SELECT * FROM review_runs WHERE id=?", (run_id,)).fetchone()
            )

    def list_review_runs(self, event_id, revision=None) -> list[dict]:
        with self._tx() as connection:
            self._require_event(connection, event_id)
            return [
                self._row(row)
                for row in connection.execute(
                    "SELECT * FROM review_runs WHERE event_id=?"
                    + (" AND revision=?" if revision is not None else "")
                    + " ORDER BY created_at,id",
                    (event_id, revision) if revision is not None else (event_id,),
                ).fetchall()
            ]

    def save_receipt(self, event_id, revision, receipt, **lease) -> dict:
        payload = _json(receipt)
        with self._tx() as connection:
            self._guard(connection, event_id, **lease)
            self._require_revision(connection, event_id, revision)
            row = connection.execute(
                "SELECT * FROM receipts WHERE event_id=? AND revision=?", (event_id, revision)
            ).fetchone()
            if row is not None:
                if row["payload"] != payload:
                    raise MaintenanceStoreError("IMMUTABLE_RECEIPT")
                return self._row(row)
            connection.execute(
                "INSERT INTO receipts VALUES(?,?,?,?,?)",
                (event_id, revision, payload, _sha(payload), time.time()),
            )
            return self._row(
                connection.execute(
                    "SELECT * FROM receipts WHERE event_id=? AND revision=?", (event_id, revision)
                ).fetchone()
            )

    def get_receipt(self, event_id, revision) -> dict | None:
        with self._tx() as connection:
            return self._row(
                connection.execute(
                    "SELECT * FROM receipts WHERE event_id=? AND revision=?", (event_id, revision)
                ).fetchone()
            )

    def create_job(self, event_id, kind="maintenance", payload=None, idempotency_key=None) -> dict:
        _text(kind)
        key, encoded = (
            _text(idempotency_key or f"{event_id}:{kind}"),
            _json({} if payload is None else payload),
        )
        with self._tx() as connection:
            self._require_event(connection, event_id)
            row = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key=?", (key,)
            ).fetchone()
            if row is not None:
                if row["event_id"] != event_id or row["kind"] != kind or row["payload"] != encoded:
                    raise MaintenanceStoreError("JOB_IDEMPOTENCY_CONFLICT")
                return self._row(row)
            job_id, now = uuid4().hex, time.time()
            connection.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                (job_id, event_id, kind, key, encoded, "pending", "{}", now, now),
            )
            return self._row(self._require_job(connection, job_id))

    def get_job(self, job_id) -> dict:
        with self._tx() as connection:
            return self._row(self._require_job(connection, job_id))

    def list_jobs(self, status=None, event_id=None) -> list[dict]:
        clauses, args = [], []
        for key, value in [("status", status), ("event_id", event_id)]:
            if value is not None:
                clauses.append(f"{key}=?")
                args.append(value)
        with self._tx() as connection:
            return [
                self._row(row)
                for row in connection.execute(
                    "SELECT * FROM jobs"
                    + (" WHERE " + " AND ".join(clauses) if clauses else "")
                    + " ORDER BY created_at,id",
                    args,
                ).fetchall()
            ]

    def acquire_lease(self, job_id, owner, ttl_seconds, *, now=None) -> dict:
        _text(owner)
        instant = _instant(now)
        self._ttl(ttl_seconds)
        if not math.isfinite(instant + ttl_seconds):
            raise MaintenanceStoreError("INVALID_LEASE_TTL")
        with self._tx() as connection:
            self._require_job(connection, job_id)
            old = connection.execute("SELECT * FROM leases WHERE job_id=?", (job_id,)).fetchone()
            if old is not None and old["expires_at"] > instant:
                raise MaintenanceStoreError("LEASE_HELD")
            token = old["fencing_token"] + 1 if old is not None else 1
            connection.execute(
                "INSERT INTO leases VALUES(?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET "
                "owner=excluded.owner,fencing_token=excluded.fencing_token,"
                "expires_at=excluded.expires_at",
                (job_id, owner, token, instant + ttl_seconds),
            )
            return dict(
                connection.execute("SELECT * FROM leases WHERE job_id=?", (job_id,)).fetchone()
            )

    @staticmethod
    def _ttl(ttl_seconds):
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise MaintenanceStoreError("INVALID_LEASE_TTL")

    def renew_lease(self, job_id, owner, fencing_token, ttl_seconds, *, now=None) -> dict:
        instant = _instant(now)
        self._ttl(ttl_seconds)
        if not math.isfinite(instant + ttl_seconds):
            raise MaintenanceStoreError("INVALID_LEASE_TTL")
        with self._tx() as connection:
            self._assert_lease(connection, job_id, owner, fencing_token, instant)
            connection.execute(
                "UPDATE leases SET expires_at=? WHERE job_id=?", (instant + ttl_seconds, job_id)
            )
            return dict(
                connection.execute("SELECT * FROM leases WHERE job_id=?", (job_id,)).fetchone()
            )

    def release_lease(self, job_id, owner, fencing_token, *, now=None) -> dict:
        instant = _instant(now)
        with self._tx() as connection:
            self._assert_lease(connection, job_id, owner, fencing_token, instant)
            connection.execute("UPDATE leases SET expires_at=? WHERE job_id=?", (instant, job_id))
            return dict(
                connection.execute("SELECT * FROM leases WHERE job_id=?", (job_id,)).fetchone()
            )

    def transition_job(
        self, job_id, expected_status, new_status, owner, fencing_token, details=None, *, now=None
    ) -> dict:
        _text(new_status)
        with self._tx() as connection:
            self._assert_lease(connection, job_id, owner, fencing_token, now)
            cursor = connection.execute(
                "UPDATE jobs SET status=?,details=?,updated_at=? WHERE id=? AND status=?",
                (
                    new_status,
                    _json({} if details is None else details),
                    _instant(now),
                    job_id,
                    expected_status,
                ),
            )
            if cursor.rowcount != 1:
                raise MaintenanceStoreError("JOB_STATUS_CONFLICT")
            return self._row(self._require_job(connection, job_id))

    def _save_immutable(self, table, object_id, payload, code) -> dict:
        _text(object_id)
        encoded = _json(payload)
        with self._tx() as connection:
            row = connection.execute(f"SELECT * FROM {table} WHERE id=?", (object_id,)).fetchone()
            if row is not None:
                if row["payload"] != encoded:
                    raise MaintenanceStoreError(code)
                return self._row(row)
            connection.execute(
                f"INSERT INTO {table} VALUES(?,?,?,?)",
                (object_id, encoded, _sha(encoded), time.time()),
            )
            return self._row(
                connection.execute(f"SELECT * FROM {table} WHERE id=?", (object_id,)).fetchone()
            )

    def _get_immutable(self, table, object_id) -> dict | None:
        with self._tx() as connection:
            return self._row(
                connection.execute(f"SELECT * FROM {table} WHERE id=?", (object_id,)).fetchone()
            )

    def save_policy(self, policy_id, payload) -> dict:
        return self._save_immutable("policies", policy_id, payload, "IMMUTABLE_POLICY")

    def get_policy(self, policy_id) -> dict | None:
        return self._get_immutable("policies", policy_id)

    def save_artifact(self, artifact_id, payload) -> dict:
        return self._save_immutable("artifacts", artifact_id, payload, "IMMUTABLE_ARTIFACT")

    def get_artifact(self, artifact_id) -> dict | None:
        return self._get_immutable("artifacts", artifact_id)
