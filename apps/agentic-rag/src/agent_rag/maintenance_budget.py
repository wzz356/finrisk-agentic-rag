"""Persistent batch budget shared by every model role and paid tool.

Money is integral millionths of CNY (1 CNY = 1_000_000 micro). Reservation
counts are never reset by cancellation, resume, cache replay or new instances.
The caller must perform mark_sent before dispatch and configure SDK retries=0;
a crash after that boundary deliberately forbids automatic resend.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

SDK_REQUIRED_MAX_RETRIES = 0
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class BudgetError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _money(value):
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise BudgetError("INVALID_MONEY")
    return value


def _count(value):
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise BudgetError("INVALID_REQUEST_LIMIT")
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise BudgetError("INVALID_REQUEST_IDENTIFIER")
    return value


def _provider(value):
    if not isinstance(value, dict):
        raise BudgetError("INVALID_PROVIDER_CONFIG")
    if type(value.get("max_retries", 0)) is not int or value.get("max_retries", 0) != 0:
        raise BudgetError("SDK_RETRIES_FORBIDDEN")
    return value


_SCHEMA = """
CREATE TABLE IF NOT EXISTS batch (
 id INTEGER PRIMARY KEY CHECK(id=1), config TEXT NOT NULL,
 circuit_broken INTEGER NOT NULL DEFAULT 0, circuit_reason TEXT, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
 ticket_id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
 payload_sha256 TEXT NOT NULL, role TEXT NOT NULL, event_id TEXT NOT NULL, stage TEXT NOT NULL,
 reserved_micro INTEGER NOT NULL, provider_config TEXT NOT NULL,
 state TEXT NOT NULL, actual_micro INTEGER, response TEXT,
 created_at REAL NOT NULL, sent_at REAL, settled_at REAL
);
CREATE TRIGGER IF NOT EXISTS immutable_request_binding BEFORE UPDATE OF
 ticket_id,request_key,payload_sha256,role,event_id,stage,reserved_micro,provider_config ON requests
 BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_REQUEST_BINDING'); END;
CREATE TRIGGER IF NOT EXISTS immutable_request_delete BEFORE DELETE ON requests
 BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_REQUEST'); END;
CREATE TRIGGER IF NOT EXISTS immutable_batch_config BEFORE UPDATE OF config ON batch
 BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_BATCH_CONFIG'); END;
"""


class BudgetLedger:
    def __init__(
        self,
        path: str | Path,
        policy_id: str,
        cap_micro: int | None,
        max_requests: int,
        *,
        live=False,
        price_policy=None,
        provider_config=None,
        limits=None,
    ):
        if type(live) is not bool:
            raise BudgetError("INVALID_LIVE_MODE")
        if live and (type(cap_micro) is not int or cap_micro <= 0):
            raise BudgetError("LIVE_CAP_REQUIRED")
        _money(cap_micro)
        _count(max_requests)
        _text(policy_id)
        config_provider = _provider({} if provider_config is None else provider_config)
        parsed_limits = self._validate_limits({} if limits is None else limits)
        if live:
            self._verify_price_policy(policy_id, price_policy)
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "policy_id": policy_id,
            "cap_micro": cap_micro,
            "max_requests": max_requests,
            "live": live,
            "price_policy": price_policy,
            "provider_config": config_provider,
            "limits": parsed_limits,
        }
        encoded = _json(config)
        self._config_json = encoded
        with closing(self._connect()) as connection:
            connection.executescript(_SCHEMA)
        with self._tx() as connection:
            row = connection.execute("SELECT config FROM batch WHERE id=1").fetchone()
            if row is not None and row["config"] != encoded:
                raise BudgetError("BATCH_CONFIG_CONFLICT")
            if row is None:
                connection.execute(
                    "INSERT INTO batch(id,config,created_at) VALUES(1,?,?)", (encoded, time.time())
                )

    @property
    def config(self) -> dict:
        """Return a snapshot so caller mutation cannot alter durable limits."""
        return json.loads(self._config_json)

    @staticmethod
    def _verify_price_policy(policy_id, policy):
        if not isinstance(policy, dict):
            raise BudgetError("PRICE_POLICY_REQUIRED")
        if (
            policy.get("policy_id") != policy_id
            or policy.get("verified") is not True
            or policy.get("currency") != "CNY"
            or policy.get("unit") != "micro_cny"
            or not isinstance(policy.get("rates"), dict)
            or not policy["rates"]
            or not isinstance(policy.get("source_path"), str)
            or not _HASH.fullmatch(str(policy.get("source_sha256", "")))
        ):
            raise BudgetError("PRICE_POLICY_UNVERIFIED")
        for value in policy["rates"].values():
            _money(value)
        if not any(value > 0 for value in policy["rates"].values()):
            raise BudgetError("PRICE_POLICY_UNVERIFIED")
        try:
            raw = Path(policy["source_path"]).read_bytes()
            source = json.loads(raw.decode("utf-8-sig"))
            valid = hashlib.sha256(raw).hexdigest() == policy["source_sha256"] and all(
                source.get(key) == policy.get(key)
                for key in ("policy_id", "currency", "unit", "rates")
            )
        except (OSError, ValueError, AttributeError):
            valid = False
        if not valid:
            raise BudgetError("PRICE_POLICY_UNVERIFIED")

    @staticmethod
    def _validate_limits(limits):
        if not isinstance(limits, dict) or set(limits) - {"role", "stage", "event"}:
            raise BudgetError("INVALID_DIMENSION_LIMITS")
        for buckets in limits.values():
            if not isinstance(buckets, dict):
                raise BudgetError("INVALID_DIMENSION_LIMITS")
            for key, value in buckets.items():
                _text(key)
                if not isinstance(value, dict) or set(value) - {"max_requests", "cap_micro"}:
                    raise BudgetError("INVALID_DIMENSION_LIMITS")
                if "max_requests" in value:
                    _count(value["max_requests"])
                if "cap_micro" in value:
                    _money(value["cap_micro"])
        return limits

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
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
    def _row(row, *, cache_hit=False):
        result = dict(row)
        result["provider_config"] = json.loads(result["provider_config"])
        result["response"] = (
            json.loads(result["response"]) if result["response"] is not None else None
        )
        result["cache_hit"] = cache_hit
        return result

    @staticmethod
    def _effective(row):
        if row["state"] == "released":
            return 0
        return row["actual_micro"] if row["state"] == "settled" else row["reserved_micro"]

    @classmethod
    def _totals(cls, rows):
        return {
            "request_count": len(rows),
            "spent_micro": sum(row["actual_micro"] for row in rows if row["state"] == "settled"),
            "reserved_micro": sum(
                row["reserved_micro"]
                for row in rows
                if row["state"] in {"reserved", "sent", "unknown"}
            ),
            "sent_micro": sum(row["reserved_micro"] for row in rows if row["state"] == "sent"),
            "unknown_micro": sum(
                row["reserved_micro"] for row in rows if row["state"] == "unknown"
            ),
            "charged_micro": sum(cls._effective(row) for row in rows),
        }

    @staticmethod
    def _open(connection):
        if connection.execute("SELECT circuit_broken FROM batch WHERE id=1").fetchone()[0]:
            raise BudgetError("BATCH_CIRCUIT_OPEN")

    def reserve(
        self,
        request_key,
        payload_sha256,
        role,
        event_id,
        stage,
        reserved_micro,
        *,
        provider_config=None,
    ) -> dict:
        for value in (request_key, role, event_id, stage):
            _text(value)
        if not isinstance(payload_sha256, str) or not _HASH.fullmatch(payload_sha256):
            raise BudgetError("INVALID_PAYLOAD_HASH")
        _money(reserved_micro)
        if self.config["live"] and reserved_micro == 0:
            raise BudgetError("LIVE_RESERVATION_REQUIRED")
        provider = _json(
            _provider(
                self.config["provider_config"] if provider_config is None else provider_config
            )
        )
        binding = (payload_sha256, role, event_id, stage, reserved_micro, provider)
        with self._tx() as connection:
            row = connection.execute(
                "SELECT * FROM requests WHERE request_key=?", (request_key,)
            ).fetchone()
            if row is not None:
                if (
                    tuple(
                        row[name]
                        for name in (
                            "payload_sha256",
                            "role",
                            "event_id",
                            "stage",
                            "reserved_micro",
                            "provider_config",
                        )
                    )
                    != binding
                ):
                    raise BudgetError("REQUEST_BINDING_CONFLICT")
                if row["state"] in {"unknown", "sent", "released"}:
                    raise BudgetError(
                        {
                            "unknown": "REQUEST_OUTCOME_UNKNOWN",
                            "sent": "REQUEST_IN_FLIGHT",
                            "released": "REQUEST_RELEASED",
                        }[row["state"]]
                    )
                return self._row(row, cache_hit=row["state"] == "settled")
            self._open(connection)
            rows = connection.execute("SELECT * FROM requests").fetchall()
            totals = self._totals(rows)
            if totals["charged_micro"] + reserved_micro > self.config["cap_micro"]:
                raise BudgetError("BUDGET_EXCEEDED")
            if totals["request_count"] + 1 > self.config["max_requests"]:
                raise BudgetError("REQUEST_LIMIT_EXCEEDED")
            for dimension, column, value in [
                ("role", "role", role),
                ("stage", "stage", stage),
                ("event", "event_id", event_id),
            ]:
                buckets = self.config["limits"].get(dimension, {})
                limit = buckets.get(value, buckets.get("*", {}))
                usage = self._totals([item for item in rows if item[column] == value])
                if (
                    "max_requests" in limit and usage["request_count"] + 1 > limit["max_requests"]
                ) or (
                    "cap_micro" in limit
                    and usage["charged_micro"] + reserved_micro > limit["cap_micro"]
                ):
                    raise BudgetError(f"{dimension.upper()}_LIMIT_EXCEEDED")
            ticket_id = uuid4().hex
            connection.execute(
                """INSERT INTO requests(ticket_id,request_key,payload_sha256,role,event_id,stage,
                reserved_micro,provider_config,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (ticket_id, request_key, *binding, "reserved", time.time()),
            )
            return self._row(
                connection.execute(
                    "SELECT * FROM requests WHERE ticket_id=?", (ticket_id,)
                ).fetchone()
            )

    @staticmethod
    def _ticket(connection, ticket):
        ticket_id = ticket.get("ticket_id") if isinstance(ticket, dict) else ticket
        if not isinstance(ticket_id, str):
            raise BudgetError("INVALID_TICKET")
        row = connection.execute(
            "SELECT * FROM requests WHERE ticket_id=?", (ticket_id,)
        ).fetchone()
        if row is None:
            raise BudgetError("REQUEST_NOT_FOUND")
        return row

    def mark_sent(self, ticket) -> dict:
        with self._tx() as connection:
            row = self._ticket(connection, ticket)
            if row["state"] != "reserved":
                raise BudgetError(
                    "REQUEST_RELEASED" if row["state"] == "released" else "REQUEST_ALREADY_SENT"
                )
            self._open(connection)
            connection.execute(
                "UPDATE requests SET state=?,sent_at=? WHERE ticket_id=? AND state=?",
                ("sent", time.time(), row["ticket_id"], "reserved"),
            )
            return self._row(self._ticket(connection, row["ticket_id"]))

    def settle(self, ticket, actual_micro: int | None, response=None) -> dict:
        if actual_micro is not None:
            _money(actual_micro)
        encoded = _json(response) if response is not None else None
        overrun = False
        with self._tx() as connection:
            row = self._ticket(connection, ticket)
            if row["state"] == "settled":
                if row["actual_micro"] != actual_micro or row["response"] != encoded:
                    raise BudgetError("SETTLEMENT_CONFLICT")
                return self._row(row, cache_hit=True)
            if row["state"] not in {"sent", "unknown"}:
                raise BudgetError("REQUEST_NOT_SENT")
            state = "unknown" if actual_micro is None else "settled"
            connection.execute(
                "UPDATE requests SET state=?,actual_micro=?,response=?,settled_at=? "
                "WHERE ticket_id=?",
                (state, actual_micro, encoded, time.time(), row["ticket_id"]),
            )
            if actual_micro is not None and actual_micro > row["reserved_micro"]:
                overrun = True
                connection.execute(
                    "UPDATE batch SET circuit_broken=1,circuit_reason=? WHERE id=1",
                    ("PRICE_RESERVATION_EXCEEDED",),
                )
            result = self._row(self._ticket(connection, row["ticket_id"]))
        if overrun:
            raise BudgetError("PRICE_RESERVATION_EXCEEDED")
        return result

    def release(self, ticket) -> dict:
        with self._tx() as connection:
            row = self._ticket(connection, ticket)
            if row["state"] not in {"reserved", "released"}:
                raise BudgetError("REQUEST_ALREADY_SENT")
            connection.execute(
                "UPDATE requests SET state=? WHERE ticket_id=?", ("released", row["ticket_id"])
            )
            return self._row(self._ticket(connection, row["ticket_id"]))

    def get_request(self, request_key) -> dict | None:
        with self._tx() as connection:
            row = connection.execute(
                "SELECT * FROM requests WHERE request_key=?", (request_key,)
            ).fetchone()
            return self._row(row, cache_hit=row["state"] == "settled") if row is not None else None

    def summary(self) -> dict:
        with self._tx() as connection:
            rows = connection.execute(
                "SELECT * FROM requests ORDER BY created_at,ticket_id"
            ).fetchall()
            batch = connection.execute("SELECT * FROM batch WHERE id=1").fetchone()
            result = self._totals(rows)
            result.update(
                {
                    "policy_id": self.config["policy_id"],
                    "live": self.config["live"],
                    "cap_micro": self.config["cap_micro"],
                    "max_requests": self.config["max_requests"],
                    "available_micro": self.config["cap_micro"] - result["charged_micro"],
                    "circuit_broken": bool(batch["circuit_broken"]),
                    "circuit_reason": batch["circuit_reason"],
                }
            )
            for dimension, column in [("role", "role"), ("stage", "stage"), ("event", "event_id")]:
                result[f"by_{dimension}"] = {
                    key: self._totals([row for row in rows if row[column] == key])
                    for key in sorted({row[column] for row in rows})
                }
            return result
