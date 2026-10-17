"""Opt-in case generation with a separate durable, fail-closed budget ledger."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any, NoReturn

from agent_rag.rerank_backend import DeepSeekReranker, _checked_usage, _TransportResult, _usage_cost

_PURPOSE = "case_generation"
_VERSION = 1
_SECRET = re.compile(r"sk-[A-Za-z0-9_-]{16,}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _safe_audit(value: Any, secret: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            _safe_audit(str(key), secret): _safe_audit(item, secret)
            for key, item in value.items()
            if str(key).strip().lower()
            not in {"authorization", "proxy-authorization", "api_key", "api-key"}
        }
    if isinstance(value, list):
        return [_safe_audit(item, secret) for item in value]
    if isinstance(value, str):
        return _SECRET.sub("[REDACTED]", value.replace(secret, "[REDACTED]") if secret else value)
    return value


def _reject_constant(value: str) -> NoReturn:
    raise ValueError("INVALID_JSON_CONSTANT")


class UpdateModelClient:
    """Every request key permits at most one transport dispatch across processes.

    ``transport`` follows the low-level rerank signature ``(payload, timeout)``.
    A valid completion with unknown usage is reusable, but its reservation remains.
    A failed or interrupted dispatch is never retried under its existing key.
    """

    def __init__(
        self,
        ledger_path: Path,
        cap_cny: float = 1,
        max_requests: int = 6,
        key_file: Path | None = None,
        transport: Callable[[dict[str, Any], float], Any] | None = None,
    ) -> None:
        if (
            isinstance(cap_cny, bool)
            or not isinstance(cap_cny, (int, float))
            or not math.isfinite(cap_cny)
            or cap_cny <= 0
            or type(max_requests) is not int
            or not 0 < max_requests < 2**63
            or (transport is not None and not callable(transport))
        ):
            raise ValueError("UPDATE_MODEL_INVALID_CONFIGURATION")
        self._cap_micro = int(
            (Decimal(str(cap_cny)) * 1_000_000).to_integral_value(rounding=ROUND_FLOOR)
        )
        if not 0 < self._cap_micro < 2**63:
            raise ValueError("UPDATE_MODEL_INVALID_CONFIGURATION")
        self.ledger_path = Path(ledger_path)
        self._max_requests = max_requests
        self._key_file = Path(key_file) if key_file is not None else None
        self._transport = transport
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise RuntimeError("UPDATE_MODEL_LEDGER_INVALID") from None
        self._initialize()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            connection = sqlite3.connect(self.ledger_path, timeout=30, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except (sqlite3.Error, OSError, OverflowError):
            raise RuntimeError("UPDATE_MODEL_LEDGER_INVALID") from None
        finally:
            if connection is not None:
                connection.close()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if tables and tables != {"policy", "requests"}:
                raise RuntimeError("UPDATE_MODEL_LEDGER_INVALID")
            if not tables:
                connection.execute("""CREATE TABLE policy (
                    id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL,
                    purpose TEXT NOT NULL, cap_micro INTEGER NOT NULL,
                    max_requests INTEGER NOT NULL, halt_reason TEXT)""")
                connection.execute("""CREATE TABLE requests (
                    request_key TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL,
                    request_json TEXT NOT NULL, reserved_micro INTEGER NOT NULL,
                    state TEXT NOT NULL, sent_at TEXT NOT NULL, completed_at TEXT,
                    usage_json TEXT, cost_micro INTEGER NOT NULL DEFAULT 0,
                    response_json TEXT, content TEXT, error_code TEXT)""")
                connection.execute(
                    "INSERT INTO policy VALUES (1, ?, ?, ?, ?, NULL)",
                    (_VERSION, _PURPOSE, self._cap_micro, self._max_requests),
                )
            policy = connection.execute("SELECT * FROM policy WHERE id=1").fetchone()
            if policy is None or policy["version"] != _VERSION or policy["purpose"] != _PURPOSE:
                raise RuntimeError("UPDATE_MODEL_LEDGER_INVALID")
            if (
                policy["cap_micro"] != self._cap_micro
                or policy["max_requests"] != self._max_requests
            ):
                raise RuntimeError("UPDATE_MODEL_POLICY_MISMATCH")

    @staticmethod
    def _existing(connection: sqlite3.Connection, request_key: str, digest: str) -> str | None:
        row = connection.execute(
            "SELECT * FROM requests WHERE request_key=?", (request_key,)
        ).fetchone()
        if row is None:
            return None
        if row["payload_sha256"] != digest:
            raise RuntimeError("UPDATE_MODEL_REQUEST_KEY_CONFLICT")
        content = row["content"]
        if row["state"] == "succeeded" and isinstance(content, str):
            return content
        raise RuntimeError("UPDATE_MODEL_REQUEST_ALREADY_SENT")

    def _load_key(self) -> str:
        try:
            if self._key_file is not None:
                keys: set[str] = set(
                    _SECRET.findall(self._key_file.read_text(encoding="utf-8-sig"))
                )
                if len(keys) != 1:
                    raise ValueError("AMBIGUOUS_KEY")
                key = next(iter(keys))
            else:
                key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if _SECRET.fullmatch(key) is None:
                raise ValueError("INVALID_KEY")
            return key
        except Exception:
            raise RuntimeError("UPDATE_MODEL_KEY_UNAVAILABLE") from None

    def _reserve(
        self, payload: dict[str, Any], request_key: str, digest: str, request_json: str
    ) -> str | None:
        reserved = (len(request_json.encode("utf-8")) + 512) * 2 + payload["max_tokens"] * 8
        with self._transaction() as connection:
            cached = self._existing(connection, request_key, digest)
            if cached is not None:
                return cached
            policy = connection.execute("SELECT * FROM policy WHERE id=1").fetchone()
            if policy["halt_reason"]:
                raise RuntimeError("UPDATE_MODEL_BUDGET_HALTED")
            totals = connection.execute("""SELECT COUNT(*) AS requests,
                COALESCE(SUM(cost_micro), 0) AS settled,
                COALESCE(SUM(CASE WHEN usage_json IS NULL
                    THEN reserved_micro ELSE 0 END), 0) AS reserved
                FROM requests""").fetchone()
            if totals["requests"] >= policy["max_requests"]:
                raise RuntimeError("UPDATE_MODEL_REQUEST_LIMIT")
            if totals["settled"] + totals["reserved"] + reserved > policy["cap_micro"]:
                raise RuntimeError("UPDATE_MODEL_BUDGET_EXHAUSTED")
            # Commit this durable claim before entering any callable that may send.
            connection.execute(
                """INSERT INTO requests
                (request_key, payload_sha256, request_json, reserved_micro, state, sent_at)
                VALUES (?, ?, ?, ?, 'sent', ?)""",
                (request_key, digest, _json(_safe_audit(payload)), reserved, _now()),
            )
        return None

    def _settle(self, request_key: str, response: Any) -> bool:
        usage = _checked_usage(response.get("usage") if isinstance(response, dict) else None)
        cost = _usage_cost(usage) if usage is not None else 0
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM requests WHERE request_key=?", (request_key,)
            ).fetchone()
            exceeded = bool(cost > row["reserved_micro"])
            connection.execute(
                "UPDATE requests SET usage_json=?, cost_micro=? WHERE request_key=?",
                (_json(usage) if usage is not None else None, cost, request_key),
            )
            if exceeded:
                connection.execute(
                    "UPDATE policy SET halt_reason='usage_exceeded_reservation' WHERE id=1"
                )
        return exceeded

    def _record_response(self, request_key: str, response: Any) -> None:
        try:
            audit = _json(response)
        except (TypeError, ValueError, RecursionError):
            audit = '{"unserializable_response": true}'
        with self._transaction() as connection:
            connection.execute(
                "UPDATE requests SET response_json=? WHERE request_key=?", (audit, request_key)
            )

    def _finish(
        self, request_key: str, *, content: str | None = None, error_code: str | None = None
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """UPDATE requests SET state=?, content=?, error_code=?, completed_at=?
                WHERE request_key=?""",
                (
                    "succeeded" if content is not None else "failed",
                    content,
                    error_code,
                    _now(),
                    request_key,
                ),
            )

    @staticmethod
    def _validate(response: Any) -> str:
        if not isinstance(response, dict) or response.get("model") not in {
            "deepseek-flash",
            "deepseek-v4.1-flash",
        }:
            raise ValueError("INVALID_RESPONSE")
        choices = response.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") != "stop"
        ):
            raise ValueError("INVALID_RESPONSE")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not isinstance(
            json.loads(content, parse_constant=_reject_constant), dict
        ):
            raise ValueError("INVALID_RESPONSE")
        return content

    def complete(
        self,
        system: str,
        user: str,
        *,
        request_key: str,
        max_tokens: int = 4096,
        replay_only: bool = False,
    ) -> str:
        if (
            any(
                not isinstance(value, str) or not value.strip()
                for value in (system, user, request_key)
            )
            or type(max_tokens) is not int
            or max_tokens <= 0
            or type(replay_only) is not bool
        ):
            raise ValueError("UPDATE_MODEL_INVALID_INPUT")
        payload = {
            "model": "deepseek-flash",
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        request_json = _json(payload)
        digest = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        with self._transaction() as connection:
            cached = self._existing(connection, request_key, digest)
        if cached is not None:
            return cached
        if replay_only:
            raise RuntimeError("UPDATE_MODEL_REPLAY_NOT_AVAILABLE")
        key = self._load_key() if self._transport is None else ""
        cached = self._reserve(payload, request_key, digest, request_json)
        if cached is not None:
            return cached
        try:
            response = (
                self._transport(payload, 30.0)
                if self._transport is not None
                else DeepSeekReranker._request(payload, 30.0, key)
            )
        except Exception:
            self._finish(request_key, error_code="UPDATE_MODEL_TRANSPORT_FAILED")
            raise RuntimeError("UPDATE_MODEL_TRANSPORT_FAILED") from None
        if isinstance(response, _TransportResult):
            response = response.value
        # Usage is settled independently of envelope, model, finish state or content validity.
        exceeded = self._settle(request_key, response)
        try:
            safe_response = _safe_audit(response, key)
        except (TypeError, ValueError, RecursionError):
            safe_response = {"unserializable_response": True}
        self._record_response(request_key, safe_response)
        if exceeded:
            self._finish(request_key, error_code="UPDATE_MODEL_USAGE_EXCEEDED_RESERVATION")
            raise RuntimeError("UPDATE_MODEL_USAGE_EXCEEDED_RESERVATION")
        try:
            content = self._validate(safe_response)
        except (TypeError, ValueError, RecursionError):
            self._finish(request_key, error_code="UPDATE_MODEL_INVALID_RESPONSE")
            raise RuntimeError("UPDATE_MODEL_INVALID_RESPONSE") from None
        self._finish(request_key, content=content)
        return content

    def summary(self) -> dict[str, Any]:
        with self._transaction() as connection:
            policy = connection.execute("SELECT * FROM policy WHERE id=1").fetchone()
            totals = connection.execute("""SELECT COUNT(*) AS requests,
                COALESCE(SUM(cost_micro), 0) AS settled,
                COALESCE(SUM(CASE WHEN usage_json IS NULL
                    THEN reserved_micro ELSE 0 END), 0) AS reserved,
                COALESCE(SUM(CASE WHEN usage_json IS NULL THEN 1 ELSE 0 END), 0) AS unknown,
                COALESCE(SUM(CASE WHEN state='succeeded' THEN 1 ELSE 0 END), 0)
                    AS successful FROM requests""").fetchone()
            return {
                "purpose": _PURPOSE,
                "cap_cny": policy["cap_micro"] / 1_000_000,
                "max_requests": policy["max_requests"],
                "requests": totals["requests"],
                "successful_requests": totals["successful"],
                "usage_unknown_requests": totals["unknown"],
                "settled_cost_cny": totals["settled"] / 1_000_000,
                "reserved_cost_cny": totals["reserved"] / 1_000_000,
                "remaining_cny": max(
                    0, policy["cap_micro"] - totals["settled"] - totals["reserved"]
                )
                / 1_000_000,
                "halt_reason": policy["halt_reason"],
            }
