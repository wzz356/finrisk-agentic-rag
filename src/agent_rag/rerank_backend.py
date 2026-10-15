"""Controlled pairwise DeepSeek Flash reranking; no network call on construction."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
PRICE_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
PRICE_CHECKED_DATE = "2026-10-15"
RANGES = {
    "specific_direct": (80, 100),
    "concise_direct": (60, 79),
    "partial": (20, 59),
    "topic_only": (1, 19),
    "none": (0, 0),
    "insufficient": (0, 0),
}
_HTTP_SLOTS = threading.BoundedSemaphore(8)
_LEDGER_LOCKS: dict[str, threading.RLock] = {}
_LEDGER_LOCKS_GUARD = threading.Lock()
_ATTEMPT_ERROR_CODES = {
    "HTTP_429", "HTTP_5XX", "HTTP_4XX", "HTTP_ERROR", "TRANSPORT_TIMEOUT",
    "TRANSPORT_ERROR", "INVALID_JSON", "INVALID_COMPLETION", "QUOTE_MISMATCH",
    "CANCELLED", "TASK_TIMEOUT", "STOPPED", "AUDIT_WRITE_FAILED",
}
_ATTEMPT_TERMINAL_STATES = {"settled", "finished_usage_unknown", "aborted_unsent"}
_PROMPT_SUFFIX = (
    "\n每次仅评下面一对，输出一个JSON对象，不用代码围栏。"
    "reason和limitation各不超过50字，保持具体；"
    "只取所需短连续原文quote，保留原换行。"
)


def payload(packet: dict, rules: str, feedback: str | None = None) -> dict:
    result = {
        "model": MODEL,
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "stream": False,
        "messages": [
            {"role": "system", "content": rules + _PROMPT_SUFFIX},
            {
                "role": "user",
                "content": json.dumps(
                    {k: packet[k] for k in ("pair_id", "query_text", "unit_text")},
                    ensure_ascii=False,
                ),
            },
        ],
    }
    if feedback:
        result["messages"].append({"role": "user", "content": feedback})
    return result


def validate_completion(packet: dict, response: dict) -> dict:
    """Preserve the experiment's score and unique whitespace restoration rules."""
    if not isinstance(response, dict) or response.get("model") not in (
        MODEL,
        "deepseek-v4.1-flash",
    ):
        raise ValueError("服务返回了未核实的模型ID")
    choices = response.get("choices", [])
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") != "stop"
    ):
        raise ValueError("API响应未正常完成，拒绝截断或其他结束状态")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("响应content不是字符串")
    opinion = json.loads(content)
    if not isinstance(opinion, dict) or opinion.get("pair_id") != packet["pair_id"]:
        raise ValueError("返回评分对象或pair_id不匹配")
    score, level = opinion.get("score"), opinion.get("level")
    if (
        type(score) is not int
        or not isinstance(level, str)
        or level not in RANGES
        or not RANGES[level][0] <= score <= RANGES[level][1]
    ):
        raise ValueError("整数评分与等级不一致")
    quote = raw_quote = opinion.get("source_quote")
    source = packet["unit_text"]
    if isinstance(quote, str) and quote and quote not in source:
        positions = [i for i, character in enumerate(source) if not character.isspace()]
        compact = "".join(source[i] for i in positions)
        needle = "".join(character for character in quote if not character.isspace())
        start = compact.find(needle) if needle else -1
        if start >= 0 and compact.find(needle, start + 1) < 0:
            quote = source[positions[start] : positions[start + len(needle) - 1] + 1]
            opinion["source_quote"] = quote
    if not isinstance(quote, str) or (quote and quote not in source) or (score >= 60 and not quote):
        raise ValueError("引文未严格绑定原始单元文本")
    if any(
        not isinstance(opinion.get(k), str) or not opinion[k].strip()
        for k in ("reason", "limitation")
    ):
        raise ValueError("缺少理由或限制说明")
    if opinion.get("human_approved") is not False:
        raise ValueError("human_approved必须为false")
    result = {
        k: opinion[k]
        for k in (
            "pair_id",
            "score",
            "level",
            "source_quote",
            "reason",
            "limitation",
            "human_approved",
        )
    }
    if raw_quote != quote:
        result.update(model_source_quote_raw=raw_quote, quote_whitespace_restored=True)
    return result


def _nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _checked_usage(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    prompt, completion, total = (
        raw.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    if (
        not all(_nonnegative_integer(n) for n in (prompt, completion, total))
        or prompt + completion != total
    ):
        return None
    hit, miss = 0, prompt
    if "prompt_cache_hit_tokens" in raw or "prompt_cache_miss_tokens" in raw:
        hit, miss = raw.get("prompt_cache_hit_tokens"), raw.get("prompt_cache_miss_tokens")
        if not all(_nonnegative_integer(n) for n in (hit, miss)) or hit + miss != prompt:
            return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
    }


def _usage_cost(usage: dict) -> int:
    # Frozen peak CNY rates, exactly as agent_harness/src/trial_budget.ts.
    return (
        (usage["prompt_cache_hit_tokens"] + 24) // 25
        + usage["prompt_cache_miss_tokens"] * 2
        + usage["completion_tokens"] * 8
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RerankStopped(RuntimeError):
    pass


class RerankAuditError(RuntimeError):
    pass


class RerankTransportError(RuntimeError):
    def __init__(self, message: str, http_status: int | None, error_code: str) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.error_code = error_code


class _TransportResult:
    def __init__(self, value: Any, http_status: int | None) -> None:
        self.value = value
        self.http_status = http_status


def _http_error_code(status: int) -> str:
    if status == 429:
        return "HTTP_429"
    if 500 <= status <= 599:
        return "HTTP_5XX"
    return "HTTP_4XX" if 400 <= status <= 499 else "HTTP_ERROR"


def _attempt_error(error: Exception, *, validating: bool = False) -> tuple[int | None, str]:
    if isinstance(error, RerankTransportError):
        return error.http_status, error.error_code
    if isinstance(error, urllib.error.HTTPError):
        return error.code, _http_error_code(error.code)
    if isinstance(error, TimeoutError) or (
        isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError)
    ):
        return None, "TRANSPORT_TIMEOUT"
    if isinstance(error, json.JSONDecodeError):
        return None, "INVALID_JSON"
    if validating:
        return None, (
            "QUOTE_MISMATCH"
            if isinstance(error, ValueError) and str(error) == "引文未严格绑定原始单元文本"
            else "INVALID_COMPLETION"
        )
    return None, "TRANSPORT_ERROR"


class RerankTaskControl:
    """A task's durable budget and deadline, shared by every physical attempt.

    The ledger lock and HTTP semaphore are shared across the service process.
    Restarting a control reuses its recorded limits, counts, costs and deadline.
    """

    def __init__(
        self,
        ledger_path: Path,
        cap_cny: float,
        max_requests: int,
        timeout_seconds: float = 180,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if (
            isinstance(cap_cny, bool)
            or not isinstance(cap_cny, (int, float))
            or not math.isfinite(cap_cny)
            or cap_cny <= 0
            or type(max_requests) is not int
            or max_requests <= 0
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("INVALID_RERANK_LIMITS")
        self.ledger_path = Path(ledger_path).resolve()
        self.attempt_audit_path = self.ledger_path.with_name(
            self.ledger_path.name + ".attempts.json"
        )
        self._retry_indices: dict[int, int] = {}
        self.cap_micro_cny = int(
            (Decimal(str(cap_cny)) * 1_000_000).to_integral_value(rounding=ROUND_FLOOR)
        )
        if self.cap_micro_cny <= 0:
            raise ValueError("INVALID_RERANK_LIMITS")
        self.max_requests = max_requests
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        with _LEDGER_LOCKS_GUARD:
            self._lock = _LEDGER_LOCKS.setdefault(
                os.path.normcase(str(self.ledger_path)), threading.RLock()
            )
        with self._lock:
            if self.ledger_path.exists():
                ledger = self._read()
            else:
                ledger = {
                    "schema_version": "finrisk-rerank-budget/1.1",
                    "model": MODEL,
                    "cap_micro_cny": self.cap_micro_cny,
                    "cap_cny": self.cap_micro_cny / 1_000_000,
                    "max_requests": max_requests,
                    "deadline_epoch_seconds": time.time() + timeout_seconds,
                    "price_source": PRICE_SOURCE,
                    "price_checked_date": PRICE_CHECKED_DATE,
                    "peak_cny_per_million": {
                        "input_cache_hit": 0.04,
                        "input_cache_miss": 2,
                        "output": 8,
                    },
                    "prompt_overhead_token_allowance": 512,
                    "reservation_kind": (
                        "UTF8_request_body_bytes_plus512_prompt_bound; "
                        "max_tokens_output; peak_price"
                    ),
                    "request_reservations": [],
                    "reserved_cost_micro_cny": 0,
                    "settled_cost_micro_cny": 0,
                    "stop_reason": None,
                }
                self._persist(ledger)
            # Monotonic time prevents wall-clock changes extending this run.
            self._deadline = time.monotonic() + max(
                0.0, ledger["deadline_epoch_seconds"] - time.time()
            )

    def _validate(self, ledger: dict) -> None:
        def invalid() -> None:
            raise ValueError("INVALID_EXISTING_RERANK_LEDGER")

        if (
            not isinstance(ledger, dict)
            or ledger.get("schema_version") != "finrisk-rerank-budget/1.1"
            or ledger.get("model") != MODEL
            or ledger.get("cap_micro_cny") != self.cap_micro_cny
            or type(ledger.get("cap_micro_cny")) is not int
            or type(ledger.get("max_requests")) is not int
            or ledger.get("max_requests") != self.max_requests
            or ledger.get("cap_cny") != self.cap_micro_cny / 1_000_000
            or ledger.get("price_source") != PRICE_SOURCE
            or ledger.get("price_checked_date") != PRICE_CHECKED_DATE
            or ledger.get("peak_cny_per_million")
            != {"input_cache_hit": 0.04, "input_cache_miss": 2, "output": 8}
            or ledger.get("prompt_overhead_token_allowance") != 512
            or type(ledger.get("deadline_epoch_seconds")) not in (int, float)
            or not math.isfinite(ledger["deadline_epoch_seconds"])
            or ledger.get("stop_reason")
            not in (
                None,
                "cancelled",
                "timeout",
                "budget_exhausted",
                "request_limit",
                "usage_exceeded_reservation",
            )
            or not isinstance(ledger.get("request_reservations"), list)
        ):
            invalid()
        reserved = settled = 0
        for index, row in enumerate(ledger["request_reservations"], 1):
            if (
                not isinstance(row, dict)
                or type(row.get("ordinal")) is not int
                or row.get("ordinal") != index
                or not _nonnegative_integer(row.get("body_bytes"))
                or row.get("max_output_tokens") != 512
                or type(row.get("reserved_micro_cny")) is not int
                or row.get("reserved_micro_cny") != (row["body_bytes"] + 512) * 2 + 512 * 8
                or not isinstance(row.get("pair_id"), str)
                or not isinstance(row.get("reserved_at"), str)
            ):
                invalid()
            state = row.get("dispatch_state")
            committed = isinstance(row.get("dispatch_committed_at"), str)
            started = isinstance(row.get("transport_started_at"), str)
            if (
                state
                not in ("reserved", "dispatch_committed", "transport_started", "settled", "aborted")
                or (state == "reserved" and (committed or started))
                or (state == "dispatch_committed" and (not committed or started))
                or (state in ("transport_started", "settled") and (not committed or not started))
                or (state == "aborted" and (started or not isinstance(row.get("aborted_at"), str)))
            ):
                invalid()
            settlement = row.get("settlement")
            if settlement is None:
                if state == "settled":
                    invalid()
                if state != "aborted":
                    reserved += row["reserved_micro_cny"]
            else:
                usage = (
                    _checked_usage(settlement.get("usage"))
                    if isinstance(settlement, dict)
                    else None
                )
                if (
                    state != "settled"
                    or usage is None
                    or type(settlement.get("cost_micro_cny")) is not int
                    or settlement.get("cost_micro_cny") != _usage_cost(usage)
                    or not isinstance(settlement.get("settled_at"), str)
                ):
                    invalid()
                settled += settlement["cost_micro_cny"]
        if (
            not _nonnegative_integer(ledger.get("reserved_cost_micro_cny"))
            or not _nonnegative_integer(ledger.get("settled_cost_micro_cny"))
            or ledger["reserved_cost_micro_cny"] != reserved
            or ledger["settled_cost_micro_cny"] != settled
        ):
            invalid()
        if self._admission_count(ledger) > self.max_requests:
            invalid()

    def _read(self) -> dict:
        try:
            ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if (
                isinstance(ledger, dict)
                and ledger.get("schema_version") == "finrisk-rerank-budget/1.0"
            ):
                # Earlier ledgers counted every durable row as a physical
                # attempt. Preserve all of those counts and unknown costs.
                rows = ledger.get("request_reservations")
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError("INVALID_EXISTING_RERANK_LEDGER")
                ledger["schema_version"] = "finrisk-rerank-budget/1.1"
                for row in rows:
                    row["dispatch_state"] = (
                        "settled" if row.get("settlement") else "transport_started"
                    )
                    row["dispatch_committed_at"] = row.get("reserved_at")
                    row["transport_started_at"] = row.get("reserved_at")
            self._validate(ledger)
            return ledger
        except (OSError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise ValueError("INVALID_EXISTING_RERANK_LEDGER") from error

    def _persist(self, ledger: dict) -> None:
        self._validate(ledger)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.ledger_path.parent,
                prefix=self.ledger_path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(ledger, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.ledger_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def remaining_seconds(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def _read_attempts(self) -> dict:
        if not self.attempt_audit_path.exists():
            return {"schema_version": "finrisk-rerank-attempts/1.0", "attempts": []}
        try:
            audit = json.loads(self.attempt_audit_path.read_text(encoding="utf-8"))
            if (
                not isinstance(audit, dict)
                or audit.get("schema_version") != "finrisk-rerank-attempts/1.0"
                or not isinstance(audit.get("attempts"), list)
            ):
                raise ValueError("INVALID_EXISTING_RERANK_ATTEMPT_AUDIT")
            ordinals = set()
            for row in audit["attempts"]:
                state = row.get("state") if isinstance(row, dict) else None
                if (
                    not isinstance(row, dict)
                    or type(row.get("ordinal")) is not int or row["ordinal"] < 1
                    or row["ordinal"] in ordinals
                    or type(row.get("retry_index")) is not int or row["retry_index"] not in (0, 1)
                    or state not in _ATTEMPT_TERMINAL_STATES | {"transport_started"}
                    or row.get("error_code") not in _ATTEMPT_ERROR_CODES | {None}
                    or type(row.get("will_retry")) is not bool
                    or row.get("usage_state") != {
                        "settled": "known", "finished_usage_unknown": "unknown",
                        "aborted_unsent": "not_sent", "transport_started": "unknown",
                    }[state]
                    or (state in _ATTEMPT_TERMINAL_STATES)
                    != isinstance(row.get("completed_at"), str)
                    or (row.get("http_status") is not None and (
                        type(row["http_status"]) is not int or not 100 <= row["http_status"] <= 599
                    ))
                ):
                    raise ValueError("INVALID_EXISTING_RERANK_ATTEMPT_AUDIT")
                ordinals.add(row["ordinal"])
            return audit
        except (OSError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise ValueError("INVALID_EXISTING_RERANK_ATTEMPT_AUDIT") from error

    def _persist_attempts(self, audit: dict) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.ledger_path.parent,
                prefix=self.attempt_audit_path.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(audit, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.attempt_audit_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _audit_attempt(
        self, ledger: dict, ordinal: int, state: str, *, http_status: int | None = None,
        error_code: str | None = None, completed_at: str | None = None, will_retry: bool = False,
    ) -> None:
        """Write only diagnostic fields; budget rows retain their existing meaning."""
        try:
            audit = self._read_attempts()
        except ValueError as error:
            raise RerankAuditError("RERANK_ATTEMPT_AUDIT_UNAVAILABLE") from error
        previous = next((r for r in audit["attempts"] if r["ordinal"] == ordinal), None)
        if previous is not None and previous["state"] in _ATTEMPT_TERMINAL_STATES:
            return  # A physical attempt has exactly one immutable terminal record.
        reservation = ledger["request_reservations"][ordinal - 1]
        row = {
            "ordinal": ordinal, "pair_id": reservation["pair_id"],
            "retry_index": self._retry_indices.get(ordinal, 0), "state": state,
            "http_status": http_status, "error_code": error_code,
            "transport_started_at": reservation.get("transport_started_at"),
            "completed_at": completed_at, "will_retry": will_retry,
            "usage_state": {
                "settled": "known", "finished_usage_unknown": "unknown",
                "aborted_unsent": "not_sent", "transport_started": "unknown",
            }[state],
        }
        if previous is not None:
            audit["attempts"].remove(previous)
        audit["attempts"].append(row)
        audit["attempts"].sort(key=lambda item: item["ordinal"])
        try:
            self._persist_attempts(audit)
        except OSError as error:
            raise RerankAuditError("RERANK_ATTEMPT_AUDIT_WRITE_FAILED") from error

    def _finish_attempt(
        self, ordinal: int, *, http_status: int | None, error_code: str | None,
        completed_at: str, will_retry: bool,
    ) -> None:
        with self._lock:
            try:
                ledger = self._read()
                state = (
                    "settled" if ledger["request_reservations"][ordinal - 1].get("settlement")
                    else "finished_usage_unknown"
                )
                self._audit_attempt(
                    ledger, ordinal, state, http_status=http_status, error_code=error_code,
                    completed_at=completed_at, will_retry=will_retry,
                )
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                # A local diagnostic failure cannot authorize another paid score.
                raise RerankAuditError('RERANK_ATTEMPT_AUDIT_UNAVAILABLE') from error

    @staticmethod
    def _admission_count(ledger: dict) -> int:
        return sum(
            row["dispatch_state"] != "aborted" or "dispatch_committed_at" in row
            for row in ledger["request_reservations"]
        )

    def _stop_reason(self, ledger: dict, *, include_limits: bool = True) -> str | None:
        if self.cancel_event.is_set() or ledger["stop_reason"] == "cancelled":
            return "cancelled"
        if self.remaining_seconds() <= 0:
            return "timeout"
        if ledger["stop_reason"] is not None:
            return ledger["stop_reason"]
        if include_limits:
            if self._admission_count(ledger) >= self.max_requests:
                return "request_limit"
            if (
                ledger["reserved_cost_micro_cny"] + ledger["settled_cost_micro_cny"]
                >= self.cap_micro_cny
            ):
                return "budget_exhausted"
        return None

    def check(self) -> bool:
        with self._lock:
            return self._stop_reason(self._read()) is None

    def cancel(self) -> None:
        with self._lock:
            # cancel() and the final dispatch commitment share one ordering.
            self.cancel_event.set()
            ledger = self._read()
            ledger["stop_reason"] = "cancelled"
            self._persist(ledger)

    def meta(self) -> dict:
        with self._lock:
            ledger = self._read()
            count = sum("transport_started_at" in row for row in ledger["request_reservations"])
            attempts = self._read_attempts()["attempts"]
            audited = {row["ordinal"] for row in attempts}
            return {
                "external_retrieval_requests": count,
                "total_external_retrieval_requests": count,
                "total_dispatch_admissions": sum(
                    "dispatch_committed_at" in row for row in ledger["request_reservations"]
                ),
                "pending_dispatch_reservations": sum(
                    row["dispatch_state"] == "reserved" for row in ledger["request_reservations"]
                ),
                "max_requests": self.max_requests,
                "cap_micro_cny": self.cap_micro_cny,
                "cap_cny": self.cap_micro_cny / 1_000_000,
                "reserved_cost_micro_cny": ledger["reserved_cost_micro_cny"],
                "settled_cost_micro_cny": ledger["settled_cost_micro_cny"],
                "booked_cost_micro_cny": ledger["reserved_cost_micro_cny"]
                + ledger["settled_cost_micro_cny"],
                "unsettled_requests": sum(
                    row["dispatch_state"] == "transport_started"
                    for row in ledger["request_reservations"]
                ),
                "in_flight_requests": sum(row["state"] == "transport_started" for row in attempts),
                "finished_usage_unknown_requests": sum(
                    row["state"] == "finished_usage_unknown" for row in attempts
                ),
                "settled_requests": sum(row["state"] == "settled" for row in attempts),
                "aborted_unsent_requests": sum(
                    row["state"] == "aborted_unsent" for row in attempts
                ),
                "unaudited_requests": sum(
                    "transport_started_at" in row and row["ordinal"] not in audited
                    for row in ledger["request_reservations"]
                ),
                "retained_reserved_micro_cny": ledger["reserved_cost_micro_cny"],
                "attempt_audit_ref": str(self.attempt_audit_path),
                "remaining_seconds": self.remaining_seconds(),
                "stop_reason": self._stop_reason(ledger),
                "price_source": ledger["price_source"],
                "price_checked_date": ledger["price_checked_date"],
                "peak_cny_per_million": copy.deepcopy(ledger["peak_cny_per_million"]),
                "ledger_ref": str(self.ledger_path),
                "not_account_statement": True,
            }

    def _reserve(
        self, request_payload: dict, pair_id: str, retry_index: int = 0,
    ) -> tuple[int, float]:
        body = json.dumps(request_payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        reservation = (len(body) + 512) * 2 + request_payload["max_tokens"] * 8
        with self._lock:
            ledger = self._read()
            reason = self._stop_reason(ledger)
            if reason:
                raise RerankStopped(reason)
            if (
                ledger["reserved_cost_micro_cny"] + ledger["settled_cost_micro_cny"] + reservation
                > self.cap_micro_cny
            ):
                ledger["stop_reason"] = "budget_exhausted"
                self._persist(ledger)
                raise RerankStopped("budget_exhausted")
            ordinal = len(ledger["request_reservations"]) + 1
            ledger["request_reservations"].append(
                {
                    "ordinal": ordinal,
                    "pair_id": pair_id,
                    "reserved_at": _utc_now(),
                    "body_bytes": len(body),
                    "max_output_tokens": request_payload["max_tokens"],
                    "reserved_micro_cny": reservation,
                    "dispatch_state": "reserved",
                }
            )
            ledger["reserved_cost_micro_cny"] += reservation
            self._persist(ledger)
            self._retry_indices[ordinal] = retry_index
            # A write can take long enough for the external cancel/deadline to
            # fire. No transport has been dispatched while this lock is held.
            timeout = min(30.0, self.remaining_seconds())
            reason = (
                "cancelled" if self.cancel_event.is_set() else ("timeout" if timeout <= 0 else None)
            )
            if reason:
                self._abort_unstarted(ledger, ordinal, reason)
                raise RerankStopped(reason)
            return ordinal, timeout

    def _abort_unstarted(
        self, ledger: dict, ordinal: int, reason: str, *, stop_task: bool = True,
    ) -> None:
        """Called with the ledger lock and before invoking any transport."""
        row = ledger["request_reservations"][ordinal - 1]
        row["dispatch_state"] = "aborted"
        row["aborted_at"] = _utc_now()
        row["abort_reason"] = reason
        row.pop("transport_started_at", None)
        ledger["reserved_cost_micro_cny"] -= row["reserved_micro_cny"]
        if stop_task:
            ledger["stop_reason"] = reason
        self._persist(ledger)
        self._audit_attempt(
            ledger, ordinal, "aborted_unsent", completed_at=row["aborted_at"],
            error_code={
                "cancelled": "CANCELLED", "timeout": "TASK_TIMEOUT",
                "audit_write_failed": "AUDIT_WRITE_FAILED",
            }.get(reason, "STOPPED"),
        )

    def _commit_dispatch(self, ordinal: int) -> None:
        """Final admission and cancel() linearize under the same lock.

        This records admission, not an HTTP attempt. The callable invocation
        begins at _start_transport; no ledger lock spans network waiting.
        """
        with self._lock:
            ledger = self._read()
            row = ledger["request_reservations"][ordinal - 1]
            if row["dispatch_state"] != "reserved":
                raise ValueError("INVALID_DISPATCH_STATE")
            reason = self._stop_reason(ledger, include_limits=False)
            if reason:
                self._abort_unstarted(ledger, ordinal, reason)
                raise RerankStopped(reason)
            row["dispatch_state"] = "dispatch_committed"
            row["dispatch_committed_at"] = _utc_now()
            self._persist(ledger)

    def _start_transport(self, ordinal: int) -> float:
        """Begin an admitted transport invocation, checking stop once more.

        After this boundary the attempt is in flight and may finish/settle
        after cancellation. A stop while persisting still aborts before any
        callable is invoked and does not increase physical request counts.
        """
        with self._lock:
            ledger = self._read()
            row = ledger["request_reservations"][ordinal - 1]
            if row["dispatch_state"] != "dispatch_committed":
                raise ValueError("INVALID_DISPATCH_STATE")
            reason = self._stop_reason(ledger, include_limits=False)
            if reason:
                self._abort_unstarted(ledger, ordinal, reason)
                raise RerankStopped(reason)
            row["dispatch_state"] = "transport_started"
            row["transport_started_at"] = _utc_now()
            self._persist(ledger)
            try:
                self._audit_attempt(ledger, ordinal, "transport_started")
            except RerankAuditError:
                # No callable ran: apply the same unstarted reservation release.
                self._abort_unstarted(ledger, ordinal, "audit_write_failed", stop_task=False)
                raise
            timeout = min(30.0, self.remaining_seconds())
            reason = (
                "cancelled" if self.cancel_event.is_set() else ("timeout" if timeout <= 0 else None)
            )
            if reason:
                self._abort_unstarted(ledger, ordinal, reason)
                raise RerankStopped(reason)
            return timeout

    def _settle(self, ordinal: int, raw_usage: Any) -> bool:
        usage = _checked_usage(raw_usage)
        if usage is None:
            return False
        with self._lock:
            ledger = self._read()
            row = ledger["request_reservations"][ordinal - 1]
            if row.get("settlement") is not None:
                return True
            if row["dispatch_state"] != "transport_started":
                raise ValueError("INVALID_SETTLEMENT_STATE")
            cost = _usage_cost(usage)
            row["dispatch_state"] = "settled"
            row["settlement"] = {"usage": usage, "cost_micro_cny": cost, "settled_at": _utc_now()}
            ledger["reserved_cost_micro_cny"] -= row["reserved_micro_cny"]
            ledger["settled_cost_micro_cny"] += cost
            if cost > row["reserved_micro_cny"] and ledger["stop_reason"] != "cancelled":
                ledger["stop_reason"] = "usage_exceeded_reservation"
            self._persist(ledger)
            return True


def _redact(value: Any, key: str = "") -> str:
    result = str(value)
    if key:
        result = result.replace(key, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED]", result)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError("拒绝带密钥的HTTP重定向")


class DeepSeekReranker:
    def __init__(
        self,
        rules: str,
        transport: Callable | None = None,
        cache_size: int = 4096,
        *,
        key_loader: Callable[[], str] | None = None,
        key_file: Path | None = None,
    ) -> None:
        if (
            not isinstance(rules, str)
            or not rules.strip()
            or type(cache_size) is not int
            or cache_size < 0
        ):
            raise ValueError("INVALID_RERANK_CONFIGURATION")
        if key_loader is not None and key_file is not None:
            raise ValueError("AMBIGUOUS_RERANK_KEY_SOURCE")
        self.rules = rules
        self.rules_sha256 = hashlib.sha256(rules.encode("utf-8")).hexdigest()
        self.model = MODEL
        self._transport = transport
        self._key_loader = key_loader
        self._key_file = Path(key_file) if key_file is not None else None
        self._cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _load_key(self) -> str:
        if self._key_loader is not None:
            key = self._key_loader()
        elif self._key_file is not None:
            keys = set(
                re.findall(r"sk-[A-Za-z0-9_-]{16,}", self._key_file.read_text(encoding="utf-8-sig"))
            )
            if len(keys) != 1:
                raise ValueError("密钥文件不能唯一识别，不输出其内容")
            key = next(iter(keys))
        else:
            key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not isinstance(key, str) or re.fullmatch(r"sk-[A-Za-z0-9_-]{16,}", key.strip()) is None:
            raise ValueError("DEEPSEEK_API_KEY_MISSING_OR_INVALID")
        return key.strip()

    @staticmethod
    def _request(request_payload: dict, timeout_seconds: float, key: str) -> _TransportResult:
        data = json.dumps(request_payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            ENDPOINT,
            data=data,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        http_status = None
        try:
            with urllib.request.build_opener(NoRedirect()).open(
                request, timeout=timeout_seconds
            ) as reply:
                http_status = getattr(reply, "status", None)
                return _TransportResult(
                    json.loads(_redact(reply.read().decode("utf-8"), key)), http_status
                )
        except urllib.error.HTTPError as error:
            try:
                detail = _redact(error.read().decode("utf-8", errors="replace"), key)[:500]
            except Exception:
                detail = "无法读取服务错误摘要"
            raise RerankTransportError(
                f"HTTP {error.code}: {detail}", error.code, _http_error_code(error.code)
            ) from None
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            _, error_code = _attempt_error(error)
            raise RerankTransportError(
                _redact(type(error).__name__ + ": " + str(error), key), http_status, error_code
            ) from None

    def _cache_key(self, query_text: str, unit: dict) -> str:
        bound = {
            "query_text": query_text,
            "unit_id": unit["unit_id"],
            "unit_text": unit["text"],
            "rules_sha256": self.rules_sha256,
            "protocol": payload({"pair_id": "", "query_text": "", "unit_text": ""}, self.rules),
        }
        return hashlib.sha256(
            json.dumps(bound, ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()

    def _score_pair(self, query_text: str, unit: dict, control: RerankTaskControl) -> dict:
        cache_key = self._cache_key(query_text, unit)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return {
                    "score": {
                        **copy.deepcopy(cached),
                        "unit_id": unit["unit_id"],
                        "file_id": unit["file_id"],
                    },
                    "requests": 0,
                    "cache_hits": 1,
                }
        packet = {
            "pair_id": "P" + cache_key[:24],
            "query_text": query_text,
            "unit_text": unit["text"],
        }
        feedback = None
        requests = 0
        error_message = "unknown_error"
        for attempt in range(2):
            acquired = False
            ordinal = None
            started = False
            validating = False
            http_status = None
            completed_at = None
            try:
                while not acquired:
                    if not control.check():
                        raise RerankStopped(control.meta()["stop_reason"] or "stopped")
                    acquired = _HTTP_SLOTS.acquire(timeout=min(0.05, control.remaining_seconds()))
                if not control.check():
                    raise RerankStopped(control.meta()["stop_reason"] or "stopped")
                key = None if self._transport is not None else self._load_key()
                request_payload = payload(packet, self.rules, feedback)
                ordinal, _ = control._reserve(request_payload, packet["pair_id"], attempt)
                control._commit_dispatch(ordinal)
                timeout = control._start_transport(ordinal)
                started = True
                requests += 1
                response = (
                    self._transport(request_payload, timeout)
                    if self._transport is not None
                    else self._request(request_payload, timeout, key)
                )
                completed_at = _utc_now()
                if isinstance(response, _TransportResult):
                    http_status, response = response.http_status, response.value
                control._settle(
                    ordinal, response.get("usage") if isinstance(response, dict) else None
                )
                validating = True
                score = validate_completion(packet, response)
                control._finish_attempt(
                    ordinal, http_status=http_status, error_code=None,
                    completed_at=completed_at, will_retry=False,
                )
                with self._cache_lock:
                    if self._cache_size:
                        self._cache[cache_key] = copy.deepcopy(score)
                        self._cache.move_to_end(cache_key)
                        while len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)
                return {
                    "score": {**score, "unit_id": unit["unit_id"], "file_id": unit["file_id"]},
                    "requests": requests,
                    "cache_hits": 0,
                }
            except RerankAuditError as error:
                error_message = str(error)
                if started:
                    try:
                        control._finish_attempt(
                            ordinal, http_status=http_status, error_code="AUDIT_WRITE_FAILED",
                            completed_at=completed_at or _utc_now(), will_retry=False,
                        )
                    except RerankAuditError:
                        pass  # A diagnostic write failure must never cause another paid attempt.
                break
            except RerankStopped as error:
                error_message = str(error)
                if started:
                    control._finish_attempt(
                        ordinal, http_status=http_status,
                        error_code={"cancelled": "CANCELLED", "timeout": "TASK_TIMEOUT"}.get(
                            error_message, "STOPPED"
                        ), completed_at=completed_at or _utc_now(), will_retry=False,
                    )
                break
            except Exception as error:
                error_message = _redact(type(error).__name__ + ": " + str(error))
                error_status, error_code = _attempt_error(error, validating=validating)
                if error_status is not None:
                    http_status = error_status
                if isinstance(error, ValueError) and ordinal is not None:
                    feedback = (
                        "上一输出未通过格式校验："
                        + error_message
                        + (
                            "。请只依据原输入和规则重新给出一个合法JSON。score必须为整数；"
                            "specific_direct=80—100，concise_direct=60—79，partial=20—59，"
                            "topic_only=1—19，none/insufficient=0；等级必须与分数对应。"
                            "引文逐字连续包含于原unit_text，保留换行；"
                            "不得改变事实、补默认评分或给参考答案。"
                        )
                    )
                will_retry = (
                    attempt == 0 and ordinal is not None
                    and http_status not in (400, 401, 402, 403)
                    and not any(code in error_message for code in (
                        "HTTP 400", "HTTP 401", "HTTP 402", "HTTP 403"
                    ))
                    and control.check()
                )
                if started:
                    control._finish_attempt(
                        ordinal, http_status=http_status, error_code=error_code,
                        completed_at=completed_at or _utc_now(), will_retry=will_retry,
                    )
                if not will_retry:
                    break
            finally:
                if acquired:
                    _HTTP_SLOTS.release()
        return {
            "error": {
                "unit_id": unit["unit_id"],
                "file_id": unit["file_id"],
                "reason": error_message,
            },
            "requests": requests,
            "cache_hits": 0,
        }

    def score_query(self, query_text: str, units: list[dict], control: RerankTaskControl) -> dict:
        if (
            not isinstance(query_text, str)
            or not isinstance(units, list)
            or not isinstance(control, RerankTaskControl)
        ):
            raise ValueError("INVALID_RERANK_INPUT")
        seen = set()
        for unit in units:
            if (
                not isinstance(unit, dict)
                or any(
                    not isinstance(unit.get(k), str) or not unit[k]
                    for k in ("unit_id", "file_id", "text")
                )
                or unit["unit_id"] in seen
            ):
                raise ValueError("INVALID_RERANK_UNITS")
            seen.add(unit["unit_id"])
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda unit: self._score_pair(query_text, unit, control), units)
            )
        scores = [row["score"] for row in results if "score" in row]
        errors = [row["error"] for row in results if "error" in row]
        meta = control.meta()
        meta.update(
            external_retrieval_requests=sum(row["requests"] for row in results),
            cache_hits=sum(row["cache_hits"] for row in results),
            rerank_mode="live",
            model=MODEL,
            rules_sha256=self.rules_sha256,
            units_requested=len(units),
            units_scored=len(scores),
        )
        return {
            "scores": scores,
            "complete": len(scores) == len(units),
            "meta": meta,
            "errors": errors,
        }
