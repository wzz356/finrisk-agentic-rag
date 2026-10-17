"""Private pair admission state; the budget ledger remains the sole fee record."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

SERVICE_GENERATION = uuid.uuid4().hex
_STATES: dict[str, dict] = {}
_STATES_LOCK = threading.Lock()
_CODES = {
    "SUCCESS", "HTTP_429", "HTTP_5XX", "HTTP_4XX", "HTTP_ERROR",
    "TRANSPORT_TIMEOUT", "TRANSPORT_ERROR", "INVALID_JSON", "INVALID_COMPLETION",
    "QUOTE_MISMATCH", "CANCELLED", "TASK_TIMEOUT", "STOPPED", "AUDIT_WRITE_FAILED",
    "CHECKPOINT_UNAVAILABLE", "ATTEMPTS_EXHAUSTED", "KEY_UNAVAILABLE", "SPAN_REFERENCE_MISMATCH",
}


class RerankCheckpointError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("CHECKPOINT_UNAVAILABLE")


class RerankCheckpointStopped(RuntimeError):
    """A waiter may stop without taking ownership away from an in-flight pair."""


def _digest(data: dict) -> str:
    return hashlib.sha256(json.dumps(
        data, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


class PairCheckpoint:
    """Coordinate only within one service generation; never take over old work.

    The shared condition uses the task's short ledger lock. Pair ownership is
    retained while HTTP runs, but that lock is released throughout the call.
    A failed write poisons admission for this generation rather than granting
    another paid attempt from an older on-disk state.
    """

    def __init__(self, control) -> None:
        self.control = control
        self.path = control.ledger_path.with_name(control.ledger_path.name + ".pairs.json")
        key = os.path.normcase(str(self.path))
        with _STATES_LOCK:
            self._state = _STATES.setdefault(key, {
                "condition": threading.Condition(control._lock), "active": set(),
                "poisoned": False, "digest": None,
            })
        self._condition = self._state["condition"]

    def _binding(self, ledger: dict) -> str:
        return _digest({
            "ledger_path": os.path.normcase(str(self.control.ledger_path)),
            **{key: ledger[key] for key in (
                "model", "cap_micro_cny", "max_requests", "deadline_epoch_seconds",
                "price_source", "price_checked_date", "peak_cny_per_million",
            )},
        })

    def _persist(self, data: dict, *, exclusive: bool = False) -> None:
        temporary = None
        try:
            if exclusive:
                # Competing processes may race to create, but only one owns it.
                with self.path.open("x", encoding="utf-8") as stream:
                    json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.path.parent,
                    prefix=self.path.name + ".", suffix=".tmp", delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            self._state["digest"] = _digest(data)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _read(self) -> dict:
        if self._state["poisoned"]:
            raise RerankCheckpointError()
        try:
            ledger = self.control._read()
            if not self.path.exists():
                if self._state["digest"] is not None or ledger["request_reservations"]:
                    raise RerankCheckpointError()
                data = {
                    "schema_version": "finrisk-rerank-checkpoint/1.0",
                    "task_binding": self._binding(ledger),
                    "service_generation": SERVICE_GENERATION,
                    "service_process_id": os.getpid(), "pairs": {},
                }
                self._persist(data, exclusive=True)
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                not isinstance(data, dict)
                or data.get("schema_version") != "finrisk-rerank-checkpoint/1.0"
                or data.get("task_binding") != self._binding(ledger)
                or data.get("service_generation") != SERVICE_GENERATION
                or data.get("service_process_id") != os.getpid()
                or not isinstance(data.get("pairs"), dict)
                or (self._state["digest"] is not None
                    and self._state["digest"] != _digest(data))
            ):
                raise RerankCheckpointError()
            covered = {}
            for key, pair in data["pairs"].items():
                attempts = pair["attempts"]
                if (
                    not isinstance(key, str) or len(key) != 64
                    or pair["pair_id"] != "P" + key[:24]
                    or not isinstance(attempts, list) or len(attempts) > 2
                    or type(pair["attempts_used"]) is not int
                    or pair["attempts_used"] != len(attempts)
                    or pair["remaining_attempts"] != 2 - len(attempts)
                    or type(pair["terminal"]) is not bool
                    or pair["last_code"] not in _CODES | {None}
                    or (len(attempts) == 2 and not pair["terminal"])
                    or (pair["score"] is not None and (
                        not isinstance(pair["score"], dict) or not pair["terminal"]
                        or pair["score"].get("pair_id") != pair["pair_id"]
                    ))
                ):
                    raise RerankCheckpointError()
                ordinals = []
                for index, attempt in enumerate(attempts):
                    ordinal = attempt["audit_ordinal"]
                    if (
                        attempt["index"] != index or attempt["code"] not in _CODES | {None}
                        or attempt["state"] not in ("claimed", "finished")
                        or (attempt["state"] == "finished") != (attempt["code"] is not None)
                        or (ordinal is not None and (type(ordinal) is not int or ordinal < 1))
                    ):
                        raise RerankCheckpointError()
                    if ordinal is not None:
                        if ordinal in covered:
                            raise RerankCheckpointError()
                        covered[ordinal] = pair["pair_id"]
                        ordinals.append(ordinal)
                if pair["audit_ordinals"] != ordinals:
                    raise RerankCheckpointError()
            for row in ledger["request_reservations"]:
                if ("dispatch_committed_at" in row or "transport_started_at" in row) and (
                    covered.get(row["ordinal"]) != row["pair_id"]
                ):
                    raise RerankCheckpointError()
            self._state["digest"] = _digest(data)
            return data
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            raise RerankCheckpointError() from error

    def _write(self, data: dict) -> None:
        try:
            self._persist(data)
        except (OSError, ValueError, TypeError) as error:
            self._state["poisoned"] = True
            raise RerankCheckpointError() from error

    def poison(self) -> None:
        """Stop new admissions after a task audit failure; retain existing fee facts."""
        with self._condition:
            self._state["poisoned"] = True
            self._state["failure_code"] = "AUDIT_WRITE_FAILED"
            self._condition.notify_all()

    @property
    def failure_code(self) -> str:
        with self._condition:
            return self._state.get("failure_code", "CHECKPOINT_UNAVAILABLE")

    @contextmanager
    def ownership(self, key: str):
        with self._condition:
            while key in self._state["active"]:
                if (self.control.cancel_event.is_set()
                        or self.control._read()["stop_reason"] == "cancelled"):
                    raise RerankCheckpointStopped("CANCELLED")
                if self.control.remaining_seconds() <= 0:
                    raise RerankCheckpointStopped("TASK_TIMEOUT")
                self._condition.wait(timeout=min(0.05, self.control.remaining_seconds()))
            self._read()
            self._state["active"].add(key)
        try:
            yield
        finally:
            with self._condition:
                self._state["active"].discard(key)
                self._condition.notify_all()

    def get(self, key: str) -> dict:
        with self._condition:
            return copy.deepcopy(self._read()["pairs"].get(key, {
                "pair_id": "P" + key[:24], "attempts": [], "attempts_used": 0,
                "remaining_attempts": 2, "terminal": False, "audit_ordinals": [],
                "score": None, "last_code": None,
            }))

    def claim(self, key: str) -> int:
        with self._condition:
            data = self._read()
            pair = self.get(key)
            if pair["terminal"] or (pair["attempts"] and pair["attempts"][-1]["state"] != "finished"):
                raise RerankCheckpointError()
            index = pair["attempts_used"]
            pair["attempts"].append({
                "index": index, "state": "claimed", "code": None, "audit_ordinal": None,
            })
            pair.update(attempts_used=index + 1, remaining_attempts=1 - index,
                        terminal=index == 1)
            data["pairs"][key] = pair
            self._write(data)
            return index

    def bind_ordinal(self, key: str, ordinal: int) -> None:
        with self._condition:
            data = self._read()
            pair = data["pairs"][key]
            pair["attempts"][-1]["audit_ordinal"] = ordinal
            pair["audit_ordinals"].append(ordinal)
            self._write(data)

    def finish(self, key: str, code: str, *, terminal: bool, score: dict | None = None) -> dict:
        with self._condition:
            data = self._read()
            pair = data["pairs"].get(key, self.get(key))
            if pair["attempts"]:
                pair["attempts"][-1].update(code=code, state="finished")
            pair.update(last_code=code, terminal=terminal or pair["remaining_attempts"] == 0,
                        score=copy.deepcopy(score))
            data["pairs"][key] = pair
            self._write(data)
            return copy.deepcopy(pair)
