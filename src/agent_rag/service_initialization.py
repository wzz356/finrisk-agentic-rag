"""Singleflight initialization with read-only readiness and terminal failures."""

import threading
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

_PHASES = {"asset_validation", "snapshot_load", "service_build"}
_ERRORS = {"SNAPSHOT_NOT_READY", "DENSE_STARTUP_FAILED", "DENSE_RESPONSE_TIMEOUT",
           "DENSE_PROTOCOL_ERROR", "LIVE_RERANK_NOT_ENABLED", "BACKEND_ASSET_HASH_MISMATCH"}


class ServiceInitialization:
    def __init__(self, factory: Callable, *, closer: Callable | None = None) -> None:
        self._factory, self._closer = factory, closer
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "not_started"
        self._service: Any = None
        self._closed = False
        self._started: float | None = None
        self._finished: float | None = None
        self._phase_started: float | None = None
        self._phase: str | None = None
        self._phases: list[dict] = []
        self._error: str | None = None

    def _status_locked(self) -> dict:
        now = self._finished or time.monotonic()
        phases = [dict(row) for row in self._phases]
        if self._phase is not None and self._phase_started is not None:
            phases.append({"phase": self._phase,
                           "elapsed_ms": round((now - self._phase_started) * 1000, 3)})
        return {"initialization_state": self._state, "initialization_phase": self._phase,
                "initialization_elapsed_ms": (round((now - self._started) * 1000, 3)
                                               if self._started is not None else 0),
                "initialization_phases": phases, "initialization_error": self._error}

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _set_phase(self, phase: str) -> None:
        if phase not in _PHASES:
            return
        with self._lock:
            if self._closed:
                return
            now = time.monotonic()
            if self._phase is not None and self._phase_started is not None:
                self._phases.append({"phase": self._phase,
                                     "elapsed_ms": round((now - self._phase_started) * 1000, 3)})
            self._phase, self._phase_started = phase, now

    def request_warmup(self) -> dict:
        with self._lock:
            if self._state == "not_started" and not self._closed:
                self._state, self._started = "loading", time.monotonic()
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name="agent-service-initialization")
                self._thread.start()
            return self._status_locked()

    def _run(self) -> None:
        try:
            service = self._factory(self._set_phase)
        except Exception as error:
            code = str(error).split(":", 1)[0]
            with self._lock:
                self._state = "failed"
                if not self._closed:
                    self._error = code if code in _ERRORS else "SERVICE_INITIALIZATION_FAILED"
                self._finished = time.monotonic()
                self._done.set()
            return
        with self._lock:
            closed = self._closed
            if not closed:
                self._service, self._state = service, "ready"
            self._finished = time.monotonic()
        if closed:
            self._close_service(service)
        self._done.set()

    def peek_ready(self) -> Any:
        with self._lock:
            return self._service if self._state == "ready" and not self._closed else None

    def get_ready(self) -> Any:
        with self._lock:
            if self._state == "ready" and not self._closed:
                return self._service
            raise HTTPException(503, detail=self._error or "SERVICE_NOT_READY")

    def wait_ready(self, timeout: float | None = None) -> Any:
        self._done.wait(timeout)
        return self.get_ready()

    def _close_service(self, service: Any) -> None:
        if self._closer is not None:
            self._closer(service)

    def close(self, timeout: float = 2) -> None:
        with self._lock:
            self._closed = True
            service, self._service = self._service, None
            self._state, self._error = "failed", "SERVICE_CLOSED"
            self._done.set()
            thread = self._thread
        if service is not None:
            self._close_service(service)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
