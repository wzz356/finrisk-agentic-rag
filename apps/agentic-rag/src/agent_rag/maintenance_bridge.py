"""Controlled JSONL subprocess bridge. The shipped CLI is fixture-only."""

from __future__ import annotations

import json
import os
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path


class NodeMaintenanceRunner:
    live_ready = False

    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()

    @staticmethod
    def token_summary(ledger):
        with sqlite3.connect(ledger.path, timeout=30) as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='maintenance_tokens'"
            ).fetchone()
            by_event = (
                dict(
                    db.execute(
                        "SELECT event_id,SUM(units) FROM maintenance_tokens GROUP BY event_id"
                    )
                )
                if exists
                else {}
            )
        return {
            "counter_kind": "conservative_byte_upper_bound",
            "reservation_rule": "no_refund",
            "reserved_units": sum(by_event.values()),
            "by_event": by_event,
        }

    @staticmethod
    def _reserve_tokens(ledger, packet, event_id, limit):
        prompt, output = packet.get("prompt_token_upper_bound"), packet.get("max_tokens")
        units = packet.get("reserved_token_units")
        if (
            type(prompt) is not int
            or type(output) is not int
            or type(units) is not int
            or prompt < 0
            or output < 1
            or units != prompt + output
            or packet.get("token_counter_kind") != "conservative_byte_upper_bound"
        ):
            raise ValueError("TOKEN_COUNTER_REQUIRED")
        with sqlite3.connect(ledger.path, timeout=30) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS maintenance_tokens "
                "(request_key TEXT PRIMARY KEY,event_id TEXT NOT NULL,"
                "payload_sha256 TEXT NOT NULL,units INTEGER NOT NULL)"
            )
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT event_id,payload_sha256,units FROM maintenance_tokens WHERE request_key=?",
                (packet["request_id"],),
            ).fetchone()
            binding = (event_id, packet["payload_sha256"], units)
            if old:
                if old != binding:
                    raise ValueError("TOKEN_REQUEST_BINDING")
                return
            used = db.execute(
                "SELECT COALESCE(SUM(units),0) FROM maintenance_tokens WHERE event_id=?",
                (event_id,),
            ).fetchone()[0]
            if used + units > limit:
                raise ValueError("EVENT_TOKEN_BUDGET_EXHAUSTED")
            db.execute(
                "INSERT INTO maintenance_tokens VALUES (?,?,?,?)", (packet["request_id"], *binding)
            )

    def run(self, task, tools, ledger, *, event_id, stage, script=None, cancelled=lambda: False):
        if task["execution_mode"] != "fixture":
            raise ValueError("LIVE_ADAPTERS_NOT_READY")
        cli = self.project_root / "agent_harness/dist/src/maintenance_cli.js"
        node = shutil.which("node")
        if not node or not cli.is_file():
            raise ValueError("MAINTENANCE_RUNNER_BUILD_REQUIRED")
        # Do not inherit model keys, external provider configuration or NODE_OPTIONS.
        allowed_env = {"PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC"}
        env = {k: v for k, v in os.environ.items() if k.upper() in {x.upper() for x in allowed_env}}
        proc = subprocess.Popen(
            [node, str(cli)],
            cwd=self.project_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        packets = queue.Queue()

        def read():
            try:
                for line in proc.stdout:
                    if len(line) > 4_000_000:
                        packets.put(ValueError("PROTOCOL_MESSAGE_TOO_LARGE"))
                        break
                    packets.put(json.loads(line))
            except Exception:
                packets.put(ValueError("INVALID_AGENT_PROTOCOL"))
            finally:
                packets.put(None)

        def drain():
            # Drain diagnostics without forwarding arbitrary exception/source text.
            while proc.stderr.read(4096):
                pass

        threading.Thread(target=read, daemon=True).start()
        threading.Thread(target=drain, daemon=True).start()

        def send(value):
            proc.stdin.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            proc.stdin.flush()

        bindings = {k: task[k] for k in ("run_id", "role", "revision")}
        tickets, result = {}, None
        deadline = time.monotonic() + task["deadline_seconds"] + 3
        try:
            send({"kind": "start", "task": task, "script": script})
            while time.monotonic() < deadline:
                if cancelled():
                    send({"kind": "cancel"})
                    raise ValueError("CANCELLED")
                try:
                    packet = packets.get(timeout=0.1)
                except queue.Empty:
                    continue
                if packet is None:
                    raise ValueError("AGENT_PROCESS_INTERRUPTED")
                if isinstance(packet, Exception):
                    raise packet
                kind = packet.get("kind")
                if kind == "final_result":
                    if any(packet.get(k) != v for k, v in bindings.items()):
                        raise ValueError("AGENT_RUN_BINDING")
                    result = packet
                    break
                if any(packet.get(k) != v for k, v in bindings.items()):
                    raise ValueError("AGENT_RUN_BINDING")
                request_id = packet.get("request_id")
                if not isinstance(request_id, str):
                    raise ValueError("INVALID_AGENT_PROTOCOL")
                if kind == "tool_request":
                    if (
                        packet.get("knowledge_commit") != task["knowledge_commit"]
                        or packet.get("scope") != task["scope"]
                    ):
                        raise ValueError("AGENT_RUN_BINDING")
                    try:
                        value = tools.call(packet["name"], packet["args"])
                        send({"kind": "tool_result", "request_id": request_id, "result": value})
                    except (ValueError, KeyError, TypeError):
                        send(
                            {
                                "kind": "tool_result",
                                "request_id": request_id,
                                "error": "TOOL_REQUEST_REJECTED",
                            }
                        )
                elif kind == "model_reserve":
                    if (
                        packet.get("physical_request") is not False
                        or packet.get("execution_mode") != "fixture"
                        or packet.get("reserved_micro") != 0
                    ):
                        raise ValueError("LIVE_ADAPTERS_NOT_READY")
                    self._reserve_tokens(
                        ledger, packet, event_id, task.get("event_token_budget", 240000)
                    )
                    ticket = ledger.reserve(
                        request_id, packet["payload_sha256"], task["role"], event_id, stage, 0
                    )
                    if ticket.get("cache_hit"):
                        raise ValueError("COMPLETED_REQUEST_REPLAY_REQUIRES_SAVED_RUN")
                    ledger.mark_sent(ticket)
                    tickets[ticket["ticket_id"]] = ticket
                    send(
                        {
                            "kind": "model_ticket",
                            "request_id": request_id,
                            "ticket": ticket["ticket_id"],
                        }
                    )
                elif kind == "model_settle":
                    ticket = tickets.get(packet.get("ticket"))
                    if ticket is None or request_id != ticket["request_key"]:
                        raise ValueError("INVALID_MODEL_TICKET")
                    ledger.settle(
                        ticket, None if packet.get("unknown") else 0, response={"fixture": True}
                    )
                    send({"kind": "model_settled", "request_id": request_id})
                else:
                    raise ValueError("INVALID_AGENT_PROTOCOL")
            if result is None:
                raise ValueError("AGENT_DEADLINE_EXCEEDED")
            for ticket in tickets.values():
                row = ledger.get_request(ticket["request_key"])
                if row["state"] == "sent":
                    ledger.settle(ticket, None)
            if any(
                ledger.get_request(ticket["request_key"])["state"] != "settled"
                for ticket in tickets.values()
            ):
                result.update(stop_reason="budget_uncertain", output=None)
            return result
        finally:
            for ticket in tickets.values():
                row = ledger.get_request(ticket["request_key"])
                if row["state"] == "sent":
                    ledger.settle(ticket, None)
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
            for handle in (proc.stdin, proc.stdout, proc.stderr):
                handle.close()
