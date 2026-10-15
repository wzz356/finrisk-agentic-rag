"""Start temporary Python and Pi services and check the offline public demo."""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from build_demo import build

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "partial", "failed", "interrupted", "rejected"}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    node = shutil.which("node")
    if not node or not (ROOT / "agent_harness/dist/src/server.js").exists():
        raise RuntimeError("Install Node dependencies and run npm build first")
    ports = [free_port(), free_port()]
    while ports[1] == ports[0]:
        ports[1] = free_port()
    base = f"http://127.0.0.1:{ports[0]}"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FINRISK_AGENT_", "DEEPSEEK_"))}
    processes = []
    summaries = []
    with tempfile.TemporaryDirectory(prefix="finrisk-public-smoke-") as temporary:
        tmp = Path(temporary)
        build(tmp / "snapshot")
        env.update(
            FINRISK_AGENT_SERVICE_TOKEN="offline-smoke-local-service",
            FINRISK_AGENT_SNAPSHOT=str(tmp / "snapshot"),
            FINRISK_AGENT_TASK_ROOT=str(tmp / "tasks"),
            FINRISK_AGENT_PYTHON_URL=base,
            FINRISK_AGENT_NODE_URL=f"http://127.0.0.1:{ports[1]}",
            FINRISK_AGENT_NODE_PORT=str(ports[1]),
            FINRISK_AGENT_ALLOW_LIVE="0",
            FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE="0",
        )
        def request(path, payload=None, headers=None):
            req = urllib.request.Request(
                base + path,
                data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
                headers={"Content-Type": "application/json", **(headers or {})},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.load(response)
        logs = (tmp / "services.log").open("w", encoding="utf-8")
        try:
            for command in [
                [sys.executable, "-m", "uvicorn", "agent_rag.app:app", "--app-dir", str(ROOT / "src"),
                 "--host", "127.0.0.1", "--port", str(ports[0])],
                [node, str(ROOT / "agent_harness/dist/src/server.js")],
            ]:
                processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=logs, stderr=logs))
            deadline = time.monotonic() + 20
            while True:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError("SERVICE_EXITED")
                try:
                    status = request("/api/agent-rag/status")
                    urllib.request.urlopen(urllib.request.Request(
                        env["FINRISK_AGENT_NODE_URL"] + "/internal/harness/health",
                        headers={"Authorization": "Bearer " + env["FINRISK_AGENT_SERVICE_TOKEN"]}), timeout=2).close()
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("STARTUP_TIMEOUT")
                    time.sleep(0.1)
            assert status["live_enabled"] is False
            assert status["retrieval_live_enabled"] is False
            with urllib.request.urlopen(base + "/agent-rag", timeout=5) as response:
                assert "合成案例" in response.read().decode()
            with urllib.request.urlopen(base + "/agent-rag/progress-view.js", timeout=5) as response:
                assert response.status == 200
            try:
                request("/api/agent-rag/tasks", {"text": "离线检查。",
                        "experiment_profile_id": "live-deepseek-flash-bm25-v1"})
                raise AssertionError("LIVE_SHOULD_BE_REJECTED")
            except urllib.error.HTTPError as error:
                assert error.code == 422
            for example in json.loads((ROOT / "examples/requests.json").read_text(encoding="utf-8")):
                admitted = request("/api/agent-rag/tasks", example)
                deadline = time.monotonic() + 120
                while True:
                    task = request(admitted["poll_url"])
                    if task["task_status"] in TERMINAL:
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("TASK_TIMEOUT: " + json.dumps(task, ensure_ascii=False))
                    time.sleep(0.1)
                assert task["task_status"] in {"completed", "partial"}, task
                report = task["report"]
                assert report["simulated"] is True
                assert report["resources"]["physical_provider_requests"] == 0
                assert task.get("progress_history"), "Missing visible progress history"
                task_dir = tmp / "tasks" / admitted["task_id"]
                registry = json.loads((task_dir / "input.json").read_text(encoding="utf-8"))
                assert registry, "Missing frozen input"
                if example["mock_scenario"] == "generic":
                    assert report["overall_judgment"] == "review_required"
                summaries.append({
                    "scenario": example["mock_scenario"], "status": task["task_status"],
                    "judgment": report["overall_judgment"], "physical_provider_requests": 0,
                    "has_progress": True,
                })
            # Read one registered source through the public review endpoint.
            snapshot = json.loads((tmp / "snapshot/manifest.json").read_text(encoding="utf-8"))
            assert snapshot["synthetic"]
            import gzip
            with gzip.open(tmp / "snapshot/index.json.gz", "rt", encoding="utf-8") as stream:
                source_id = json.load(stream)["sources"][0]["source_id"]
            source = request(f"/api/agent-rag/tasks/{admitted['task_id']}/source/{source_id}")
            assert source["status"] == "ok", source
            assert "合成" in source["data"]["evidence"]["quote"]
            assert source["data"]["evidence"]["text_quality"] == "synthetic_demo_not_real_penalty"
            print(json.dumps({"passed": True, "tasks": summaries, "live_rejected": True,
                              "source_verified": True, "api_calls": 0}, ensure_ascii=False, indent=2))
        except BaseException:
            logs.flush()
            evidence = ROOT / "runtime/smoke-failure"
            evidence.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp, evidence, dirs_exist_ok=True)
            print((tmp / "services.log").read_text(encoding="utf-8")[:6000], file=sys.stderr)
            print("Diagnostic copy saved to runtime/smoke-failure", file=sys.stderr)
            raise
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            logs.close()


if __name__ == "__main__":
    main()
