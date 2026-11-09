import hashlib
import importlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from test_budget_profiles import live_environment
from test_v2_service import make_service

from agent_rag.contracts import canonical_json, tool_schemas_for_profile
from agent_rag.input_registry import text_hash

PROJECT = Path(__file__).resolve().parents[2]
PLANNED = "live-deepseek-flash-planned-v1"


@pytest.fixture
def node(monkeypatch, tmp_path):
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE", "FINRISK_AGENT_DEEPSEEK_KEY_FILE"):
        monkeypatch.delenv(key, raising=False)
    sentinel = tmp_path / "OFFLINE_NO_SUCH_KEY_FILE"
    assert not sentinel.exists()
    monkeypatch.setenv("DEEPSEEK_API_KEY_FILE", str(sentinel))
    monkeypatch.setenv("FINRISK_AGENT_DEEPSEEK_KEY_FILE", str(sentinel))
    result = shutil.which("node")
    if result is None:
        pytest.skip("Node must be on PATH for the offline cross-language acceptance run")
    return result


def node_contract(node, bundle):
    module = (PROJECT / "agent_harness/dist/src/agent_loop_runner.js").as_uri()
    code = """
import { readFileSync } from 'node:fs';
const { plannedExecutionContract } = await import(process.argv[1]);
const task = JSON.parse(readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(plannedExecutionContract(task.config)));
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", code, module],
        input=json.dumps(bundle), capture_output=True, text=True, encoding="utf8", timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def server_boundary(node, service, task_id, tmp_path):
    requests = []
    bundle_route = f"/internal/agent-rag/tasks/{task_id}/bundle"

    class Bundles(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path != bundle_route:
                self.send_error(404)
                return
            body = json.dumps(service.bundle(task_id)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    python = ThreadingHTTPServer(("127.0.0.1", 0), Bundles)
    thread = threading.Thread(target=python.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    environment = dict(os.environ)
    environment.update(
        FINRISK_AGENT_SERVICE_TOKEN="offline-contract-test",
        FINRISK_AGENT_PYTHON_URL=f"http://127.0.0.1:{python.server_port}",
        FINRISK_AGENT_NODE_PORT=str(port), FINRISK_AGENT_TASK_ROOT=str(service.root),
    )
    with (tmp_path / "node-server.log").open("w", encoding="utf8") as output:
        process = subprocess.Popen(
            [node, "--import", offline_fetch_guard(tmp_path).as_uri(),
             str(PROJECT / "agent_harness/dist/src/server.js")],
            env=environment, stdout=output, stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", trust_env=False,
                headers={"Authorization": "Bearer offline-contract-test"}, timeout=5,
            ) as client:
                deadline = time.monotonic() + 8
                while True:
                    assert process.poll() is None, "offline Node server exited"
                    try:
                        if client.get("/internal/harness/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert time.monotonic() < deadline, "offline Node server not ready"
                    time.sleep(0.05)
                response = client.post("/internal/harness/run", json={"task_id": task_id})
                assert response.status_code == 422
                # The synthetic dictionary intentionally stops the normal runTask path
                # immediately AFTER the real live-contract gate, BEFORE key loading.
                assert response.json()["error"] == "CRITERIA_DICTIONARY_MISMATCH"
                manifest_path = service.root / task_id / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf8"))
                manifest["profile_sha256"] = "0" * 64
                manifest_path.write_text(json.dumps(manifest), encoding="utf8")
                drift = client.post("/internal/harness/run", json={"task_id": task_id})
                assert drift.status_code == 422
                assert drift.json()["error"] == "EXECUTION_CONTRACT_MISMATCH"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            python.shutdown()
            python.server_close()
            thread.join(timeout=2)
    assert requests == [bundle_route, bundle_route]
    assert not list(service.root.glob("*/report.json"))
    assert not list(service.root.glob("*/state.json"))
    assert not (tmp_path / "models.json").exists()


@pytest.mark.parametrize("engine", ["controlled_agent", "fixed_rag"])
def test_standard_planned_admission_passes_all_nine_hashes_to_server(
    tmp_path, monkeypatch, node, engine
):
    live_environment(monkeypatch, tmp_path)
    service = make_service(tmp_path, "live")
    # Deliberate fixture mismatch: normal runTask stops after the actual
    # contract check and before trial initialization or key loading.
    service.snapshot.manifest["dictionary_sha256"] = "0" * 64
    service.snapshot_manifest["dictionary_sha256"] = "0" * 64
    request = {"text": "材料", "user_context": {"purpose_hint": "事实描述"},
               "experiment_profile_id": PLANNED, "engine": engine}
    record, _ = service.admit(request, None)
    task_id = record["task_id"]
    manifest = json.loads((service.root / task_id / "manifest.json").read_text(encoding="utf8"))
    bundle = service.bundle(task_id)
    expected = node_contract(node, bundle)
    assert len(expected) == 9 and bundle["execution_contract"] == expected
    assert bundle["snapshot"]["dictionary_sha256"] == "0" * 64
    assert all(manifest[key] == value for key, value in expected.items())
    assert type(bundle["config"]["model_trial_cap_cny"]) is float
    assert bundle["config"]["engine"] == record["engine"] == engine
    assert bundle["config"]["orchestration_mode"] == (
        "planned_fixed" if engine == "fixed_rag" else "planned_adaptive"
    )
    assert bundle["config"]["fast_path_enabled"] is False
    assert manifest["config_sha256"] == text_hash(canonical_json(bundle["config"]))
    assert manifest["profile_sha256"] != manifest["config_sha256"]
    assert bundle["input"]["text"] == request["text"]
    assert bundle["input"]["user_context"]["purpose_hint"] == "事实描述"
    assert bundle["config"]["max_model_requests"] == 12
    assert bundle["config"]["rerank_max_requests"] == 800
    server_boundary(node, service, task_id, tmp_path)


def test_old_default_contract_stays_unchanged(tmp_path):
    service = make_service(tmp_path)
    record, _ = service.admit({"text": "材料"}, None)
    bundle = service.bundle(record["task_id"])
    assert service.config["profile_id"] == bundle["config"]["profile_id"]
    assert bundle["config"]["profile_id"] == "mock-v2-flash-replay-v1"
    assert "orchestration_mode" not in bundle["config"]
    assert set(bundle["execution_contract"]) == {
        "model_wire_sha256", "criteria_sha256", "prompt_sha256", "tool_contract_sha256",
        "request_admission_policy_sha256",
    }
    assert bundle["execution_contract"]["prompt_sha256"] == hashlib.sha256(
        (PROJECT / "configs/agent_rag/system-prompt.txt").read_bytes()
    ).hexdigest()
    assert bundle["execution_contract"]["tool_contract_sha256"] == text_hash(
        canonical_json(tool_schemas_for_profile(bundle["config"]["retrieval_profile"]))
    )


def test_shared_inventory_drift_refused_before_paid_control(tmp_path, monkeypatch):
    helper = importlib.import_module("agent_rag.planned_execution_contract")
    inventory = json.loads(helper.INVENTORY.read_text(encoding="utf8"))
    inventory["execution_contract"]["criteria_sha256"] = "0" * 64
    drift = tmp_path / "drift.json"
    drift.write_text(json.dumps(inventory), encoding="utf8")
    monkeypatch.setattr(helper, "INVENTORY", drift)
    live_environment(monkeypatch, tmp_path)
    service = make_service(tmp_path, "live")
    with pytest.raises(ValueError, match="PLANNED_SERVICE_CONTRACT_DRIFT"):
        service.admit({"text": "材料", "experiment_profile_id": PLANNED}, None)
    assert not list(service.root.glob("*/admission.json"))
    assert not list(service.root.glob("*/rerank-ledger.json"))


def test_new_contract_runtime_dependencies_are_frozen():
    from test_agentic_v3_experiment import api

    frozen = set(api().freeze_runtime(PROJECT))
    for name in (
        "src/agent_rag/planned_execution_contract.py",
        "contracts/agent_rag/planned-execution-contract.json",
        "agent_harness/src/agent_loop_service_contract.ts",
        "agent_harness/dist/src/agent_loop_service_contract.js",
    ):
        assert PROJECT / name in frozen



def offline_fetch_guard(tmp_path):
    guard = tmp_path / "offline-fetch-guard.mjs"
    guard.write_text("""
const originalFetch = globalThis.fetch;
globalThis.fetch = (input, init) => {
  const url = new URL(typeof input === "string" ? input
    : input instanceof URL ? input.href : input.url);
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
    return Promise.reject(new Error("OFFLINE_EXTERNAL_FETCH_FORBIDDEN"));
  }
  return originalFetch(input, { ...init, redirect: "error" });
};
globalThis.finriskOfflineFetchGuard = true;
""", encoding="utf8")
    return guard


def test_offline_child_guards_before_service_run(node, tmp_path):
    module = (PROJECT / "agent_harness/dist/src/deepseek_gateway.js").as_uri()
    code = """
import assert from 'node:assert/strict';
assert.equal(globalThis.finriskOfflineFetchGuard, true);
await assert.rejects(fetch('https://api.deepseek.com/chat/completions'),
  /OFFLINE_EXTERNAL_FETCH_FORBIDDEN/);
const { loadDeepSeekKey } = await import(process.argv[1]);
assert.throws(() => loadDeepSeekKey(), e => e.code === 'ENOENT'
  && e.path === process.env.DEEPSEEK_API_KEY_FILE
  && e.path.endsWith('OFFLINE_NO_SUCH_KEY_FILE'));
process.stdout.write('offline egress guard active; explicit nonexistent key sentinel verified');
"""
    result = subprocess.run(
        [node, "--import", offline_fetch_guard(tmp_path).as_uri(),
         "--input-type=module", "-e", code, module],
        capture_output=True, text=True, encoding="utf8", timeout=30, check=True,
    )
    assert "offline egress guard active" in result.stdout
    assert not (tmp_path / "models.json").exists()
