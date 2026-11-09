import importlib
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


def initializer(factory, **kwargs):
    name = 'agent_rag.service_initialization'
    assert importlib.util.find_spec(name) is not None, '初始化必须与status请求分离'
    return importlib.import_module(name).ServiceInitialization(factory, **kwargs)


def test_twenty_concurrent_state_reads_do_not_start_factory():
    calls = []
    init = initializer(lambda stage: calls.append(True))
    with ThreadPoolExecutor(max_workers=20) as pool:
        states = list(pool.map(lambda _: init.status(), range(20)))
    assert {row['initialization_state'] for row in states} == {'not_started'}
    assert calls == []
    with pytest.raises(HTTPException) as error:
        init.get_ready()
    assert error.value.status_code == 503


def test_twenty_warmups_share_one_blocked_factory_and_record_phases():
    entered, release = threading.Event(), threading.Event()
    calls = []
    service = object()
    def factory(stage):
        calls.append(True)
        stage('snapshot_load')
        entered.set()
        assert release.wait(2)
        stage('service_build')
        return service
    init = initializer(factory)
    try:
        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(lambda _: init.request_warmup(), range(20)))
        assert entered.wait(1)
        assert calls == [True]
        assert init.status()['initialization_state'] == 'loading'
        with pytest.raises(HTTPException) as error:
            init.get_ready()
        assert error.value.status_code == 503
    finally:
        release.set()
    assert init.wait_ready(2) is service
    assert init.status()['initialization_state'] == 'ready'
    assert [row['phase'] for row in init.status()['initialization_phases']] == [
        'snapshot_load', 'service_build']
    assert all(row['elapsed_ms'] >= 0 for row in init.status()['initialization_phases'])


def test_failed_factory_is_terminal_and_error_text_is_whitelisted():
    calls = []
    def factory(stage):
        calls.append(True)
        raise RuntimeError('Bearer SECRET prompt body')
    init = initializer(factory)
    init.request_warmup()
    with pytest.raises(HTTPException):
        init.wait_ready(2)
    for _ in range(20):
        assert init.request_warmup()['initialization_state'] == 'failed'
        assert init.status()['initialization_error'] == 'SERVICE_INITIALIZATION_FAILED'
    assert calls == [True]
    assert 'SECRET' not in str(init.status())


def test_close_only_closes_own_service_once():
    closed = []
    own, other = object(), object()
    first = initializer(lambda stage: own, closer=closed.append)
    second = initializer(lambda stage: other, closer=closed.append)
    first.request_warmup()
    second.request_warmup()
    assert first.wait_ready(2) is own
    assert second.wait_ready(2) is other
    first.close()
    first.close()
    assert closed == [own]
    assert second.get_ready() is other
    second.close()


def test_close_during_loading_closes_eventual_service_without_retry():
    entered, release = threading.Event(), threading.Event()
    closed = []
    own = object()
    def factory(stage):
        entered.set()
        assert release.wait(2)
        return own
    init = initializer(factory, closer=closed.append)
    init.request_warmup()
    assert entered.wait(1)
    init.close(timeout=0)
    release.set()
    init.close(timeout=2)
    assert closed == [own]
    assert init.request_warmup()['initialization_state'] == 'failed'


def test_status_readiness_and_authenticated_warmup_share_initialization(monkeypatch):
    import agent_rag.router as module
    entered, release = threading.Event(), threading.Event()
    calls = []
    service = SimpleNamespace(config={'execution_mode': 'mock', 'retrieval_profile':
        'v2-hybrid-flash-v1', 'rerank_mode': 'replay', 'profile_id': 'ready-profile'},
        snapshot=SimpleNamespace(manifest={}))
    def factory(stage):
        calls.append(True)
        entered.set()
        assert release.wait(2)
        return service
    init = initializer(factory)
    monkeypatch.setattr(module, '_initialization', init)
    monkeypatch.setenv('FINRISK_AGENT_SERVICE_TOKEN', 'unit-secret')
    monkeypatch.setenv('FINRISK_AGENT_WARMUP_V2', '1')
    app = FastAPI()
    app.include_router(module.router)
    try:
        with TestClient(app) as client:
            def check(_):
                status = client.get('/api/agent-rag/status')
                readiness = client.get('/api/agent-rag/readiness')
                assert status.status_code == 200
                assert readiness.status_code == 503
            with ThreadPoolExecutor(max_workers=20) as pool:
                list(pool.map(check, range(20)))
            assert calls == []
            assert client.post('/internal/agent-rag/warmup').status_code == 401
            assert calls == []
            headers = {'Authorization': 'Bearer unit-secret'}
            with ThreadPoolExecutor(max_workers=20) as pool:
                replies = list(pool.map(lambda _: client.post('/internal/agent-rag/warmup',
                    headers=headers), range(20)))
            assert all(reply.status_code == 202 for reply in replies)
            assert entered.wait(1)
            assert calls == [True]
            assert client.post('/api/agent-rag/tasks', json={'text': '测试'}).status_code == 503
            release.set()
            assert init.wait_ready(2) is service
            assert client.get('/api/agent-rag/readiness').status_code == 200
            ready = client.get('/api/agent-rag/status').json()
            assert ready['initialization_state'] == 'ready'
            assert ready['default_profile_id'] == 'ready-profile'
            assert ready['score_mode'] == 'replay'
            assert ready['live_model'] == 'deepseek-flash'
    finally:
        release.set()
        init.close()


def test_router_failed_readiness_does_not_retry_or_echo_secrets(monkeypatch):
    import agent_rag.router as module
    calls = []
    def factory(stage):
        calls.append(True)
        raise ValueError('SECRET auth header')
    init = initializer(factory)
    monkeypatch.setattr(module, '_initialization', init)
    monkeypatch.setenv('FINRISK_AGENT_WARMUP_V2', '1')
    monkeypatch.setenv('FINRISK_AGENT_SERVICE_TOKEN', 'unit-secret')
    app = FastAPI()
    app.include_router(module.router)
    with TestClient(app) as client:
        client.post('/internal/agent-rag/warmup', headers={'Authorization': 'Bearer unit-secret'})
        with pytest.raises(HTTPException):
            init.wait_ready(2)
        for _ in range(20):
            response = client.get('/api/agent-rag/readiness')
            assert response.status_code == 503
            assert 'SECRET' not in response.text
            assert client.post('/api/agent-rag/tasks', json={'text': '测试'}).status_code == 503
        assert calls == [True]


@pytest.mark.parametrize('final_state', ['ready', 'failed'])
def test_startup_wait_warms_once_and_records_only_safe_diagnostics(tmp_path, final_state):
    import json
    import shutil
    import subprocess
    from pathlib import Path
    script = Path(__file__).resolve().parents[2] / 'start-agent-rag.ps1'
    source = script.read_text(encoding='utf-8-sig')
    assert 'function Wait-AgentRagReadiness' in source, '启动须单次预热再有限轮询'
    wrapper = tmp_path / 'startup-contract.ps1'
    wrapper.write_text(r'''
param([string]$StartupScript,[string]$DiagnosticPath,[string]$FinalState)
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($StartupScript,[ref]$tokens,[ref]$parseErrors)
$definition = $ast.Find({param($item) $item -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq 'Wait-AgentRagReadiness'}, $true)
. ([scriptblock]::Create($definition.Extent.Text))
$script:requests = [System.Collections.Generic.List[string]]::new()
function Invoke-RestMethod {
    param($Uri,$Method,$Headers,$TimeoutSec,$ErrorAction)
    if ($TimeoutSec -gt 2) { throw 'request timeout exceeded' }
    $script:requests.Add("$Method $Uri")
    $state = if ($Uri.EndsWith('/status')) { 'not_started' } elseif ($Uri.EndsWith('/warmup')) { 'loading' } else { $FinalState }
    return @{initialization_state=$state; initialization_phase='snapshot_load'; initialization_elapsed_ms=10; initialization_error=$(if ($state -eq 'failed') {'SECRET'} else {$null}); secret='Bearer SECRET'}
}
function Start-Sleep { param($Milliseconds) }
$failed = $false
try {
    Wait-AgentRagReadiness -BaseUrl 'http://127.0.0.1:1234' -Headers @{Authorization='Bearer SECRET'} -TimeoutSeconds 5 -DiagnosticPath $DiagnosticPath
} catch { $failed = $true }
@{requests=@($script:requests); failed=$failed} | ConvertTo-Json -Compress
''', encoding='utf-8')
    diagnostic = tmp_path / 'initialization.json'
    executable = shutil.which('pwsh') or shutil.which('powershell')
    assert executable is not None
    result = subprocess.run([executable, '-NoProfile', '-File', str(wrapper),
        '-StartupScript', str(script), '-DiagnosticPath', str(diagnostic),
        '-FinalState', final_state], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout)
    assert len([row for row in outcome['requests'] if row.startswith('Post ')]) == 1
    assert outcome['failed'] is (final_state == 'failed')
    assert 'SECRET' not in result.stdout + result.stderr
    assert diagnostic.exists() is (final_state == 'failed')
    if diagnostic.exists():
        assert 'SECRET' not in diagnostic.read_text(encoding='utf-8-sig')


def test_router_cleans_snapshot_worker_if_task_service_construction_fails(tmp_path, monkeypatch):
    import agent_rag.router as module
    snapshot_root = tmp_path / 'snapshot'
    snapshot_root.mkdir()
    (snapshot_root / 'manifest.json').write_text('{"schema_version":"finrisk-v2-snapshot/1.0"}')
    closed = []
    snapshot = SimpleNamespace(backend=SimpleNamespace(dense=SimpleNamespace(close=lambda: closed.append(True))))
    monkeypatch.setenv('FINRISK_AGENT_SNAPSHOT', str(snapshot_root))
    monkeypatch.setattr(module, 'load_v2_snapshot', lambda path: snapshot)
    def fail(*args):
        raise ValueError('SECRET config error')
    monkeypatch.setattr(module, 'TaskService', fail)
    init = initializer(module._build_service)
    init.request_warmup()
    with pytest.raises(HTTPException):
        init.wait_ready(2)
    assert closed == [True]
    assert init.status()['initialization_error'] == 'SERVICE_INITIALIZATION_FAILED'
    for _ in range(20):
        init.request_warmup()
    assert closed == [True]


def test_startup_wait_parses_real_http_503_and_never_reposts_warmup(tmp_path):
    import json
    import shutil
    import subprocess
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def respond(self, code, state):
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'initialization_state': state,
                'initialization_error': 'SERVICE_INITIALIZATION_FAILED' if state == 'failed' else None}).encode())
        def do_POST(self):
            requests.append(('POST', self.path))
            self.respond(202, 'loading')
        def do_GET(self):
            requests.append(('GET', self.path))
            self.respond(200 if self.path.endswith('/status') else 503,
                         'not_started' if self.path.endswith('/status') else 'failed')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = Path(__file__).resolve().parents[2] / 'start-agent-rag.ps1'
    wrapper = tmp_path / 'http503.ps1'
    wrapper.write_text(r'''
param([string]$StartupScript,[string]$BaseUrl,[string]$DiagnosticPath)
$ErrorActionPreference='Stop'
$tokens=$null; $parseErrors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile($StartupScript,[ref]$tokens,[ref]$parseErrors)
$definition=$ast.Find({param($item) $item -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq 'Wait-AgentRagReadiness'},$true)
. ([scriptblock]::Create($definition.Extent.Text))
try { Wait-AgentRagReadiness -BaseUrl $BaseUrl -Headers @{} -TimeoutSeconds 5 -DiagnosticPath $DiagnosticPath; exit 2 }
catch { exit 0 }
''', encoding='utf-8')
    diagnostic = tmp_path / 'failed.json'
    try:
        executable = shutil.which('pwsh') or shutil.which('powershell')
        result = subprocess.run([executable, '-NoProfile', '-File', str(wrapper),
            '-StartupScript', str(script), '-BaseUrl', f'http://127.0.0.1:{server.server_port}',
            '-DiagnosticPath', str(diagnostic)], capture_output=True, text=True, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert requests == [('GET', '/api/agent-rag/status'), ('POST', '/internal/agent-rag/warmup'),
                        ('GET', '/api/agent-rag/readiness')]
    assert json.loads(diagnostic.read_text(encoding='utf-8-sig'))['error_code'] == 'SERVICE_INITIALIZATION_FAILED'
