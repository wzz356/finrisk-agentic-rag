param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '.agent-venv\Scripts\python.exe'),
    [string]$NodePath = 'node',
    [string]$SnapshotPath = (Join-Path $PSScriptRoot 'runtime\agent_rag\snapshot-v1'),
    [string]$KnowledgeRegistry = '',
    [string]$TaskRoot = (Join-Path $PSScriptRoot 'runtime\agent_rag\tasks'),
    [string]$ServiceLogPath = (Join-Path $PSScriptRoot 'runtime\agent_rag\service-logs'),
    [int]$PythonPort = 8765,
    [int]$NodePort = 8766,
    [switch]$WarmupV2,
    [ValidateRange(1,3600)][int]$StartupTimeoutSeconds = 300,
    [switch]$EnableLive,
    [string]$KeyFile = ''
)
$ErrorActionPreference = 'Stop'

function Remember-AgentOwnedProcessTree {
    param([Diagnostics.Process]$OwnedProcess)
    if (-not $OwnedProcess) { return }
    $OwnedProcess.Refresh()
    if ($OwnedProcess.HasExited) { return }
    if (-not $OwnedProcess.PSObject.Properties['FinRiskOwnedChildren']) {
        $OwnedProcess | Add-Member -NotePropertyName FinRiskOwnedChildren -NotePropertyValue ([Collections.Generic.List[Diagnostics.Process]]::new())
    }
    foreach ($row in @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($OwnedProcess.Id)" -ErrorAction Stop)) {
        $child = Get-Process -Id $row.ProcessId -ErrorAction SilentlyContinue
        if (-not $child) { continue }
        try { $started = $child.StartTime } catch { continue }
        if ($started -lt $OwnedProcess.StartTime) { continue }
        $known = $OwnedProcess.FinRiskOwnedChildren | Where-Object { $_.Id -eq $child.Id -and -not $_.HasExited }
        if (-not $known) { $OwnedProcess.FinRiskOwnedChildren.Add($child) } else { $child = @($known)[0] }
        Remember-AgentOwnedProcessTree $child
    }
}

function Stop-AgentOwnedProcessTree {
    param([Diagnostics.Process]$OwnedProcess)
    if (-not $OwnedProcess) { return }
    if ($OwnedProcess.PSObject.Properties['FinRiskOwnedChildren']) {
        foreach ($child in $OwnedProcess.FinRiskOwnedChildren) { Stop-AgentOwnedProcessTree $child }
    }
    $OwnedProcess.Refresh()
    if (-not $OwnedProcess.HasExited) {
        # The retained Process handle proves this is the instance we started.
        & (Get-Command taskkill.exe -ErrorAction Stop).Source /PID $OwnedProcess.Id /T /F *> $null
        if ($LASTEXITCODE -ne 0) {
            $OwnedProcess.Refresh()
            if (-not $OwnedProcess.HasExited) { throw 'OWNED_PROCESS_TREE_STOP_FAILED' }
        }
        return
    }
    # The parent can already have crashed. Only descendants born during its
    # actual lifetime belong to it; a reused PID must never claim new children.
    $exitTime = $OwnedProcess.ExitTime
    if (-not $exitTime) { return } # Already-reaped descendants were covered by retained handles above.
    $started = $OwnedProcess.StartTime.ToUniversalTime()
    $ended = $exitTime.ToUniversalTime()
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($OwnedProcess.Id)" -ErrorAction Stop)
    foreach ($row in $children) {
        $child = Get-Process -Id $row.ProcessId -ErrorAction SilentlyContinue
        if ($child) {
            try { $created = $child.StartTime.ToUniversalTime() } catch { continue }
            if ($created -ge $started -and $created -le $ended) { Stop-AgentOwnedProcessTree $child }
        }
    }
}

function Wait-AgentRagReadiness {
    param([string]$BaseUrl, [hashtable]$Headers, [int]$TimeoutSeconds,
          [string]$DiagnosticPath, $PythonProcess = $null, $NodeProcess = $null)
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $lastState = @{ initialization_state = 'not_started' }
    $startupError = 'STARTUP_TIMEOUT'
    try {
        # First wait for the lightweight HTTP listener; this never loads the snapshot.
        $listening = $false
        while (-not $listening) {
            foreach ($process in @($PythonProcess, $NodeProcess)) {
                if ($process) {
                    Remember-AgentOwnedProcessTree $process
                    $process.Refresh()
                    if ($process.HasExited) { $startupError = 'SERVICE_PROCESS_EXITED'; throw $startupError }
                }
            }
            $remaining = $TimeoutSeconds - $timer.Elapsed.TotalSeconds
            $requestTimeout = [Math]::Min(2, [Math]::Floor($remaining))
            if ($requestTimeout -lt 1) { throw $startupError }
            try {
                $lastState = Invoke-RestMethod -Uri "$BaseUrl/api/agent-rag/status" -Method Get -TimeoutSec $requestTimeout -ErrorAction Stop
                $listening = $true
            } catch {
                if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) { throw $startupError }
                Start-Sleep -Milliseconds 200
            }
        }
        $requestTimeout = [Math]::Min(2, [Math]::Floor($TimeoutSeconds - $timer.Elapsed.TotalSeconds))
        if ($requestTimeout -lt 1) { throw $startupError }
        # The warmup command is sent exactly once; a lost response never causes a retry.
        $startupError = 'WARMUP_REQUEST_FAILED'
        $lastState = Invoke-RestMethod -Uri "$BaseUrl/internal/agent-rag/warmup" -Method Post -Headers $Headers -TimeoutSec $requestTimeout -ErrorAction Stop
        $startupError = 'STARTUP_TIMEOUT'
        while ($true) {
            foreach ($process in @($PythonProcess, $NodeProcess)) {
                if ($process) {
                    Remember-AgentOwnedProcessTree $process
                    $process.Refresh()
                    if ($process.HasExited) { $startupError = 'SERVICE_PROCESS_EXITED'; throw $startupError }
                }
            }
            $requestTimeout = [Math]::Min(2, [Math]::Floor($TimeoutSeconds - $timer.Elapsed.TotalSeconds))
            if ($requestTimeout -lt 1) { throw $startupError }
            try {
                $lastState = Invoke-RestMethod -Uri "$BaseUrl/api/agent-rag/readiness" -Method Get -TimeoutSec $requestTimeout -ErrorAction Stop
            } catch {
                # Readiness intentionally uses HTTP 503 while loading/failed.
                try { $lastState = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction Stop }
                catch { $lastState = @{ initialization_state = 'loading' } }
            }
            if ($lastState.initialization_state -eq 'ready') { return }
            if ($lastState.initialization_state -eq 'failed') {
                $startupError = 'SERVICE_INITIALIZATION_FAILED'
                throw $startupError
            }
            Start-Sleep -Milliseconds 200
        }
    } catch {
        # Never persist request headers or exception bodies in startup diagnostics.
        $safeState = @{ error_code = $startupError; elapsed_ms = [Math]::Round($timer.Elapsed.TotalMilliseconds, 3) }
        $allowedValues = @{
            initialization_state = @('not_started', 'loading', 'ready', 'failed')
            initialization_phase = @('asset_validation', 'snapshot_load', 'service_build')
            initialization_error = @('SERVICE_CLOSED', 'SERVICE_INITIALIZATION_FAILED', 'SNAPSHOT_NOT_READY', 'DENSE_STARTUP_FAILED', 'DENSE_RESPONSE_TIMEOUT', 'DENSE_PROTOCOL_ERROR', 'LIVE_RERANK_NOT_ENABLED', 'BACKEND_ASSET_HASH_MISMATCH')
        }
        foreach ($field in $allowedValues.Keys) {
            $value = $lastState.$field
            if ($value -is [string] -and $allowedValues[$field] -contains $value) { $safeState[$field] = $value }
        }
        $safeState | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $DiagnosticPath -Encoding UTF8
        throw "V2服务启动未完成；阶段诊断已保存：$DiagnosticPath"
    }
}

if (-not (Test-Path -LiteralPath $PythonPath)) { throw '独立Python环境不存在，请先执行运行说明中的安装步骤。' }
if ($KnowledgeRegistry) {
    $registryPath = (Resolve-Path -LiteralPath $KnowledgeRegistry).Path
    $SnapshotPath = & $PythonPath -c 'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from agent_rag.knowledge_registry import current_snapshot; print(current_snapshot(Path(sys.argv[2])))' (Join-Path $PSScriptRoot 'src') $registryPath
    if ($LASTEXITCODE -ne 0) { throw '知识库版本指针无效，请先完成发布或回滚。' }
    $env:FINRISK_AGENT_KB_REGISTRY = $registryPath
    $WarmupV2 = $true
} else {
    Remove-Item Env:FINRISK_AGENT_KB_REGISTRY -ErrorAction SilentlyContinue
}
if (-not (Test-Path -LiteralPath (Join-Path $SnapshotPath 'manifest.json'))) { throw '来源快照不存在，请先执行运行说明中的构建步骤。' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'agent_harness\node_modules\@earendil-works\pi-agent-core'))) { throw 'Pi依赖未安装，请先执行npm ci。' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'agent_harness\dist\src\server.js'))) { throw '运行层未编译，请先执行npm run build。' }
if ($PythonPort -eq $NodePort) { throw 'Python与Node需要不同端口。' }
foreach ($port in @($PythonPort, $NodePort)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "端口$port已占用，请改用空闲端口。" }
}
$resolvedNode = (Get-Command $NodePath -ErrorAction Stop).Source
$version = & $resolvedNode -p 'process.versions.node'
if ([version]$version -lt [version]'22.19.0') { throw 'Node版本须不低于22.19.0。' }
& $PythonPath -c 'import fastapi,httpx,jsonschema,uvicorn'
if ($LASTEXITCODE -ne 0) { throw 'Python依赖不完整，请按运行说明安装requirements-agent-rag.txt。' }
$env:FINRISK_AGENT_SERVICE_TOKEN = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
$env:FINRISK_AGENT_SNAPSHOT = (Resolve-Path -LiteralPath $SnapshotPath).Path
$env:FINRISK_AGENT_TASK_ROOT = $TaskRoot
$env:FINRISK_AGENT_PYTHON_URL = "http://127.0.0.1:$PythonPort"
$env:FINRISK_AGENT_NODE_URL = "http://127.0.0.1:$NodePort"
$env:FINRISK_AGENT_NODE_PORT = "$NodePort"
$env:FINRISK_AGENT_ALLOW_LIVE = if ($EnableLive) { '1' } else { '0' }
$env:FINRISK_AGENT_WARMUP_V2 = if ($WarmupV2) { '1' } else { '0' }
$env:DEEPSEEK_API_KEY_FILE = $KeyFile
$logPath = $ServiceLogPath
New-Item -ItemType Directory -Path $logPath -Force | Out-Null
$pythonProcess = $null
$nodeProcess = $null
try {
    $pythonArgs = @('-m', 'uvicorn', 'agent_rag.app:app', '--app-dir', ('"' + (Join-Path $PSScriptRoot 'src') + '"'), '--host', '127.0.0.1', '--port', "$PythonPort", '--workers', '1')
    $pythonProcess = Start-Process -FilePath $PythonPath -ArgumentList $pythonArgs -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logPath 'python.stdout.log') -RedirectStandardError (Join-Path $logPath 'python.stderr.log')
    $nodeProcess = Start-Process -FilePath $resolvedNode -ArgumentList @('"' + (Join-Path $PSScriptRoot 'agent_harness\dist\src\server.js') + '"') -WorkingDirectory (Join-Path $PSScriptRoot 'agent_harness') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logPath 'node.stdout.log') -RedirectStandardError (Join-Path $logPath 'node.stderr.log')
    $null = $pythonProcess.StartTime
    $null = $nodeProcess.StartTime
    if ($WarmupV2) {
        Wait-AgentRagReadiness -BaseUrl "http://127.0.0.1:$PythonPort" -Headers @{ Authorization = "Bearer $env:FINRISK_AGENT_SERVICE_TOKEN" } -TimeoutSeconds $StartupTimeoutSeconds -DiagnosticPath (Join-Path $logPath 'initialization.json') -PythonProcess $pythonProcess -NodeProcess $nodeProcess
        Write-Host 'V2来源服务已就绪。'
    }
    Write-Host "演示页面：http://127.0.0.1:$PythonPort/agent-rag"
    Write-Host $(if ($EnableLive) { '真实Agent入口已启用；提交live profile会调用收费接口。' } elseif ($env:FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE -eq '1') { 'Agent为模拟模式；V2检索会调用真实收费重排接口，按任务账本限制。' } else { '当前为模拟模式，模型API调用为0。' })
    Write-Host '按Ctrl+C停止两个本轮服务。'
    while (-not $pythonProcess.HasExited -and -not $nodeProcess.HasExited) { Remember-AgentOwnedProcessTree $pythonProcess; Remember-AgentOwnedProcessTree $nodeProcess; Start-Sleep -Seconds 1; $pythonProcess.Refresh(); $nodeProcess.Refresh() }
} finally {
    try { Stop-AgentOwnedProcessTree $pythonProcess }
    finally { Stop-AgentOwnedProcessTree $nodeProcess }
}
