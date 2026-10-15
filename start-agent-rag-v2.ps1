param(
    [Parameter(Mandatory=$true)][string]$SnapshotPath,
    [string]$PythonPath = (Join-Path $PSScriptRoot '.agent-venv\Scripts\python.exe'),
    [string]$NodePath = 'node',
    [ValidateRange(1,65535)][int]$PythonPort = 8773,
    [ValidateRange(1,65535)][int]$NodePort = 8774,
    [ValidateRange(1,3600)][int]$StartupTimeoutSeconds = 300,
    [string]$TaskRoot = (Join-Path $PSScriptRoot 'runtime\agent_rag\v2\tasks'),
    [switch]$AllowPaidRetrieval,
    [double]$RerankCapCny = 0,
    [ValidateRange(1,240)][int]$RerankMaxRequests = 240,
    [switch]$EnableLiveAgent,
    [string]$ModelTrialLedger = '',
    [string]$KeyFile = ''
)
$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath (Join-Path $SnapshotPath 'manifest.json') -Encoding UTF8 -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 'finrisk-v2-snapshot/1.0') { throw '需要独立V2来源快照。' }
if ($manifest.backend.mode -notin @('replay','live')) { throw 'V2重排模式只能为replay或live。' }
if ($PythonPort -eq $NodePort) { throw 'Python与Node需要不同端口。' }
foreach ($port in @($PythonPort,$NodePort)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "端口$port已占用，请改用空闲端口。" }
}
$paid = $manifest.backend.mode -eq 'live'
if ($paid -and (-not $AllowPaidRetrieval -or $RerankCapCny -le 0 -or [double]::IsInfinity($RerankCapCny) -or [double]::IsNaN($RerankCapCny))) { throw '真实重排必须显式指定-AllowPaidRetrieval和正数-RerankCapCny。' }
if ($EnableLiveAgent -and -not $paid) { throw '真实Agent须配合真实重排快照；回放不冒充真实工具。' }
$env:FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE = if ($paid) { '1' } else { '0' }
$env:FINRISK_AGENT_RERANK_CAP_CNY = "$RerankCapCny"
$env:FINRISK_AGENT_RERANK_MAX_REQUESTS = "$RerankMaxRequests"
$env:FINRISK_AGENT_DEEPSEEK_KEY_FILE = $KeyFile
if ($EnableLiveAgent) {
    if (-not $ModelTrialLedger) { $ModelTrialLedger = Join-Path $TaskRoot 'model-trial-ledger.json' }
    $env:FINRISK_AGENT_TRIAL_LEDGER = $ModelTrialLedger
} else {
    Remove-Item Env:FINRISK_AGENT_TRIAL_LEDGER -ErrorAction SilentlyContinue
}
$logPath = Join-Path (Split-Path -Parent $TaskRoot) ('service-logs-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))
Write-Host $(if ($paid) { "V2真实重排：单任务费用上限$RerankCapCny 元、最多$RerankMaxRequests 次评分请求。" } else { 'V2冻结评分回放；不发起模型API请求。仅冻结查询有已存评分。' })
if ($EnableLiveAgent) { Write-Host 'Agent模型使用独立持久试跑账本：整本上限1元/18次；单任务模型仍最多6次。历史额度不重置。' }
& (Join-Path $PSScriptRoot 'start-agent-rag.ps1') -PythonPath $PythonPath -NodePath $NodePath -SnapshotPath $SnapshotPath -TaskRoot $TaskRoot -ServiceLogPath $logPath -PythonPort $PythonPort -NodePort $NodePort -WarmupV2 -StartupTimeoutSeconds $StartupTimeoutSeconds -EnableLive:$EnableLiveAgent -KeyFile $KeyFile
