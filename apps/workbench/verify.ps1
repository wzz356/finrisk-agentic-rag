$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "尚未安装测试依赖。请先运行 install.bat 或 ./install.ps1。"
}

Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot ".tmp") | Out-Null
$env:TEMP = Join-Path $projectRoot ".tmp"
$env:TMP = $env:TEMP

& $venvPython "scripts\verify_delivery.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $venvPython -m ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $venvPython -m mypy src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $venvPython -m pytest -q --basetemp=.tmp/pytest-verify
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCommand) {
    & $nodeCommand.Source --check "src\web\static\app.js"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $nodeCommand.Source --check "src\web\static\source-map.js"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $nodeCommand.Source --test "tests/web/*.test.mjs"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $nodeCommand.Source --check "agent_runtime\worker.mjs"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $nodeCommand.Source --check "agent_runtime\runtime.mjs"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $nodeCommand.Source --test "agent_runtime/test/*.test.mjs"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    throw "未安装 Node.js，无法验证 Pi Agent。"
}

Write-Host "FinRisk 全部交付验证通过。"
