$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "尚未安装项目依赖。请先运行 install.bat 或 ./install.ps1。"
}

$envPath = Join-Path $projectRoot ".env"
if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") -Destination $envPath
}

Set-Location -LiteralPath $projectRoot
Write-Host "FinRisk 正在启动：http://127.0.0.1:8000"
& $venvPython (Join-Path $projectRoot "scripts\run_web.py")
exit $LASTEXITCODE
