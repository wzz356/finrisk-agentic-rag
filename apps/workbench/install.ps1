param(
    [switch]$RuntimeOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "未找到 Python。请先安装 Python 3.11 或更高版本，并将 python 加入 PATH。"
}

$python = $pythonCommand.Source
& $python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 版本过低；FinRisk 要求 Python 3.11 或更高版本。"
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $python -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "创建虚拟环境失败。"
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "升级 pip 失败。"
}

$requirements = if ($RuntimeOnly) { "requirements.lock.txt" } else { "requirements-dev.lock.txt" }
& $venvPython -m pip install --requirement (Join-Path $projectRoot $requirements)
if ($LASTEXITCODE -ne 0) {
    throw "安装 Python 依赖失败。"
}

$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if (-not $npmCommand -or -not $nodeCommand) {
    throw "Pi Agent 需要 Node.js 22.19 或更高版本及 npm；安装后重新运行 install.ps1。"
}
& $nodeCommand.Source -e "const [major,minor]=process.versions.node.split('.').map(Number);process.exit(major>22||(major===22&&minor>=19)?0:1)"
if ($LASTEXITCODE -ne 0) { throw "Pi Agent 要求 Node.js 22.19 或更高版本。" }
Push-Location -LiteralPath (Join-Path $projectRoot "agent_runtime")
try {
    & $npmCommand.Source ci --ignore-scripts --no-audit --no-fund --cache .cache/npm
    if ($LASTEXITCODE -ne 0) { throw "安装 Pi Agent 依赖失败。" }
} finally {
    Pop-Location
}

$envPath = Join-Path $projectRoot ".env"
if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") -Destination $envPath
}

& $venvPython (Join-Path $projectRoot "scripts\verify_delivery.py") --allow-missing-knowledge
if ($LASTEXITCODE -ne 0) {
    throw "交付数据校验失败。"
}

Write-Host "FinRisk 依赖安装完成。知识库就绪后运行 start.bat 或 ./start.ps1 启动系统。"
