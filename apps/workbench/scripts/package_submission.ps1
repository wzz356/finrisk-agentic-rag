param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot "release"
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$resolvedOutput = (Resolve-Path $OutputDirectory).Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$packageName = "FinRisk_可运行项目成果_$stamp"
$stage = Join-Path $resolvedOutput $packageName
$zipPath = Join-Path $resolvedOutput "$packageName.zip"
New-Item -ItemType Directory -Path $stage | Out-Null

$includePaths = @(
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "README.md",
    "DELIVERY_README.md",
    "pyproject.toml",
    "uv.lock",
    "requirements.lock.txt",
    "requirements-dev.lock.txt",
    "install.bat",
    "install.ps1",
    "install.sh",
    "start.bat",
    "start.ps1",
    "start.sh",
    "verify.bat",
    "verify.ps1",
    "verify.sh",
    "configs",
    "agent_runtime",
    "data/eval",
    "data/risk",
    "data/runtime_knowledge",
    "deploy",
    "docs",
    "reports",
    "scripts",
    "src",
    "tests"
)
$excludedDirectories = @(
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "release",
    "node_modules",
    ".cache",
    "cache",
    "runtime",
    "project_cleanup_20261018",
    ".tmp",
    "local-preview"
)

foreach ($relativePath in $includePaths) {
    $source = Join-Path $projectRoot $relativePath
    if (-not (Test-Path -LiteralPath $source)) {
        throw "交付必需路径不存在：$relativePath"
    }

    if (Test-Path -LiteralPath $source -PathType Leaf) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $stage $relativePath)
        continue
    }

    Get-ChildItem -LiteralPath $source -Recurse -File -Force | ForEach-Object {
        $relativeFile = $_.FullName.Substring($projectRoot.Length).TrimStart('\', '/')
        $pathParts = $relativeFile -split '[\\/]'
        $hasExcludedDirectory = $false
        foreach ($part in $pathParts) {
            if ($excludedDirectories -contains $part) {
                $hasExcludedDirectory = $true
                break
            }
        }
        if ($hasExcludedDirectory -or $_.Name -eq ".env" -or $_.Extension -in @(".pyc", ".pyo", ".log", ".zip")) {
            return
        }

        $destination = Join-Path $stage $relativeFile
        $destinationDirectory = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $destination
    }
}

$contentLines = Get-ChildItem -LiteralPath $stage -Recurse -File |
    ForEach-Object { $_.FullName.Substring($stage.Length).TrimStart('\', '/') -replace '\\', '/' } |
    Sort-Object
[System.IO.File]::WriteAllLines(
    (Join-Path $stage "PACKAGE_CONTENTS.txt"),
    $contentLines,
    [System.Text.UTF8Encoding]::new($false)
)

$hashLines = Get-ChildItem -LiteralPath $stage -Recurse -File |
    Where-Object { $_.Name -ne "PACKAGE_MANIFEST.sha256" } |
    Sort-Object FullName |
    ForEach-Object {
        $relativeFile = $_.FullName.Substring($stage.Length).TrimStart('\', '/') -replace '\\', '/'
        "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant(), $relativeFile
    }
[System.IO.File]::WriteAllLines(
    (Join-Path $stage "PACKAGE_MANIFEST.sha256"),
    $hashLines,
    [System.Text.UTF8Encoding]::new($false)
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $stage,
    $zipPath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)

if (-not (Test-Path -LiteralPath $zipPath) -or (Get-Item -LiteralPath $zipPath).Length -le 0) {
    throw "压缩包生成失败。"
}

$resolvedStage = (Resolve-Path $stage).Path
$stagePrefix = $resolvedOutput.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $resolvedStage.StartsWith($stagePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "拒绝清理不在 release 目录内的暂存路径：$resolvedStage"
}
Remove-Item -LiteralPath $resolvedStage -Recurse -Force

$zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
Write-Host "PACKAGE_PATH=$zipPath"
Write-Host "PACKAGE_SHA256=$zipHash"
