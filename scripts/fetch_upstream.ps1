[CmdletBinding()]
param(
    [string]$RepositoryUrl = "https://github.com/AlphaGPU/leetgpu-challenges.git",
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Destination) {
    $Destination = Join-Path $projectRoot "upstream\leetgpu-challenges"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
$challengeRoot = Join-Path $Destination "challenges"

if (Test-Path -LiteralPath $Destination) {
    if (Test-Path -LiteralPath $challengeRoot) {
        Write-Host "Upstream challenge repository is already available at $Destination"
        exit 0
    }
    throw "Destination already exists but is not a LeetGPU challenge repository: $Destination"
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to download the upstream challenge repository."
}

$parent = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $parent | Out-Null
& git clone --depth 1 $RepositoryUrl $Destination
if ($LASTEXITCODE -ne 0) {
    throw "Failed to clone $RepositoryUrl"
}
if (-not (Test-Path -LiteralPath $challengeRoot)) {
    throw "The cloned repository does not contain a challenges directory."
}

Write-Host "Upstream challenges are ready at $challengeRoot"
