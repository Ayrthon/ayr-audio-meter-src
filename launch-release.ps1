# Build and run the release binary in THIS repo's target\release\ (same path the installer uses).
# Use this when launching from Cursor so builds are not redirected to a sandbox target dir.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\launch-release.ps1

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$env:CARGO_TARGET_DIR = Join-Path $Root "target"

Write-Host "==> cargo build --release" -ForegroundColor Cyan
cargo build --release

$Exe = Join-Path $Root "target\release\audio-meter-rs.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Missing $Exe - build failed."
}

Write-Host ('==> Starting ' + $Exe) -ForegroundColor Cyan
Start-Process -FilePath $Exe -WorkingDirectory $Root -WindowStyle Normal
