# Build release binary + compile Windows installer (Inno Setup 6).
# Prerequisites: Rust toolchain, Inno Setup 6 from https://jrsoftware.org/isdl.php
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File installer\build-installer.ps1
#
# The installed "Audio Meter.exe" is the same `target\release\audio-meter-rs.exe` from this build.
# `.cargo\config.toml` uses static CRT on gnullvm/gnu hosts so users do not need libunwind.dll.
# Optional: set AUDIO_METER_BUILD for the in-app credit line, e.g.
#   $env:AUDIO_METER_BUILD = "$(git describe --tags --always)"; .\installer\build-installer.ps1

$ErrorActionPreference = "Stop"
# Avoid `Resolve-Path (Join-Path $PSScriptRoot "..")` — under PowerShell the
# UNC / `\\wsl.localhost\...` provider rewrites that into a malformed
# `Microsoft.PowerShell.Core\FileSystem::\\...` path that breaks `cargo build`.
# `Split-Path -Parent` keeps the path normal on both local drives and UNC.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Cursor/agent sandboxes may set CARGO_TARGET_DIR to a temp folder — force the normal project
# `target\` so this exe matches local builds and `target\release\audio-meter-rs.exe`.
$env:CARGO_TARGET_DIR = Join-Path $Root "target"

Write-Host "==> cargo build --release" -ForegroundColor Cyan
cargo build --release

$ReleaseExe = Join-Path $Root "target\release\audio-meter-rs.exe"
if (-not (Test-Path $ReleaseExe)) {
    Write-Error "Missing $ReleaseExe - release build failed."
}

$Candidates = @(
    Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"
    Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"
)
$ISCC = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $ISCC) {
    Write-Error @"
Inno Setup 6 compiler (ISCC.exe) not found.
Install Inno Setup 6: https://jrsoftware.org/isdl.php
Then re-run this script.
"@
}

$Iss = Join-Path $PSScriptRoot "audio-meter.iss"
Write-Host "==> & `"$ISCC`" `"$Iss`"" -ForegroundColor Cyan
& $ISCC $Iss

$Dist = Join-Path $Root "dist"
Write-Host ""
Write-Host "Done. Installer:" -ForegroundColor Green
Get-ChildItem $Dist -Filter "AYR-Audio-Meter-Setup-*.exe" | ForEach-Object { Write-Host "  $($_.FullName)" }
