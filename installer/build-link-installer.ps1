# Build release binary + compile the AYR Audio Link sender installer.
# Prerequisites: Rust toolchain, Inno Setup 6 from https://jrsoftware.org/isdl.php
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File installer\build-link-installer.ps1
#
# Output: dist\AYR-Audio-Link-Setup-<version>.exe
#
# Design mirrors build-installer.ps1 (the meter's builder). Keep the two in
# lock-step — any change made to one should almost certainly land in the
# other (target-dir handling, UNC path workaround, ISCC discovery).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Same CARGO_TARGET_DIR pin as the meter's builder: Cursor / agent sandboxes
# may preconfigure this to a temp folder, which would mean the installer
# references an exe that isn't where the .iss file expects. Always build
# into the canonical repo `target\`.
$env:CARGO_TARGET_DIR = Join-Path $Root "target"

Write-Host "==> cargo build --release -p ayr-audio-link" -ForegroundColor Cyan
cargo build --release -p ayr-audio-link

$ReleaseExe = Join-Path $Root "target\release\ayr-audio-link.exe"
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

$Iss = Join-Path $PSScriptRoot "ayr-audio-link.iss"
Write-Host "==> & `"$ISCC`" `"$Iss`"" -ForegroundColor Cyan
& $ISCC $Iss

$Dist = Join-Path $Root "dist"
Write-Host ""
Write-Host "Done. Installer:" -ForegroundColor Green
Get-ChildItem $Dist -Filter "AYR-Audio-Link-Setup-*.exe" | ForEach-Object { Write-Host "  $($_.FullName)" }
