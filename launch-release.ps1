# Build and run the release binary in THIS repo's target\release\ (same path the installer uses).
# Use this when launching from Cursor so builds are not redirected to a sandbox target dir.
#
# Staging step: after build, the .exe is copied to %LOCALAPPDATA%\AudioMeter\ and launched from
# there. Running the binary directly from \\wsl.localhost\... places it in the Internet security
# zone, which makes Windows show the "We can't verify who created this file" Open File - Security
# Warning every single launch. Local-drive execution is trusted by SmartScreen, so the warning
# goes away for dev builds. The original path under target\release\audio-meter-rs.exe is left
# untouched so installer\build-installer.ps1 can still package it.
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

$Cache = Join-Path $env:LOCALAPPDATA "AudioMeter"
if (-not (Test-Path $Cache)) {
    New-Item -ItemType Directory -Path $Cache | Out-Null
}

# Close dev/staged instances so Copy-Item is not blocked and we always run the fresh build.
Write-Host "==> Stopping running Audio Meter (if any)" -ForegroundColor Cyan
$cacheFull = (Resolve-Path $Cache).Path
foreach ($p in Get-Process -ErrorAction SilentlyContinue) {
    try {
        $exePath = $p.MainModule.FileName
    } catch {
        continue
    }
    if (-not $exePath) { continue }
    $dir = [IO.Path]::GetDirectoryName($exePath)
    $leaf = [IO.Path]::GetFileName($exePath)
    if ($dir.Equals($cacheFull, [StringComparison]::OrdinalIgnoreCase) -and ($leaf -like 'audio-meter-rs*.exe')) {
        Write-Host ('    Stopping PID ' + $p.Id + ' ' + $leaf) -ForegroundColor DarkGray
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Milliseconds 400

$LocalExe = Join-Path $Cache "audio-meter-rs.exe"
Copy-Item -Path $Exe -Destination $LocalExe -Force
# Strip any Mark-of-the-Web ADS that may have been attached (no-op when absent).
try { Unblock-File -Path $LocalExe -ErrorAction Stop } catch {}

Write-Host ('==> Starting ' + $LocalExe) -ForegroundColor Cyan
Start-Process -FilePath $LocalExe -WorkingDirectory $Cache -WindowStyle Normal
