# Fast iteration: debug build + staged launch (same %LOCALAPPDATA%\AudioMeter\ copy as release,
# so SmartScreen does not nag about \\wsl.localhost\ paths).
#
# Use this instead of launch-release.ps1 while editing UI - incremental debug builds are much
# quicker than --release.
#
# Optional: -SkipBuild  (used by launch-dev-watch.ps1 after cargo watch already ran `cargo build`)
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\launch-dev.ps1

param([switch]$SkipBuild)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$Manifest = Join-Path $Root "Cargo.toml"
if (-not (Test-Path -LiteralPath $Manifest)) {
    Write-Error "No Cargo.toml next to launch-dev.ps1 - wrong working folder. Expected: $Manifest"
}
Write-Host "==> Package root: $Root" -ForegroundColor DarkGray
Write-Host "==> Manifest:    $Manifest" -ForegroundColor DarkGray

# Keep debug artifacts on NTFS (not the WSL-mounted repo path) so incremental compilation works.
# Building with `target` under `\\wsl$\...` breaks incremental locks - that forced full rebuilds every time.
$DevTargetRoot = Join-Path $env:LOCALAPPDATA "AudioMeter\cargo-target-dev"
$env:CARGO_TARGET_DIR = $DevTargetRoot
Remove-Item Env:CARGO_INCREMENTAL -ErrorAction SilentlyContinue

if (-not $SkipBuild) {
    Write-Host "==> cargo build (debug) package audio-meter-rs -> $DevTargetRoot" -ForegroundColor Cyan
    cargo build -p audio-meter-rs --bin audio-meter-rs
}

$Exe = Join-Path $DevTargetRoot "debug\audio-meter-rs.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Missing $Exe - build failed."
}
$builtSize = (Get-Item -LiteralPath $Exe).Length
Write-Host ("==> Built exe: {0} ({1:N0} bytes)" -f $Exe, $builtSize) -ForegroundColor DarkGray

$Cache = Join-Path $env:LOCALAPPDATA "AudioMeter"
if (-not (Test-Path $Cache)) {
    New-Item -ItemType Directory -Path $Cache | Out-Null
}

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
try { Unblock-File -Path $LocalExe -ErrorAction Stop } catch {}

$built = (Get-Item $Exe).LastWriteTimeUtc.ToString("yyyy-MM-dd HH:mm:ss 'UTC'")
$staged = (Get-Item $LocalExe).LastWriteTimeUtc.ToString("yyyy-MM-dd HH:mm:ss 'UTC'")
Write-Host ("==> Staged exe timestamp: {0} (build output was {1})" -f $staged, $built) -ForegroundColor DarkGray

Write-Host ('==> Starting ' + $LocalExe) -ForegroundColor Cyan
Start-Process -FilePath $LocalExe -WorkingDirectory $Cache -WindowStyle Normal
