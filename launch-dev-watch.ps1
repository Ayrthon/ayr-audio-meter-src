# Watch src/ and Cargo files; on each change run launch-dev.ps1 (`cargo build` debug + stage + restart).
#
# One-time (Windows, same Rust as your builds):
#   cargo install cargo-watch
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\launch-dev-watch.ps1
# WSL:    ./launch-dev-watch.sh

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$DevTargetRoot = Join-Path $env:LOCALAPPDATA "AudioMeter\cargo-target-dev"
$env:CARGO_TARGET_DIR = $DevTargetRoot
Remove-Item Env:CARGO_INCREMENTAL -ErrorAction SilentlyContinue

$cargoWatch = Get-Command cargo-watch -ErrorAction SilentlyContinue
if (-not $cargoWatch) {
    Write-Host "error: cargo-watch not on PATH." -ForegroundColor Red
    Write-Host "Install once:  cargo install cargo-watch" -ForegroundColor Yellow
    exit 1
}

$LaunchDev = Join-Path $Root "launch-dev.ps1"
Write-Host "==> cargo watch: save a file -> incremental build -> restart (Ctrl+C to stop)" -ForegroundColor Cyan
Write-Host "    Target dir (fast incremental): $DevTargetRoot" -ForegroundColor DarkGray
Write-Host "    Watches: src, Cargo.toml, Cargo.lock" -ForegroundColor DarkGray

# Do NOT use `-x build -s ...`: cargo-watch runs that via CMD.exe, which refuses UNC cwd
# (`\\wsl$\...`) and breaks `cargo build`. Run only PowerShell; `launch-dev.ps1` does `cargo build`.
cargo watch --delay 0.35 -w src -w Cargo.toml -w Cargo.lock -- powershell -NoProfile -ExecutionPolicy Bypass -File "$LaunchDev"
