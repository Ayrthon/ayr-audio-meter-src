# Build release binary + compile Windows installer (Inno Setup 6).
# Prerequisites: Rust toolchain, Inno Setup 6 from https://jrsoftware.org/isdl.php
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File installer\build-installer.ps1
#
# The installed "Audio Meter.exe" is built with the **MSVC** target so `winres` links cleanly.
# (Default `x86_64-pc-windows-gnullvm` release builds can fail final link with `-lresource` when
# embedding the Windows icon.) Binary: `%LOCALAPPDATA%\AudioMeter\cargo-target-msvc\...\` when the
# repo is on a WSL UNC path (see script); otherwise `target\x86_64-pc-windows-msvc\release\`.
# `.cargo\config.toml` static-links the MSVC CRT so end users don't need the VC++ redistributable.
# Optional: set AUDIO_METER_BUILD for the in-app credit line, e.g.
#   $env:AUDIO_METER_BUILD = "$(git describe --tags --always)"; .\installer\build-installer.ps1

$ErrorActionPreference = "Stop"

# `winres` invokes the Windows SDK `rc.exe`. A plain PowerShell session (especially when launched
# from WSL) may not have `WindowsSdkDir` / Kit roots set, which yields a broken path like
# `\bin\x64\rc.exe` and fails the build script. Import the MSVC + SDK environment from vcvars.
function Import-VcVars64 {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        return
    }
    $installPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null
    if (-not $installPath) {
        $installPath = & $vswhere -latest -products * -property installationPath 2>$null
    }
    if (-not $installPath) {
        return
    }
    $vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path -LiteralPath $vcvars)) {
        return
    }
    # `cmd.exe` rejects a UNC current directory ("defaulting to Windows directory"), which prevents
    # vcvars from wiring up `WindowsSdkDir` / `rc.exe`. Always invoke it from a local path.
    Push-Location $env:TEMP
    try {
        cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                Set-Item -Path "env:$($matches[1])" -Value $matches[2]
            }
        }
    } finally {
        Pop-Location
    }
}
Import-VcVars64

# Avoid `Resolve-Path (Join-Path $PSScriptRoot "..")` — under PowerShell the
# UNC / `\\wsl.localhost\...` provider rewrites that into a malformed
# `Microsoft.PowerShell.Core\FileSystem::\\...` path that breaks `cargo build`.
# `Split-Path -Parent` keeps the path normal on both local drives and UNC.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Prefer the repo's `target\` so outputs match local builds. When the project lives on the WSL UNC
# share (`\\wsl$`, `\\wsl.localhost\...`), MSVC rustc often fails with "Access is denied" writing
# under that path (parallel dep builds). In that case build to a Windows-local target dir.
$IsWslUnc = $Root -match '^\\\\wsl(\$|\.)'
if ($IsWslUnc) {
    $LocalTargetRoot = Join-Path $env:LOCALAPPDATA "AudioMeter\cargo-target-msvc"
    if (-not (Test-Path $LocalTargetRoot)) {
        New-Item -ItemType Directory -Path $LocalTargetRoot -Force | Out-Null
    }
    $env:CARGO_TARGET_DIR = $LocalTargetRoot
    Write-Host "WSL UNC repo: using CARGO_TARGET_DIR=$($env:CARGO_TARGET_DIR)" -ForegroundColor DarkGray
} else {
    # Cursor/agent sandboxes may set CARGO_TARGET_DIR — force normal project `target\`.
    $env:CARGO_TARGET_DIR = Join-Path $Root "target"
}

$RustTarget = "x86_64-pc-windows-msvc"
Write-Host "==> cargo build --release --target $RustTarget" -ForegroundColor Cyan
cargo build --release --target $RustTarget

$ReleaseExe = Join-Path $env:CARGO_TARGET_DIR "$RustTarget\release\audio-meter-rs.exe"
if (-not (Test-Path $ReleaseExe)) {
    Write-Error "Missing $ReleaseExe - release build failed. Install the MSVC Rust target: rustup target add $RustTarget"
}

# Inno runs with `installer\` as CWD and resolves `..\target\...` under the repo. When
# `CARGO_TARGET_DIR` is outside the repo (WSL UNC + local MSVC target dir), copy the exe next to
# `Cargo.toml` so `{#ReleaseExe}` exists and remains a stable relative path.
$InnoExe = Join-Path $Root "target\$RustTarget\release\audio-meter-rs.exe"
if ($ReleaseExe -ne $InnoExe) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $InnoExe) | Out-Null
    Copy-Item -LiteralPath $ReleaseExe -Destination $InnoExe -Force
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
$ReleaseExeRel = "..\target\$RustTarget\release\audio-meter-rs.exe"
Write-Host "==> & `"$ISCC`" /DReleaseExe=$ReleaseExeRel `"$Iss`"" -ForegroundColor Cyan
& $ISCC "/DReleaseExe=$ReleaseExeRel" $Iss

$Dist = Join-Path $Root "dist"
Write-Host ""
Write-Host "Done. Installer:" -ForegroundColor Green
Get-ChildItem $Dist -Filter "AYR-Audio-Meter-Setup-*.exe" | ForEach-Object { Write-Host "  $($_.FullName)" }
