#!/usr/bin/env bash
# Drive the Windows native build/launch of Audio Meter from a WSL shell.
#
# This is the WSL-side wrapper for launch-release.ps1 (see that script and
# .cursor/rules/launch-and-installer-build.mdc for context). It invokes the
# Windows PowerShell toolchain against the repo mounted over \\wsl.localhost,
# so the produced binary is target\release\audio-meter-rs.exe - the same path
# installer\build-installer.ps1 packages.
#
# Usage (from repo root, inside WSL):
#   ./launch-release.sh
#
# Requirements (install once on the Windows side):
#   winget install --id Rustlang.Rustup -e
#   winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override \
#     "--wait --quiet --norestart --includeRecommended \
#      --add Microsoft.VisualStudio.Workload.VCTools \
#      --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
#      --add Microsoft.VisualStudio.Component.Windows11SDK.22621"

set -euo pipefail

if ! command -v powershell.exe >/dev/null 2>&1; then
  echo "error: powershell.exe not on PATH - are you inside WSL with Windows interop enabled?" >&2
  exit 1
fi

if ! command -v wslpath >/dev/null 2>&1; then
  echo "error: wslpath missing - this script must run inside WSL." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PS1_WIN="$(wslpath -w "$SCRIPT_DIR/launch-release.ps1")"

echo "==> WSL: invoking Windows PowerShell"
echo "    $PS1_WIN"
echo "    (build + stage to %LOCALAPPDATA%\\AudioMeter\\ then Start-Process)"
echo ""

# Invoke Windows PowerShell. Prepend %USERPROFILE%\.cargo\bin so cargo is on
# PATH regardless of whether the user has restarted their Windows shell since
# running rustup-init. The PS one-liner escapes all $env vars to survive the
# bash double-quoted context; only $PS1_WIN expands here.
#
# Do not `exec` so we can print the exit code if the build/launch fails.
# Temporarily allow non-zero from PowerShell so `set -e` does not skip `status=$?`.
set +e
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
  "\$env:Path = \"\$env:USERPROFILE\\.cargo\\bin;\$env:Path\"; & '$PS1_WIN'"
status=$?
set -e
if [ "$status" -ne 0 ]; then
  echo "" >&2
  echo "error: PowerShell exited with status $status (cargo build or staging failed)." >&2
  echo "  Run this script in your own WSL terminal to see full output, or from Windows PowerShell" >&2
  echo "  after cd to the repo: .\\launch-release.ps1" >&2
fi
exit "$status"
