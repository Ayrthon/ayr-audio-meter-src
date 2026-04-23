#!/usr/bin/env bash
# WSL wrapper: one-shot debug build + staged launch (fast iteration vs launch-release.sh).
#
# Usage (from repo root, inside WSL):
#   ./launch-dev.sh

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
PS1_WIN="$(wslpath -w "$SCRIPT_DIR/launch-dev.ps1")"

echo "==> WSL: invoking Windows PowerShell"
echo "    $PS1_WIN"
echo ""

set +e
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
  "\$env:Path = \"\$env:USERPROFILE\\.cargo\\bin;\$env:Path\"; & '$PS1_WIN'"
status=$?
set -e
if [ "$status" -ne 0 ]; then
  echo "" >&2
  echo "error: PowerShell exited with status $status." >&2
  echo "  Run in your own WSL terminal, or from Windows: .\\launch-dev.ps1" >&2
fi
exit "$status"
