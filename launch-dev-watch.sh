#!/usr/bin/env bash
# WSL wrapper: cargo watch + debug build + staged restart on file changes.
#
# One-time on Windows:  cargo install cargo-watch
#
# Usage (from repo root, inside WSL):
#   ./launch-dev-watch.sh

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
PS1_WIN="$(wslpath -w "$SCRIPT_DIR/launch-dev-watch.ps1")"

exec powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
  "\$env:Path = \"\$env:USERPROFILE\\.cargo\\bin;\$env:Path\"; & '$PS1_WIN'"
