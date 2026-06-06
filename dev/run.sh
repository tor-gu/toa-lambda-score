#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/.." python "$SCRIPT_DIR/run_local.py" "$@"
