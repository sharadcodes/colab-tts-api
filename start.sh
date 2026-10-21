#!/usr/bin/env bash
# start.sh — dispatcher for the per-model launchers.
#
# Each model has its own script (start_chatterbox.sh, start_chatterbox_full.sh,
# start_vibevoice.sh, start_dia.sh) with its own isolated uv venv. This wrapper
# preserves the original `MODEL=... ./start.sh` interface by delegating to the
# right script.
#
# Usage:
#   ./start.sh                          # defaults to chatterbox
#   MODEL=chatterbox_full ./start.sh
#   MODEL=vibevoice ./start.sh
#   MODEL=dia ./start.sh
#   MODEL=breeze ./start.sh
#
# Or call the per-model scripts directly (preferred):
#   ./start_chatterbox.sh
#   ./start_chatterbox_full.sh
#   ./start_vibevoice.sh
#   ./start_dia.sh
#   ./start_breeze.sh

set -euo pipefail
MODEL="${MODEL:-chatterbox}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODEL" in
  chatterbox)      exec "$HERE/start_chatterbox.sh"      ;;
  chatterbox_full) exec "$HERE/start_chatterbox_full.sh" ;;
  vibevoice)       exec "$HERE/start_vibevoice.sh"       ;;
  dia)             exec "$HERE/start_dia.sh"             ;;
  breeze)          exec "$HERE/start_breeze.sh"          ;;
  *)
    echo "ERROR: Unknown MODEL='$MODEL'. Must be: chatterbox | chatterbox_full | vibevoice | dia | breeze"
    echo "Tip: run a per-model script directly, e.g. ./start_chatterbox.sh"
    exit 1
    ;;
esac
