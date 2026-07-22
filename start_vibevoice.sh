#!/usr/bin/env bash
# start_vibevoice.sh — launch the VibeVoice-1.5B TTS API.
#
# Creates an isolated uv venv at venv/vibevoice, clones VibeVoice, installs
# torch + the editable package, downloads the 1.5B weights from Hugging Face,
# starts a Cloudflare quick tunnel + the FastAPI server, and prints the URL.
#
# Usage:
#   ./start_vibevoice.sh
#   VIBEVOICE_STEPS=40 ./start_vibevoice.sh
#
# Env vars (all optional):
#   VIBEVOICE_STEPS     : int                             (default: 20)
#   PORT                : int                             (default: 8000)
#   HOST                : str                             (default: 0.0.0.0)
#   VENV_DIR            : path                            (default: ./venv/vibevoice)
#   CUDA_TAG            : cu118 | cu124 | cu126           (default: cu124)

set -euo pipefail
MODEL="vibevoice"
VIBEVOICE_STEPS="${VIBEVOICE_STEPS:-20}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"

banner
gpu_check
ensure_uv
ensure_venv
install_common

# ─── VibeVoice deps ───────────────────────────
echo "Installing VibeVoice-1.5B..."
if [ ! -d "$HERE/VibeVoice" ]; then
  git clone https://github.com/davidamacey/VibeVoice.git "$HERE/VibeVoice"
fi
uv_install "torch" "torchaudio" "--index-url" "$TORCH_INDEX"
uv_install "-e" "$HERE/VibeVoice"
if [ ! -d "$HERE/models/VibeVoice-1.5B" ]; then
  "$VENV_PY" -m huggingface_hub.commands.huggingface_cli \
    download microsoft/VibeVoice-1.5B --local-dir "$HERE/models/VibeVoice-1.5B"
fi
export VIBEVOICE_MODEL_PATH="$HERE/models/VibeVoice-1.5B"

verify_imports
ensure_cloudflared
start_tunnel
print_endpoints

export VIBEVOICE_STEPS="$VIBEVOICE_STEPS"
run_server
