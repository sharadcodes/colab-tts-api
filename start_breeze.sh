#!/usr/bin/env bash
# start_breeze.sh — launch the Breeze TTS 2 API.
#
# Creates an isolated uv venv at venv/breeze, clones the breeze-tts inference
# repo, installs its requirements, downloads the Breeze-TTS-2 checkpoint from
# Hugging Face, starts a Cloudflare quick tunnel + the FastAPI server, and
# prints the URL.
#
# Eager inference needs ~7.7 GiB VRAM — fits a Colab free T4.
#
# Usage:
#   ./start_breeze.sh
#   BREEZE_CFG_SCALE=4 BREEZE_INSTRUCTION="A warm, calm voice" ./start_breeze.sh
#
# Env vars (all optional):
#   BREEZE_CFG_SCALE    : float                           (default: 1.0; use ~4 with instruction)
#   BREEZE_INSTRUCTION  : str                             (default: ""; voice design/direction prompt)
#   BREEZE_SEED         : int                             (default: 42)
#   PORT                : int                             (default: 8000)
#   HOST                : str                             (default: 0.0.0.0)
#   VENV_DIR            : path                            (default: ./venv/breeze)
#   CUDA_TAG            : cu126 | cu128                   (default: cu128; torch 2.9.x needs >= cu126)
#
set -euo pipefail
MODEL="breeze"
CUDA_TAG="${CUDA_TAG:-cu128}"
BREEZE_CFG_SCALE="${BREEZE_CFG_SCALE:-1.0}"
BREEZE_INSTRUCTION="${BREEZE_INSTRUCTION:-}"
BREEZE_SEED="${BREEZE_SEED:-42}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"

banner
gpu_check
ensure_uv
ensure_venv
install_common

# ─── Breeze TTS 2 deps ────────────────────────
echo "Installing Breeze TTS 2..."
if [ ! -d "$HERE/breeze-tts" ]; then
  git clone https://github.com/breezeblue-ai/breeze-tts.git "$HERE/breeze-tts"
fi
uv_install "torch==2.9.1" "torchaudio==2.9.1" "--index-url" "$TORCH_INDEX"
uv_install -r "$HERE/breeze-tts/requirements.txt"
if [ ! -d "$HERE/models/Breeze-TTS-2" ]; then
  "$VENV_PY" -m huggingface_hub.commands.huggingface_cli \
    download BreezeBlue/Breeze-TTS-2 --local-dir "$HERE/models/Breeze-TTS-2"
fi
export BREEZE_MODEL_PATH="$HERE/models/Breeze-TTS-2"
# breeze_infer / models packages live in the cloned repo, not on PyPI
export PYTHONPATH="$HERE/breeze-tts:${PYTHONPATH:-}"

verify_imports
ensure_cloudflared
start_tunnel
print_endpoints

export BREEZE_CFG_SCALE="$BREEZE_CFG_SCALE"
export BREEZE_INSTRUCTION="$BREEZE_INSTRUCTION"
export BREEZE_SEED="$BREEZE_SEED"
run_server
