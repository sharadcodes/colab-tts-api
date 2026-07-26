#!/usr/bin/env bash
# start_chatterbox_full.sh — launch the full-size 500M English Chatterbox TTS API.
#
# Creates an isolated uv venv at venv/chatterbox_full, installs torch 2.6.0 trio +
# chatterbox-tts (transformers pinned to 4.46.3), starts a Cloudflare quick
# tunnel + the FastAPI server, and prints the public URL.
#
# Usage:
#   ./start_chatterbox_full.sh
#
# Env vars (all optional):
#   CHATTERBOX_EXAGGERATION : float                       (default: 0.5)
#   CHATTERBOX_CFG_WEIGHT   : float                       (default: 0.5)
#   CHATTERBOX_TEMPERATURE  : float                       (default: 0.8)
#   CHATTERBOX_REPETITION_PENALTY : float                 (default: 1.2)
#   CHATTERBOX_MIN_P        : float                       (default: 0.05)
#   CHATTERBOX_TOP_P        : float                       (default: 1.0)
#   PORT                    : int                         (default: 8000)
#   HOST                    : str                         (default: 0.0.0.0)
#   VENV_DIR                : path                        (default: ./venv/chatterbox_full)
#   CUDA_TAG                : cu118 | cu124 | cu126       (default: cu124)

set -euo pipefail
MODEL="chatterbox_full"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"

banner
gpu_check
ensure_uv
ensure_venv
install_common

# ─── Chatterbox deps ──────────────────────────
echo "Installing torch 2.6.0 trio + chatterbox (transformers pinned to 4.46.3)..."
uv_install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
           "--index-url" "$TORCH_INDEX"
# Pin transformers to 4.46.3 (chatterbox's exact pin); newer versions
# remove top-level LlamaModel and break chatterbox imports.
uv_install "transformers==4.46.3"
# Install perth from GitHub source — the PyPI `resemble-perth` wheel is broken
# (PerthImplicitWatermarker resolves to None → "NoneType is not callable").
# The GitHub repo installs as the `perth` package, which is what chatterbox
# actually imports. See: https://github.com/resemble-ai/Perth/issues/7
uv_install "setuptools"
uv_install "git+https://github.com/resemble-ai/Perth.git"
# Install chatterbox without deps so it doesn't pull the broken PyPI perth,
# then install its remaining deps manually (skip resemble-perth).
uv_install "chatterbox-tts" --no-deps
uv_install "librosa==0.11.0" "s3tokenizer" "diffusers==0.29.0" \
           "conformer>=0.3.2" "safetensors>=0.5.3" "pyloudnorm" "omegaconf"

verify_imports
ensure_cloudflared
start_tunnel
print_endpoints

run_server
