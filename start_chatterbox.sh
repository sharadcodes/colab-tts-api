#!/usr/bin/env bash
# start_chatterbox.sh — launch the Chatterbox TTS API.
#
# Creates an isolated uv venv at venv/chatterbox, installs torch 2.6.0 trio +
# chatterbox-tts (transformers pinned to 4.46.3), starts a Cloudflare quick
# tunnel + the FastAPI server, and prints the public URL.
#
# Usage:
#   ./start_chatterbox.sh
#   CHATTERBOX_VARIANT=en ./start_chatterbox.sh
#
# Env vars (all optional):
#   CHATTERBOX_VARIANT  : turbo | en | mtl               (default: turbo)
#   PORT                : int                             (default: 8000)
#   HOST                : str                             (default: 0.0.0.0)
#   VENV_DIR            : path                            (default: ./venv/chatterbox)
#   CUDA_TAG            : cu118 | cu124 | cu126           (default: cu124)

set -euo pipefail
MODEL="chatterbox"
CHATTERBOX_VARIANT="${CHATTERBOX_VARIANT:-turbo}"

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

# pkuseg + pykakasi are only needed by the multilingual (mtl) variant for
# Chinese/Japanese text segmentation. pkuseg==0.0.25 fails to build under
# isolation because it imports numpy in setup.py without declaring it, so
# install with --no-build-isolation (numpy is already in the venv from
# install_common). pykakasi is a pure-Python wheel — no isolation issue.
if [ "$CHATTERBOX_VARIANT" = "mtl" ]; then
  echo "Installing multilingual deps (pkuseg, pykakasi)..."
  uv_install "pkuseg" --no-build-isolation
  uv_install "pykakasi"
fi

verify_imports
ensure_cloudflared
start_tunnel
print_endpoints

export CHATTERBOX_VARIANT="$CHATTERBOX_VARIANT"
run_server
