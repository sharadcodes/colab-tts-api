#!/usr/bin/env bash
# start_dia.sh — launch the Dia-1.6B TTS API.
#
# Creates an isolated uv venv at venv/dia, installs torch + Dia from GitHub,
# starts a Cloudflare quick tunnel + the FastAPI server, and prints the URL.
#
# Usage:
#   ./start_dia.sh
#   DIA_CFG_SCALE=4.0 DIA_TEMPERATURE=1.8 DIA_TOP_P=0.90 ./start_dia.sh
#
# Env vars (all optional):
#   DIA_CFG_SCALE       : float                           (default: 4.0)
#   DIA_TEMPERATURE     : float                           (default: 1.8)
#   DIA_TOP_P           : float                           (default: 0.90)
#   PORT                : int                             (default: 8000)
#   HOST                : str                             (default: 0.0.0.0)
#   VENV_DIR            : path                            (default: ./venv/dia)
#   CUDA_TAG            : cu118 | cu124 | cu126           (default: cu124)

set -euo pipefail
MODEL="dia"
DIA_CFG_SCALE="${DIA_CFG_SCALE:-4.0}"
DIA_TEMPERATURE="${DIA_TEMPERATURE:-1.8}"
DIA_TOP_P="${DIA_TOP_P:-0.90}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"

banner
gpu_check
ensure_uv
ensure_venv
install_common

# ─── Dia deps ─────────────────────────────────
echo "Installing Dia-1.6B..."
uv_install "torch" "torchaudio" "--index-url" "$TORCH_INDEX"
uv_install "git+https://github.com/nari-labs/dia.git"

verify_imports
ensure_cloudflared
start_tunnel
print_endpoints

export DIA_CFG_SCALE="$DIA_CFG_SCALE"
export DIA_TEMPERATURE="$DIA_TEMPERATURE"
export DIA_TOP_P="$DIA_TOP_P"
run_server
