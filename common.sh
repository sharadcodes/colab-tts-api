#!/usr/bin/env bash
# common.sh — shared helpers for the per-model start_* scripts.
#
# Sourced (never executed directly). Provides:
#   - config defaults + per-model venv layout
#   - gpu_check, ensure_uv, ensure_venv, uv_install
#   - ensure_cloudflared, start_tunnel, print_endpoints, run_server
#
# Each start_<model>.sh sets MODEL and (optionally) model-specific defaults,
# then sources this file and calls run_server at the end.

set -euo pipefail

# ─── Defaults ─────────────────────────────────
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_TAG="${CUDA_TAG:-cu124}"

# Per-model venv: swap-and-use. Each model lives in its own isolated env
# under VENV_ROOT/<model>; switching models reuses an existing env instead of
# reinstalling. Override VENV_ROOT to relocate all envs together.
VENV_ROOT="${VENV_ROOT:-$HERE/venv}"
VENV_DIR="${VENV_DIR:-$VENV_ROOT/$MODEL}"
VENV_PY="$VENV_DIR/bin/python"

TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"

# ─── Banner ───────────────────────────────────
banner() {
  echo "════════════════════════════════════════════════════════"
  echo "  Colab TTS Voice Cloning API"
  echo "  Model: $MODEL  |  Port: $PORT  |  CUDA: $CUDA_TAG"
  echo "  Venv:  $VENV_DIR"
  echo "════════════════════════════════════════════════════════"
}

# ─── GPU check ────────────────────────────────
gpu_check() {
  if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: no CUDA GPU detected. On Colab: Runtime → Change runtime type → T4 GPU."
    exit 1
  fi
  GPU_NAME="$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
  VRAM_GB="$(python3 -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.total_memory/1024**3:.1f}")')"
  echo "GPU: $GPU_NAME  |  VRAM: ${VRAM_GB}GB"
}

# ─── uv bootstrap ─────────────────────────────
ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    pip install -q uv
  fi
  UV="$(command -v uv)"
}

# ─── venv + install helper ────────────────────
ensure_venv() {
  if [ ! -f "$VENV_PY" ]; then
    echo "Creating uv venv at $VENV_DIR..."
    "$UV" venv "$VENV_DIR" --python 3.11
  fi
}

uv_install() { "$UV" pip install --python "$VENV_PY" -q "$@"; }

# ─── Common deps (FastAPI + audio) ────────────
install_common() {
  echo "Installing common deps (FastAPI + audio)..."
  uv_install "fastapi" "uvicorn[standard]" "python-multipart" "soundfile" "numpy<2"
}

# ─── Verify imports ───────────────────────────
verify_imports() {
  echo "Verifying venv imports..."
  "$VENV_PY" -c "import torch, numpy, fastapi; print(f'torch={torch.__version__} numpy={numpy.__version__} cuda={torch.cuda.is_available()}')"
}

# ─── cloudflared ──────────────────────────────
ensure_cloudflared() {
  CF_BIN="$HERE/cloudflared"
  if [ ! -f "$CF_BIN" ]; then
    echo "Downloading cloudflared..."
    wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -O "$CF_BIN"
    chmod +x "$CF_BIN"
  fi
}

# ─── Tunnel ───────────────────────────────────
start_tunnel() {
  echo "Starting Cloudflare quick tunnel..."
  "$CF_BIN" tunnel --url "http://$HOST:$PORT" > /tmp/cf_tunnel.log 2>&1 &
  TUNNEL_PID=$!

  PUBLIC_URL=""
  for _ in $(seq 1 30); do
    sleep 1
    PUBLIC_URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cf_tunnel.log | head -1 || true)"
    [ -n "$PUBLIC_URL" ] && break
  done

  if [ -z "$PUBLIC_URL" ]; then
    echo "ERROR: Cloudflare tunnel did not produce a URL. Tunnel log:"
    cat /tmp/cf_tunnel.log
    kill "$TUNNEL_PID" 2>/dev/null || true
    exit 1
  fi
}

# ─── Print endpoints ──────────────────────────
print_endpoints() {
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  PUBLIC URL: $PUBLIC_URL"
  echo "════════════════════════════════════════════════════════"
  echo ""
  echo "Endpoints:"
  echo "  POST $PUBLIC_URL/tts"
  echo "  GET  $PUBLIC_URL/health"
  echo "  GET  $PUBLIC_URL/models"
  echo ""
  echo "Test (no cloning):"
  echo "  curl -X POST $PUBLIC_URL/tts -F 'text=Hello world' --output out.wav"
  echo ""
  echo "Test (voice cloning):"
  echo "  curl -X POST $PUBLIC_URL/tts -F 'text=Hello in my voice' -F 'ref_audio=@your_voice.wav' --output cloned.wav"
  echo ""
}

# ─── Run server (foreground) ──────────────────
# Caller must export any model-specific env vars before invoking.
run_server() {
  export TTS_MODEL="$MODEL"

  echo "Starting server (model loading may take 30-60s)..."
  echo "Tunnel PID: $TUNNEL_PID"
  echo ""

  cleanup() {
    echo ""
    echo "Stopping server + tunnel..."
    kill "$TUNNEL_PID" 2>/dev/null || true
    exit 0
  }
  trap cleanup INT TERM

  exec "$VENV_PY" -m uvicorn server:app \
    --host "$HOST" --port "$PORT" --app-dir "$HERE"
}
