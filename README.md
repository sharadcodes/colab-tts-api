# Colab TTS Voice Cloning API

One-command launchers for a FastAPI TTS voice-cloning server behind a Cloudflare
quick tunnel. Runs on Colab, local Linux, or any CUDA VPS. Each model has its
own launcher script and its own isolated `uv` venv — switch models without
reinstalling or removing the others.

## Supported models

| Model          | Size  | Launcher                   | Notes                              |
| -------------- | ----- | -------------------------- | ---------------------------------- |
| chatterbox     | 350M  | `start_chatterbox.sh`      | Fast. Variants: `turbo`/`en`/`mtl` |
| chatterbox_full| 500M  | `start_chatterbox_full.sh` | Full-size English Chatterbox       |
| vibevoice      | 1.5B  | `start_vibevoice.sh`       | Expressive, voice cloning          |
| dia            | 1.6B  | `start_dia.sh`             | English, needs `ref_text` to clone |
| breeze         | ~2B   | `start_breeze.sh`          | EN/ZH, clone/design/direction, needs `ref_text` to clone |

## Requirements

- CUDA GPU (≥6GB VRAM; ≥10GB for dia)
- Linux (Colab / VPS / WSL2) — scripts are bash
- Internet access (Cloudflare tunnel + model downloads on first run)

## Quick start

Run the per-model script directly (preferred):

```bash
./start_chatterbox.sh
./start_chatterbox_full.sh
./start_vibevoice.sh
./start_dia.sh
./start_breeze.sh
```

Or use the dispatcher (preserves the original `MODEL=...` interface):

```bash
MODEL=chatterbox      ./start.sh
MODEL=chatterbox_full ./start.sh
MODEL=vibevoice       ./start.sh
MODEL=dia             ./start.sh
MODEL=breeze          ./start.sh
```

Output prints a public URL like `https://<random>.trycloudflare.com` once the
tunnel is up and the model is loaded (30–60s on first run).

## Configuration

All env vars are optional. Defaults shown. Per-model vars only apply to the
relevant launcher.

| Var                             | Default              | Values                          |
| ------------------------------- | -------------------- | ------------------------------- |
| `CHATTERBOX_VARIANT`            | `turbo`              | `turbo`/`en`/`mtl`              |
| `CHATTERBOX_EXAGGERATION`       | `0.5`                | float (full/en/mtl only)        |
| `CHATTERBOX_CFG_WEIGHT`         | `0.5`                | float (full/en/mtl only)        |
| `CHATTERBOX_TEMPERATURE`        | `0.8`                | float                           |
| `CHATTERBOX_REPETITION_PENALTY` | `1.2`                | float                           |
| `CHATTERBOX_MIN_P`              | `0.05`               | float (full/en/mtl only)        |
| `CHATTERBOX_TOP_P`              | `1.0`                | float                           |
| `VIBEVOICE_STEPS`               | `20`                 | int (lower = faster)            |
| `DIA_CFG_SCALE`                 | `4.0`                | float                           |
| `DIA_TEMPERATURE`               | `1.8`                | float                           |
| `DIA_TOP_P`                     | `0.90`               | float                           |
| `BREEZE_CFG_SCALE`              | `1.0`                | float (use ~4 with instruction) |
| `BREEZE_INSTRUCTION`            | `""`                 | voice design/direction prompt   |
| `BREEZE_SEED`                   | `42`                 | int                             |
| `PORT`                          | `8000`               | int                             |
| `HOST`                          | `0.0.0.0`            | str                             |
| `VENV_ROOT`                     | `./venv`             | path (per-model envs live here) |
| `VENV_DIR`                      | `<root>/<model>`     | override per-model venv         |
| `CUDA_TAG`                      | `cu124`              | `cu118`/`cu124`/`cu126`         |

### Per-model venvs (swap-and-use)

Each model installs into its own venv under `VENV_ROOT/<model>/`:

```
venv/chatterbox/
venv/chatterbox_full/
venv/vibevoice/
venv/dia/
venv/breeze/
```

Switching models reuses an already-installed env — no reinstall, no removal.
First run for a model downloads deps + weights; subsequent runs skip both.

## Endpoints

| Method | Path                  | Purpose                          |
| ------ | --------------------- | -------------------------------- |
| `GET`  | `/`                   | Built-in dense test UI           |
| `GET`  | `/health`             | Liveness + active model          |
| `GET`  | `/models`             | Backend info                     |
| `GET`  | `/voices`             | List server-side voice names     |
| `POST` | `/voices/upload`      | Upload a named ref clip          |
| `POST` | `/tts`                | Synthesize speech → WAV bytes    |
| `POST` | `/v1/audio/speech`    | OpenAI-compatible TTS endpoint   |
| `GET`  | `/v1/audio/voices`    | OpenAI-compatible voice listing  |

### Custom API (`/tts`)

No cloning:

```bash
curl -X POST https://<url>.trycloudflare.com/tts \
  -F 'text=Hello world' --output out.wav
```

Voice cloning:

```bash
curl -X POST https://<url>.trycloudflare.com/tts \
  -F 'text=Hello in my voice' \
  -F 'ref_audio=@your_voice.wav' \
  --output cloned.wav
```

Server-side voice (from `voices/` folder):

```bash
curl -X POST https://<url>.trycloudflare.com/tts \
  -F 'text=Hello world' -F 'voice=narrator' --output out.wav
```

Upload a voice once, reuse by name:

```bash
curl -X POST https://<url>.trycloudflare.com/voices/upload \
  -F 'name=narrator' -F 'audio=@voice.wav'
```

Dia cloning also requires `ref_text` (transcript of the reference audio):

```bash
curl -X POST https://<url>.trycloudflare.com/tts \
  -F 'text=[S1] Welcome to the show.' \
  -F 'ref_audio=@ref.wav' \
  -F 'ref_text=[S1] Welcome to the show.' \
  --output cloned.wav
```

### OpenAI-compatible API (`/v1/audio/speech`)

Works with any client that expects the OpenAI TTS API. JSON body, returns
audio bytes directly.

```bash
curl -X POST https://<url>.trycloudflare.com/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world","voice":"narrator"}' \
  --output out.mp3
```

| Field             | Default  | Notes                                          |
| ----------------- | -------- | ---------------------------------------------- |
| `model`           | `tts-1`  | Ignored — server uses its configured backend   |
| `input`           | required | Text to synthesize                             |
| `voice`           | `default`| Server-side voice name, or `default`           |
| `response_format` | `wav`    | `wav` / `mp3` / `flac` / `opus`                |
| `speed`           | `1.0`    | Ignored (not supported by backends)            |

**Python (openai SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<url>.trycloudflare.com/v1",
    api_key="unused",
)
response = client.audio.speech.create(
    model="tts-1",
    voice="narrator",
    input="Hello world",
)
response.stream_to_file("out.mp3")
```

## Colab usage

Open `colab_tts_api.ipynb` in Colab, set `MODEL` in the config cell, run all
cells. The launch cell clones this repo and runs `start.sh`, streaming the
public URL into the notebook output.

## Files

```
colab-tts-api/
├── start.sh                   # thin dispatcher (MODEL=... ./start.sh)
├── start_chatterbox.sh        # chatterbox launcher (torch 2.6.0 trio + chatterbox-tts)
├── start_chatterbox_full.sh   # full-size 500M English chatterbox launcher
├── start_vibevoice.sh         # vibevoice launcher (VibeVoice-1.5B from GitHub + HF)
├── start_dia.sh               # dia launcher (Dia-1.6B from GitHub)
├── start_breeze.sh            # breeze launcher (Breeze TTS 2 from GitHub + HF)
├── common.sh                  # shared helpers (venv, tunnel, server, banner)
├── server.py             # FastAPI server with TTSBackend ABC + 5 backends
├── ui.html               # built-in test UI served at GET /
├── colab_tts_api.ipynb   # Colab launcher notebook
├── clients/
│   ├── gui.py            # Tkinter GUI client
│   └── generate_from_file.py  # CLI batch generator from a text file
├── voices/               # drop reference audio clips here for named voice reuse
└── README.md
```

## Clients

- **GUI** — `python clients/gui.py` (Tkinter, needs `pip install requests`)
- **CLI** — `python clients/generate_from_file.py <url> <input.txt>` (needs `pip install requests`)

Both support server-side voices (from `voices/`), ref audio upload, and ref text
for Dia. See `--help` on the CLI or the "Upload..." button in the GUI.

## Adding a backend

1. Subclass `TTSBackend` in `server.py`, implement `load()` + `synthesize()`,
   and register it in `BACKENDS`.
2. Copy `start_chatterbox.sh` to `start_<name>.sh`, change `MODEL`, and swap
   the install block for your model's deps.
3. Add a `case` branch in `start.sh` dispatching to the new script.
4. Add a row to the table at the top of this README.
