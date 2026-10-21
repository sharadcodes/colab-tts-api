"""Modular TTS voice-cloning API server.

Runs inside an isolated uv venv (created by start.sh).
Launched as: python -m uvicorn server:app --host 0.0.0.0 --port 8000

Config via env vars:
  TTS_MODEL           : chatterbox | chatterbox_full | vibevoice | dia | breeze  (default: chatterbox)
  CHATTERBOX_VARIANT  : turbo | en | mtl                                (default: turbo)
  CHATTERBOX_EXAGGERATION : float                              (default: 0.5)  (full/en/mtl)
  CHATTERBOX_CFG_WEIGHT   : float                              (default: 0.5)  (full/en/mtl)
  CHATTERBOX_TEMPERATURE  : float                              (default: 0.8)  (all)
  CHATTERBOX_REPETITION_PENALTY : float                        (default: 1.2)  (all)
  CHATTERBOX_MIN_P        : float                              (default: 0.05) (full/en/mtl)
  CHATTERBOX_TOP_P        : float                              (default: 1.0)  (all)
  VIBEVOICE_STEPS     : int                                  (default: 20)
  DIA_CFG_SCALE       : float                                (default: 4.0)
  DIA_TEMPERATURE     : float                                (default: 1.8)
  DIA_TOP_P           : float                                (default: 0.90)
  BREEZE_MODEL_PATH   : path                                 (default: /content/models/Breeze-TTS-2)
  BREEZE_CFG_SCALE    : float                                (default: 1.0; ~4 with instruction)
  BREEZE_INSTRUCTION  : str                                  (default: "")
  BREEZE_SEED         : int                                  (default: 42)

Add new backends by subclassing TTSBackend and registering in BACKENDS.
"""

from __future__ import annotations

import abc
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Base interface
# ──────────────────────────────────────────────
class TTSBackend(abc.ABC):
    name: str = "base"
    needs_ref_text: bool = False

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def synthesize(
        self, text: str, ref_audio_path: str | None = None, ref_text: str | None = None
    ) -> tuple[bytes, int]: ...

    def info(self) -> dict:
        return {"model": self.name, "needs_ref_text": self.needs_ref_text}


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ──────────────────────────────────────────────
# Chatterbox
# ──────────────────────────────────────────────
class ChatterboxBackend(TTSBackend):
    name = "chatterbox"
    needs_ref_text = False

    def load(self) -> None:
        variant = os.environ.get("CHATTERBOX_VARIANT", "turbo")
        if variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        elif variant == "en":
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_pretrained(device="cuda")
        elif variant == "mtl":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        else:
            raise ValueError(f"Unknown chatterbox variant: {variant}")
        self.sr = self.model.sr
        print(f"Chatterbox loaded (variant={variant}, sr={self.sr})")

    def synthesize(self, text, ref_audio_path=None, ref_text=None):
        kwargs = {}
        if ref_audio_path:
            kwargs["audio_prompt_path"] = ref_audio_path
        wav = self.model.generate(text, **kwargs)
        audio = wav.cpu().numpy().squeeze()
        return _wav_bytes(audio, self.sr), self.sr


class ChatterboxFullBackend(ChatterboxBackend):
    """Full-size 500M English Chatterbox model."""

    name = "chatterbox_full"

    def load(self) -> None:
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.sr = self.model.sr
        print(f"Chatterbox full loaded (sr={self.sr})")

    def synthesize(self, text, ref_audio_path=None, ref_text=None):
        kwargs = {
            "exaggeration": float(os.environ.get("CHATTERBOX_EXAGGERATION", "0.5")),
            "cfg_weight": float(os.environ.get("CHATTERBOX_CFG_WEIGHT", "0.5")),
            "temperature": float(os.environ.get("CHATTERBOX_TEMPERATURE", "0.8")),
            "repetition_penalty": float(os.environ.get("CHATTERBOX_REPETITION_PENALTY", "1.2")),
            "min_p": float(os.environ.get("CHATTERBOX_MIN_P", "0.05")),
            "top_p": float(os.environ.get("CHATTERBOX_TOP_P", "1.0")),
        }
        if ref_audio_path:
            kwargs["audio_prompt_path"] = ref_audio_path
        wav = self.model.generate(text, **kwargs)
        audio = wav.cpu().numpy().squeeze()
        return _wav_bytes(audio, self.sr), self.sr


# ──────────────────────────────────────────────
# VibeVoice-1.5B
# ──────────────────────────────────────────────
class VibeVoiceBackend(TTSBackend):
    name = "vibevoice"
    needs_ref_text = False

    def load(self) -> None:
        import torchaudio
        from vibevoice import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor import VibeVoiceProcessor

        model_path = os.environ.get(
            "VIBEVOICE_MODEL_PATH", "/content/models/VibeVoice-1.5B"
        )
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained_hf(
            model_path, device="cuda", torch_dtype=torch.bfloat16
        )
        self.model.eval()
        self.model.set_ddpm_inference_steps(
            int(os.environ.get("VIBEVOICE_STEPS", "20"))
        )
        self.sr = 24000
        self._torchaudio = torchaudio
        print(f"VibeVoice-1.5B loaded (sr={self.sr})")

    def synthesize(self, text, ref_audio_path=None, ref_text=None):
        if not text.strip().startswith("Speaker"):
            text = f"Speaker 0: {text}"
        kwargs = {"text": text, "return_tensors": "pt"}
        if ref_audio_path:
            wav, sr = self._torchaudio.load(ref_audio_path)
            if sr != self.sr:
                wav = self._torchaudio.functional.resample(wav, sr, self.sr)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            kwargs["voice_samples"] = [wav.squeeze(0).numpy()]
        inputs = self.processor(**kwargs).to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, tokenizer=self.processor.tokenizer, cfg_scale=3.0,
                return_speech=True,
            )
        audio = output.speech_outputs[0].cpu().float().numpy()
        return _wav_bytes(audio, self.sr), self.sr


# ──────────────────────────────────────────────
# Dia-1.6B
# ──────────────────────────────────────────────
class DiaBackend(TTSBackend):
    name = "dia"
    needs_ref_text = True

    def load(self) -> None:
        from dia.model import Dia

        self.model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B-0626", compute_dtype="float16"
        )
        self.sr = 44100
        print(f"Dia-1.6B loaded (sr={self.sr})")

    def synthesize(self, text, ref_audio_path=None, ref_text=None):
        if "[S" not in text:
            text = f"[S1] {text}"
        gen_kwargs = dict(
            cfg_scale=float(os.environ.get("DIA_CFG_SCALE", "4.0")),
            temperature=float(os.environ.get("DIA_TEMPERATURE", "1.8")),
            top_p=float(os.environ.get("DIA_TOP_P", "0.90")),
            cfg_filter_top_k=50,
            verbose=True,
        )
        if ref_audio_path:
            if not ref_text:
                raise ValueError(
                    "Dia requires ref_text (transcript of reference audio)."
                )
            if "[S" not in ref_text:
                ref_text = f"[S1] {ref_text}"
            output = self.model.generate(
                ref_text + " " + text, audio_prompt=ref_audio_path, **gen_kwargs
            )
        else:
            output = self.model.generate(text, **gen_kwargs)
        audio = np.array(output).squeeze()
        return _wav_bytes(audio, self.sr), self.sr


# ──────────────────────────────────────────────
# Breeze TTS 2
# ──────────────────────────────────────────────
class BreezeBackend(TTSBackend):
    """Breeze TTS 2 — bilingual EN/ZH, voice clone / design / direction.

    Cloning requires ref_text (exact transcript of the reference audio).
    Without ref audio it runs voice design; BREEZE_INSTRUCTION steers it.
    """

    name = "breeze"
    needs_ref_text = True

    def load(self) -> None:
        from breeze_infer.runtime import (
            load_runtime,
            resolve_device,
            update_generation_config_for_breeze,
        )
        from models.fast_streaming import (
            FastBreezeStreamingRuntime,
            FastStreamingConfig,
        )

        model_path = Path(
            os.environ.get("BREEZE_MODEL_PATH", "/content/models/Breeze-TTS-2")
        )
        tokenizer, model, audio_tokenizer = load_runtime(
            model_path, device=resolve_device(), attn_implementation="eager"
        )
        update_generation_config_for_breeze(model)
        config = FastStreamingConfig(
            max_new_tokens=1500, max_seq_len=2048, repetition_penalty=1.1
        )
        self.runtime = FastBreezeStreamingRuntime(
            model, audio_tokenizer, config, tokenizer=tokenizer
        )
        self.tokenizer = tokenizer
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        self.sr = self.runtime.sample_rate
        print(f"Breeze TTS 2 loaded (sr={self.sr})")

    def synthesize(self, text, ref_audio_path=None, ref_text=None):
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import (
            get_template,
            prepare_inputs,
            select_template_name,
        )

        if ref_audio_path and not (ref_text and ref_text.strip()):
            raise ValueError(
                "Breeze requires ref_text (exact transcript of the reference audio)."
            )

        request = {"id": "api-request", "text": text, "speaker": "S0"}
        instruction = os.environ.get("BREEZE_INSTRUCTION", "").strip()
        if instruction:
            request["instruction"] = instruction
        if ref_audio_path:
            request["ref_audio_path"] = ref_audio_path
            request["ref_text"] = ref_text.strip()

        seed = int(os.environ.get("BREEZE_SEED", "42"))
        set_all_seeds(seed)
        inputs = prepare_inputs(
            self.tokenizer,
            self.audio_tokenizer,
            self.model,
            [request],
            get_template(select_template_name(request)),
            guidance_scale=float(os.environ.get("BREEZE_CFG_SCALE", "1.0")),
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        chunks = [
            chunk.audio
            for chunk in self.runtime.iter_audio_chunks(
                inputs, request_id="api-request", seed=seed
            )
        ]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return _wav_bytes(audio, self.sr), self.sr


# ──────────────────────────────────────────────
# Registry — add new backends here
# ──────────────────────────────────────────────
BACKENDS = {
    "chatterbox": ChatterboxBackend,
    "chatterbox_full": ChatterboxFullBackend,
    "vibevoice": VibeVoiceBackend,
    "dia": DiaBackend,
    "breeze": BreezeBackend,
}


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
MODEL = os.environ.get("TTS_MODEL", "chatterbox")
VOICES_DIR = Path(os.environ.get("VOICES_DIR", Path(__file__).resolve().parent / "voices"))
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
app = FastAPI(title="Colab TTS Voice Cloning API", version="1.0")
backend: TTSBackend = BACKENDS[MODEL]()


def _list_voices() -> list[str]:
    if not VOICES_DIR.is_dir():
        return []
    return sorted(
        p.stem for p in VOICES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    )


def _resolve_voice(name: str) -> str | None:
    if not VOICES_DIR.is_dir():
        return None
    for p in VOICES_DIR.iterdir():
        if p.is_file() and p.stem == name and p.suffix.lower() in _AUDIO_EXTS:
            return str(p)
    return None


def _resolve_voice_text(name: str) -> str | None:
    """Return the transcript for a named voice, if one was uploaded."""
    if not VOICES_DIR.is_dir():
        return None
    txt = VOICES_DIR / f"{name}.txt"
    if txt.is_file():
        return txt.read_text(encoding="utf-8").strip()
    return None


@app.on_event("startup")
def _load() -> None:
    print(f"Loading backend: {MODEL}...")
    backend.load()
    print(f"Ready: {backend.info()}")
    voices = _list_voices()
    if voices:
        print(f"Voices available: {', '.join(voices)}")
    else:
        print("No voices in voices/ dir — voice cloning via upload only.")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.get("/models")
async def models():
    return backend.info()


@app.get("/voices")
async def voices():
    return {"voices": _list_voices()}


@app.post("/voices/upload")
async def upload_voice(
    name: str = Form(...),
    audio: UploadFile = File(...),
    ref_text: str = Form(""),
):
    """Upload a reference audio clip (+ optional transcript) to voices/.

    Saves:
      voices/<name>.<ext>   — the audio clip
      voices/<name>.txt     — the transcript (if ref_text provided)

    The voice becomes reusable by name in /tts and /v1/audio/speech.
    The transcript is required for Dia cloning, ignored by other backends.
    """
    if not name.replace("_", "").replace("-", "").isalnum():
        return JSONResponse(
            status_code=400,
            content={"error": "Voice name must be alphanumeric (dashes/underscores ok)."},
        )
    suffix = os.path.splitext(audio.filename or "ref.wav")[1] or ".wav"
    if suffix.lower() not in _AUDIO_EXTS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported audio format: {suffix}"},
        )
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VOICES_DIR / f"{name}{suffix}"
    # Remove any existing file with the same name but different extension
    for p in VOICES_DIR.iterdir():
        if p.is_file() and p.stem == name and p.suffix.lower() in _AUDIO_EXTS:
            p.unlink()
    content = await audio.read()
    out_path.write_bytes(content)

    # Save transcript alongside the audio (used by Dia for cloning)
    has_text = False
    if ref_text.strip():
        (VOICES_DIR / f"{name}.txt").write_text(ref_text.strip(), encoding="utf-8")
        has_text = True

    print(f"Voice uploaded: {name} -> {out_path} ({len(content) / 1024:.1f} KB)"
          + (" + transcript" if has_text else ""))
    return {
        "status": "ok",
        "voice": name,
        "path": str(out_path),
        "bytes": len(content),
        "has_transcript": has_text,
    }


@app.post("/tts")
async def tts(
    text: str = Form(...),
    ref_audio: UploadFile | None = File(None),
    ref_text: str = Form(""),
    voice: str = Form(""),
):
    ref_path = None
    is_voice_file = False
    try:
        if voice:
            resolved = _resolve_voice(voice)
            if resolved is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Voice '{voice}' not found in {VOICES_DIR}"},
                )
            ref_path = resolved
            is_voice_file = True
            # Auto-load the voice's transcript if the caller didn't provide one
            if not ref_text:
                ref_text = _resolve_voice_text(voice) or ""
        elif ref_audio:
            suffix = os.path.splitext(ref_audio.filename or "ref.wav")[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await ref_audio.read())
                ref_path = f.name
        wav_bytes, sr = backend.synthesize(
            text, ref_audio_path=ref_path, ref_text=ref_text or None
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(sr), "X-Model": MODEL},
        )
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"\n{'='*60}\nERROR in /tts:\n{tb}{'='*60}\n", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})
    finally:
        if ref_path and not is_voice_file and os.path.exists(ref_path):
            os.unlink(ref_path)


# ──────────────────────────────────────────────
# OpenAI-compatible endpoint
# ──────────────────────────────────────────────
# Mirrors POST /v1/audio/speech from the OpenAI TTS API.
# JSON body: {"model": "<ignored, uses server backend>", "input": "text",
#             "voice": "<server voice name or 'default'>", "response_format": "wav|mp3|flac|opus"}
# Returns audio bytes directly (same as OpenAI).
#
# Voice cloning via ref_audio is not supported through this endpoint (OpenAI's
# spec has no such field). Use /tts for cloning, or upload a voice to voices/
# and reference it by name here.

_OPENAI_FORMATS = {
    "wav": ("audio/wav", "WAV", "PCM_16"),
    "mp3": ("audio/mpeg", "MP3", None),
    "flac": ("audio/flac", "FLAC", "PCM_16"),
    # opus requires libsndfile support; fall back to wav if unavailable
    "opus": ("audio/ogg", "OGG", "OPUS"),
}


class OpenAISpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="Ignored — server uses its configured backend.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(default="default", description="Server-side voice name from voices/ folder, or 'default'.")
    response_format: str = Field(default="wav", description="wav | mp3 | flac | opus")
    speed: float = Field(default=1.0, description="Ignored (not supported by backends).")


@app.post("/v1/audio/speech")
async def openai_speech(req: OpenAISpeechRequest):
    ref_path = None
    is_voice_file = False
    voice_text: str | None = None
    try:
        if req.voice and req.voice != "default":
            resolved = _resolve_voice(req.voice)
            if resolved is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": f"Voice '{req.voice}' not found.", "type": "invalid_request_error"}},
                )
            ref_path = resolved
            is_voice_file = True
            # Auto-load the voice's transcript (needed for Dia cloning)
            voice_text = _resolve_voice_text(req.voice)

        wav_bytes, sr = backend.synthesize(
            req.input, ref_audio_path=ref_path, ref_text=voice_text if req.voice != "default" else None
        )

        fmt = req.response_format.lower()
        if fmt not in _OPENAI_FORMATS:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Unsupported format: {fmt}", "type": "invalid_request_error"}},
            )
        media_type, sf_format, sf_subtype = _OPENAI_FORMATS[fmt]

        if fmt == "wav":
            out_bytes = wav_bytes
        else:
            # Re-encode from the WAV bytes produced by the backend
            audio_arr, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            buf = io.BytesIO()
            try:
                sf.write(buf, audio_arr, sr, format=sf_format, subtype=sf_subtype)
            except Exception:
                # Fallback to WAV if the requested codec isn't available
                out_bytes = wav_bytes
                media_type = "audio/wav"
            else:
                out_bytes = buf.getvalue()

        return Response(
            content=out_bytes,
            media_type=media_type,
            headers={"X-Sample-Rate": str(sr), "X-Model": MODEL},
        )
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"\n{'='*60}\nERROR in /v1/audio/speech:\n{tb}{'='*60}\n", flush=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "traceback": tb}},
        )
    finally:
        if ref_path and not is_voice_file and os.path.exists(ref_path):
            os.unlink(ref_path)


@app.get("/v1/audio/voices")
async def openai_voices():
    """OpenAI-compatible voice listing (non-standard but useful for clients)."""
    voices = _list_voices()
    return {"voices": voices}
