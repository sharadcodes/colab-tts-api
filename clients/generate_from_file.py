#!/usr/bin/env python3
"""CLI for the Colab TTS API.

Two modes:
  1. Generate — synthesize audio from a text file, line by line.
  2. Upload   — upload a voice clip (+ optional transcript) to the server.

Usage:
    # Generate
    python generate_from_file.py <url> input.txt
    python generate_from_file.py <url> input.txt -n 5
    python generate_from_file.py <url> input.txt --ref voice.wav
    python generate_from_file.py <url> input.txt -o out/ --prefix line_

    # Dia (needs ref_text for cloning)
    python generate_from_file.py <url> input.txt \
        --ref voice.mp3 --ref-text "transcript of voice.mp3"

    # Server-side voice (from voices/ folder on the server)
    python generate_from_file.py <url> input.txt --voice narrator

    # Upload a voice to the server (reusable by name later)
    python generate_from_file.py <url> --upload-voice voice.wav
    python generate_from_file.py <url> --upload-voice voice.wav \
        --voice-name narrator --ref-text-file transcript.txt

Requirements:
    pip install requests
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}


def synthesize(
    url: str,
    text: str,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    voice: str | None = None,
) -> bytes:
    data: dict[str, str] = {"text": text}
    if ref_text:
        data["ref_text"] = ref_text
    if voice:
        data["voice"] = voice
    files = {}
    if ref_audio and ref_audio.exists() and not voice:
        mime = _AUDIO_MIME.get(ref_audio.suffix.lower(), "application/octet-stream")
        files["ref_audio"] = (ref_audio.name, ref_audio.read_bytes(), mime)
    r = requests.post(f"{url.rstrip('/')}/tts", data=data, files=files, timeout=300)
    r.raise_for_status()
    return r.content


def upload_voice(
    url: str,
    audio_path: Path,
    name: str | None = None,
    ref_text: str | None = None,
) -> dict:
    voice_name = name or audio_path.stem
    mime = _AUDIO_MIME.get(audio_path.suffix.lower(), "application/octet-stream")
    data: dict[str, str] = {"name": voice_name}
    if ref_text:
        data["ref_text"] = ref_text
    with open(audio_path, "rb") as f:
        files = {"audio": (audio_path.name, f, mime)}
        r = requests.post(
            f"{url.rstrip('/')}/voices/upload",
            data=data,
            files=files,
            timeout=60,
        )
    r.raise_for_status()
    return r.json()


def cmd_generate(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    ref_text = args.ref_text
    if args.ref_text_file:
        if not args.ref_text_file.exists():
            print(f"ERROR: ref-text-file not found: {args.ref_text_file}", file=sys.stderr)
            return 1
        ref_text = args.ref_text_file.read_text(encoding="utf-8").strip()
    if args.ref and not ref_text and not args.voice:
        print(
            "WARNING: --ref given without --ref-text; Dia cloning will fail. "
            "Chatterbox/VibeVoice will work fine.",
            file=sys.stderr,
        )

    lines = [ln.strip() for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.lines > 0:
        lines = lines[: args.lines]
    if not lines:
        print("ERROR: no non-empty lines to synthesize", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    total = len(lines)
    width = len(str(total))

    for i, line in enumerate(lines, 1):
        out_path = args.out / f"{args.prefix}{i:0{width}d}{args.ext}"
        print(f"[{i}/{total}] {line[:60]}{'...' if len(line) > 60 else ''}")
        try:
            audio = synthesize(args.url, line, args.ref, ref_text, args.voice)
            out_path.write_bytes(audio)
            print(f"  -> {out_path} ({len(audio) / 1024:.1f} KB)")
        except requests.RequestException as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1

    print(f"\nDone: {total} file(s) in {args.out}/")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    if not args.upload_voice.exists():
        print(f"ERROR: voice audio not found: {args.upload_voice}", file=sys.stderr)
        return 1

    ref_text = args.ref_text
    if args.ref_text_file:
        if not args.ref_text_file.exists():
            print(f"ERROR: ref-text-file not found: {args.ref_text_file}", file=sys.stderr)
            return 1
        ref_text = args.ref_text_file.read_text(encoding="utf-8").strip()

    name = args.voice_name or args.upload_voice.stem
    print(f"Uploading voice '{name}' from {args.upload_voice}..."
          + (" + transcript" if ref_text else ""))
    try:
        result = upload_voice(args.url, args.upload_voice, name, ref_text)
        print(f"  Done: {result}")
        return 0
    except requests.RequestException as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("url", help="API base URL (e.g. https://abc.trycloudflare.com)")

    # Generate mode
    ap.add_argument("input", nargs="?", type=Path, help="Text file to read lines from (generate mode)")
    ap.add_argument("-n", "--lines", type=int, default=0, help="Number of lines to synthesize (0 = all)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument("--prefix", default="line_", help="Output filename prefix")
    ap.add_argument("--ref", type=Path, help="Reference audio file for voice cloning (wav/mp3/flac/ogg/m4a)")
    ap.add_argument("--ref-text", help="Transcript of reference audio (required by Dia for cloning)")
    ap.add_argument("--ref-text-file", type=Path, help="File containing the transcript of reference audio")
    ap.add_argument("--voice", help="Server-side voice name (from voices/ folder on the server)")
    ap.add_argument("--ext", default=".wav", help="Output file extension")

    # Upload mode
    ap.add_argument("--upload-voice", type=Path, help="Upload this audio clip to the server's voices/ folder")
    ap.add_argument("--voice-name", help="Name for the uploaded voice (default: audio filename stem)")

    args = ap.parse_args()

    if args.upload_voice:
        return cmd_upload(args)
    if not args.input:
        ap.print_help(sys.stderr)
        return 1
    return cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
