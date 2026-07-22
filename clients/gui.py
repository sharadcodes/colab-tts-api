#!/usr/bin/env python3
"""Tkinter GUI for the Colab TTS API.

Generate TTS audio from a text file, line by line, with optional voice
cloning. Works with any running backend (chatterbox / vibevoice / dia).

Voices: drop audio clips in the server's voices/ folder (e.g. voices/narrator.wav).
The GUI fetches the list from /voices and lets you pick by name.

Requirements:
    pip install requests
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import requests

_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}


class TTSApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Colab TTS Generator")
        self.root.geometry("780x660")
        self.root.minsize(600, 540)

        self.input_path = tk.StringVar()
        self.ref_path = tk.StringVar()
        self.ref_text = tk.StringVar()
        self.out_dir = tk.StringVar(value="out")
        self.prefix = tk.StringVar(value="line_")
        self.url = tk.StringVar()
        self.num_lines = tk.StringVar(value="0")
        self.ext = tk.StringVar(value=".wav")
        self.voice_var = tk.StringVar()

        self.msg_queue: queue.Queue[str] = queue.Queue()
        self.running = False

        self._build_ui()
        self.root.after(100, self._drain_queue)

    # ─── UI ──────────────────────────────────────
    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ── API URL ──
        url_frame = ttk.LabelFrame(self.root, text="API URL")
        url_frame.pack(fill="x", **pad)
        url_row = ttk.Frame(url_frame)
        url_row.pack(fill="x", padx=8, pady=6)
        ttk.Entry(url_row, textvariable=self.url).pack(side="left", fill="x", expand=True)
        ttk.Button(url_row, text="Refresh voices", command=self._fetch_voices, width=14).pack(
            side="left", padx=(4, 0)
        )

        # ── Voice selection ──
        voice_frame = ttk.LabelFrame(self.root, text="Voice (server-side voices/ folder)")
        voice_frame.pack(fill="x", **pad)
        voice_row = ttk.Frame(voice_frame)
        voice_row.pack(fill="x", padx=8, pady=6)
        ttk.Label(voice_row, text="Voice:").pack(side="left")
        self.voice_combo = ttk.Combobox(
            voice_row, textvariable=self.voice_var, values=[], width=20, state="readonly"
        )
        self.voice_combo.pack(side="left", padx=(4, 16))
        ttk.Button(voice_row, text="Upload...", command=self._on_upload_voice, width=10).pack(
            side="left", padx=(0, 4)
        )
        ttk.Label(
            voice_row,
            text="(leave empty to use ref audio below or default voice)",
            foreground="gray",
        ).pack(side="left")

        # ── Files ──
        files_frame = ttk.LabelFrame(self.root, text="Files")
        files_frame.pack(fill="x", **pad)

        rows = [
            ("Input text:", self.input_path, self._pick_input, "open"),
            ("Reference audio:", self.ref_path, self._pick_ref, "open"),
            ("Ref text file:", self.ref_text, self._pick_ref_text, "open"),
            ("Output dir:", self.out_dir, self._pick_out_dir, "dir"),
        ]
        for label, var, cmd, _kind in rows:
            r = ttk.Frame(files_frame)
            r.pack(fill="x", padx=8, pady=3)
            ttk.Label(r, text=label, width=16).pack(side="left")
            ttk.Entry(r, textvariable=var).pack(side="left", fill="x", expand=True)
            ttk.Button(r, text="Browse", command=cmd, width=8).pack(side="left", padx=(4, 0))

        # ── Options ──
        opt_frame = ttk.LabelFrame(self.root, text="Options")
        opt_frame.pack(fill="x", **pad)

        opt_row1 = ttk.Frame(opt_frame)
        opt_row1.pack(fill="x", padx=8, pady=4)
        ttk.Label(opt_row1, text="Lines (0=all):").pack(side="left")
        ttk.Entry(opt_row1, textvariable=self.num_lines, width=6).pack(side="left", padx=(4, 16))
        ttk.Label(opt_row1, text="Prefix:").pack(side="left")
        ttk.Entry(opt_row1, textvariable=self.prefix, width=12).pack(side="left", padx=(4, 16))
        ttk.Label(opt_row1, text="Extension:").pack(side="left")
        ttk.Combobox(
            opt_row1, textvariable=self.ext, values=[".wav", ".mp3"], width=6
        ).pack(side="left", padx=(4, 0))

        # ── Generate ──
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)
        self.gen_btn = ttk.Button(btn_frame, text="Generate", command=self._on_generate)
        self.gen_btn.pack(side="left", padx=8)
        self.stop_btn = ttk.Button(
            btn_frame, text="Stop", command=self._on_stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=4)

        # ── Progress ──
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=12, pady=(8, 4))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var).pack(anchor="w", padx=12)

        # ── Log ──
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=12, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=4, pady=4, side="left")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)

    # ─── Fetch voices from server ────────────────
    def _fetch_voices(self) -> None:
        url = self.url.get().strip()
        if not url:
            messagebox.showerror("Error", "Enter the API URL first.")
            return
        self._log("Fetching voice list from server...")

        def _do() -> None:
            try:
                r = requests.get(f"{url.rstrip('/')}/voices", timeout=10)
                r.raise_for_status()
                voices = r.json().get("voices", [])
                self.root.after(0, lambda: self._update_voices(voices))
            except requests.RequestException as e:
                self._log(f"ERROR fetching voices: {e}")

        threading.Thread(target=_do, daemon=True).start()

    def _update_voices(self, voices: list[str]) -> None:
        self.voice_combo["values"] = voices
        if voices:
            self._log(f"Voices available: {', '.join(voices)}")
        else:
            self._log("No voices found in server's voices/ folder.")

    # ─── Upload voice to server ──────────────────
    def _on_upload_voice(self) -> None:
        url = self.url.get().strip()
        if not url:
            messagebox.showerror("Error", "Enter the API URL first.")
            return
        audio_path = filedialog.askopenfilename(
            title="Select voice audio to upload",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if not audio_path:
            return
        name = Path(audio_path).stem

        # Ask for an optional transcript file (needed for Dia)
        transcript_path = filedialog.askopenfilename(
            title="Select transcript for this voice (optional — needed for Dia)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        ref_text_val = ""
        if transcript_path:
            try:
                ref_text_val = Path(transcript_path).read_text(encoding="utf-8").strip()
            except OSError as e:
                self._log(f"  ERROR reading transcript: {e}")
                return

        self._log(f"Uploading '{name}' from {audio_path}..."
                  + (" + transcript" if ref_text_val else ""))

        def _do() -> None:
            try:
                with open(audio_path, "rb") as f:
                    files = {"audio": (Path(audio_path).name, f, "application/octet-stream")}
                    data = {"name": name}
                    if ref_text_val:
                        data["ref_text"] = ref_text_val
                    r = requests.post(
                        f"{url.rstrip('/')}/voices/upload",
                        data=data,
                        files=files,
                        timeout=60,
                    )
                r.raise_for_status()
                self._log(f"  Uploaded: {r.json()}")
                self._fetch_voices()
            except requests.RequestException as e:
                self._log(f"  ERROR: {e}")

        threading.Thread(target=_do, daemon=True).start()

    # ─── File pickers ────────────────────────────
    def _pick_input(self) -> None:
        p = filedialog.askopenfilename(
            title="Select input text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if p:
            self.input_path.set(p)

    def _pick_ref(self) -> None:
        p = filedialog.askopenfilename(
            title="Select reference audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if p:
            self.ref_path.set(p)

    def _pick_ref_text(self) -> None:
        p = filedialog.askopenfilename(
            title="Select reference transcript",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if p:
            self.ref_text.set(p)

    def _pick_out_dir(self) -> None:
        p = filedialog.askdirectory(title="Select output directory")
        if p:
            self.out_dir.set(p)

    # ─── Logging ─────────────────────────────────
    def _log(self, msg: str) -> None:
        self.msg_queue.put(msg)

    def _drain_queue(self) -> None:
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self.log.configure(state="normal")
                self.log.insert("end", msg + "\n")
                self.log.see("end")
                self.log.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)

    # ─── Generate ────────────────────────────────
    def _on_generate(self) -> None:
        if self.running:
            return
        url = self.url.get().strip()
        input_p = self.input_path.get().strip()
        if not url:
            messagebox.showerror("Error", "API URL is required.")
            return
        if not input_p:
            messagebox.showerror("Error", "Input text file is required.")
            return
        inp = Path(input_p)
        if not inp.exists():
            messagebox.showerror("Error", f"Input file not found:\n{inp}")
            return

        try:
            n = int(self.num_lines.get()) if self.num_lines.get().strip() else 0
        except ValueError:
            messagebox.showerror("Error", "Lines must be an integer.")
            return

        self.running = True
        self.gen_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress["value"] = 0
        self._log("Starting generation...")

        t = threading.Thread(target=self._worker, args=(url, inp, n), daemon=True)
        t.start()

    def _on_stop(self) -> None:
        self.running = False
        self._log("Stop requested — will halt after current line.")

    def _worker(self, url: str, inp: Path, n: int) -> None:
        try:
            lines = [
                ln.strip()
                for ln in inp.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            if n > 0:
                lines = lines[:n]
            if not lines:
                self._log("ERROR: no non-empty lines in input file.")
                self._finish()
                return

            out_dir = Path(self.out_dir.get().strip() or "out")
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = self.prefix.get().strip() or "line_"
            ext = self.ext.get().strip() or ".wav"

            voice_name = self.voice_var.get().strip()
            ref_p = Path(self.ref_path.get().strip()) if self.ref_path.get().strip() else None
            ref_text_val: str | None = None
            rt = self.ref_text.get().strip()
            if rt:
                rtp = Path(rt)
                if rtp.exists():
                    ref_text_val = rtp.read_text(encoding="utf-8").strip()
                else:
                    ref_text_val = rt

            if voice_name:
                self._log(f"Using server voice: {voice_name}")
            elif ref_p and not ref_text_val:
                self._log("WARNING: ref audio without ref_text — Dia cloning will fail.")

            ref_bytes: bytes | None = None
            ref_name = ""
            ref_mime = "audio/wav"
            if ref_p and ref_p.exists():
                ref_bytes = ref_p.read_bytes()
                ref_name = ref_p.name
                ref_mime = _AUDIO_MIME.get(ref_p.suffix.lower(), "application/octet-stream")

            total = len(lines)
            width = len(str(total))
            self.root.after(0, lambda: self.progress.configure(maximum=total))

            for i, line in enumerate(lines, 1):
                if not self.running:
                    self._log("Stopped.")
                    break

                self.root.after(
                    0,
                    lambda i=i, t=total: self.status_var.set(
                        f"Generating {i}/{t}..."
                    ),
                )
                self._log(f"[{i}/{total}] {line[:70]}{'...' if len(line) > 70 else ''}")

                data: dict[str, str] = {"text": line}
                if ref_text_val:
                    data["ref_text"] = ref_text_val
                if voice_name:
                    data["voice"] = voice_name
                files = {}
                if ref_bytes and not voice_name:
                    files["ref_audio"] = (ref_name, ref_bytes, ref_mime)

                try:
                    r = requests.post(
                        f"{url.rstrip('/')}/tts",
                        data=data,
                        files=files,
                        timeout=300,
                    )
                    r.raise_for_status()
                    out_path = out_dir / f"{prefix}{i:0{width}d}{ext}"
                    out_path.write_bytes(r.content)
                    self._log(f"  -> {out_path} ({len(r.content) / 1024:.1f} KB)")
                except requests.RequestException as e:
                    self._log(f"  ERROR: {e}")
                    self._log("Aborting.")
                    break

                self.root.after(0, lambda v=i: self.progress.configure(value=v))

            self._log("Done.")
        except Exception as e:
            self._log(f"FATAL: {e}")
        finally:
            self._finish()

    def _finish(self) -> None:
        self.root.after(0, self._ui_finish)

    def _ui_finish(self) -> None:
        self.running = False
        self.gen_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Ready.")


def main() -> None:
    root = tk.Tk()
    TTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
