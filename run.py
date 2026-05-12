#!/usr/bin/env python3
"""
Custom Piper HTTP server with runtime model switching.
Supports the existing /?text= API and adds POST /voice to switch models.

Env vars:
  MODEL_DOWNLOAD_LINK   - HuggingFace model URL (used at startup)
  MODEL_TARGET_FOLDER   - folder to store model files (default: /app/models)
  SPEAKER               - speaker ID (default: 0)
  SENTENCE_SILENCE      - silence between sentences in seconds (default: 0.0)
  CUDA                  - enable GPU inference if available (default: true)
  LENGTH_SCALE          - phoneme length (default: none = Piper default)
  NOISE_SCALE           - generator noise (default: none = Piper default)
  NOISE_W               - phoneme width noise (default: none = Piper default)
"""

import subprocess
import os
import sys
import io
import wave
import logging
import json
import atexit
from pathlib import Path

from flask import Flask, request, jsonify

from piper import PiperVoice

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("piper-http-custom")

# ---------------------------------------------------------------------------
# Model management (singleton voice)
# ---------------------------------------------------------------------------

_voice: PiperVoice | None = None
_model_path: str | None = None
_synth_args: dict = {}


def download_model(link: str, target_folder: str) -> str:
    """Download model.onnx (+ .json) from huggingface if not already cached."""
    script_folder = os.path.dirname(os.path.realpath(__file__))
    download_script = os.path.join(script_folder, "download/download-piper-voices.py")
    model_path = os.path.join(target_folder, "model.onnx")

    if os.path.exists(model_path):
        _LOGGER.info("Model already exists at %s – skipping download", model_path)
    else:
        _LOGGER.info("Downloading model from %s", link)
        os.makedirs(target_folder, exist_ok=True)
        subprocess.run(["python", download_script, link, target_folder], check=True)

    return model_path


def _unload_voice():
    """Unload the current voice and free GPU memory."""
    global _voice
    if _voice is not None:
        _LOGGER.info("Unloading current model...")
        del _voice
        _voice = None
        # Clear CUDA cache if available
        try:
            import torch
            torch.cuda.empty_cache()
            _LOGGER.info("CUDA cache cleared")
        except ImportError:
            pass


def load_voice(
    model_path: str,
    use_cuda: bool = True,
    speaker_id: int | None = None,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w: float | None = None,
    sentence_silence: float = 0.0,
) -> PiperVoice:
    """Load (or switch to) a new voice model."""
    global _voice, _model_path, _synth_args

    # First, fully unload the old one
    _unload_voice()

    # Check if cuda is actually available
    if use_cuda:
        try:
            import torch
            use_cuda = torch.cuda.is_available()
            _LOGGER.info("CUDA available: %s", use_cuda)
        except ImportError:
            use_cuda = False

    _LOGGER.info("Loading model: %s (cuda=%s)", model_path, use_cuda)
    voice = PiperVoice.load(model_path, use_cuda=use_cuda)

    # Build the synthesis-arguments dict (only non-None values)
    _synth_args = {
        k: v
        for k, v in {
            "speaker_id": speaker_id,
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_w": noise_w,
            "sentence_silence": sentence_silence,
        }.items()
        if v is not None
    }

    _voice = voice
    _model_path = model_path
    _LOGGER.info("Voice loaded – synth_args=%s", _synth_args)
    return voice


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def handle_synthesize():
    """Synthesise speech from text.  Compatible with the original piper HTTP API.

    GET  /?text=Hello   → synthesise "Hello"
    POST /              → read text from request body
    """
    if _voice is None:
        return "No voice model loaded", 503

    text = (
        request.data.decode("utf-8")
        if request.method == "POST"
        else request.args.get("text", "")
    ).strip()

    if not text:
        return "No text provided", 400

    _LOGGER.debug("Synthesising: %s", text)
    try:
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                _voice.synthesize(text, wav_file, **_synth_args)
            return wav_io.getvalue()
    except Exception as exc:
        _LOGGER.error("Synthesis failed: %s", exc)
        return f"Synthesis error: {exc}", 500


@app.route("/voice", methods=["GET", "POST"])
def handle_voice():
    """Get or switch the current voice model.

    GET  /voice         → return current model path + synthesis config
    POST /voice         → switch to a new model

    POST JSON payload:
    {
      "link": "https://...model.onnx?download=true",
      "target_folder": "/app/models",           (optional)
      "cuda": true,                              (optional)
      "speaker_id": 0,                           (optional)
      "length_scale": 1.2,                       (optional)
      "noise_scale": 0.667,                      (optional)
      "noise_w": 0.8,                            (optional)
      "sentence_silence": 0.5                    (optional)
    }
    """
    if request.method == "GET":
        return jsonify({
            "model_path": _model_path,
            "synthesis_config": _synth_args,
        })

    # --- POST: switch voice ---
    data = request.get_json(silent=True)
    if not data:
        # also accept form-encoded
        data = request.form.to_dict()

    link = data.get("link") or data.get("model_path")
    if not link:
        return jsonify({"error": "Missing 'link' (model download URL)"}), 400

    target_folder = data.get("target_folder", "/app/models")

    try:
        model_path = download_model(link, target_folder)
    except Exception as exc:
        _LOGGER.error("Download failed: %s", exc)
        return jsonify({"error": f"Download failed: {exc}"}), 500

    try:
        load_voice(
            model_path,
            use_cuda=data.get("cuda", True),
            speaker_id=data.get("speaker_id"),
            length_scale=data.get("length_scale"),
            noise_scale=data.get("noise_scale"),
            noise_w=data.get("noise_w"),
            sentence_silence=data.get("sentence_silence", 0.0),
        )
        return jsonify({"status": "ok", "model": str(model_path)})
    except Exception as exc:
        _LOGGER.error("Loading model failed: %s", exc)
        return jsonify({"error": f"Failed to load model: {exc}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": _voice is not None,
        "model_path": _model_path,
    })


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main():
    # Read environment
    link = os.environ.get("MODEL_DOWNLOAD_LINK", "")
    target_folder = os.environ.get("MODEL_TARGET_FOLDER", "/app/models")
    speaker_str = os.environ.get("SPEAKER", "0")
    sentence_silence = float(os.environ.get("SENTENCE_SILENCE", "0.0"))
    cuda_str = os.environ.get("CUDA", "true")
    length_scale = _env_float("LENGTH_SCALE")
    noise_scale = _env_float("NOISE_SCALE")
    noise_w = _env_float("NOISE_W")

    use_cuda = cuda_str.lower() in ("true", "1", "yes")
    speaker_id = int(speaker_str) if speaker_str and speaker_str != "none" else None

    # Auto-download model at startup (if link is provided)
    if link:
        model_path = download_model(link, target_folder)
        load_voice(
            model_path,
            use_cuda=use_cuda,
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
            sentence_silence=sentence_silence,
        )
    else:
        _LOGGER.warning(
            "MODEL_DOWNLOAD_LINK not set – no model loaded at startup. "
            "Use POST /voice to load one later."
        )

    # Clean up on exit
    atexit.register(_unload_voice)

    _LOGGER.info("Starting Piper HTTP on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


def _env_float(key: str) -> float | None:
    val = os.environ.get(key)
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
