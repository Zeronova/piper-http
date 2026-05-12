#!/usr/bin/env python3
"""
Custom Piper HTTP server with runtime model switching and web UI.

Web UI:
  GET  /                 → HTML guide page (if no ?text=)
  GET  /?text=Hallo      → WAV download
  POST /                 → text in body → WAV download

Model switching:
  GET  /voice            → current model info
  POST /voice            → switch to new model
  GET  /health           → status check

Env vars:
  MODEL_DOWNLOAD_LINK   - HuggingFace model URL (used at startup)
  MODEL_DEFAULT         - default model link shown in web UI
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
import html as html_mod
from pathlib import Path

from flask import Flask, request, jsonify, Response

from piper import PiperVoice

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("piper-http-custom")

# ---------------------------------------------------------------------------
# Model management (singleton voice)
# ---------------------------------------------------------------------------

_voice: PiperVoice | None = None
_model_path: str | None = None
_synth_args: dict = {}
_startup_env: dict = {}  # snapshot of env vars for the web UI


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


# ── Well-known voice download URLs for Piper ─────────────────
# If MODEL_DEFAULT is a short name (de_DE-thorsten-high), it gets
# expanded to the full HuggingFace URL via this template.
_PIPER_VOICE_URL_TPL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "{language}/{locale}/{voice}/{quality}/{locale}-{voice}-{quality}.onnx?download=true"
)


def resolve_model_ref(ref: str) -> str:
    """Return a download URL for *ref*.

    * If *ref* starts with ``http`` → return unchanged.
    * Otherwise treat as ``{locale}-{voice}-{quality}`` (e.g.
      ``de_DE-thorsten-high``) and build the standard HuggingFace URL.
    """
    if ref.startswith("http"):
        return ref

    # expected format: locale-voice-quality  e.g.  de_DE-thorsten-high
    parts = ref.rsplit("-", 2)
    if len(parts) != 3:
        _LOGGER.warning("Cannot parse model name %r – using as-is", ref)
        return ref

    locale, voice, quality = parts
    language = locale.split("_")[0]
    return _PIPER_VOICE_URL_TPL.format(
        language=language, locale=locale, voice=voice, quality=quality
    )


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

    _unload_voice()

    if use_cuda:
        try:
            import torch
            use_cuda = torch.cuda.is_available()
            _LOGGER.info("CUDA available: %s", use_cuda)
        except ImportError:
            use_cuda = False

    _LOGGER.info("Loading model: %s (cuda=%s)", model_path, use_cuda)
    voice = PiperVoice.load(model_path, use_cuda=use_cuda)

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
# HTML guide page
# ---------------------------------------------------------------------------

def _render_html() -> str:
    """Render the web UI guide page."""
    host_url = request.host_url.rstrip("/")
    current = _synth_args.copy()
    current["model_path"] = _model_path

    env = _startup_env
    default_ref = env.get("MODEL_DEFAULT", "")
    default_url = resolve_model_ref(default_ref) if default_ref else "–"
    default_text = "Hallo Welt, das ist ein Test."  # placeholder fallback

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Piper TTS</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px;
    line-height: 1.6;
  }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  h1 {{ color: #00d4aa; border-bottom: 2px solid #00d4aa; padding-bottom: 8px; }}
  h2 {{ color: #ffd369; margin-top: 2em; }}
  pre {{
    background: #16213e; color: #a8d8ea; padding: 14px; border-radius: 8px;
    overflow-x: auto; font-size: 14px;
  }}
  code {{ background: #16213e; padding: 2px 6px; border-radius: 4px; }}
  form {{ background: #16213e; padding: 20px; border-radius: 8px; margin: 1em 0; }}
  label {{ display: block; margin-bottom: 6px; font-weight: bold; }}
  input[type=text] {{
    width: 100%; padding: 10px; border: 1px solid #333; border-radius: 6px;
    background: #0f3460; color: #e0e0e0; font-size: 15px;
  }}
  button {{
    background: #00d4aa; color: #1a1a2e; border: none; padding: 10px 24px;
    border-radius: 6px; font-size: 15px; font-weight: bold; cursor: pointer;
    margin-top: 10px;
  }}
  button:hover {{ background: #00f0c0; }}
  .badge {{ color: #ffd369; font-weight: normal; }}
  .loading {{ color: #888; }}
  #config-display {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<div class="container">

<h1>🔊 Piper TTS</h1>

<!-- ─── Quick form ─── -->
<h2>Sofort testen</h2>
<form method="get" action="{{host_url}}/">
  <label for="text">Text eingeben:</label>
  <input type="text" name="text" id="text"
         value="{html_mod.escape(default_text)}"
         placeholder="Dein Text zum Sprechen">
  <button type="submit">🔊 Synthetisieren &amp; Download</button>
</form>

<!-- ─── curl / POST usage ─── -->
<h2>curl</h2>
<pre>curl -o output.wav "{host_url}/?text=Hallo+Welt"</pre>

<h2>POST (raw body)</h2>
<pre>curl -X POST "{host_url}/" -d "Hallo Welt" -o output.wav</pre>

<h2>Stimme wechseln</h2>
<pre>curl -X POST "{host_url}/voice" \\
  -H "Content-Type: application/json" \\
  -d '{{"link": "{html_mod.escape(default_url)}", "sentence_silence": 1.5}}'</pre>

<!-- ─── Current config ─── -->
<h2>Aktuelle Konfiguration</h2>
<pre id="config-display">Lade…</pre>

<!-- ─── Default model ─── -->
<p><strong>Default-Model:</strong> <code>{html_mod.escape(default_ref)}</code></p>
<p><strong>→ URL:</strong> <code>{html_mod.escape(default_url)}</code></p>

</div>

<script>
fetch('{host_url}/voice')
  .then(r => r.json())
  .then(d => {{
    document.getElementById('config-display').textContent =
      JSON.stringify(d, null, 2);
  }})
  .catch(e => {{
    document.getElementById('config-display').textContent =
      'Fehler: ' + e.message;
  }});
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def handle_synthesize():
    """
    GET  /              → HTML guide page (browser)
    GET  /?text=Hallo   → WAV download
    POST /              → text in body → WAV download
    """
    if _voice is None:
        return "No voice model loaded", 503

    # ── Browser: no text parameter → show guide page ──
    if request.method == "GET" and not request.args.get("text"):
        return _render_html()

    # ── Extract text ──
    text = (
        request.data.decode("utf-8")
        if request.method == "POST"
        else request.args.get("text", "")
    ).strip()

    if not text:
        # POST without body → guide page
        if request.method == "POST":
            return _render_html()
        return "No text provided", 400

    _LOGGER.debug("Synthesising: %s", text)
    try:
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                _voice.synthesize(text, wav_file, **_synth_args)
            wav_data = wav_io.getvalue()

        return Response(
            wav_data,
            mimetype="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="piper-tts.wav"',
                "Content-Length": str(len(wav_data)),
            },
        )
    except Exception as exc:
        _LOGGER.error("Synthesis failed: %s", exc)
        return f"Synthesis error: {exc}", 500


@app.route("/voice", methods=["GET", "POST"])
def handle_voice():
    """Get or switch the current voice model.

    GET  /voice         → current model path + synthesis config
    POST /voice         → switch to a new model

    POST JSON payload (all optional):
    {
      "link": "https://...model.onnx?download=true",   # wenn leer → MODEL_DEFAULT
      "target_folder": "/app/models",
      "cuda": true,
      "speaker_id": 0,
      "length_scale": 1.2,
      "noise_scale": 0.667,
      "noise_w": 0.8,
      "sentence_silence": 0.5
    }

    If *link* is omitted (or the body is empty), the server loads
    the voice defined by the ``MODEL_DEFAULT`` env var.
    """
    if request.method == "GET":
        return jsonify({
            "model_path": _model_path,
            "synthesis_config": _synth_args,
        })

    # --- POST: switch voice ---
    data = request.get_json(silent=True)
    if not data:
        data = request.form.to_dict()

    link = (data.get("link") or data.get("model_path") or "").strip()

    # Fallback to MODEL_DEFAULT when no link was provided
    if not link:
        default_ref = os.environ.get("MODEL_DEFAULT", "")
        if not default_ref:
            return jsonify({"error": "Missing 'link'. Set MODEL_DEFAULT env var or pass a link."}), 400
        link = resolve_model_ref(default_ref)
        _LOGGER.info("No link provided – falling back to MODEL_DEFAULT: %s", link)

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
    # Snapshot env for web UI
    global _startup_env
    _startup_env = {k: v for k, v in os.environ.items() if k.startswith("MODEL_")}
    # Also snapshot the synthesis env vars
    for key in ("SPEAKER", "SENTENCE_SILENCE", "LENGTH_SCALE", "NOISE_SCALE", "NOISE_W", "CUDA"):
        if key in os.environ:
            _startup_env[key] = os.environ[key]

    # Resolve MODEL_DEFAULT name → URL for later use
    model_default_name = os.environ.get("MODEL_DEFAULT", "")
    model_default_url = resolve_model_ref(model_default_name) if model_default_name else ""

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

    # Auto-download & load model at startup
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
