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
  SPEAKER               - speaker ID (default: 1)
  SENTENCE_SILENCE      - silence between sentences in seconds (default: 0.5)
  CUDA                  - enable GPU inference if available (default: true)
  LENGTH_SCALE          - phoneme length (default: 1.0)
  NOISE_SCALE           - generator noise (default: none = Piper default)
  NOISE_W               - phoneme width noise (default: none = Piper default)
  OUTPUT_FORMAT         - optional: "ogg" to convert via ffmpeg, empty/"wav" to skip (default: none)
  OUTPUT_QUALITY        - ffmpeg audio quality (-q:a), used when OUTPUT_FORMAT is set (default: 5)
  OUTPUT_SAMPLE_RATE    - ffmpeg sample rate (-ar), used when OUTPUT_FORMAT is set (default: 22050)
  OUTPUT_PAD_START      - seconds of silence before audio (default: 0)
  OUTPUT_PAD_END        - seconds of silence after audio (default: 0)
"""

import subprocess
import os
import sys
import io
import wave
import logging
import json
import atexit
import threading
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
    """Download model (.onnx + .json) from huggingface if not already cached.

    Uses the original filename from the download URL (e.g. ``de_DE-thorsten-high.onnx``).
    Falls back to the old ``model.onnx`` for backward compatibility.
    Returns the path to the downloaded model file.
    """
    script_folder = os.path.dirname(os.path.realpath(__file__))
    download_script = os.path.join(script_folder, "download/download-piper-voices.py")

    # Extract the intended filename from the download URL
    from urllib.parse import urlparse
    model_basename = os.path.basename(urlparse(link).path)  # "de_DE-thorsten-high.onnx?download=true" → "de_DE-thorsten-high.onnx"
    if not model_basename.endswith(".onnx"):
        model_basename = "model.onnx"  # fallback

    model_path = os.path.join(target_folder, model_basename)
    legacy_path = os.path.join(target_folder, "model.onnx")

    # Check cache: prefer the original filename, fall back to legacy
    if os.path.exists(model_path):
        _LOGGER.info("Model already exists at %s – skipping download", model_path)
    elif os.path.exists(legacy_path):
        _LOGGER.info("Using legacy model.onnx – renaming to %s", model_basename)
        os.rename(legacy_path, model_path)
        legacy_json = os.path.join(target_folder, "model.onnx.json")
        model_json = model_path + ".json"
        if os.path.exists(legacy_json):
            os.rename(legacy_json, model_json)
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
        # CUDA-Cache wird von onnxruntime beim Session-Close automatisch geräumt
        # torch.cuda.empty_cache() entfällt – PyTorch ist nicht installiert


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
            import onnxruntime
            use_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
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


# ── Helpers for model switching ─────────────────────────────

def _arg_float(val: str | None, default: float | None = None) -> float | None:
    """Parse a query-string float, returning *default* on missing/empty."""
    if val is None or val.strip() == "":
        return default
    try:
        return float(val.strip())
    except (ValueError, TypeError):
        return default


_model_lock = threading.Lock()


def _switch_voice(
    link: str = "",
    target_folder: str | None = None,
    use_cuda: bool = True,
    speaker_id: int | None = None,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w: float | None = None,
    sentence_silence: float | None = None,
) -> tuple[Response, int]:
    """Download & load a new Piper voice.  Returns a JSON response tuple."""
    with _model_lock:
        link = link.strip()
        if not link:
            default_ref = os.environ.get("MODEL_DEFAULT", "")
            if not default_ref:
                return jsonify({"error": "Missing model ref or MODEL_DEFAULT"}), 400
            link = resolve_model_ref(default_ref)

        target_folder = target_folder or os.environ.get("MODEL_TARGET_FOLDER", "/app/models")

        try:
            model_path = download_model(link, target_folder)
        except Exception as exc:
            _LOGGER.error("Download failed: %s", exc)
            return jsonify({"error": f"Download failed: {exc}"}), 500

        # ── Cache: skip reload if same model + same params ──────────
        requested_args: dict = {
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
        if model_path == _model_path and requested_args == _synth_args:
            _LOGGER.info("Model %s already loaded – skipping reload", model_path)
            return jsonify({"status": "ok", "model": str(model_path), "cached": True}), 200

        try:
            load_voice(
                model_path,
                use_cuda=use_cuda,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                sentence_silence=sentence_silence,
            )
            return jsonify({"status": "ok", "model": str(model_path)}), 200
        except Exception as exc:
            _LOGGER.error("Loading model failed: %s", exc)
            return jsonify({"error": f"Failed to load model: {exc}"}), 500


# ---------------------------------------------------------------------------
# HTML guide page
# ---------------------------------------------------------------------------

def _render_models_table_rows(target_folder: str) -> str:
    """Render table rows for .onnx files in the model folder."""
    rows = []
    try:
        if os.path.isdir(target_folder):
            for fname in sorted(os.listdir(target_folder)):
                if not fname.endswith(".onnx"):
                    continue
                full = os.path.join(target_folder, fname)
                size_mb = os.path.getsize(full) / (1024 * 1024)
                active = "✅" if full == _model_path else "–"
                rows.append(
                    f'<tr style="border-bottom:1px solid #222;">'
                    f'<td style="padding:4px 8px;">{html_mod.escape(fname)}</td>'
                    f'<td style="padding:4px 8px;">{size_mb:.1f} MB</td>'
                    f'<td style="padding:4px 8px;">{active}</td>'
                    f'</tr>'
                )
    except OSError:
        pass
    if not rows:
        return '<tr><td colspan="3" style="padding:8px; color:#888;">Keine Modelle gefunden</td></tr>'
    return "\n".join(rows)


def _render_model_options(target_folder: str) -> str:
    """Render <option> elements for a <select> dropdown of available models."""
    options = []
    try:
        if os.path.isdir(target_folder):
            for fname in sorted(os.listdir(target_folder)):
                if not fname.endswith(".onnx"):
                    continue
                full = os.path.join(target_folder, fname)
                size_mb = os.path.getsize(full) / (1024 * 1024)
                onnx_name = fname
                selected = ' selected' if full == _model_path else ''
                options.append(
                    f'<option value="{html_mod.escape(onnx_name)}"{selected}>'
                    f'{html_mod.escape(onnx_name)} ({size_mb:.1f} MB)'
                    f'</option>'
                )
    except OSError:
        pass
    if not options:
        return '<option value="">– Keine Modelle gefunden –</option>'
    return "\n".join(options)


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

<!-- ─── Combined quick-test + voice switch ─── -->
<h2>Stimme testen</h2>
<form action="{host_url}/" method="get" id="synth-form">
  <label for="model-select">Stimme (lokal verfügbar):</label>
  <select name="model" id="model-select"
          style="width:100%;padding:10px;border:1px solid #333;border-radius:6px;
                 background:#0f3460;color:#e0e0e0;font-size:15px;">
    {_render_model_options(target_folder=os.environ.get("MODEL_TARGET_FOLDER", "/app/models"))}
  </select>
  <div style="margin-top:4px;color:#888;font-size:13px;">
    Aktuell: {html_mod.escape(os.path.basename(_model_path or '?'))}
  </div>

  <label for="text" style="margin-top:16px;">Text eingeben:</label>
  <input type="text" name="text" id="text"
         value="{html_mod.escape(default_text)}"
         placeholder="Dein Text zum Sprechen">

  <div style="display:flex;gap:12px;margin-top:10px;">
    <div style="flex:1;">
      <label for="length-scale" style="font-size:13px;">length_scale</label>
      <input type="text" name="length_scale" id="length-scale"
             placeholder="z.B. 1.2 (Default aus Config)"
             style="width:100%;padding:8px;border:1px solid #333;border-radius:6px;
                    background:#0f3460;color:#e0e0e0;font-size:13px;">
    </div>
    <div style="flex:1;">
      <label for="sentence-silence" style="font-size:13px;">sentence_silence</label>
      <input type="text" name="sentence_silence" id="sentence-silence"
             placeholder="z.B. 0.3 (Default aus Config)"
             style="width:100%;padding:8px;border:1px solid #333;border-radius:6px;
                    background:#0f3460;color:#e0e0e0;font-size:13px;">
    </div>
    <div style="flex:1;">
      <label for="speaker-id" style="font-size:13px;">speaker_id</label>
      <input type="text" name="speaker_id" id="speaker-id"
             placeholder="z.B. 0 (Default aus Config)"
             style="width:100%;padding:8px;border:1px solid #333;border-radius:6px;
                    background:#0f3460;color:#e0e0e0;font-size:13px;">
    </div>
    <div style="flex:1;">
      <label for="format" style="font-size:13px;">format</label>
      <input type="text" name="format" id="format" value="ogg"
             style="width:100%;padding:8px;border:1px solid #333;border-radius:6px;
                    background:#0f3460;color:#e0e0e0;font-size:13px;">
    </div>
  </div>

  <div id="synth-status" class="loading" style="margin-top:8px;"></div>
  <button type="submit">🔊 Stimme wechseln &amp; synthetisieren</button>
</form>
<script>
document.getElementById('synth-form').addEventListener('submit', function(e) {{
  e.preventDefault();
  var status = document.getElementById('synth-status');
  var model = document.getElementById('model-select').value;
  var text = document.getElementById('text').value.trim();
  if (!text) {{ status.textContent = 'Bitte Text eingeben'; return; }}
  status.textContent = 'Wechsle Stimme …';
  var button = this.querySelector('button');
  button.disabled = true;

  fetch('{host_url}/voice', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ link: model }})
  }})
  .then(function(r) {{ return r.json(); }})
  .then(function(d) {{
    if (d.status !== 'ok') {{
      status.textContent = 'Stimme fehlgeschlagen: ' + (d.error || 'Unbekannt');
      button.disabled = false;
      return;
    }}
    status.textContent = 'Synthetisiere …';
    var params = new URLSearchParams();
    params.set('text', text);
    var ls = document.getElementById('length-scale').value.trim();
    var ss = document.getElementById('sentence-silence').value.trim();
    var sid = document.getElementById('speaker-id').value.trim();
    var fmt = document.getElementById('format').value.trim();
    if (ls) params.set('length_scale', ls);
    if (ss) params.set('sentence_silence', ss);
    if (sid) params.set('speaker_id', sid);
    if (fmt) params.set('format', fmt);
    // Download auslösen ohne die Seite zu verlassen
    var a = document.createElement('a');
    a.href = '{host_url}/?' + params.toString();
    a.download = 'piper-tts.' + (fmt || 'wav');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    status.textContent = '✅ Synthese fertig – Datei wird heruntergeladen';
    button.disabled = false;
  }})
  .catch(function(e) {{
    status.textContent = e.message;
    button.disabled = false;
  }});
}});
</script>

<!-- ─── Bot / Agent API Guide ─── -->
<h2>API für Bots &amp; Automation</h2>
<p style="color:#888;">Alle Endpunkte, die ein anderer Agent (oder curl) braucht:</p>

<h3 style="color:#a8d8ea;">Grundaufbau</h3>
<pre>GET  /?text=Hallo+Welt                              → Audio-Download
POST /                                               → Text im Body = Audio-Download
POST /?length_scale=1.0&amp;format=ogg                   → Mit Parametern</pre>
<p>Sämtliche Synthese- und Output-Parameter werden als Query-Parameter an <code>GET /</code> oder <code>POST /</code> angehängt.</p>

<h3 style="color:#a8d8ea;">Model Management</h3>
<pre>GET  /voice    → Aktuelles Modell + Konfiguration (JSON)
POST /voice    → Stimme wechseln (JSON Body, siehe curl)
GET  /models   → Verfügbare Modelle (JSON)
GET  /health   → Status + Version + CUDA (JSON)</pre>
<p><code>POST /voice</code> Body:</p>
<pre>{{
  "link": "de_DE-thorsten-high",
  "sentence_silence": 1.5,
  "speaker_id": 0,
  "length_scale": 1.1
}}</pre>
<p>
  <code>link</code> akzeptiert:<br>
  • HuggingFace-URL (z.B. <code>https://huggingface.co/rhasspy/piper-voices/…/de_DE-thorsten-high.onnx</code>)<br>
  • Dateiname (aus <code>GET /models</code>)<br>
  • Kurzname (z.B. <code>de_DE-thorsten-high</code>)
</p>
<p>Auch <code>?model=</code> im TTS-Request wechselt per-Request die Stimme, ohne den globalen Zustand zu ändern.</p>

<h3 style="color:#a8d8ea;">Synthese-Parameter</h3>
<table style="width:100%;border-collapse:collapse;margin-bottom:1em;">
<tr style="border-bottom:1px solid #333;text-align:left;">
  <th style="padding:4px 8px;">Parameter</th>
  <th style="padding:4px 8px;">Typ</th>
  <th style="padding:4px 8px;">Beschreibung</th>
</tr>
<tr><td style="padding:4px 8px;"><code>speaker_id</code></td><td style="padding:4px 8px;">int</td><td style="padding:4px 8px;">Sprecher-ID (0 oder 1, je nach Modell)</td></tr>
<tr><td style="padding:4px 8px;"><code>sentence_silence</code></td><td style="padding:4px 8px;">float</td><td style="padding:4px 8px;">Pause zwischen Sätzen in Sekunden (Default: 0.5)</td></tr>
<tr><td style="padding:4px 8px;"><code>length_scale</code></td><td style="padding:4px 8px;">float</td><td style="padding:4px 8px;">Sprechgeschwindigkeit (1.0 = normal)</td></tr>
<tr><td style="padding:4px 8px;"><code>noise_scale</code></td><td style="padding:4px 8px;">float</td><td style="padding:4px 8px;">Generator-Rauschen (Piper-Standard)</td></tr>
<tr><td style="padding:4px 8px;"><code>noise_w</code></td><td style="padding:4px 8px;">float</td><td style="padding:4px 8px;">Phonem-Breiten-Rauschen</td></tr>
</table>

<h3 style="color:#a8d8ea;">Output-Format</h3>
<table style="width:100%;border-collapse:collapse;margin-bottom:1em;">
<tr style="border-bottom:1px solid #333;text-align:left;">
  <th style="padding:4px 8px;">Parameter</th>
  <th style="padding:4px 8px;">Typ</th>
  <th style="padding:4px 8px;">Beschreibung</th>
</tr>
<tr><td style="padding:4px 8px;"><code>format</code></td><td style="padding:4px 8px;">string</td><td style="padding:4px 8px;">Ausgabeformat: <code>wav</code> (Default), <code>ogg</code>, <code>mp3</code>, <code>opus</code>, <code>flac</code>, <code>aac</code></td></tr>
<tr><td style="padding:4px 8px;"><code>upsample</code></td><td style="padding:4px 8px;">bool</td><td style="padding:4px 8px;">Auf mind. 22kHz hochsamplen (<code>true</code>/<code>false</code>)</td></tr>
</table>
<p style="color:#888;font-size:13px;">Die Samplerate wird vom Piper-Modell vorgegeben. <code>upsample=true</code> skaliert auf ≥22kHz hoch.</p>

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

    Optional query params (override current voice config per request):
      ?speaker_id=0&length_scale=1.1&noise_scale=0.667&noise_w=0.8&sentence_silence=0.5
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

    # ── Merge per-request override params ──
    overrides = {}
    for key in ("speaker_id", "length_scale", "noise_scale", "noise_w", "sentence_silence"):
        val = _arg_float(request.args.get(key))
        if val is not None:
            overrides[key] = int(val) if key == "speaker_id" else val

    # ── Per-request model switch (optional) ──
    model_param = request.args.get("model", "").strip()
    if model_param:
        _LOGGER.info("Per-request model switch to %s with overrides=%s", model_param, overrides)
        result, code = _switch_voice(link=model_param, **overrides)
        if code != 200:
            _LOGGER.warning("Model switch failed: %s", result.get_data(as_text=True))
            return result, code

    synth_kwargs = {**_synth_args, **overrides}

    _LOGGER.debug("Synthesising: %s  overrides=%s -> %s", text[:50], overrides, synth_kwargs)

    # ── Output format config ──────────────────────────────────────
    output_format = os.environ.get("OUTPUT_FORMAT", "").strip().lower()
    output_quality = os.environ.get("OUTPUT_QUALITY", "5")

    # Per-request override via ?format=
    output_format = request.args.get("format", output_format).strip().lower()

    # ── Padding config ────────────────────────────────────────────
    pad_start = float(request.args.get("pad_start",
        os.environ.get("OUTPUT_PAD_START", "0")))
    pad_end = float(request.args.get("pad_end",
        os.environ.get("OUTPUT_PAD_END", "0")))

    try:
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                _voice.synthesize(text, wav_file, **synth_kwargs)
            wav_data = wav_io.getvalue()

        # ── Determine target sample rate ──────────────────────────
        with io.BytesIO(wav_data) as r:
            with wave.open(r, "rb") as wf:
                native_rate = wf.getframerate()

        sr_param = request.args.get("sample_rate", "").strip()
        upsample_param = request.args.get("upsample", "").strip().lower() in ("true", "1", "yes")
        if sr_param:
            output_sample_rate = int(sr_param)
        elif upsample_param:
            output_sample_rate = max(native_rate, 22050)
        else:
            output_sample_rate = int(os.environ.get("OUTPUT_SAMPLE_RATE", str(native_rate)))

        # ── Optional ffmpeg conversion ─────────────────────────
        if output_format and output_format != "wav":
            CODEC_MAP = {
                "ogg": "libvorbis",
                "opus": "libopus",
                "mp3": "libmp3lame",
                "mpeg": "libmp3lame",
                "aac": "aac",
                "flac": "flac",
            }
            MIME_MAP = {
                "ogg": "audio/ogg",
                "opus": "audio/opus",
                "mp3": "audio/mpeg",
                "aac": "audio/aac",
                "flac": "audio/flac",
                "wav": "audio/wav",
            }
            ext = output_format
            mimetype = MIME_MAP.get(output_format, "audio/" + output_format)

            ffmpeg_cmd = [
                "ffmpeg",
                "-i", "pipe:0",
            ]
            # ── Silence padding via audio filter ──────────────
            if pad_start > 0 or pad_end > 0:
                filters = []
                if pad_start > 0:
                    filters.append(f"adelay={int(pad_start * 1000)}")
                if pad_end > 0:
                    filters.append(f"apad=pad_dur={pad_end}")
                ffmpeg_cmd += ["-af", ",".join(filters)]
            ffmpeg_cmd += [
                "-codec:a", CODEC_MAP.get(output_format, "copy"),
                "-q:a", output_quality,
                "-ar", str(output_sample_rate),
                "-f", output_format,
                "pipe:1",
                "-y",
            ]

            proc = subprocess.run(
                ffmpeg_cmd,
                input=wav_data,
                capture_output=True,
                check=True,
            )
            audio_data = proc.stdout
        else:
            ext = "wav"
            mimetype = "audio/wav"
            audio_data = wav_data

        return Response(
            audio_data,
            mimetype=mimetype,
            headers={
                "Content-Disposition": f'attachment; filename="piper-tts.{ext}"',
                "Content-Length": str(len(audio_data)),
            },
        )
    except Exception as exc:
        _LOGGER.error("Synthesis failed: %s", exc)
        return f"Synthesis error: {exc}", 500


@app.route("/voice", methods=["GET", "POST"])
def handle_voice():
    """Get or switch the current voice model.

    GET  /voice                      → current model path + synthesis config
    GET  /voice?model=de_DE-thorsten-high  → switch model by name (query params)
    POST /voice                     → switch model (JSON body)

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
        # ?model= → switch model
        model_param = request.args.get("model", "").strip()
        if model_param:
            resp, code = _switch_voice(
                link=model_param,
                target_folder=request.args.get("target_folder"),
                use_cuda=(request.args.get("cuda") or "").lower() in ("true", "1", "yes"),
                speaker_id=int(request.args["speaker_id"]) if request.args.get("speaker_id") else None,
                length_scale=_arg_float(request.args.get("length_scale")),
                noise_scale=_arg_float(request.args.get("noise_scale")),
                noise_w=_arg_float(request.args.get("noise_w")),
                sentence_silence=_arg_float(request.args.get("sentence_silence"), default=0.0),
            )
            return resp, code

        # Otherwise just show current state
        return jsonify({
            "model_path": _model_path,
            "synthesis_config": _synth_args,
        })

    # --- POST: switch voice ---
    data = request.get_json(silent=True)
    if not data:
        data = request.form.to_dict()

    # Coerce cuda from string if necessary
    cuda_val = data.get("cuda", True)
    if isinstance(cuda_val, str):
        use_cuda = cuda_val.lower() in ("true", "1", "yes")
    else:
        use_cuda = bool(cuda_val)

    resp, code = _switch_voice(
        link=data.get("link") or data.get("model_path") or "",
        target_folder=data.get("target_folder"),
        use_cuda=use_cuda,
        speaker_id=data.get("speaker_id"),
        length_scale=data.get("length_scale"),
        noise_scale=data.get("noise_scale"),
        noise_w=data.get("noise_w"),
        sentence_silence=data.get("sentence_silence", 0.0),
    )
    return resp, code


@app.route("/health", methods=["GET"])
def health():
    """Simple health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": _voice is not None,
        "model_path": _model_path,
    })


# ── Well-known Piper voices (list for the web UI) ─────────────────
_WELL_KNOWN_VOICES: list[dict] = [
    {"name": "de_DE-thorsten-high",      "label": "Thorsten (High)",       "gender": "male",   "language": "Deutsch"},
    {"name": "de_DE-thorsten-medium",    "label": "Thorsten (Medium)",     "gender": "male",   "language": "Deutsch"},
    {"name": "de_DE-thorsten-low",       "label": "Thorsten (Low)",        "gender": "male",   "language": "Deutsch"},
    {"name": "de_DE-thorsten-tesslow",   "label": "Thorsten (Tesseract)",  "gender": "male",   "language": "Deutsch"},
    {"name": "de_DE-eva_k-x-low",        "label": "Eva K. (X-Low)",        "gender": "female", "language": "Deutsch"},
    {"name": "de_DE-kerstin-low",        "label": "Kerstin (Low)",         "gender": "female", "language": "Deutsch"},
    {"name": "de_DE-mls-medium",         "label": "MLS (Medium)",          "gender": "unknown","language": "Deutsch"},
    {"name": "de_DE-pavoque-low",        "label": "Pavoque (Low)",         "gender": "unknown","language": "Deutsch"},
    {"name": "de_DE-ramona-low",         "label": "Ramona (Low)",          "gender": "female", "language": "Deutsch"},
    {"name": "en_GB-alan-low",           "label": "Alan (GB, Low)",        "gender": "male",   "language": "English"},
    {"name": "en_GB-alan-medium",        "label": "Alan (GB, Medium)",     "gender": "male",   "language": "English"},
    {"name": "en_US-amy-low",            "label": "Amy (US, Low)",         "gender": "female", "language": "English"},
    {"name": "en_US-amy-medium",         "label": "Amy (US, Medium)",      "gender": "female", "language": "English"},
    {"name": "en_US-lessac-medium",      "label": "Lessac (US, Medium)",   "gender": "female", "language": "English"},
    {"name": "en_US-lessac-high",        "label": "Lessac (US, High)",     "gender": "female", "language": "English"},
    {"name": "en_US-libritts-high",      "label": "LibriTTS (US, High)",   "gender": "unknown","language": "English"},
]


@app.route("/models", methods=["GET"])
def list_models():
    """List available models (files on disk + well-known voices)."""
    target_folder = os.environ.get("MODEL_TARGET_FOLDER", "/app/models")

    on_disk = []
    if os.path.isdir(target_folder):
        for fname in sorted(os.listdir(target_folder)):
            if fname.endswith(".onnx"):
                full = os.path.join(target_folder, fname)
                size_mb = round(os.path.getsize(full) / (1024 * 1024), 1)
                is_active = (full == _model_path)
                on_disk.append({"file": fname, "path": full, "size_mb": size_mb, "active": is_active})

    return jsonify({
        "target_folder": target_folder,
        "active_model": _model_path,
        "on_disk": on_disk,
        "well_known": _WELL_KNOWN_VOICES,
    })


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main():
    # Snapshot env for web UI
    global _startup_env
    _startup_env = {k: v for k, v in os.environ.items() if k.startswith("MODEL_")}
    # Also snapshot the synthesis env vars
    for key in ("SPEAKER", "SENTENCE_SILENCE", "LENGTH_SCALE", "NOISE_SCALE", "NOISE_W", "CUDA",
                "OUTPUT_FORMAT", "OUTPUT_QUALITY", "OUTPUT_SAMPLE_RATE", "OUTPUT_PAD_START", "OUTPUT_PAD_END"):
        if key in os.environ:
            _startup_env[key] = os.environ[key]

    # Resolve MODEL_DEFAULT name → URL for later use
    model_default_name = os.environ.get("MODEL_DEFAULT", "")
    model_default_url = resolve_model_ref(model_default_name) if model_default_name else ""

    # Read environment
    link = os.environ.get("MODEL_DOWNLOAD_LINK", "")
    target_folder = os.environ.get("MODEL_TARGET_FOLDER", "/app/models")
    speaker_str = os.environ.get("SPEAKER", "1")
    sentence_silence = float(os.environ.get("SENTENCE_SILENCE", "0.5"))
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
