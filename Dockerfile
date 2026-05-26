FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ARG BUILD_DATE

# Python 3.10 + pip + git (Ubuntu 22.04 default)
RUN apt update && apt install -y \
    python3 python3-pip python3-venv python-is-python3 \
    git ffmpeg espeak-ng \
    && apt clean

# Piper-Source nach /piper-src – NICHT nach /app/piper/,
# sonst schattet das leere Verzeichnis das echte piper-Package (PEP 420).
RUN git clone https://github.com/rhasspy/piper /piper-src
RUN python3 -m pip install --upgrade pip

# Piper-Package installieren (editable inkl. HTTP-Extras)
WORKDIR /piper-src/src/python_run
RUN python3 -m pip install -e ".[http]" --no-cache-dir

# CUDA-Unterstützung: CPU-onnxruntime durch GPU-Version ersetzen
# CUDA 12.3 + cuDNN 9 via Base-Image, onnxruntime-gpu >= 1.21
RUN python3 -m pip uninstall -y onnxruntime 2>/dev/null; \
    python3 -m pip install onnxruntime-gpu --no-cache-dir

# Weitere Hilfsmittel
RUN python3 -m pip install wget num2words --no-cache-dir

# App-Dateien
WORKDIR /app
COPY run.py /app
COPY download /app/download

EXPOSE 5000

# ── Environment variables (defaults) ──────────────────────────────────
ENV MODEL_DOWNLOAD_LINK="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx?download=true"
ENV MODEL_DEFAULT="de_DE-thorsten-high"
ENV MODEL_TARGET_FOLDER="/app/models"
ENV SPEAKER="1"
ENV SENTENCE_SILENCE="0.0"
ENV LENGTH_SCALE="1.1"
ENV NOISE_SCALE="0.667"
ENV NOISE_W="0.8"
ENV CUDA="true"

CMD ["python3", "-u", "/app/run.py"]
