FROM python:3.11-slim

ARG BUILD_DATE

# Piper-Source nach /piper-src – NICHT nach /app/piper/,
# sonst schattet das leere Verzeichnis das echte piper-Package (PEP 420).
RUN apt update && apt install -y git
RUN git clone https://github.com/rhasspy/piper /piper-src
RUN pip install --upgrade pip

# Piper-Package installieren (editable)
WORKDIR /piper-src/src/python_run
RUN pip install -e . --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_http.txt --no-cache-dir

# Weitere Abhängigkeiten (TTS-Ausgabe, Model-Download)
RUN apt install -y ffmpeg espeak-ng
RUN pip install wget

# App-Dateien
WORKDIR /app
COPY run.py /app
COPY download /app/download

EXPOSE 5000

# ── Environment variables (defaults) ──────────────────────────────────
ENV MODEL_DOWNLOAD_LINK="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx?download=true"
ENV MODEL_DEFAULT="de_DE-thorsten-high"
ENV MODEL_TARGET_FOLDER="/app/models"
ENV SPEAKER="0"
ENV SENTENCE_SILENCE="0.0"
ENV LENGTH_SCALE="1.1"
ENV NOISE_SCALE="0.667"
ENV NOISE_W="0.8"
ENV CUDA="true"

CMD ["python", "-u", "/app/run.py"]
