FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Get the latest version of the code
RUN apt update && apt install -y git
RUN git clone https://github.com/rhasspy/piper

# Update pip and install the required packages
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app/piper/src/python_run

# Install the package
RUN pip install -e .

# Install the requirements
RUN pip install -r requirements.txt

# Install http server
RUN pip install -r requirements_http.txt

# Install wget pip package
RUN pip install wget

# Copy the run.py file into the container
COPY run.py /app
# Copy the download folder into the container
COPY download /app/download

# Expose the port 5000
EXPOSE 5000

# ── Environment variables (defaults) ──────────────────────────────────
ENV MODEL_DOWNLOAD_LINK="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/pavoque/low/de_DE-pavoque-low.onnx?download=true"
ENV MODEL_TARGET_FOLDER="/app/models"
ENV SPEAKER="0"
ENV SENTENCE_SILENCE="0.0"
ENV CUDA="true"

# Start the custom HTTP server
CMD ["python", "-u", "/app/run.py"]
