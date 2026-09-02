# SIH26166 - one-container deployment (Hugging Face Space, Docker SDK).
# Serves the FastAPI backend AND the frontend same-origin on one port, so
# the Space URL alone is the complete shareable app.
#
# Data strategy: the ~29 GB of mission data is NOT in this image (it lives
# in a public HF dataset repo - see deploy/download_data.py) and is pulled
# once at container boot into the Space's ephemeral disk.
FROM python:3.11-slim

# OpenCV runtime libs (libGL/libglib) + onnxruntime's libgomp
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first (cached layer - rebuilds are fast)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
        huggingface_hub

# app code (repo is a few MB - ONNX model + label/browse caches included)
COPY . /app

# data root the backend already resolves relative to the repo (../data)
ENV HF_DATA_REPO="" \
    KAGGLE_DATA_SLUG="" \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# download the data snapshot at BOOT (not build) so restarts can skip it
# via the .restored marker; then run the API.  download_data.py is
# resilient: if the repo/slug is unset or unreachable it degrades to the
# small committed caches instead of killing the container.
CMD ["sh", "-c", "python deploy/download_data.py; uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]
