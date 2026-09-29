# Owner: Pranav Sukale
# RAGMedX-MA backend image (CPU / mock-LLM default).
#
# Build context is the backend/ directory:
#   docker build -t ragmedx-backend ./backend
#
# For GPU serving, base this on an nvidia/cuda image and install
# requirements.txt (CUDA torch wheels) instead of requirements-docker-cpu.txt.

FROM python:3.12-slim

# System deps: build tools for a few wheels; curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace model cache lives on a mounted volume so weights survive
    # container restarts instead of re-downloading on every boot.
    HF_HOME=/app/data/cache/huggingface

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements-docker-cpu.txt ./requirements-docker-cpu.txt
RUN pip install --upgrade pip \
    && pip install -r requirements-docker-cpu.txt

# Copy the application source.
COPY . .

# Data directory (SQLite db, FAISS index, uploads, HF cache) is a volume mount.
RUN mkdir -p /app/data

EXPOSE 8000

# Default runtime configuration; override via compose env_file / environment.
ENV RAGMEDX_LLM_MODE=mock \
    RAGMEDX_FAISS_INDEX_DIR=data/faiss_index

# Liveness probe hits the public health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
