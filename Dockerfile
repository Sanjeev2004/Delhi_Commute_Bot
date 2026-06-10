# ────────────────────────────────────────────────────────────────────────────
# DelhiCommuteBot — Dockerfile
# ────────────────────────────────────────────────────────────────────────────
# Multi-stage-ish Python 3.11 slim image.  Installs system deps needed
# by faiss-cpu, copies source, and runs uvicorn.
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (gcc/g++ for faiss-cpu C extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (layer caching) ────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ──────────────────────────────────────────────────
COPY src/ src/
COPY data/ data/

# ── Default command ───────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
