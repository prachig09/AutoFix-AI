# syntax=docker/dockerfile:1
# =============================================================================
# AutoFix AI - Cloud Run image (CPU-only, single worker, low RAM)
#
# Build context must contain:  requirements.txt, app/main.py, model.joblib
# (model.joblib is produced by `python train_model.py`).
# =============================================================================
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE : no .pyc files -> smaller image
# PYTHONUNBUFFERED        : logs reach Cloud Logging immediately
# PIP_NO_CACHE_DIR        : never keep pip's download cache in a layer
# *_NUM_THREADS=1         : stop numpy/BLAS/OpenMP spawning threads that
#                           inflate RAM and fight for the single vCPU
# PORT                    : default for local `docker run`; Cloud Run overrides it
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8080

# libgomp1 = OpenMP runtime that XGBoost's wheel links against. It is NOT
# included in python:3.11-slim; without it `import xgboost` fails at startup.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the trained model binary.
COPY --chown=app:app app/ ./app/
COPY --chown=app:app model.joblib ./model.joblib

USER app

EXPOSE 8080

# Shell form via `sh -c` so ${PORT} expands; `exec` makes uvicorn PID 1 so it
# receives SIGTERM from Cloud Run and shuts down gracefully (lets the
# background BigQuery insert finish).
# --workers 1      : one process = one copy of the model in RAM
# --no-access-log  : access logs contain client IPs (DPDP); app logs are structured
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --no-access-log"]
