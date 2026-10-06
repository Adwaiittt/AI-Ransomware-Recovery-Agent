# syntax=docker/dockerfile:1
#
# One image for both services (api + watcher); the command differs.
# Build stage installs deps, trains the detector, and caches the embedding
# model, so `docker compose up` works offline with no extra steps.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ---------------------------------------------------------------------------
FROM base AS builder
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf-cache

# CPU-only torch first: the default Linux wheel bundles CUDA (~2.5 GB extra).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install only the dependency list from pyproject.toml (not the package itself),
# so this layer is cached until dependencies change, not on every code edit.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt

# Pre-download the embedding model into the image (runtime runs offline).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
COPY ml ./ml
COPY simulator ./simulator
COPY scripts ./scripts

# Deterministic training (seeded) -> ml/artifacts/model.joblib + metrics.json
RUN python -m ml.generate_dataset && python -m ml.train && rm -rf ml/data

# ---------------------------------------------------------------------------
FROM base AS runtime
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache
COPY --from=builder --chown=app:app /app /app

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    LOG_FORMAT=json

# Writable runtime dirs (mounted as volumes in compose), owned by the app user.
RUN mkdir -p /app/data /app/sandbox/watched /app/restore_out \
 && chown -R app:app /app/data /app/sandbox /app/restore_out

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# --no-access-log: our RequestIdMiddleware already logs one JSON line per request.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
