# NyayMitra API image (CPU). Build: docker build -t nyaymitra-api .
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    APP_ENV=production \
    # Only trust X-Forwarded-For from this proxy address (rate limits are per client IP).
    # Set to your reverse proxy's IP when behind one.
    FORWARDED_ALLOW_IPS=127.0.0.1

WORKDIR /app

# CPU-only torch first (the default Linux wheel pulls multi-GB CUDA libraries): the
# explicit +cpu version only exists on the PyTorch index; its dependencies come from
# PyPI (--extra-index-url). Then the pinned lock file (torch==X is satisfied by X+cpu).
COPY requirements.lock .
RUN TORCH_VERSION=$(grep -E '^torch==' requirements.lock | cut -d= -f3) \
 && pip install "torch==${TORCH_VERSION}+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.lock

# Bake the embedding model into the image (no download at startup; works offline)
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
ENV HF_HUB_OFFLINE=1

COPY alembic.ini ./
COPY backend ./backend
COPY data/legal_corpus ./data/legal_corpus
COPY data/referral ./data/referral
COPY data/templates ./data/templates
COPY docker/entrypoint.sh ./entrypoint.sh

RUN useradd --uid 10001 --create-home app \
 && mkdir -p /app/data/chroma_db \
 && chmod +x /app/entrypoint.sh \
 && chown -R app:app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
# Single worker: rate limiter and approval checkpointer are per-process (see CLAUDE.md).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
