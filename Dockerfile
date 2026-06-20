# syntax=docker/dockerfile:1
#
# NOTE: the RAG dependencies (torch via sentence-transformers) make this image
# large. The Flask API boots and serves /api/health without them; /api/chat
# reports 503 until they are installed and the keys/index are configured.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup --system --gid 1000 app && \
    adduser --system --uid 1000 --gid 1000 app

WORKDIR /app

# setup.py + src are needed for the editable `-e .` install.
COPY requirements.txt setup.py ./
COPY src ./src
RUN pip install -r requirements.txt

COPY --chown=app:app . .

USER app

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "wsgi:app"]
