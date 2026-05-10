# ============================================================================
# Argus MCP Server — production image
# ============================================================================
# NOTE: do not add a `# syntax=docker/dockerfile:X.Y` pragma here — Render's
# build environment has a known issue with the dockerfile frontend version
# resolution that causes "frontend grpc server closed unexpectedly". The
# default builder works fine for everything we use.

FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY argus ./argus

RUN pip install --upgrade pip && \
    pip wheel --wheel-dir /wheels .

# ----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARGUS_HOST=0.0.0.0 \
    ARGUS_PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1001 argus

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY --chown=argus:argus argus ./argus

USER argus

# Build reference KB at image build time (small enough to bake in)
RUN python -m argus.reference.build_kb

EXPOSE 8080

# Liveness check — confirms the TCP port is listening. Fly's [[services]]
# tcp_checks is the authoritative healthcheck in production; this is just
# for local `docker run` smoke tests.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('127.0.0.1',8080)); s.close()" || exit 1

CMD ["python", "-m", "argus.server"]
