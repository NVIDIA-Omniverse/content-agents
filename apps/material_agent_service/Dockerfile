# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Material Agent Service Dockerfile
FROM python:3.13.11-slim

ARG APP_VERSION=unknown
ARG GIT_COMMIT=unknown

# CVE-2025-13836: http.client memory DoS — fixed in 3.13.11+
RUN python3 -c "import sys; v=sys.version_info; assert v >= (3,13,11), \
    f'CVE-2025-13836: Python {sys.version} is vulnerable, need >=3.13.11'"
# CVE-2026-4519: webbrowser.open() command injection — not exploitable
# (webbrowser module unused by this service). Fix pending in future 3.13.x release.

# Install system dependencies. --no-install-recommends keeps the image lean
# and avoids pulling in unexpected packages (docker:S6500).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy entire repository (context is repo root from docker-compose)
COPY . /app

# Local Scene Optimizer backend. Populate `.build-resources/scene_optimizer_core`
# on the host by running `scripts/fetch_build_resources.sh` before
# `docker compose build`; the directory is COPY'd in above with the rest of
# the repo. When absent, the `optimize_usd` step falls back to NVCF at runtime.
RUN SO=/app/.build-resources/scene_optimizer_core; \
    if [ -d "$SO/python" ] && [ -d "$SO/lib" ] && \
       [ -d "$SO/extraLibs" ] && [ -d "$SO/usdpy" ]; then \
      echo "Scene Optimizer Core present at $SO"; \
    else \
      echo "Scene Optimizer Core not provided — optimize_usd will fall back to NVCF"; \
    fi
ENV WU_SO_PACKAGE_DIR=/app/.build-resources/scene_optimizer_core

# Install world_understanding package and dependencies.
# --no-config + --no-sources give us a clean, reproducible install path:
# ignore any host-level uv config and tool.uv.sources indirection.
# SETUPTOOLS_SCM_PRETEND_VERSION bypasses git versioning.
RUN cd /app && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv pip install --system -e ".[telemetry]" --no-config --no-sources

# Install material_agent package if it exists
RUN if [ -d "/app/apps/material_agent" ]; then \
      cd /app/apps/material_agent && UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
      uv pip install --system -e . --no-config --no-sources; \
    fi

# Install material_agent_service package (versions from pyproject.toml)
RUN cd /app/apps/material_agent_service && UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 uv pip install --system -e . --no-config --no-sources

# Copy React frontend build output from Stage 1
# Create directories for materials and sessions
RUN mkdir -p /var/material-agent/sessions /app/materials

# Create non-root user for security
RUN useradd -m -u 1000 material-agent && \
    chown -R material-agent:material-agent /var/material-agent /app

# Add Python paths for material_agent imports
ENV PYTHONPATH="/app:/app/apps:/app/apps/material_agent_service:${PYTHONPATH}"

# Switch to non-root user
USER material-agent

# Set working directory to service
WORKDIR /app/apps/material_agent_service

# Environment variables with defaults
ENV MA_SESSION_STORAGE_PATH=/var/material-agent/sessions \
    MA_SESSION_TTL_HOURS=24 \
    MA_MAX_ACTIVE_SESSIONS=8 \
    MA_MAX_UPLOAD_SIZE_MB=500 \
    MA_GEOM_LIMIT=0 \
    MA_DEBUG=false \
    MA_PROGRESS_POLL_INTERVAL_SECONDS=2 \
    MA_IMAGE_WIDTH=512 \
    MA_IMAGE_HEIGHT=512 \
    MA_RENDERER_BACKEND=remote \
    MA_MAX_WORKERS=16

LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"

ENV APP_VERSION=${APP_VERSION}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run service with proper module path
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]