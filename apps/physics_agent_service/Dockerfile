# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Physics Agent Service Dockerfile
# The main service is CPU-only. Rendering is delegated to the sidecar
# ovrtx-rendering-api container via RENDER_ENDPOINT.
FROM python:3.12-slim

ARG APP_VERSION=unknown
ARG GIT_COMMIT=unknown

# Install uv for faster package management
RUN pip install --no-cache-dir uv

# Install system dependencies. The gnupg/libgpgme packages pull the
# CVE-2025-68973 fix; keep them listed explicitly so the security test
# in tests/test_dockerfile_security.py sees them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    gnupg \
    gpg \
    gpg-agent \
    gpgconf \
    gpgsm \
    gpg-wks-client \
    dirmngr \
    libgpgme11t64 \
    && rm -rf /var/lib/apt/lists/*

# CVE-2026-4519: webbrowser.open() command injection — not exploitable
# (webbrowser module unused by this service). Fix pending in future 3.12.x.

# Set working directory
WORKDIR /app

# Copy only the source trees needed to build and run this service. Keep
# `.build-resources` present so an optional Scene Optimizer bundle can be
# staged there before `docker compose build` without copying unrelated local
# files from the repository root.
COPY pyproject.toml README.md VERSION.md /app/
COPY world_understanding /app/world_understanding
COPY apps/physics_agent /app/apps/physics_agent
COPY apps/physics_agent_service /app/apps/physics_agent_service
COPY .build-resources /app/.build-resources

# Local Scene Optimizer backend. Populate `.build-resources/scene_optimizer_core`
# on the host by running `scripts/fetch_build_resources.sh` before
# `docker compose build`. When absent, the `optimize_usd` step falls back to
# NVCF at runtime.
RUN SO=/app/.build-resources/scene_optimizer_core; \
    if [ -d "$SO/python" ] && [ -d "$SO/lib" ] && \
       [ -d "$SO/extraLibs" ] && [ -d "$SO/usdpy" ]; then \
      echo "Scene Optimizer Core present at $SO"; \
    else \
      echo "Scene Optimizer Core not provided — optimize_usd will fall back to NVCF"; \
    fi
ENV WU_SO_PACKAGE_DIR=/app/.build-resources/scene_optimizer_core

# Install world_understanding package and dependencies.
# --no-config + --no-sources keep the install self-contained inside the repo.
RUN cd /app && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv pip install --system -e ".[telemetry]" --no-config --no-sources

# Install physics_agent package if it exists
RUN if [ -d "/app/apps/physics_agent" ]; then \
    cd /app/apps/physics_agent && UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv pip install --system -e . --no-config --no-sources; \
    fi

# Install physics_agent_service package
RUN cd /app/apps/physics_agent_service && UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv pip install --system -e . --no-config --no-sources

# Create directories for sessions
RUN mkdir -p /var/physics-agent/sessions

# Create non-root user for security
RUN useradd -m -u 10000 physics-agent && \
    chown -R physics-agent:physics-agent /var/physics-agent

# Keep /app read-only for the service user without Python trying to create
# __pycache__ beside editable source files at runtime.
ENV PYTHONDONTWRITEBYTECODE=1

# Add Python paths for physics_agent imports
ENV PYTHONPATH="/app:/app/apps:/app/apps/physics_agent_service:${PYTHONPATH}"

# Switch to non-root user
USER physics-agent

# Set working directory to service
WORKDIR /app/apps/physics_agent_service

# Environment variables with defaults
ENV PA_SESSION_STORAGE_PATH=/var/physics-agent/sessions \
    PA_SESSION_TTL_HOURS=24 \
    PA_MAX_ACTIVE_SESSIONS=1 \
    PA_MAX_UPLOAD_SIZE_MB=500 \
    PA_DEBUG=false \
    PA_RENDER_BACKEND=remote \
    PA_MAX_WORKERS=16

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
