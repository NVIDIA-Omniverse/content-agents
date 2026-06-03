# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Texture Agent Service Dockerfile
# No GPU/CUDA needed — texture generation uses API calls (Gemini/NVIDIA Inference)
FROM python:3.12-bookworm

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management. Keep pip above the image-scan floor.
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.1" uv

# CVE-2026-4519: webbrowser.open() command injection - not exploitable
# (webbrowser module unused by this service). Fix pending in future 3.12.x.

# Set working directory
WORKDIR /app

# Copy only the source trees needed to build and run this service rather than
# recursively copying the full repository build context.
COPY pyproject.toml README.md VERSION.md /app/
COPY world_understanding /app/world_understanding
COPY apps/texture_agent /app/apps/texture_agent
COPY apps/texture_agent_service /app/apps/texture_agent_service

# Initialize a git repo so uv-dynamic-versioning can resolve the version
# (.git is excluded by .dockerignore)
RUN cd /app && git init && git config user.email "build@docker" && git config user.name "build" && git add -A && git commit -m "docker build"

# Install world_understanding package and dependencies.
# --no-config + --no-sources keep the install self-contained inside the repo
# instead of resolving repo-local packages from an external index.
RUN cd /app && SETUPTOOLS_SCM_PRETEND_VERSION=0.2.0 UV_DYNAMIC_VERSIONING_BYPASS=0.2.0 \
    uv pip install --system -e "." --no-config --no-sources

# Install texture_agent package if it exists
RUN if [ -d "/app/apps/texture_agent" ]; then \
    cd /app/apps/texture_agent && UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv pip install --system -e . --no-config --no-sources; \
    fi

# Install texture_agent_service package
RUN cd /app/apps/texture_agent_service && UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv pip install --system -e . --no-config --no-sources

# Create directories for sessions
RUN mkdir -p /var/texture-agent/sessions

# Create non-root user for security
RUN useradd -m -u 10000 texture-agent && \
    chown -R texture-agent:texture-agent /var/texture-agent

# Keep /app read-only for the service user without Python trying to create
# __pycache__ beside editable source files at runtime.
ENV PYTHONDONTWRITEBYTECODE=1

# Add Python paths for texture_agent imports
ENV PYTHONPATH="/app:/app/apps:/app/apps/texture_agent_service:${PYTHONPATH}"

# Switch to non-root user
USER texture-agent

# Set working directory to service
WORKDIR /app/apps/texture_agent_service

# Environment variables with defaults
ENV TA_SESSION_STORAGE_PATH=/var/texture-agent/sessions \
    TA_SESSION_TTL_HOURS=24 \
    TA_MAX_ACTIVE_SESSIONS=4 \
    TA_MAX_UPLOAD_SIZE_MB=500 \
    TA_TEXTURE_BACKEND=simple_image_gen

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run service with proper module path
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8001"]
