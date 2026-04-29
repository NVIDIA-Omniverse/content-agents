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

# Install uv for faster package management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy entire repository (context is repo root from docker-compose)
COPY . /app

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
    chown -R texture-agent:texture-agent /var/texture-agent /app

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
