# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# OVRTX Rendering API Service Dockerfile
#
# Build context must be the repository root:
#   docker build -f apps/ovrtx_rendering_api/Dockerfile -t ovrtx-rendering-api .

FROM ubuntu:24.04

ARG APP_VERSION=0.1.0

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (curl for health check, git for pip installs,
# libgl1 + libvulkan1 for OVRTX GPU rendering, xvfb for headless display).
# --no-install-recommends keeps the image lean and avoids pulling in
# unexpected packages (SonarQube docker:S6500).
#
# CVE-2025-13836: Ubuntu 24.04 applies security backports to python3.12.
# Use Ubuntu's distro-patched Python 3.12 instead of the official Python image:
# OVRTX needs cp312, while container scans can flag the official image's
# /usr/local Python CPE before distro backports are visible.
# CVE-2026-4519: webbrowser.open() command injection - not exploitable
# (webbrowser module unused by this service). Ubuntu 24.04 patch pending.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libvulkan1 \
    tini \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv for faster package management. Keep pip above the image-scan floor.
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.1" uv

WORKDIR /app

# Copy only the source trees needed to build and run this service. Keep
# `.build-resources` present so an optional Scene Optimizer bundle can be
# staged there before `docker build` without copying unrelated local files.
COPY pyproject.toml README.md VERSION.md /app/
COPY world_understanding /app/world_understanding
COPY apps/ovrtx_rendering_api /app/apps/ovrtx_rendering_api
COPY .build-resources /app/.build-resources

# Local Scene Optimizer backend. Populate `.build-resources/scene_optimizer_core`
# on the host by running `scripts/fetch_build_resources.sh` before
# `docker build`. When absent, the `optimize_usd` step falls back to NVCF at
# runtime.
RUN SO=/app/.build-resources/scene_optimizer_core; \
    if [ -d "$SO/python" ] && [ -d "$SO/lib" ] && \
       [ -d "$SO/extraLibs" ] && [ -d "$SO/usdpy" ]; then \
      echo "Scene Optimizer Core present at $SO"; \
    else \
      echo "Scene Optimizer Core not provided — optimize_usd will fall back to NVCF"; \
    fi
ENV WU_SO_PACKAGE_DIR=/app/.build-resources/scene_optimizer_core

# Install world_understanding package (provides render_ovrtx module).
# --no-config + --no-sources give us a clean, reproducible install path:
# ignore any host-level uv config and tool.uv.sources indirection.internal network access.
RUN cd /app && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv pip install -e ".[telemetry]" --no-config --no-sources

# Install ovrtx_rendering_api service
RUN cd /app/apps/ovrtx_rendering_api && uv pip install -e . --no-config --no-sources

# Set Python paths
ENV PYTHONPATH="/app:/app/apps/ovrtx_rendering_api:${PYTHONPATH}"

# OVRTX configuration. 500 is the Kit-parity convergence plateau at
# ~39 dB PSNR; lower it per-request via RenderSettings.num_sensor_updates
# for cheap previews.
ENV OVRTX_LOG_LEVEL=warn \
    OVRTX_NUM_SENSOR_UPDATES=500 \
    OVRTX_RENDER_MODE=pt

# Tell NVIDIA Container Toolkit to mount Vulkan/GL driver libraries
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all

# NVIDIA Vulkan ICD (tells Vulkan loader where to find the NVIDIA driver)
COPY apps/ovrtx_rendering_api/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY apps/ovrtx_rendering_api/docker-entrypoint.sh /usr/local/bin/ovrtx-entrypoint
RUN chmod 0644 /etc/vulkan/icd.d/nvidia_icd.json
RUN chmod 0555 /usr/local/bin/ovrtx-entrypoint

# Pre-provision the isolated OVRTX venv so the daemon starts immediately.
# ovrtx bundles its own USD C libs that conflict with the app's pxr provider,
# so it must live in a separate venv. Use the package provisioning function so
# Docker builds and runtime validation share one explicit OVRTX
# version/dependency source.
ENV WU_OVRTX_VENV_DIR=/app/.ovrtx_venv
RUN WU_OVRTX_LOCK_DIR=/tmp/wu-ovrtx-locks \
    python -m world_understanding.functions.graphics.render_ovrtx --provision-only && \
    rm -rf /tmp/wu-ovrtx-locks
ENV WU_OVRTX_AUTO_PROVISION=0
RUN python -m world_understanding.functions.graphics.render_ovrtx --provision-only

# Runtime user. Keep application code and the prebuilt OVRTX venv root-owned.
# OVRTX resolves its NVIDIA shader cache through a package-relative path, so
# build that path as a symlink once and send all cache writes to renderer HOME.
RUN groupadd --gid 10001 renderer && \
    useradd --uid 10001 --gid renderer --create-home \
        --home-dir /home/renderer --shell /usr/sbin/nologin renderer && \
    install -d -o renderer -g renderer /home/renderer/.cache/ovrtx-nv-shadercache && \
    install -d -o renderer -g renderer -m 700 /tmp/.X11-unix && \
    OVRTX_BIN=/app/.ovrtx_venv/lib/python3.12/site-packages/ovrtx/bin && \
    mkdir -p "$OVRTX_BIN/cache" && \
    rm -rf "$OVRTX_BIN/cache/nv_shadercache" && \
    ln -s /home/renderer/.cache/ovrtx-nv-shadercache "$OVRTX_BIN/cache/nv_shadercache" && \
    chgrp -R renderer /app && \
    chmod -R g+rX /app

ENV HOME=/home/renderer \
    XDG_CACHE_HOME=/home/renderer/.cache \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/apps/ovrtx_rendering_api

USER renderer

LABEL org.opencontainers.image.version="${APP_VERSION}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=12 \
  CMD curl -fsS http://localhost:8000/health | grep -q '"gpu_initialized":true' || exit 1

# Start Xvfb for headless Vulkan rendering, then launch uvicorn.
ENTRYPOINT ["tini", "-g", "--"]
CMD ["ovrtx-entrypoint", "python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
