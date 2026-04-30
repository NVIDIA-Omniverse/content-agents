# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# OVRTX Rendering API Service Dockerfile
#
# Build context must be the repository root:
#   docker build -f apps/ovrtx_rendering_api/Dockerfile -t ovrtx-rendering-api .

FROM python:3.12-slim

ARG APP_VERSION=0.1.0

# Install system dependencies (curl for health check, git for pip installs,
# libgl1 + libvulkan1 for OVRTX GPU rendering, xvfb for headless display).
# --no-install-recommends keeps the image lean and avoids pulling in
# unexpected packages (SonarQube docker:S6500).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libvulkan1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy entire repository (context is repo root)
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

# Install world_understanding package (provides render_ovrtx module).
# --no-config + --no-sources give us a clean, reproducible install path:
# ignore any host-level uv config and tool.uv.sources indirection.internal network access.
RUN cd /app && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv pip install --system -e ".[telemetry]" --no-config --no-sources

# Install ovrtx_rendering_api service
RUN cd /app/apps/ovrtx_rendering_api && uv pip install --system -e . --no-config --no-sources

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

# Pre-provision the isolated OVRTX venv so the daemon starts immediately.
# ovrtx bundles its own USD C libs that conflict with usd-core, so it
# must live in a separate venv.
ENV WU_OVRTX_VENV_DIR=/app/.ovrtx_venv
RUN uv venv /app/.ovrtx_venv --python 3.12 && \
    uv pip install --python /app/.ovrtx_venv/bin/python --no-config --no-sources \
        "ovrtx==0.2.0.280040" numpy pillow && \
    ln -sf /app/.ovrtx_venv/lib/python3.12/site-packages/ovrtx/bin/library \
           /app/.ovrtx_venv/lib/python3.12/site-packages/library

# NOTE: Running as root for Xvfb access. The official ovrtx-render-worker
# also runs as root for display server access. We deliberately do NOT
# create a non-root user here — there used to be a `useradd renderer` +
# chown that was never followed by `USER renderer`, so it was dead code
# that confused reviewers. TODO: revisit with rootless Xvfb or host
# display passthrough so we can drop privileges properly.

WORKDIR /app/apps/ovrtx_rendering_api

LABEL org.opencontainers.image.version="${APP_VERSION}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=12 \
  CMD curl -fsS http://localhost:8000/health | grep -q '"gpu_initialized":true' || exit 1

# Start Xvfb for headless Vulkan rendering, then launch uvicorn. Docker
# restarts keep the container filesystem, so a previous Xvfb can leave stale
# :99 lock/socket files behind and prevent the replacement display from
# starting.
CMD ["sh", "-c", "rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 && Xvfb :99 -screen 0 1024x768x24 +extension GLX &>/dev/null & sleep 1 && DISPLAY=:99 exec python -m uvicorn service.main:app --host 0.0.0.0 --port 8000"]
