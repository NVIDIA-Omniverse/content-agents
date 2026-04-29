#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch build resources that are not on public PyPI.
#
# Downloads the public `scene_optimizer_core_usd_25.11_py_3.12` zip from
# NVIDIA's CloudFront and unpacks it into `.build-resources/scene_optimizer_core/`.
# The Dockerfiles mount that directory into the image and point
# `WU_SO_PACKAGE_DIR` at it so the `optimize_usd` step has a working local
# backend. If the directory is absent, the Dockerfiles skip the setup and
# `optimize_usd` falls back to the NVCF cloud backend (or fails if neither is
# configured).
#
# Idempotent: exits early if the unpacked package is already present.
#
# Override the default package version/platform by setting `SO_CORE_URL`:
#   SO_CORE_URL=https://.../scene_optimizer_core_*.zip ./scripts/fetch_build_resources.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_RESOURCES="$REPO_ROOT/.build-resources"
PKG_DIR="$BUILD_RESOURCES/scene_optimizer_core"

# Default: Linux x86_64, Scene Optimizer Core 110.1.0 (Kit 110.1 parity).
# Jira: OMPE-67494. For aarch64 or Windows builds, override SO_CORE_URL.
DEFAULT_URL="https://d4i3qtqj3r0z5.cloudfront.net/scene_optimizer_core_usd_25.11_py_3.12%40110.1.0%2Bmr7.102.662bf9b8.gl.manylinux_2_35_x86_64.release.zip"
URL="${SO_CORE_URL:-$DEFAULT_URL}"

mkdir -p "$BUILD_RESOURCES"

# Skip if already unpacked. The `usdpy` dir is a good sentinel — it only
# exists in the public package layout (vs. legacy internal bundle layouts).
if [[ -d "$PKG_DIR/python" && -d "$PKG_DIR/lib" && -d "$PKG_DIR/extraLibs" && -d "$PKG_DIR/usdpy" ]]; then
    echo "scene_optimizer_core already unpacked at $PKG_DIR — skipping fetch"
    exit 0
fi

# Clean up any partial/stale unpack
rm -rf "$PKG_DIR"

ZIP_PATH="$BUILD_RESOURCES/scene_optimizer_core.zip"
echo "Fetching Scene Optimizer Core from:"
echo "  $URL"
curl --fail --location --silent --show-error --output "$ZIP_PATH" "$URL"

echo "Unpacking into $PKG_DIR ..."
mkdir -p "$PKG_DIR"
unzip -q "$ZIP_PATH" -d "$PKG_DIR"
rm -f "$ZIP_PATH"

# Sanity check: make sure the expected layout is in place.
for sub in python lib extraLibs usdpy; do
    if [[ ! -d "$PKG_DIR/$sub" ]]; then
        echo "ERROR: unpacked package missing expected subdir: $PKG_DIR/$sub" >&2
        echo "       (check that SO_CORE_URL points at a Scene Optimizer Core zip)" >&2
        exit 1
    fi
done

du -sh "$PKG_DIR"
echo "Done. You can now run: docker compose build"
