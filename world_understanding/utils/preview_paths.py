# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preview thumbnail filename helpers shared by agent services."""

from __future__ import annotations

import hashlib
from pathlib import Path

from world_understanding.utils.compat_hash import legacy_md5_hex


def normalize_render_image_path(image_path: str | Path) -> str:
    """Normalize dataset image paths to the render-relative hash input."""
    normalized_path = str(image_path).replace("\\", "/")
    prefix = "usd/renders/"
    if normalized_path.startswith(prefix):
        normalized_path = normalized_path[len(prefix) :]
    return normalized_path


def preview_filename_for_render_path(render_path: str | Path) -> str:
    """Return the preferred Sonar-friendly preview filename for new sessions."""
    normalized_path = normalize_render_image_path(render_path)
    filename = Path(normalized_path).name
    path_hash = hashlib.blake2s(normalized_path.encode(), digest_size=4).hexdigest()
    return f"{path_hash}_{filename}"


def legacy_preview_filename_for_render_path(render_path: str | Path) -> str:
    """Return the historical preview filename for persisted old sessions."""
    normalized_path = normalize_render_image_path(render_path)
    filename = Path(normalized_path).name
    path_hash = legacy_md5_hex(normalized_path, length=8)
    return f"{path_hash}_{filename}"


def find_existing_preview_filename(
    preview_dir: Path,
    render_path: str | Path,
) -> str | None:
    """Find an already-created preview filename for a render image.

    Old sessions may contain MD5-prefixed preview files created before the
    Sonar hardening. Serving the persisted filename is the compatibility
    boundary; callers should not recalculate a different name and 404.
    """
    normalized_path = normalize_render_image_path(render_path)
    preferred = preview_filename_for_render_path(normalized_path)
    preferred_path = preview_dir / preferred
    if preferred_path.is_file():
        return preferred
    if not preview_dir.exists():
        return None

    legacy = legacy_preview_filename_for_render_path(normalized_path)
    if (preview_dir / legacy).is_file():
        return legacy
    return None


def resolve_preview_filename(preview_dir: Path, render_path: str | Path) -> str:
    """Return an existing compatible preview filename or the preferred new one."""
    return find_existing_preview_filename(preview_dir, render_path) or (
        preview_filename_for_render_path(render_path)
    )
