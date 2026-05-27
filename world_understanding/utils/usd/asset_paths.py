# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for USD asset paths that may be resolved outside Python.

USD layers, references, and material attributes can be evaluated by native
ArResolver code in renderers. These helpers keep that boundary conservative:
generated output should author local relative paths, not resolver schemes or
host absolute paths that bypass Python-side URL and filesystem checks.
"""

from __future__ import annotations

import os
from pathlib import Path


def is_windows_drive_path(path: str) -> bool:
    """Return true for Windows drive-absolute paths such as ``C:/foo``."""
    return (
        len(path) >= 3
        and path[0].isalpha()
        and path[1] == ":"
        and path[2] in {"/", "\\"}
    )


def usd_asset_uri_scheme(path: str) -> str:
    """Return a resolver/URI scheme for a USD asset path, if one is present."""
    if not path or is_windows_drive_path(path):
        return ""

    colon_index = path.find(":")
    if colon_index <= 0:
        return ""

    first_separator = len(path)
    for separator in ("/", "\\"):
        separator_index = path.find(separator)
        if separator_index >= 0:
            first_separator = min(first_separator, separator_index)
    if first_separator < colon_index:
        return ""

    scheme = path[:colon_index]
    if not scheme[0].isalpha():
        return ""
    if not all(char.isalnum() or char in {"+", ".", "-"} for char in scheme):
        return ""
    return scheme.lower()


def is_uri_asset_path(path: str) -> bool:
    """Return true when a USD asset path uses a resolver/URI scheme."""
    return bool(usd_asset_uri_scheme(path))


def is_absolute_asset_path(path: str) -> bool:
    """Return whether an asset path is absolute on POSIX or Windows."""
    return path.startswith("/") or os.path.isabs(path) or is_windows_drive_path(path)


def is_unsafe_resolver_asset_path(path: str) -> bool:
    """Return true for paths that should not be authored into generated USD."""
    return is_uri_asset_path(path) or is_absolute_asset_path(path)


def is_relative_to(path: Path, base: Path) -> bool:
    """Compatibility wrapper for ``Path.is_relative_to``."""
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def resolve_relative_asset_path_under_base(path: str, base_dir: Path) -> Path:
    """Resolve a local relative USD asset path and require it to stay in base_dir."""
    if not path:
        raise ValueError("empty asset path")
    if is_uri_asset_path(path):
        raise ValueError(f"resolver URI schemes are not allowed: {path}")
    if is_absolute_asset_path(path):
        raise ValueError(f"absolute asset paths are not allowed: {path}")

    resolved_base = base_dir.resolve()
    resolved_path = (resolved_base / path).resolve()
    if not is_relative_to(resolved_path, resolved_base):
        raise ValueError(f"asset path escapes its source directory: {path}")
    return resolved_path
