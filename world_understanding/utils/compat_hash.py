# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility-only helpers for persisted non-security identifiers."""

from __future__ import annotations

import hashlib


def legacy_md5_hex(data: str | bytes, *, length: int) -> str:
    """Return the legacy MD5 prefix used by persisted non-security IDs.

    This is intentionally narrow: use it only when changing the digest would
    break existing filenames, session IDs, manifests, or historical benchmark
    continuity. New non-security identifiers should prefer BLAKE2/SHA-256.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()[:length]
