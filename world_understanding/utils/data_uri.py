# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility for deciding whether to use data URI encoding instead of S3 upload."""

import logging
import os

logger = logging.getLogger(__name__)


def should_use_data_uri(use_data_uri: bool | None = None) -> bool:
    """Determine whether to use data URI encoding instead of S3 upload.

    Decision hierarchy:
        1. Explicit parameter (if not ``None``).
        2. ``MA_RENDERING_USE_DATA_URI`` environment variable
           (``"true"`` → data URI, ``"false"`` → S3 upload).
        3. Default: data URI unless ``WU_S3_BUCKET`` is set in the environment.

    ``WU_S3_BUCKET`` is the explicit signal that the caller wants the S3
    upload branch. Ambient AWS credentials alone are not sufficient —
    credentials without a bucket cannot be used for upload, and AWS env
    vars are common on developer machines for unrelated projects.

    Args:
        use_data_uri: Explicit override. ``True`` / ``False`` to force behaviour,
            ``None`` (default) to fall through to env var / auto-detection.

    Returns:
        ``True`` if data URI mode should be used, ``False`` otherwise.
    """
    if use_data_uri is not None:
        return use_data_uri

    env_val = os.getenv("MA_RENDERING_USE_DATA_URI", "").lower()
    if env_val == "true":
        return True
    if env_val == "false":
        return False

    return not os.environ.get("WU_S3_BUCKET")
