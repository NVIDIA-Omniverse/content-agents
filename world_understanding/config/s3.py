# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Centralized S3 configuration defaults.

Provides a single source of truth for S3 bucket, region, and profile defaults
used throughout the world_understanding library and its applications.

Override via environment variables:
    WU_S3_BUCKET  — S3 bucket name (required if using S3 features)
    WU_S3_REGION  — AWS region (default: "us-east-2")
    WU_S3_PROFILE — AWS CLI profile name (required if using S3 features)
"""

import os

WU_S3_BUCKET: str = os.environ.get("WU_S3_BUCKET") or ""
WU_S3_REGION: str = os.environ.get("WU_S3_REGION") or "us-east-2"
WU_S3_PROFILE: str = os.environ.get("WU_S3_PROFILE") or ""
