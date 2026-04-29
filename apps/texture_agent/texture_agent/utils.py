# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for the texture agent."""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Get the package version."""
    try:
        return version("texture-agent")
    except PackageNotFoundError:
        return "0.0.0-dev"
