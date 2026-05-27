# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation Agent package utilities."""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed Validation Agent version."""

    try:
        return version("validation-agent")
    except PackageNotFoundError:
        return "0.0.1-dev"
