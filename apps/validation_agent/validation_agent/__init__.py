# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation Agent package.

The stable contracts and runtime implementation live in
``world_understanding.validation``. This package provides the standalone CLI
and package identity used by the agent collection.
"""

from .utils import get_version

__version__ = get_version()
__all__ = ["__version__", "get_version"]
