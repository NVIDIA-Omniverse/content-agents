# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared CLI utilities for World Understanding agents.

This module provides common CLI infrastructure used across all agents,
including logging setup, error handling, and display utilities.
"""

from .logging import setup_logging

__all__ = [
    "setup_logging",
]
