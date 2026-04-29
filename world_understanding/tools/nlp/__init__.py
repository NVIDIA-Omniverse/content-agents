# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Natural language processing tools."""

from . import chat_tool

# Import the tools to trigger their registration
from .chat_tool import chat_tool as _chat_tool

__all__ = ["chat_tool"]
