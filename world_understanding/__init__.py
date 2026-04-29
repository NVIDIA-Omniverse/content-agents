# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""World Understanding - Python library for Vision-Language Agents."""

from . import agentic, functions, nat, registry, tools, utils
from .utils.misc_utils import get_version

# Version info
__version__ = get_version()

__all__ = ["agentic", "functions", "nat", "registry", "tools", "utils", "__version__"]
