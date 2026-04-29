# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration system for Physics Agent."""

from physics_agent.config.path_resolver import ProjectPathResolver
from physics_agent.config.schema import (
    STEP_ORDER,
    STEP_OUTPUT_DIRS,
    get_default_config,
    get_step_defaults,
)
from physics_agent.config.unified_config import UnifiedPipelineConfigTask
from physics_agent.config.validator import ConfigValidator

__all__ = [
    "ConfigValidator",
    "ProjectPathResolver",
    "STEP_ORDER",
    "STEP_OUTPUT_DIRS",
    "UnifiedPipelineConfigTask",
    "get_default_config",
    "get_step_defaults",
]
