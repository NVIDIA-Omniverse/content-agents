# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration models and utilities for World Understanding agents."""

from .base_path_resolver import BasePathResolver
from .loader import ConfigError, ConfigLoader, load_config
from .usd_dataset import (
    PrimFilters,
    RendererConfig,
    RenderingModeConfig,
    USDDatasetConfig,
)
from .utils import (
    API_KEY_ENV_VAR_MAP,
    LOCAL_NIM_API_KEY_PLACEHOLDER,
    ensure_tuple,
    get_api_key_for_backend,
    get_api_key_for_model_config,
    get_openai_api_key_for_base_url,
    is_local_base_url,
    is_local_nim_api_key_placeholder,
    is_placeholder_api_key,
    resolve_path_from_config,
    safe_divide,
)

__all__ = [
    # Models
    "USDDatasetConfig",
    "RendererConfig",
    "RenderingModeConfig",
    "PrimFilters",
    # Path Resolvers
    "BasePathResolver",
    # Utilities
    "API_KEY_ENV_VAR_MAP",
    "LOCAL_NIM_API_KEY_PLACEHOLDER",
    "resolve_path_from_config",
    "get_api_key_for_backend",
    "get_api_key_for_model_config",
    "get_openai_api_key_for_base_url",
    "is_local_nim_api_key_placeholder",
    "is_placeholder_api_key",
    "is_local_base_url",
    "ensure_tuple",
    "safe_divide",
    # Loaders
    "ConfigLoader",
    "ConfigError",
    "load_config",
]
