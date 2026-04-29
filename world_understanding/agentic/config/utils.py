# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration utility functions.

This module provides common configuration utilities used across all agents
for path resolution, API key management, and more.
"""

import os
from pathlib import Path


def resolve_path_from_config(
    path_key: str,
    config: dict,
    config_path: Path,
    override_key: str | None = None,
    context: dict | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a path from configuration with override support.

    This handles the common pattern of:
    1. Check for override in context
    2. Use config value if available
    3. Make relative paths absolute relative to config file
    4. Optionally validate existence

    Args:
        path_key: Key name in config dict (e.g., "usd_path", "output_dir")
        config: Configuration dictionary
        config_path: Path to the configuration file (for relative path resolution)
        override_key: Optional key name in context for override value
        context: Optional context dictionary containing overrides
        must_exist: If True, raise FileNotFoundError if path doesn't exist

    Returns:
        Resolved absolute Path object

    Raises:
        ValueError: If neither override nor config provides a value
        FileNotFoundError: If must_exist=True and path doesn't exist

    Example:
        ```python
        # In a config task
        usd_path = resolve_path_from_config(
            path_key="usd_path",
            config=config,
            config_path=config_path,
            override_key="source_override",
            context=context,
            must_exist=True
        )
        ```
    """
    context = context or {}
    override_key = override_key or f"{path_key}_override"

    # Try override first
    path_value = context.get(override_key)

    # Fall back to config
    if path_value is None:
        path_value = config.get(path_key)

    # Validate we got a value
    if path_value is None:
        raise ValueError(
            f"'{path_key}' not specified. Provide it via config or context['{override_key}']"
        )

    # Convert to Path and resolve
    path_obj = Path(path_value)

    # Make relative paths absolute relative to config file directory
    if not path_obj.is_absolute():
        if config_path.is_file():
            path_obj = config_path.parent / path_obj
        else:
            path_obj = config_path / path_obj

    # Resolve to absolute path
    path_obj = path_obj.resolve()

    # Validate existence if required
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"{path_key} not found: {path_obj}")

    return path_obj


def get_api_key_for_backend(backend: str, model_type: str = "model") -> str:
    """Get API key for a backend with validation.

    Retrieves API keys from environment variables with backend-specific
    error messages.

    Args:
        backend: Backend name (e.g., "nim", "perflab_azure_openai")
        model_type: Type of model for error message (e.g., "VLM", "LLM")

    Returns:
        API key string

    Raises:
        ValueError: If required API key is not set

    Example:
        ```python
        api_key = get_api_key_for_backend("nim", "VLM")
        # Retrieves NVIDIA_API_KEY or raises error
        ```
    """
    # Map backends to environment variable names
    # For Azure backends, we check multiple possible env vars (NSTORAGE_API_KEY, AZURE_OPENAI_API_KEY)
    env_var_map = {
        "nim": ["NVIDIA_API_KEY"],
        "perflab_azure_openai": ["NSTORAGE_API_KEY", "AZURE_OPENAI_API_KEY"],
        "azure_openai": ["AZURE_OPENAI_API_KEY", "NSTORAGE_API_KEY"],
        "nvidia_inference": ["INFERENCE_NVIDIA_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
    }

    env_vars = env_var_map.get(backend)

    if env_vars is None:
        # Backend doesn't require API key or uses default mechanism
        return ""

    # Try each environment variable in order
    api_key: str | None = None
    for env_var in env_vars:
        api_key = os.getenv(env_var)
        if api_key:
            break

    if not api_key:
        # Create helpful error message with all attempted env vars
        env_vars_str = " or ".join(env_vars)
        raise ValueError(f"{env_vars_str} not set for {backend} {model_type}")

    assert api_key is not None
    return api_key


def ensure_tuple(value: list | tuple | None, default: list | tuple) -> tuple:
    """Ensure value is a tuple, converting from list if needed.

    Args:
        value: Value to convert (list, tuple, or None)
        default: Default value to use if value is None

    Returns:
        Tuple version of the value

    Example:
        ```python
        color = ensure_tuple([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        # Returns (1.0, 0.0, 0.0)

        color = ensure_tuple(None, [0.5, 0.5, 0.5])
        # Returns (0.5, 0.5, 0.5)
        ```
    """
    if value is None:
        value = default
    return tuple(value) if isinstance(value, list) else value


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Value to return if denominator is zero

    Returns:
        Result of division or default value

    Example:
        ```python
        ratio = safe_divide(10, 5)     # Returns 2.0
        ratio = safe_divide(10, 0)     # Returns 0.0
        ratio = safe_divide(10, 0, 1)  # Returns 1.0
        ```
    """
    return numerator / denominator if denominator > 0 else default
