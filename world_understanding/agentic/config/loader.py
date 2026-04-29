# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic configuration loader with validation.

This module provides a unified configuration loading mechanism that works
with Pydantic models and supports YAML loading, override application, and
path resolution.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


class ConfigLoader[T: BaseModel]:
    """Generic configuration loader with validation.

    This class provides a unified way to load, validate, and process
    configuration files using Pydantic models.

    Example:
        ```python
        from world_understanding.agentic.config import USDDatasetConfig, ConfigLoader

        # Create loader for USD dataset configs
        loader = ConfigLoader(USDDatasetConfig)

        # Load config with optional overrides
        config = loader.load(
            config_path=Path("config.yaml"),
            overrides={"output_dir": "/tmp/output"},
            context={"source_override": "/path/to/usd"}
        )
        ```
    """

    def __init__(self, model_class: type[T]):
        """Initialize the config loader.

        Args:
            model_class: Pydantic model class to use for validation
        """
        self.model_class = model_class

    def load(
        self,
        config_path: Path,
        overrides: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> T:
        """Load and validate configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
            overrides: Optional dictionary of CLI overrides to apply
            context: Optional context dictionary for additional resolution

        Returns:
            Validated configuration model instance

        Raises:
            ConfigError: If loading or validation fails
            FileNotFoundError: If config file doesn't exist

        Example:
            ```python
            loader = ConfigLoader(USDDatasetConfig)
            config = loader.load(
                Path("my_config.yaml"),
                overrides={"output_dir": "/tmp/output"}
            )
            ```
        """
        # Validate config file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML config: {e}") from e

        if data is None:
            data = {}

        # Apply overrides
        if overrides:
            data = self._apply_overrides(data, overrides)

        # Resolve paths relative to config file (if model supports it)
        if hasattr(self.model_class, "_resolve_paths"):
            data = self._resolve_paths(data, config_path.parent)

        # Validate and create model
        try:
            return self.model_class(**data)
        except ValidationError as e:
            raise ConfigError(f"Invalid configuration: {e}") from e

    def _apply_overrides(
        self, data: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply CLI overrides to configuration data.

        Supports nested overrides using dot notation (e.g., "renderer.backend").

        Args:
            data: Original configuration data
            overrides: Override dictionary

        Returns:
            Configuration data with overrides applied
        """
        result = data.copy()

        for key, value in overrides.items():
            # Support nested keys with dot notation
            if "." in key:
                parts = key.split(".")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                result[key] = value

        return result

    def _resolve_paths(self, data: dict[str, Any], config_dir: Path) -> dict[str, Any]:
        """Resolve relative paths to absolute paths.

        Args:
            data: Configuration data
            config_dir: Directory containing config file

        Returns:
            Configuration data with resolved paths
        """
        result = data.copy()

        # Common path fields to resolve
        path_fields = ["usd_path", "output_dir", "materials_library_path"]

        for field in path_fields:
            if field in result and result[field] is not None:
                path_value = Path(result[field])
                if not path_value.is_absolute():
                    result[field] = str((config_dir / path_value).resolve())

        return result


def load_config[T: BaseModel](
    model_class: type[T],
    config_path: Path,
    overrides: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> T:
    """Convenience function to load a config in one call.

    Args:
        model_class: Pydantic model class for validation
        config_path: Path to YAML configuration file
        overrides: Optional CLI overrides
        context: Optional context dictionary

    Returns:
        Validated configuration model instance

    Example:
        ```python
        from world_understanding.agentic.config import USDDatasetConfig, load_config

        config = load_config(
            USDDatasetConfig,
            Path("config.yaml"),
            overrides={"output_dir": "/tmp/output"}
        )
        ```
    """
    loader = ConfigLoader(model_class)
    return loader.load(config_path, overrides=overrides, context=context)
