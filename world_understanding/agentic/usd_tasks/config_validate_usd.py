# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for validate_usd step."""

import logging
from pathlib import Path
from typing import Any

import yaml

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.agentic.usd_tasks.validate_usd import ON_FAILURE_MODES
from world_understanding.functions.graphics.validate_usd import (
    DEFAULT_VALIDATION_CATEGORIES,
)

logger = logging.getLogger(__name__)


class ValidateUSDConfigTask(Task):
    """Load and validate configuration for USD validation step.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - input_usd_path: Path to input USD to validate
        - validation_config: Validation parameters
        - on_failure: "warn" | "block" | "fix"
        - output_dir: Directory for validation outputs
        - original_usd_path: (validate_output only) Path to original input USD
        - baseline_validation: (validate_output only) Cached baseline
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        """Load validation configuration.

        Args:
            context: Workflow context with config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with configuration values

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If required fields are missing
        """
        listener = get_listener(context)

        config_path = context.get("config_path")
        if not config_path:
            raise ValueError("config_path is required in context")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        listener.info(f"Loading validate_usd configuration from {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Extract paths (already resolved by UnifiedPipelineConfigTask)
        if "input_usd_path" not in config:
            raise ValueError("input_usd_path is required in validate_usd config")

        context["input_usd_path"] = config["input_usd_path"]

        if "output_dir" in config:
            context["output_dir"] = config["output_dir"]

        # Pass through original_usd_path and baseline_validation
        # (injected by pipeline executor for validate_output)
        if "original_usd_path" in config:
            context["original_usd_path"] = config["original_usd_path"]
        if "baseline_validation" in config:
            context["baseline_validation"] = config["baseline_validation"]

        # on_failure mode
        on_failure = config.get("on_failure", "warn")
        if on_failure not in ON_FAILURE_MODES:
            raise ValueError(
                f"Invalid on_failure mode: {on_failure!r}. "
                f"Must be one of {ON_FAILURE_MODES}"
            )
        context["on_failure"] = on_failure

        # Build validation config
        validation_config = config.get("validation_config", {})

        # Ensure categories have defaults
        if "categories" not in validation_config:
            validation_config["categories"] = list(DEFAULT_VALIDATION_CATEGORIES)

        # Validate categories
        invalid = [
            c
            for c in validation_config.get("categories", [])
            if c not in DEFAULT_VALIDATION_CATEGORIES
        ]
        if invalid:
            raise ValueError(
                f"Unknown validation categories: {invalid}. "
                f"Available: {DEFAULT_VALIDATION_CATEGORIES}"
            )

        # Default poll_seconds
        if "poll_seconds" not in validation_config:
            validation_config["poll_seconds"] = 300

        context["validation_config"] = validation_config

        listener.info(f"Input USD: {context['input_usd_path']}")
        cats = ", ".join(validation_config.get("categories", []))
        listener.info(f"Categories: {cats}")
        listener.info(f"On failure: {on_failure}")

        return context
