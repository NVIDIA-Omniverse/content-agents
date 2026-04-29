# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for restore_usd step."""

import logging
from pathlib import Path
from typing import Any

import yaml

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class RestoreUSDConfigTask(Task):
    """Load and validate configuration for predictions restoration step.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - original_usd_path: Path to original USD (auto-wired by executor)
        - predictions_path: Path to input predictions.jsonl (auto-wired by executor)
        - output_predictions_path: Path for restored predictions.jsonl
        - optimization_metadata: Metadata from optimize_usd (injected by executor)
    """

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load restoration configuration.

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

        listener.info(f"Loading restore_usd configuration from {config_path}")

        # Load config
        with open(config_path, encoding="utf-8") as f:
            # Use a permissive loader that handles Python-specific YAML tags
            # from the scene optimizer config (e.g. UsdFormat enum values).
            try:
                config = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                f.seek(0)
                loader = yaml.FullLoader
                config = yaml.load(f, Loader=loader)  # noqa: S506

        if not config:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Validate required fields
        if "original_usd_path" not in config:
            raise ValueError("original_usd_path is required in restore_usd config")
        if "predictions_path" not in config:
            raise ValueError("predictions_path is required in restore_usd config")
        if "output_predictions_path" not in config:
            raise ValueError(
                "output_predictions_path is required in restore_usd config"
            )
        if "optimization_metadata" not in config:
            raise ValueError("optimization_metadata is required in restore_usd config")

        # Extract paths (already resolved by UnifiedPipelineConfigTask)
        context["original_usd_path"] = config["original_usd_path"]
        context["predictions_path"] = config["predictions_path"]
        context["output_predictions_path"] = config["output_predictions_path"]
        context["optimization_metadata"] = config["optimization_metadata"]

        listener.info(f"Original USD: {context['original_usd_path']}")
        listener.info(f"Input predictions: {context['predictions_path']}")
        listener.info(f"Output predictions: {context['output_predictions_path']}")

        return context
