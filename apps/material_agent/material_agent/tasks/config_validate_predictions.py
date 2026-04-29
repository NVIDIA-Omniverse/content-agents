# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for validate_predictions step."""

import logging
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class ValidatePredictionsConfigTask(Task):
    """Load and validate configuration for prediction validation step.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - predictions_path: Path to predictions JSONL file
        - material_names: List of valid material names
        - llm_config: Optional LLM config for repair
    """

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        config_path = context.get("config_path")
        if not config_path:
            raise ValueError("config_path is required in context")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        listener.info(f"Loading validate_predictions config from {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError(f"Empty configuration file: {config_path}")

        # predictions_path is required (auto-wired by executor)
        if "predictions_path" not in config:
            raise ValueError(
                "predictions_path is required in validate_predictions config"
            )
        context["predictions_path"] = config["predictions_path"]

        # material_names — list of valid names
        if "material_names" not in config:
            raise ValueError(
                "material_names is required in validate_predictions config"
            )
        context["material_names"] = config["material_names"]

        # Optional LLM config for repair
        if "llm" in config:
            context["llm_config"] = config["llm"]

        listener.info(f"Predictions: {context['predictions_path']}")
        listener.info(f"Material library: {len(context['material_names'])} entries")

        return context
