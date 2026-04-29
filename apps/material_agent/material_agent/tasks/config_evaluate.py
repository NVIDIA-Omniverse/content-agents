# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for evaluation workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class EvaluateConfigTask(Task):
    """Compatibility config task for evaluation workflows."""

    def __init__(self):
        """Initialize the evaluate config loading task."""
        self.name = "EvaluateConfigLoading"
        self.description = "Load evaluation configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load evaluation configuration.

        Args:
            context: Workflow context containing config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config_path = context.get("config_path")
        if not config_path:
            raise ValueError("config_path not provided in context")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        listener.info(f"Loading evaluation configuration from {config_path}")

        # Load YAML configuration
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Configuration file is empty")

        # Resolve paths - try both relative to config dir and relative to cwd
        config_dir = config_path.parent.resolve()

        def resolve_path(path_str: str | None) -> Path | None:
            """Resolve a path, trying both relative to config dir and cwd."""
            if not path_str:
                return None
            path = Path(path_str)
            if path.is_absolute():
                return path
            # Try relative to cwd first (more common for result paths)
            cwd_path = Path.cwd() / path
            if cwd_path.exists():
                return cwd_path.resolve()
            # Fall back to relative to config dir
            config_relative = config_dir / path
            return config_relative.resolve()

        # Pass through the config
        context["config"] = config

        # Resolve predictions_path
        context["predictions_path"] = resolve_path(config.get("predictions_path"))

        # Resolve dataset_path
        context["dataset_path"] = resolve_path(config.get("dataset_path"))

        context["llm_judge_config"] = config.get("llm_judge", {})

        # Resolve output_dir
        context["output_dir"] = resolve_path(config.get("output_dir"))

        return context
