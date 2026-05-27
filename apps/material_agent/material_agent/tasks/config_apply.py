# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for apply workflows.

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


class ApplyConfigTask(Task):
    """Compatibility config task for apply workflows.

    This task loads config from a YAML file (which may be a temp file
    created by the unified pipeline system).
    """

    def __init__(self):
        """Initialize the apply config loading task."""
        self.name = "ApplyConfigLoading"
        self.description = "Load apply configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load apply configuration.

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

        listener.info(f"Loading apply configuration from {config_path}")

        # Load YAML configuration
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Configuration file is empty")

        # Pass through the config
        context["config"] = config

        # Extract key fields
        context["input_usd_path"] = config.get("input_usd_path")
        context["predictions_path"] = config.get("predictions_path")
        context["output_usd_path"] = config.get("output_usd_path")
        context["materials_mapping"] = config.get("materials_mapping", {})
        context["usd_search_config"] = config.get("usd_search", {})
        context["aws_profile"] = config.get("aws_profile")
        context["layer_only"] = config.get("layer_only", False)
        context["flatten_output"] = config.get("flatten_output", True)
        context["skip_instance_check"] = config.get("skip_instance_check", False)
        allow_empty_predictions = config.get("allow_empty_predictions", False)
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "apply.allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )
        context["allow_empty_predictions"] = allow_empty_predictions
        fail_on_unknown_material = config.get("fail_on_unknown_material", False)
        if not isinstance(fail_on_unknown_material, bool):
            raise ValueError(
                "apply.fail_on_unknown_material must be a boolean, got "
                f"{type(fail_on_unknown_material).__name__}"
            )
        context["fail_on_unknown_material"] = fail_on_unknown_material
        context["render_config"] = config.get("render", {})
        context["llm_config"] = config.get("llm", {})

        # Set render_enabled flag based on render config
        render_config = context["render_config"]
        context["render_enabled"] = (
            render_config.get("enabled", False) if render_config else False
        )

        return context
