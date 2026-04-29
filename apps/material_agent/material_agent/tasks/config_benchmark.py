# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for benchmark workflows.

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


class BenchmarkConfigTask(Task):
    """Compatibility config task for benchmark workflows."""

    def __init__(self):
        """Initialize the benchmark config loading task."""
        self.name = "BenchmarkConfigLoading"
        self.description = "Load benchmark configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load benchmark configuration.

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

        listener.info(f"Loading benchmark configuration from {config_path}")

        # Load YAML configuration
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Configuration file is empty")

        # Pass through the config
        context["config"] = config
        context["dataset_path"] = config.get("dataset")
        context["output_dir"] = config.get("output_dir")
        context["vlm_config"] = config.get("vlm", {})
        context["llm_config"] = config.get("llm", {})
        context["llm_judge_config"] = config.get("llm_judge", {})
        context["max_workers"] = config.get("max_workers", 64)

        # Load system prompt from file if system_prompt_file is provided
        system_prompt = config.get("system_prompt")
        system_prompt_file = config.get("system_prompt_file")

        if system_prompt_file and not system_prompt:
            # Load from file
            system_prompt_path = Path(system_prompt_file)
            if system_prompt_path.exists():
                with open(system_prompt_path, encoding="utf-8") as f:
                    system_prompt = f.read()
                listener.info(f"Loaded system prompt from: {system_prompt_path}")
                # Also set it back in config so VLMInferenceTask can find it
                config["system_prompt"] = system_prompt
            else:
                listener.warning(
                    f"System prompt file not found: {system_prompt_path}, will use default"
                )

        context["system_prompt"] = system_prompt

        return context
