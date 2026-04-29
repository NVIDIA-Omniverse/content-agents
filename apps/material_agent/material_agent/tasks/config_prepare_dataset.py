# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for prepare dataset workflows.

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


class PrepareDatasetConfigTask(Task):
    """Compatibility config task for prepare dataset workflows."""

    def __init__(self):
        """Initialize the prepare dataset config loading task."""
        self.name = "PrepareDatasetConfigLoading"
        self.description = "Load prepare dataset configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load prepare dataset configuration.

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

        listener.info(f"Loading prepare dataset configuration from {config_path}")

        # Load YAML configuration
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Configuration file is empty")

        # Pass through the config
        context["config"] = config
        context["usd_dir"] = Path(config.get("usd_dir", ""))
        context["vector_store_path"] = (
            Path(config["vector_store"]) if config.get("vector_store") else None
        )
        context["dataset_path"] = Path(config.get("dataset", ""))
        context["config_path"] = config_path

        # Use models from config if provided, otherwise discover from usd_dir
        if "models" in config and config["models"]:
            context["models"] = config["models"]
            listener.info(f"Using models from config: {config['models']}")
        else:
            # Discover models from usd_dir (legacy behavior)
            usd_dir = context["usd_dir"]
            if usd_dir and usd_dir.exists():
                models = self._discover_models_from_usd_dir(usd_dir)
                context["models"] = models
                listener.info(f"Discovered {len(models)} models from usd_dir")
            else:
                context["models"] = []
                listener.warning("No models found - usd_dir doesn't exist")

        return context

    def _discover_models_from_usd_dir(self, usd_dir: Path) -> list[str]:
        """Discover model numbers from USD directory structure."""
        if not usd_dir.exists():
            return []

        models = []
        for item in usd_dir.iterdir():
            if item.is_dir():
                dataset_json = item / "dataset.json"
                prims_jsonl = item / "prims.jsonl"
                usd_model_json = item / "usd_model.json"

                if (
                    dataset_json.exists()
                    and prims_jsonl.exists()
                    and usd_model_json.exists()
                ):
                    models.append(item.name)

        models.sort()
        return models
