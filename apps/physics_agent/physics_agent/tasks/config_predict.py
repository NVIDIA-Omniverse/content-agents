# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict configuration task for Physics Agent."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from physics_agent.api.defaults import PREDICT_DEFAULTS, apply_defaults

logger = logging.getLogger(__name__)


class PredictConfigTask(Task):
    """Load and validate prediction configuration.

    Input context keys:
        - config_path: Path to YAML config file
        OR
        - config_dict: Configuration dictionary

    Output context keys:
        - dataset: List of dataset entries
        - dataset_path: Path to dataset file
        - output_dir: Output directory for predictions
        - vlm_config: VLM configuration
        - llm_config: LLM configuration (optional)
        - system_prompt: System prompt for VLM
        - output_key: Key for classification output
    """

    def __init__(self):
        """Initialize the config task."""
        self.name = "PredictConfig"
        self.description = "Load and validate prediction configuration"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Load and validate configuration.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with configuration
        """
        config = self._load_config(context)

        # Apply defaults
        config = apply_defaults(config, PREDICT_DEFAULTS)

        # Resolve paths
        config_path = context.get("config_path")
        if config_path:
            config_dir = Path(config_path).parent
        else:
            config_dir = Path.cwd()

        # Load dataset
        dataset_path = config.get("dataset")
        if dataset_path:
            dataset_path = self._resolve_path(dataset_path, config_dir)
            dataset = self._load_dataset(dataset_path)
        else:
            raise ValueError("No dataset specified in configuration")

        # Resolve output directory
        output_dir = config.get("output_dir")
        if output_dir:
            output_dir = self._resolve_path(output_dir, config_dir)
        else:
            output_dir = dataset_path.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract system prompt (if dataset.json exists with v0.2 format)
        system_prompt = self._extract_system_prompt(dataset_path)

        # Get output_key (configurable)
        output_key = config.get("output_key", "classification")

        # Update context
        context["config"] = config  # Required for ModelProvisioningTask
        context.update(
            {
                "dataset": dataset,
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "image_base_dir": str(dataset_path.parent),
                "vlm_config": config.get("vlm", {}),
                "llm_config": config.get("llm", {}),
                "system_prompt": system_prompt,
                "output_key": output_key,
                "max_workers": config.get("max_workers"),
                "resume": context.get("resume", False),
                "stream_predictions": context.get("stream_predictions", True),
            }
        )

        # Extract report compression configuration if present
        report_config = config.get("report", {})
        if isinstance(report_config, dict):
            if "image_max_size" in report_config:
                context["report_image_max_size"] = report_config["image_max_size"]
            if "image_format" in report_config:
                context["report_image_format"] = report_config["image_format"]
            if "image_quality" in report_config:
                context["report_image_quality"] = report_config["image_quality"]

        logger.info("Loaded configuration for prediction")
        logger.info("Dataset: %s (%d entries)", dataset_path, len(dataset))
        logger.info("Output directory: %s", output_dir)
        logger.info("Output key: %s", output_key)

        return context

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        """Load configuration from file or dict.

        Args:
            context: Workflow context

        Returns:
            Configuration dictionary
        """
        if "config_dict" in context:
            return context["config_dict"]

        config_path = context.get("config_path")
        if not config_path:
            raise ValueError("No config_path or config_dict in context")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_path(self, path: str, config_dir: Path) -> Path:
        """Resolve path relative to config directory.

        Args:
            path: Path string
            config_dir: Configuration directory

        Returns:
            Resolved Path
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (config_dir / path_obj).resolve()

    def _load_dataset(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load dataset from JSONL file.

        Args:
            dataset_path: Path to dataset file

        Returns:
            List of dataset entries
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        dataset = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))

        return dataset

    def _extract_system_prompt(self, dataset_path: Path) -> str | None:
        """Extract system prompt from dataset.json (v0.2 format).

        Args:
            dataset_path: Path to dataset JSONL file

        Returns:
            System prompt string or None
        """
        # Check for dataset.json in same directory
        dataset_json = dataset_path.parent / "dataset.json"
        if dataset_json.exists():
            try:
                with open(dataset_json, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("system_prompt")
            except Exception as e:
                logger.warning("Failed to load system prompt from dataset.json: %s", e)

        return None
