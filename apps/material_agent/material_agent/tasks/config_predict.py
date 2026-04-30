# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for prediction workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import drop_stale_endpoint_credentials

logger = logging.getLogger(__name__)


class PredictConfigTask(Task):
    """Compatibility config task for prediction workflows.

    This task loads config from a YAML file (which may be a temp file
    created by the unified pipeline system).
    """

    def __init__(self):
        """Initialize the predict config loading task."""
        self.name = "PredictConfigLoading"
        self.description = "Load prediction configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load prediction configuration.

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

        listener.info(f"Loading prediction configuration from {config_path}")

        # Load YAML configuration
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Configuration file is empty")

        # Simply pass through the config - it's already been resolved
        # by UnifiedPipelineConfigTask if coming from unified system
        context["config"] = config

        # Extract key fields for backward compatibility
        context["dataset_path"] = config.get("dataset")
        context["output_dir"] = config.get("output_dir")
        vlm_config = config.get("vlm", {})
        # Inject local NIM endpoint if configured via env var.
        # Setting MA_VLM_NIM_BASE_URL forces backend=nim regardless of config
        # (same pattern as material_agent_service pipeline_router.py).
        nim_base_url = os.environ.get("MA_VLM_NIM_BASE_URL")
        vlm_backend = (vlm_config.get("backend") or "").strip().lower()
        if nim_base_url and vlm_backend not in ("", "echo", "mock"):
            if vlm_backend != "nim":
                listener.info(
                    f"MA_VLM_NIM_BASE_URL set - overriding VLM backend "
                    f"from '{vlm_config.get('backend', '')}' to 'nim'"
                )
            drop_stale_endpoint_credentials(
                vlm_config, preserve_local_nim_placeholder=True
            )
            vlm_config["backend"] = "nim"
            vlm_config["base_url"] = nim_base_url
        config["vlm"] = vlm_config

        llm_config = config.get("llm", {})
        llm_nim_base_url = os.environ.get("MA_LLM_NIM_BASE_URL")
        llm_uses_vlm_sidecar = False
        if not llm_nim_base_url:
            llm_nim_base_url = os.environ.get("MA_VLM_NIM_BASE_URL")
            llm_uses_vlm_sidecar = bool(llm_nim_base_url)
        llm_backend = (llm_config.get("backend") or "").strip().lower()
        if llm_nim_base_url and llm_backend not in ("", "echo", "mock"):
            if llm_backend != "nim":
                listener.info(
                    f"MA_LLM_NIM_BASE_URL/MA_VLM_NIM_BASE_URL set - overriding "
                    f"LLM backend from '{llm_config.get('backend', '')}' to 'nim'"
                )
            drop_stale_endpoint_credentials(
                llm_config, preserve_local_nim_placeholder=True
            )
            llm_config["backend"] = "nim"
            llm_config["base_url"] = llm_nim_base_url
            if llm_uses_vlm_sidecar and vlm_config.get("model"):
                llm_config["model"] = vlm_config["model"]
        config["llm"] = llm_config

        context["vlm_config"] = vlm_config
        context["llm_config"] = llm_config
        context["max_workers"] = config.get("max_workers", 64)
        context["prediction_batch_size"] = config.get("prediction_batch_size", 1)

        # Load system prompt from multiple sources (priority order):
        # 1. Direct system_prompt in config
        # 2. dataset.json inference.prompts[0].system_prompt (v0.2 format - PREFERRED)
        # 3. system_prompt_file (legacy fallback only)
        system_prompt = config.get("system_prompt")

        # Try loading from dataset.json (v0.2 format) first if no direct system_prompt
        if not system_prompt:
            dataset_path = config.get("dataset")
            if dataset_path:
                # Dataset path points to dataset.jsonl, get the config file
                dataset_dir = Path(dataset_path).parent
                dataset_config_path = dataset_dir / "dataset.json"

                if dataset_config_path.exists():
                    try:
                        import json

                        with open(dataset_config_path, encoding="utf-8") as f:
                            dataset_config = json.load(f)

                        # Extract system prompt from v0.2 format
                        prompts = dataset_config.get("inference", {}).get("prompts", [])
                        if prompts and len(prompts) > 0:
                            system_prompt = prompts[0].get("system_prompt", "")
                            if system_prompt:
                                listener.info(
                                    "Loaded system prompt from dataset.json (v0.2 format)"
                                )
                                config["system_prompt"] = system_prompt
                    except Exception as e:
                        listener.warning(
                            f"Failed to load system prompt from dataset.json: {e}"
                        )

        # Legacy fallback: try system_prompt_file only if still no system prompt
        if not system_prompt:
            system_prompt_file = config.get("system_prompt_file")
            if system_prompt_file:
                # Load from file (legacy support)
                system_prompt_path = Path(system_prompt_file)
                if system_prompt_path.exists():
                    with open(system_prompt_path, encoding="utf-8") as f:
                        system_prompt = f.read()
                    listener.info(
                        f"Loaded system prompt from file (legacy): {system_prompt_path}"
                    )
                    config["system_prompt"] = system_prompt
                else:
                    # Only warn if we couldn't load from dataset.json either
                    # (i.e., we actually need this file)
                    listener.warning(
                        f"System prompt file not found: {system_prompt_path}. "
                        "Unable to load system prompt from either dataset.json or file. "
                        "Will use default system prompt."
                    )

        context["system_prompt"] = system_prompt

        # Extract report compression configuration if present
        report_config = config.get("report", {})
        if isinstance(report_config, dict):
            if "image_max_size" in report_config:
                context["report_image_max_size"] = report_config["image_max_size"]
            if "image_format" in report_config:
                context["report_image_format"] = report_config["image_format"]
            if "image_quality" in report_config:
                context["report_image_quality"] = report_config["image_quality"]

        return context
