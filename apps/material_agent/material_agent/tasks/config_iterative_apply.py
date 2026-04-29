# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for iterative apply workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from material_agent.api.defaults import (
    ITERATION_DEFAULTS,
    PREDICT_DEFAULTS,
    apply_defaults,
)

logger = logging.getLogger(__name__)


class IterativeApplyConfigTask(Task):
    """Compatibility config task for iterative apply workflows."""

    def __init__(self):
        """Initialize the iterative apply config loading task."""
        self.name = "IterativeApplyConfigLoading"
        self.description = "Load iterative apply configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load iterative apply configuration.

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

        listener.info(f"Loading iterative apply configuration from {config_path}")

        # Load YAML configuration
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Configuration file is empty")

        # Pass through the config
        context["config"] = config
        context["input_usd_path"] = config.get("input_usd_path")
        context["output_usd_path"] = config.get("output_usd_path")
        # Set final_output_usd_path for IterativeApplyCompletionTask
        context["final_output_usd_path"] = config.get("output_usd_path")
        context["dataset_path"] = config.get("dataset")

        # Get iteration settings from nested 'iteration' section or top-level
        iteration_config = config.get("iteration", {})
        context["max_iterations"] = iteration_config.get(
            "max_iterations"
        ) or config.get("max_iterations", 5)
        save_intermediate = iteration_config.get("save_intermediate", True)
        context["save_intermediate"] = save_intermediate

        # Map iterations_dir / intermediate_dir to intermediate_output_dir
        iterations_dir = iteration_config.get("intermediate_dir") or config.get(
            "iterations_dir"
        )
        if iterations_dir:
            context["intermediate_output_dir"] = iterations_dir
            context["iterations_dir"] = (
                iterations_dir  # Keep for backward compatibility
            )

        # Extract settings from nested predict config with defaults applied
        predict_config = config.get("predict", {})
        predict_config_with_defaults = apply_defaults(predict_config, PREDICT_DEFAULTS)

        context["vlm_config"] = predict_config_with_defaults.get("vlm", {})
        context["llm_config"] = predict_config_with_defaults.get("llm", {})
        context["max_workers"] = predict_config_with_defaults.get("max_workers", 64)
        context["prediction_batch_size"] = predict_config_with_defaults.get(
            "prediction_batch_size", 1
        )

        # Add VLM and LLM configs to main config for ModelProvisioningTask
        config["vlm"] = predict_config_with_defaults.get("vlm", {})
        config["llm"] = predict_config_with_defaults.get("llm", {})

        # Load system prompt from file if system_prompt_file is provided
        system_prompt = predict_config.get("system_prompt")
        system_prompt_file = predict_config.get("system_prompt_file")

        if system_prompt_file and not system_prompt:
            # Load from file
            system_prompt_path = Path(system_prompt_file)
            if system_prompt_path.exists():
                with open(system_prompt_path, encoding="utf-8") as f:
                    system_prompt = f.read()
                listener.info(f"Loaded system prompt from: {system_prompt_path}")
            else:
                listener.warning(
                    f"System prompt file not found: {system_prompt_path}, "
                    "will use default"
                )

        # Store system prompt in both locations for compatibility
        context["system_prompt"] = system_prompt
        # VLMInferenceTask expects it here (context["config"] already set above)
        context["config"]["system_prompt"] = system_prompt

        # Extract report compression configuration from predict config
        report_config = predict_config.get("report", {})
        if isinstance(report_config, dict):
            if "image_max_size" in report_config:
                context["report_image_max_size"] = report_config["image_max_size"]
            if "image_format" in report_config:
                context["report_image_format"] = report_config["image_format"]
            if "image_quality" in report_config:
                context["report_image_quality"] = report_config["image_quality"]

        # Extract settings from nested apply config
        apply_config = config.get("apply", {})
        context["layer_only"] = apply_config.get("layer_only", False)
        context["flatten_output"] = apply_config.get("flatten_output", True)
        context["aws_profile"] = apply_config.get("aws_profile")
        context["usd_search_config"] = apply_config.get("usd_search", {})

        # Build materials_mapping from top-level materials section or apply config
        materials_mapping = apply_config.get("materials_mapping", {})
        if not materials_mapping:
            materials_mapping = self._load_materials_mapping(
                config, config_path, listener
            )
        context["materials_mapping"] = materials_mapping

        # Extract render settings (now a sibling of apply, not nested within it)
        render_config = config.get("render", {})
        context["render_enabled"] = render_config.get("enabled", False)
        context["render_config"] = render_config

        # Extract judge settings with defaults applied
        judge_config = config.get("judge", {})
        judge_config_with_defaults = apply_defaults(
            judge_config, ITERATION_DEFAULTS["judge"]
        )
        context["judge_config"] = judge_config_with_defaults
        context["reference_images"] = judge_config.get("reference_images", [])

        # Add judge config to main config for ModelProvisioningTask.
        # When the judge has a VLM, set vlm_judge; otherwise set llm_judge.
        if "vlm" in judge_config_with_defaults:
            config["vlm_judge"] = judge_config_with_defaults["vlm"]
        else:
            config["llm_judge"] = judge_config_with_defaults

        return context

    def _load_materials_mapping(
        self, config: dict[str, Any], config_path: Path, listener: Any
    ) -> dict[str, str]:
        """Load materials mapping from top-level materials section.

        Supports:
        - materials.path: Path to external materials YAML file
        - materials.library_path + materials.entries: Inline definition

        Args:
            config: Full configuration dictionary
            config_path: Path to the config file (for resolving relative paths)
            listener: Event listener for logging

        Returns:
            Dictionary mapping material names to bindings, plus
            material_library_path key for library-based materials
        """
        materials_config = config.get("materials", {})
        if not materials_config:
            return {}

        config_dir = config_path.parent

        # If materials.path points to external YAML, load it
        materials_path = materials_config.get("path")
        if materials_path:
            materials_yaml_path = Path(materials_path)
            if not materials_yaml_path.is_absolute():
                materials_yaml_path = config_dir / materials_yaml_path
            if materials_yaml_path.exists():
                listener.info(f"Loading materials from: {materials_yaml_path}")
                with open(materials_yaml_path, encoding="utf-8") as f:
                    materials_config = yaml.safe_load(f) or {}
                # Resolve library_path relative to the materials YAML
                if "library_path" in materials_config:
                    lib_path = Path(materials_config["library_path"])
                    if not lib_path.is_absolute():
                        lib_path = materials_yaml_path.parent / lib_path
                    materials_config["library_path"] = str(lib_path.resolve())
            else:
                listener.warning(f"Materials file not found: {materials_yaml_path}")
                return {}

        # Convert library_path + entries into materials_mapping dict
        library_path = materials_config.get("library_path")
        entries = materials_config.get("entries", [])
        if not library_path or not entries:
            return {}

        # Resolve library_path relative to config dir if needed
        if not Path(library_path).is_absolute():
            library_path = str(config_dir / library_path)

        mapping: dict[str, str] = {"material_library_path": library_path}
        for entry in entries:
            name = entry.get("name", "")
            binding = entry.get("binding", "")
            if name and binding:
                mapping[name] = binding

        listener.info(
            f"Loaded {len(mapping) - 1} materials from library: "
            f"{Path(library_path).name}"
        )
        return mapping
