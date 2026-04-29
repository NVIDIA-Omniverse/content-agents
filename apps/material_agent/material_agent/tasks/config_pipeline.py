# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for pipeline workflows."""

import logging
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from material_agent.api.defaults import PIPELINE_STEP_NAMES

logger = logging.getLogger(__name__)


class PipelineConfigTask(Task):
    """Task to load and validate configuration for pipeline workflows.

    This task reads a YAML configuration file for pipelines and validates its structure.
    The pipeline config can contain sections for each step: build_dataset_usd,
    build_dataset_pdf_vectorstore, build_dataset_prepare_dataset, predict/benchmark,
    apply, refine.

    Available steps:
    - build_dataset_usd: Prepare USD data with rendering
    - build_dataset_prepare_dataset: Create dataset from rendered USD
    - predict/benchmark: VLM-based material prediction
    - apply: Apply predicted materials to USD (single pass)
    - refine: Iterative material assignment with VLM judge refinement

    Each step section can either:
    1. Reference an external config file via 'config' key
    2. Provide inline configuration directly

    Path Resolution:
        - All relative paths in config file are treated as relative to the config file location
        - working_dir in pipeline section overrides the base directory for path resolution
        - Absolute paths are used as-is

    Input context keys:
        - config_path: Path to the YAML pipeline configuration file
        - skip_steps: Optional list of step names to skip
        - only_steps: Optional list of step names to run exclusively

    Output context keys:
        - pipeline_config: Parsed and validated pipeline configuration
        - pipeline_name: Name of the pipeline
        - pipeline_description: Description of the pipeline
        - working_dir: Working directory for relative path resolution
        - steps_to_run: Ordered list of steps to execute
        - step_configs: Dictionary of resolved configurations for each step
        - keep_temp_files: Whether to preserve temporary files after completion
    """

    # Use centralized step names
    VALID_STEPS = PIPELINE_STEP_NAMES

    def __init__(self):
        """Initialize the pipeline config loading task."""
        self.name = "PipelineConfigLoading"
        self.description = "Load and validate pipeline configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load and validate pipeline configuration.

        Args:
            context: Workflow context containing config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If configuration file not found
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config_path = context.get("config_path")
        if not config_path:
            raise ValueError("config_path not provided in context")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Pipeline configuration file not found: {config_path}"
            )

        listener.info(f"Loading pipeline configuration from {config_path}")

        # Load YAML configuration
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Pipeline configuration file is empty")

        # Extract pipeline metadata
        pipeline_meta = config.get("pipeline", {})
        pipeline_name = pipeline_meta.get("name", "unnamed_pipeline")
        pipeline_description = pipeline_meta.get("description", "")

        # Determine working directory for path resolution
        working_dir = pipeline_meta.get("working_dir", ".")
        working_dir = Path(working_dir)
        if not working_dir.is_absolute():
            working_dir = config_path.parent / working_dir
        working_dir = working_dir.resolve()

        # Check if temporary files should be preserved (default: True)
        keep_temp_files = pipeline_meta.get("keep_temp_files", True)

        listener.info(f"Pipeline: {pipeline_name}")
        if pipeline_description:
            listener.info(f"Description: {pipeline_description}")
        listener.info(f"Working directory: {working_dir}")
        if keep_temp_files:
            listener.info("Temporary files will be preserved after completion")

        # Parse unified materials section
        materials_data = self._parse_materials(config, config_path)
        if materials_data:
            listener.info(
                f"Loaded {len(materials_data['entries'])} materials from unified definition"
            )
            if materials_data.get("library_path"):
                listener.info(f"  Material library: {materials_data['library_path']}")

        # Validate and extract step configurations
        steps_to_run, step_configs = self._process_steps(
            config, config_path, working_dir, context, materials_data, listener
        )

        if not steps_to_run:
            raise ValueError("No valid steps found in pipeline configuration")

        listener.info(f"Steps to execute: {', '.join(steps_to_run)}")

        # Update context
        context["pipeline_config"] = config
        context["pipeline_name"] = pipeline_name
        context["pipeline_description"] = pipeline_description
        context["working_dir"] = working_dir
        context["steps_to_run"] = steps_to_run
        context["step_configs"] = step_configs
        context["materials_data"] = materials_data  # Store for use by steps
        context["keep_temp_files"] = keep_temp_files

        return context

    def _parse_materials(
        self, config: dict[str, Any], config_path: Path
    ) -> dict[str, Any] | None:
        """Parse unified materials section from pipeline config.

        Args:
            config: Full pipeline configuration
            config_path: Path to the pipeline config file

        Returns:
            Parsed materials data or None if not present
        """
        materials_section = config.get("materials")
        if not materials_section:
            return None

        if not isinstance(materials_section, dict):
            raise ValueError("'materials' section must be a dictionary")

        # Parse library path (optional)
        library_path = materials_section.get("library_path")
        if library_path:
            library_path = Path(library_path)
            # Resolve relative to config file location
            if not library_path.is_absolute():
                library_path = config_path.parent / library_path
            library_path = str(library_path.resolve())

        # Parse material entries
        entries = materials_section.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("'materials.entries' must be a list")

        parsed_entries = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Material entry {i} must be a dictionary")

            name = entry.get("name")
            description = entry.get("description", "")
            binding = entry.get("binding", "")

            if not name:
                raise ValueError(f"Material entry {i} missing 'name' field")

            parsed_entries.append(
                {
                    "name": name,
                    "description": description,
                    "binding": binding,
                }
            )

        return {
            "library_path": library_path,
            "entries": parsed_entries,
        }

    def _process_steps(
        self,
        config: dict[str, Any],
        config_path: Path,
        working_dir: Path,
        context: dict[str, Any],
        materials_data: dict[str, Any] | None = None,
        listener: Any = None,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Process and resolve step configurations.

        Args:
            config: Full pipeline configuration
            config_path: Path to the pipeline config file
            working_dir: Working directory for relative paths
            context: Workflow context
            materials_data: Optional materials data from pipeline config
            listener: Event listener for logging

        Returns:
            Tuple of (steps_to_run, step_configs)
        """
        skip_steps = set(context.get("skip_steps", []))
        only_steps = context.get("only_steps", [])

        steps_to_run = []
        step_configs = {}

        # Process steps in order
        for step_name in self.VALID_STEPS:
            # Skip if not in config
            if step_name not in config:
                continue

            # Handle mutually exclusive predict/benchmark
            if step_name == "predict" and "benchmark" in config:
                listener.warning(
                    "Both 'predict' and 'benchmark' found. Using 'predict'."
                )
            elif step_name == "benchmark" and "predict" in steps_to_run:
                listener.info("Skipping 'benchmark' as 'predict' is already configured")
                continue

            # Apply skip/only filters
            if skip_steps and step_name in skip_steps:
                listener.info(f"Skipping step: {step_name} (--skip)")
                continue

            if only_steps and step_name not in only_steps:
                listener.debug(f"Skipping step: {step_name} (not in --only)")
                continue

            # Get step configuration
            step_config = config[step_name]
            if not isinstance(step_config, dict):
                raise ValueError(f"Step '{step_name}' must be a dictionary")

            # Resolve configuration (external reference or inline)
            resolved_config = self._resolve_step_config(
                step_name, step_config, config_path, working_dir, listener
            )

            # Inject materials data into specific steps
            if materials_data:
                resolved_config = self._inject_materials_into_step(
                    step_name, resolved_config, materials_data, listener
                )

            steps_to_run.append(step_name)
            step_configs[step_name] = resolved_config

        return steps_to_run, step_configs

    def _resolve_step_config(
        self,
        step_name: str,
        step_config: dict[str, Any],
        config_path: Path,
        working_dir: Path,
        listener: Any = None,
    ) -> dict[str, Any]:
        """Resolve step configuration from external file or inline.

        Args:
            step_name: Name of the step
            step_config: Step configuration section
            config_path: Path to the pipeline config file
            working_dir: Working directory for relative paths
            listener: Event listener for logging

        Returns:
            Resolved configuration dictionary
        """
        # If 'config' key exists, load external config file
        if "config" in step_config:
            external_config_path = step_config["config"]
            external_config_path = Path(external_config_path)

            # Resolve relative to pipeline config file (not working_dir)
            if not external_config_path.is_absolute():
                external_config_path = config_path.parent / external_config_path

            if not external_config_path.exists():
                raise FileNotFoundError(
                    f"External config for step '{step_name}' not found: {external_config_path}"
                )

            listener.info(
                f"Loading external config for {step_name}: {external_config_path}"
            )

            with open(external_config_path, encoding="utf-8") as f:
                resolved_config = yaml.safe_load(f)

            if not resolved_config:
                raise ValueError(f"External config for '{step_name}' is empty")

            # Store reference to external config path for path resolution
            resolved_config["_external_config_path"] = external_config_path

            return resolved_config

        # Otherwise, use inline configuration
        listener.info(f"Using inline config for {step_name}")

        # Make a copy to avoid modifying original
        resolved_config = dict(step_config)

        # Mark as inline config
        resolved_config["_inline_config"] = True
        resolved_config["_pipeline_config_path"] = config_path

        return resolved_config

    def _inject_materials_into_step(
        self,
        step_name: str,
        step_config: dict[str, Any],
        materials_data: dict[str, Any],
        listener: Any = None,
    ) -> dict[str, Any]:
        """Inject materials data into step configuration.

        Args:
            step_name: Name of the step
            step_config: Step configuration
            materials_data: Parsed materials data from pipeline config
            listener: Event listener for logging

        Returns:
            Updated step configuration with materials injected
        """
        # For build_dataset_prepare_dataset: inject materials_list
        if step_name == "build_dataset_prepare_dataset":
            if "materials_list" not in step_config:
                # Extract list of material names with descriptions for prompts
                materials_list = [entry["name"] for entry in materials_data["entries"]]
                step_config["materials_list"] = materials_list
                listener.debug(
                    f"Injected {len(materials_list)} materials into {step_name}"
                )

            # Also inject formatted materials for prompt substitution
            if "prompts" in step_config:
                materials_formatted = self._format_materials_for_prompt(
                    materials_data["entries"]
                )
                step_config["_materials_formatted"] = materials_formatted

        # For validate/harmonize: inject material_names
        elif step_name in ("validate_predictions", "harmonize_predictions"):
            if "material_names" not in step_config:
                step_config["material_names"] = [
                    entry["name"] for entry in materials_data["entries"]
                ]
                listener.debug(
                    f"Injected {len(step_config['material_names'])} material_names into {step_name}"
                )

        # For apply and refine: inject materials_mapping
        elif step_name in ["apply", "refine"]:
            # Build materials_mapping from entries
            materials_mapping = {}

            # Add library path if present
            if materials_data.get("library_path"):
                materials_mapping["material_library_path"] = materials_data[
                    "library_path"
                ]

            # Add name -> binding mappings
            for entry in materials_data["entries"]:
                materials_mapping[entry["name"]] = entry["binding"]

            # For refine step, inject into the 'apply' subsection
            if step_name == "refine":
                if "apply" not in step_config:
                    step_config["apply"] = {}
                if "materials_mapping" not in step_config["apply"]:
                    step_config["apply"]["materials_mapping"] = materials_mapping
                    listener.debug(
                        f"Injected materials_mapping with {len(materials_data['entries'])} entries into refine.apply"
                    )
            # For apply step, inject at top level
            elif step_name == "apply":
                if "materials_mapping" not in step_config:
                    step_config["materials_mapping"] = materials_mapping
                    listener.debug(
                        f"Injected materials_mapping with {len(materials_data['entries'])} entries into {step_name}"
                    )

        return step_config

    def _format_materials_for_prompt(self, entries: list[dict[str, Any]]) -> str:
        """Format materials list for prompt injection.

        Args:
            entries: List of material entries with name and description

        Returns:
            Formatted string ready for {materials_list} substitution
        """
        lines = []
        for entry in entries:
            name = entry["name"]
            description = entry.get("description", "")
            if description:
                lines.append(f"{name}: {description}")
            else:
                lines.append(name)

        return "\n".join(lines)
