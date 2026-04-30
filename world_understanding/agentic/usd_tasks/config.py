# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD data preparation configuration task."""

import logging
from pathlib import Path
from typing import Any

import yaml

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from .defaults import USD_RENDERING_DEFAULTS

logger = logging.getLogger(__name__)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return value


class USDDataPrepConfigTask(Task):
    """Load and validate USD data preparation configuration from YAML."""

    def __init__(self):
        self.name = "USDDataPrepConfig"
        self.description = "Load and validate USD data preparation configuration"

    def run(self, context: dict[str, Any], object_store: ObjectStore) -> dict[str, Any]:
        """Load configuration and populate context for USD data preparation.

        Expected context inputs:
            - config_path: Path to YAML configuration file
            - source_override: Optional USD path override
            - output_dir_override: Optional output directory override
            - prim_filters: Optional filters for prim selection
            - extract_prim_metadata: Optional flag to extract metadata

        Updates context with:
            - usd_path: Path to USD file
            - output_dir: Output directory for dataset
            - render_output_dir: Directory for rendered images
            - dataset_output_dir: Directory for dataset manifest
            - prim_filters: Filters for prim selection
            - extract_metadata: Whether to extract prim metadata
            - renderer_config: Renderer configuration

        Path Resolution:
            - All relative paths in config file are treated as relative to
              the config file location
            - Command-line overrides are relative to the current working
              directory
            - Absolute paths are used as-is
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config_path = Path(context["config_path"])  # Required from CLI
        listener.info(f"Loading USD configuration from {config_path}")

        # Load YAML configuration
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            listener.warning(f"Config file {config_path} not found, using defaults")
            config = {}

        # Get USD path (from override or config)
        source_override = context.get("source_override")
        if source_override:
            # Command-line overrides are relative to current directory
            usd_path = Path(source_override)
            if not usd_path.is_absolute():
                usd_path = usd_path.resolve()
            listener.info(f"Using USD path override: {usd_path}")
        elif config.get("usd_path"):
            usd_path = Path(config["usd_path"])
            # Config paths are relative to config file location
            if not usd_path.is_absolute():
                usd_path = config_path.parent / usd_path
        else:
            # USD path is required - no default
            raise ValueError(
                "USD path not specified. Please provide 'usd_path' in the configuration file "
                "or use --usd-path command line option."
            )

        context["usd_path"] = usd_path

        # Get output directory (from override or config)
        output_dir_override = context.get("output_dir_override")
        if output_dir_override:
            # Command-line overrides are relative to current directory
            output_dir = Path(output_dir_override)
            if not output_dir.is_absolute():
                output_dir = output_dir.resolve()
            listener.info(f"Using output directory override: {output_dir}")
        elif config.get("output_dir"):
            output_dir = Path(config["output_dir"])
            # Config paths are relative to config file location
            if not output_dir.is_absolute():
                output_dir = config_path.parent / output_dir
        else:
            # Default output dir is relative to config file location
            output_dir = config_path.parent / "output/dataset"

        context["output_dir"] = output_dir
        context["render_output_dir"] = output_dir / "renders"

        # Note: dataset files (dataset.json, prims.jsonl) will be saved
        # directly in output_dir, not in a subdirectory

        # Create output directories
        context["render_output_dir"].mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get prim filters
        context["prim_filters"] = config.get(
            "prim_filters",
            context.get(
                "prim_filters",
                {
                    "types": ["UsdGeom.Mesh"],
                    "skip_instances": True,
                    "skip_prototypes": False,
                },
            ),
        )

        # Get metadata extraction flag
        context["extract_metadata"] = config.get(
            "extract_metadata", context.get("extract_prim_metadata", False)
        )

        # Get display color extraction flag
        context["extract_display_color"] = config.get(
            "extract_display_color", context.get("extract_display_color", False)
        )

        # Get display color statistics flag
        context["include_display_color_statistics"] = config.get(
            "include_display_color_statistics",
            context.get("include_display_color_statistics", False),
        )

        # Get material bindings extraction flag
        context["extract_material_bindings"] = config.get(
            "extract_material_bindings", context.get("extract_material_bindings", True)
        )

        # Get hierarchy extraction flag
        context["extract_hierarchy"] = config.get(
            "extract_hierarchy", context.get("extract_hierarchy", True)
        )

        # Get USD model building flag
        context["build_usd_model"] = config.get(
            "build_usd_model", context.get("build_usd_model", True)
        )

        # Get USD model export flag
        context["export_usd_model"] = config.get(
            "export_usd_model", context.get("export_usd_model", True)
        )

        # Get skip existing flag (for resuming renders)
        # Check both "resume" (unified pipeline) and "skip_existing" (direct usage)
        # They mean the same thing: skip already rendered prims
        # Priority: context[skip_existing] > context[resume] > config[skip_existing] > False
        skip_existing = context.get(
            "skip_existing", context.get("resume", config.get("skip_existing", False))
        )
        context["skip_existing"] = skip_existing

        # Get skip existing materials flag
        # Filter prims with direct material bindings during traversal
        # Priority: context[skip_existing_materials] > config[skip_existing_materials] > False
        skip_existing_materials = context.get(
            "skip_existing_materials",
            config.get("skip_existing_materials", False),
        )
        context["skip_existing_materials"] = skip_existing_materials

        # Get batch size for rendering efficiency
        context["batch_size"] = config.get("batch_size", context.get("batch_size", 10))

        # Get async render request concurrency. This is separate from num_workers:
        # num_workers controls local task/thread parallelism, while this limits
        # simultaneous remote render requests in the async traversal path.
        context["max_concurrent_requests"] = _positive_int(
            config.get(
                "max_concurrent_requests", context.get("max_concurrent_requests", 128)
            ),
            "max_concurrent_requests",
        )

        # Get number of workers for parallel batch processing
        # Check both "max_workers" (unified pipeline) and "num_workers" (direct usage)
        context["num_workers"] = _positive_int(
            config.get(
                "num_workers", context.get("max_workers", context.get("num_workers", 1))
            ),
            "num_workers",
        )

        # Get renderer configuration - merge with defaults
        # Check context first (for unified pipeline), then config file
        renderer_config_override = context.get("renderer", config.get("renderer", {}))
        context["renderer_config"] = {
            **USD_RENDERING_DEFAULTS,  # Start with centralized defaults
            **renderer_config_override,  # Override with user config or context
        }

        listener.info("USD configuration loaded:")
        listener.info(f"  USD path: {context['usd_path']}")
        listener.info(f"  Output directory: {context['output_dir']}")
        listener.info(f"  Extract metadata: {context['extract_metadata']}")
        listener.info(
            f"  Extract display color: {context.get('extract_display_color', False)}"
        )
        listener.info(
            f"  Extract material bindings: {context['extract_material_bindings']}"
        )
        listener.info(f"  Extract hierarchy: {context['extract_hierarchy']}")
        listener.info(f"  Build USD model: {context['build_usd_model']}")
        listener.info(f"  Export USD model: {context['export_usd_model']}")
        listener.info(f"  Skip existing: {context['skip_existing']}")
        listener.info(f"  Batch size: {context['batch_size']}")
        listener.info(
            f"  Max concurrent requests: {context['max_concurrent_requests']}"
        )
        listener.info(f"  Number of workers: {context['num_workers']}")
        listener.info(f"  Renderer backend: {context['renderer_config']['backend']}")
        camera_type = context["renderer_config"].get("camera_view_type", "corner")
        listener.info(f"  Camera view type: {camera_type}")

        return context
