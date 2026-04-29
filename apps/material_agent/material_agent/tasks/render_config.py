# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for loading render configuration from YAML."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class RenderConfigTask(Task):
    """Load render configuration from a YAML file.

    This task is the entry point for the render workflow. It loads the configuration
    file and prepares the context for subsequent tasks.

    Input context keys:
        - config_path: Path to the YAML configuration file (required)
        - input_usd_override: Optional override for input USD path
        - output_path_override: Optional override for output path

    Output context keys:
        - input_usd_path: Path to the USD file to render
        - output_base_path: Base path for rendered images
        - render_config: Rendering configuration dictionary with:
            - backend: Rendering backend (remote, ovrtx)
            - image_width: Image width in pixels
            - image_height: Image height in pixels
            - camera_corners: List of camera corners to render from
            - camera_margin: Camera margin multiplier
            - background_color: Background color as [R, G, B]
            - flatten_before_render: Whether to flatten USD before rendering
        - flatten_before_render: Boolean flag
    """

    def __init__(self):
        """Initialize the render config task."""
        self.name = "RenderConfig"
        self.description = "Load render configuration"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load and validate the render configuration.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with render configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        import yaml

        config_path = context.get("config_path")
        if not config_path:
            raise ValueError("config_path is required in context")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        listener.info(f"Loading render configuration from {config_path}")

        # Load config
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract render configuration
        # Handle three cases:
        # 1. Direct render config (from pipeline executor temp file)
        # 2. Nested under "render" key (standalone config)
        # 3. Nested under "steps.render" (full unified config)

        if "backend" in config and "enabled" in config:
            # Case 1: Direct render config (already extracted by pipeline executor)
            render_config = config
            input_usd_path = config.get("input_usd_path")
            output_path = config.get("output_path")
        elif "render" in config:
            # Case 2: Standalone render config
            render_config = config["render"]
            input_usd_path = config.get("input_usd_path")
            output_path = config.get("output_path")
        elif "steps" in config and "render" in config["steps"]:
            # Case 3: Unified config format
            render_config = config["steps"]["render"]

            # Infer input from output config
            if "output" in config:
                output_config = config["output"]
                input_usd_path = output_config.get("usd_path")
            elif "input" in config:
                # Fallback to input
                input_usd_path = config["input"].get("usd_path")
            else:
                input_usd_path = None

            # Get project working dir for output
            if "project" in config:
                working_dir = Path(config["project"].get("working_dir", "."))
                output_path = working_dir / "renders"
            else:
                output_path = None
        else:
            raise ValueError(
                "No 'render' configuration found in config file. "
                "Expected either direct render config, 'render' key, or 'steps.render' key."
            )

        # Apply overrides from context
        input_usd_override = context.get("input_usd_override")
        if input_usd_override:
            input_usd_path = input_usd_override
            listener.info(f"Using input USD override: {input_usd_path}")

        output_path_override = context.get("output_path_override")
        if output_path_override:
            output_path = output_path_override
            listener.info(f"Using output path override: {output_path}")

        # Validate inputs
        if not input_usd_path:
            raise ValueError("input_usd_path not specified in config")

        # Convert to Path - paths from unified config temp files are already absolute
        input_usd_path = Path(input_usd_path)

        # If path is relative (unlikely from unified config), resolve relative to config file
        if not input_usd_path.is_absolute():
            input_usd_path = (config_path.parent / input_usd_path).resolve()

        listener.info(f"Input USD for rendering: {input_usd_path}")

        if not input_usd_path.exists():
            raise FileNotFoundError(f"Input USD file not found: {input_usd_path}")

        # Prepare output path
        if output_path:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = (config_path.parent / output_path).resolve()
        else:
            # Default to same directory as input USD
            output_path = input_usd_path.parent

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract render settings with defaults
        backend = render_config.get("backend", "remote")
        image_width = render_config.get("image_width", 1024)
        image_height = render_config.get("image_height", image_width)
        camera_corners = render_config.get("camera_corners", ["+x+y+z"])
        camera_margin = render_config.get("camera_margin", 1.2)
        background_color = render_config.get("background_color", [1.0, 1.0, 1.0])
        flatten_before_render = render_config.get("flatten_before_render", True)

        # Ensure camera_corners is a list
        if isinstance(camera_corners, str):
            camera_corners = [camera_corners]

        listener.info("Render configuration loaded successfully:")
        listener.info(f"  Input USD: {input_usd_path}")
        listener.info(f"  Output directory: {output_path}")
        listener.info(f"  Backend: {backend}")
        listener.info(f"  Image size: {image_width}x{image_height}")
        listener.info(f"  Camera corners: {', '.join(camera_corners)}")
        listener.info(f"  Flatten before render: {flatten_before_render}")

        # Update context
        context["input_usd_path"] = str(input_usd_path)
        context["output_base_path"] = str(output_path)
        render_config_out: dict[str, Any] = {
            "backend": backend,
            "image_width": image_width,
            "image_height": image_height,
            "camera_corners": camera_corners,
            "camera_margin": camera_margin,
            "background_color": background_color,
        }
        # Pass through prim_path for camera scoping on large scenes
        if render_config.get("prim_path"):
            render_config_out["prim_path"] = render_config["prim_path"]
        # Pass through clear_materials to strip original bindings before render
        if render_config.get("clear_materials"):
            render_config_out["clear_materials"] = True
        context["render_config"] = render_config_out
        context["flatten_before_render"] = flatten_before_render

        return context
