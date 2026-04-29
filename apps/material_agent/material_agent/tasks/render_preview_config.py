# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for loading render-preview configuration from YAML.

This thin config task maps the material agent step config keys to the
context keys expected by ``RenderScenePreviewTask``.
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class RenderPreviewConfigTask(Task):
    """Load render-preview configuration from a YAML file.

    Mirrors ``RenderConfigTask`` but targets the ``render_preview`` step,
    which uses :class:`RenderScenePreviewTask` from the shared library.

    Input context keys:
        - config_path: Path to the YAML configuration file (required)

    Output context keys:
        - usd_path: Path to the USD file to render
        - output_dir: Directory for preview images
        - render_config: Dictionary consumed by RenderScenePreviewTask
    """

    def __init__(self) -> None:
        self.name = "RenderPreviewConfig"
        self.description = "Load render-preview configuration"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load and validate the render-preview configuration.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with render preview configuration
        """
        import yaml

        listener = get_listener(context, logger_name=__name__)

        config_path = context.get("config_path")
        if not config_path:
            raise ValueError("config_path is required in context")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        listener.info(f"Loading render-preview configuration from {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # The config is already extracted by the unified pipeline executor
        # so it's a flat dict with all keys at the top level.
        usd_path = config.get("usd_path")
        if not usd_path:
            raise ValueError("usd_path not specified in render_preview config")

        usd_path = Path(usd_path)
        if not usd_path.is_absolute():
            usd_path = (config_path.parent / usd_path).resolve()

        if not usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        output_dir = config.get("output_dir")
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = (config_path.parent / output_dir).resolve()
        else:
            output_dir = usd_path.parent / "preview"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build render_config dict for RenderScenePreviewTask
        render_config: dict[str, Any] = {
            "backend": config.get("backend", "remote"),
            "image_width": config.get("image_width", 512),
            "image_height": config.get("image_height", 512),
            "cameras": config.get("cameras", ["+x+y+z"]),
            "camera_margin": config.get("camera_margin", 1.0),
            "background_color": config.get("background_color", [1.0, 1.0, 1.0]),
            "should_reset_materials": config.get("should_reset_materials", True),
            "use_lights": config.get("use_lights", True),
            "flatten_before_render": config.get("flatten_before_render", False),
        }

        listener.info(f"  USD: {usd_path}")
        listener.info(f"  Output: {output_dir}")
        listener.info(f"  Backend: {render_config['backend']}")
        listener.info(
            f"  Size: {render_config['image_width']}x{render_config['image_height']}"
        )
        listener.info(f"  Cameras: {render_config['cameras']}")

        context["usd_path"] = str(usd_path)
        context["output_dir"] = str(output_dir)
        context["render_config"] = render_config

        # Pass through prim_filters if present (same schema as build_dataset_usd)
        prim_filters = config.get("prim_filters")
        if prim_filters:
            context["prim_filters"] = prim_filters
            listener.info(f"  Prim filters: {prim_filters}")

        return context
