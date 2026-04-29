# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Render the final textured USD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class RenderOutputTask(Task):
    """Render the final textured USD for visual verification.

    Context keys read:
        output_usd_paths (list[str]): From ApplyTexturesTask.
        render_config (dict): backend, image_width, image_height, etc.
        working_dir (str): Working directory.

    Context keys written:
        rendered_image_paths (list[str]): Paths to rendered images.
    """

    def __init__(self) -> None:
        self.name = "RenderOutput"
        self.description = "Render the final textured USD"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        output_usd_paths: list[str] = context.get("output_usd_paths", [])
        config: dict = context.get("render_config", {})
        working_dir = Path(context["working_dir"])

        if not output_usd_paths:
            logger.info("No output USDs to render")
            context["rendered_image_paths"] = []
            return context

        image_width = config.get("image_width", 1024)
        image_height = config.get("image_height", image_width)

        out_dir = working_dir / "renders"
        out_dir.mkdir(parents=True, exist_ok=True)

        from pxr import Usd
        from world_understanding.functions.graphics.render_nvcf import (
            render_all_cameras,
        )
        from world_understanding.utils.usd.material import (
            convert_custom_mdl_to_builtin,
        )

        rendered: list[str] = []

        for usd_path in output_usd_paths:
            logger.info("Rendering %s", usd_path)
            try:
                stage = Usd.Stage.Open(str(usd_path))
                if not stage:
                    logger.warning("Failed to open stage: %s", usd_path)
                    continue

                # Flatten for NVCF
                flat_layer = stage.Flatten()
                flat_stage = Usd.Stage.Open(flat_layer)
                convert_custom_mdl_to_builtin(flat_stage)

                results = render_all_cameras(
                    stage=flat_stage,
                    image_width=image_width,
                    image_height=image_height,
                )

                for i, result in enumerate(results):
                    images = result.get("images", [])
                    for j, img in enumerate(images):
                        out_name = f"render_{i}_{j}.png"
                        out_path = out_dir / out_name
                        img.save(str(out_path))
                        rendered.append(str(out_path))
                        logger.info("  Saved render: %s", out_path)

            except Exception:
                logger.exception("Failed to render %s", usd_path)

        context["rendered_image_paths"] = rendered
        logger.info("Rendered %d images", len(rendered))
        return context
