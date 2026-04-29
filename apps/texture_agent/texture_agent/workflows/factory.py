# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow factory functions for the texture agent pipeline."""

from __future__ import annotations

from typing import Any

from texture_agent.config.schema import STEP_ORDER
from texture_agent.tasks import (
    ApplyTexturesTask,
    BlendTexturesTask,
    DiscoverMaterialsTask,
    GeneratePromptsTask,
    GenerateTexturesTask,
    PrepareUVsTask,
    RenderMaterialPreviewsTask,
    RenderOutputTask,
)

# Map step names to task classes
_STEP_TASKS = {
    "prepare_uvs": PrepareUVsTask,
    "discover_materials": DiscoverMaterialsTask,
    "generate_prompts": GeneratePromptsTask,
    "render_previews": RenderMaterialPreviewsTask,
    "generate_textures": GenerateTexturesTask,
    "blend_textures": BlendTexturesTask,
    "apply_textures": ApplyTexturesTask,
    "render": RenderOutputTask,
}


def create_texture_pipeline_workflow(
    context: dict[str, Any],
    skip: list[str] | None = None,
    only: list[str] | None = None,
) -> list[Any]:
    """Create the ordered list of tasks for the texture pipeline.

    Args:
        context: Pipeline context (with 'steps' config).
        skip: Step names to skip.
        only: If set, run only these steps.

    Returns:
        List of instantiated Task objects in execution order.
    """
    steps_config = context.get("steps", {})
    skip = set(skip or [])
    only_set = set(only) if only else None

    tasks = []
    for step_name in STEP_ORDER:
        # Check enabled
        step_cfg = steps_config.get(step_name, {})
        if not step_cfg.get("enabled", True):
            continue

        # Apply skip/only filters
        if step_name in skip:
            continue
        if only_set and step_name not in only_set:
            continue

        task_cls = _STEP_TASKS.get(step_name)
        if task_cls:
            tasks.append(task_cls())

    return tasks


def run_pipeline(
    context: dict[str, Any],
    skip: list[str] | None = None,
    only: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the texture pipeline end-to-end.

    Args:
        context: Pipeline context from config_to_context().
        skip: Step names to skip.
        only: If set, run only these steps.
        dry_run: If True, print execution plan without running.

    Returns:
        Updated context dict with all outputs.
    """
    import logging

    logger = logging.getLogger(__name__)

    tasks = create_texture_pipeline_workflow(context, skip=skip, only=only)

    if dry_run:
        logger.info("Dry run -- execution plan:")
        for i, task in enumerate(tasks, 1):
            logger.info("  %d. %s: %s", i, task.name, task.description)
        return context

    logger.info("Running texture pipeline (%d steps)", len(tasks))
    for i, task in enumerate(tasks, 1):
        logger.info("[%d/%d] %s", i, len(tasks), task.name)
        context = task.run(context)
        logger.info("[%d/%d] %s complete", i, len(tasks), task.name)

    logger.info("Pipeline complete")
    return context
