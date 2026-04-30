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


def _valid_steps_message() -> str:
    return ", ".join(STEP_ORDER)


def _normalize_step_filter(
    values: list[str] | None,
    option_name: str,
) -> set[str]:
    if not values:
        return set()

    normalized: set[str] = set()
    unknown: list[str] = []
    for raw_value in values:
        step_name = raw_value.strip()
        if not step_name:
            raise ValueError(
                f"{option_name} contains an empty step name. "
                f"Valid steps: {_valid_steps_message()}"
            )
        if step_name not in STEP_ORDER:
            unknown.append(step_name)
            continue
        normalized.add(step_name)

    if unknown:
        invalid = ", ".join(repr(step_name) for step_name in unknown)
        raise ValueError(
            f"Invalid {option_name} step name(s): {invalid}. "
            f"Valid steps: {_valid_steps_message()}"
        )

    return normalized


def _normalize_step_filters(
    skip: list[str] | None,
    only: list[str] | None,
) -> tuple[set[str], set[str] | None]:
    skip_set = _normalize_step_filter(skip, "--skip")
    only_set = _normalize_step_filter(only, "--only")

    if skip_set and only_set:
        raise ValueError("--skip and --only cannot be used together; choose one.")

    return skip_set, only_set or None


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
    skip_set, only_set = _normalize_step_filters(skip, only)

    tasks = []
    for step_name in STEP_ORDER:
        # Check enabled
        step_cfg = steps_config.get(step_name, {})
        if not step_cfg.get("enabled", True):
            continue

        # Apply skip/only filters
        if step_name in skip_set:
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
