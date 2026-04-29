# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from material_agent.api.types import APIResult, AssignmentStats, DownloadStats

logger = logging.getLogger(__name__)


@dataclass
class ApplyInput:
    """Input parameters for apply API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        input_usd_override: Optional path to override input USD from config
        predictions_override: Optional path to override predictions from config
        output_usd_override: Optional path to override output USD from config
        layer_only: Output layer only (not full stage)
        render_enabled: Enable rendering after apply
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    input_usd_override: Path | None = None
    predictions_override: Path | None = None
    output_usd_override: Path | None = None
    layer_only: bool = False
    render_enabled: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        # Handle config as either Path or dict
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not self.config.exists():
                raise FileNotFoundError(f"Config file not found: {self.config}")

        if self.input_usd_override:
            self.input_usd_override = Path(self.input_usd_override)

        if self.predictions_override:
            self.predictions_override = Path(self.predictions_override)

        if self.output_usd_override:
            self.output_usd_override = Path(self.output_usd_override)


@dataclass
class ApplyOutput(APIResult):
    """Output results from apply API."""

    output_usd_path: Path | None = None
    unique_materials: list[str] | None = None
    matched_materials: dict[str, list[Any]] | None = None
    resolved_materials: dict[str, str] | None = None
    materials_applied: dict[str, Any] | None = None
    assignment_stats: AssignmentStats | None = None
    download_stats: DownloadStats | None = None
    rendered_image_paths: list[Path] | None = None
    rendering_skipped: bool = True
    layer_only: bool = False
    raw_result: dict[str, Any] | None = None


async def arun_apply(params: ApplyInput) -> ApplyOutput:
    """Apply predicted materials to a USD file asynchronously.

    This is the core async implementation. The sync version delegates to this.
    This is equivalent to: pipeline --only apply

    Args:
        params: Apply input parameters

    Returns:
        ApplyOutput with results or error information
    """
    logger.info("Starting apply via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info(f"Configuration file: {params.config}")

    if params.input_usd_override:
        logger.info(f"Input USD override: {params.input_usd_override}")
    if params.predictions_override:
        logger.info(f"Predictions override: {params.predictions_override}")
    if params.output_usd_override:
        logger.info(f"Output USD override: {params.output_usd_override}")

    try:
        # Import the pipeline API to reuse logic
        from material_agent.api.pipeline import PipelineInput, arun_pipeline

        # Create pipeline params with only=apply
        pipeline_params = PipelineInput(
            config=params.config,
            skip_steps=[],
            only_steps=["apply"],
            resume=False,
            dry_run=False,
            clean=False,
            verbose=params.verbose,
        )

        # Run pipeline asynchronously
        pipeline_result = await arun_pipeline(pipeline_params)

        if pipeline_result.success:
            # Extract apply-specific results
            apply_result = pipeline_result.step_results.get("apply", {})

            # Convert assignment stats
            assignment_stats_dict = apply_result.get("assignment_stats", {})
            assignment_stats = (
                AssignmentStats(
                    materials_created=assignment_stats_dict.get("materials_created", 0),
                    materials_applied=assignment_stats_dict.get("materials_applied", 0),
                    total_prims=assignment_stats_dict.get("total_prims", 0),
                    failed=assignment_stats_dict.get("failed", 0),
                )
                if assignment_stats_dict
                else None
            )

            # Convert download stats
            download_stats_dict = apply_result.get("download_stats", {})
            download_stats = (
                DownloadStats(
                    found_local=download_stats_dict.get("found_local", 0),
                    downloaded=download_stats_dict.get("downloaded", 0),
                    failed=download_stats_dict.get("failed", 0),
                    skipped=download_stats_dict.get("skipped", 0),
                )
                if download_stats_dict
                else None
            )

            # Convert rendered images to Paths
            rendered_images = apply_result.get("rendered_image_paths", [])
            rendered_image_paths = (
                [Path(p) for p in rendered_images] if rendered_images else None
            )

            return ApplyOutput(
                success=True,
                output_usd_path=Path(apply_result["output_usd_path"])
                if apply_result.get("output_usd_path")
                else None,
                unique_materials=apply_result.get("unique_materials"),
                matched_materials=apply_result.get("matched_materials"),
                resolved_materials=apply_result.get("resolved_materials"),
                materials_applied=apply_result.get("materials_applied"),
                assignment_stats=assignment_stats,
                download_stats=download_stats,
                rendered_image_paths=rendered_image_paths,
                rendering_skipped=apply_result.get("rendering_skipped", True),
                layer_only=apply_result.get("layer_only", False),
                raw_result=apply_result,
            )
        else:
            return ApplyOutput(
                success=False,
                error=pipeline_result.error,
            )

    except Exception as e:
        logger.error(f"Error running apply: {str(e)}", exc_info=True)
        return ApplyOutput(
            success=False,
            error=str(e),
        )


def run_apply(params: ApplyInput) -> ApplyOutput:
    """Apply predicted materials to a USD file synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Apply input parameters

    Returns:
        ApplyOutput with results or error information
    """
    return asyncio.run(arun_apply(params))


async def aapply(
    config: Path | dict[str, Any],
    input_usd_override: Path | None = None,
    predictions_override: Path | None = None,
    output_usd_override: Path | None = None,
    layer_only: bool = False,
    render_enabled: bool = False,
    verbose: bool = False,
) -> ApplyOutput:
    """Async convenience function for apply API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        input_usd_override: Optional path to override input USD from config
        predictions_override: Optional path to override predictions from config
        output_usd_override: Optional path to override output USD from config
        layer_only: Output layer only (not full stage)
        render_enabled: Enable rendering after apply
        verbose: Enable verbose output

    Returns:
        ApplyOutput with results
    """
    params = ApplyInput(
        config=config,
        input_usd_override=input_usd_override,
        predictions_override=predictions_override,
        output_usd_override=output_usd_override,
        layer_only=layer_only,
        render_enabled=render_enabled,
        verbose=verbose,
    )
    return await arun_apply(params)


def apply(
    config: Path | dict[str, Any],
    input_usd_override: Path | None = None,
    predictions_override: Path | None = None,
    output_usd_override: Path | None = None,
    layer_only: bool = False,
    render_enabled: bool = False,
    verbose: bool = False,
) -> ApplyOutput:
    """Sync convenience function for apply API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        input_usd_override: Optional path to override input USD from config
        predictions_override: Optional path to override predictions from config
        output_usd_override: Optional path to override output USD from config
        layer_only: Output layer only (not full stage)
        render_enabled: Enable rendering after apply
        verbose: Enable verbose output

    Returns:
        ApplyOutput with results
    """
    return asyncio.run(
        aapply(
            config,
            input_usd_override,
            predictions_override,
            output_usd_override,
            layer_only,
            render_enabled,
            verbose,
        )
    )
