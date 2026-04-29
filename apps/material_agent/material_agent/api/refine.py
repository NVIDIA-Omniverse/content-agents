# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refine API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from material_agent.api.types import APIResult

logger = logging.getLogger(__name__)


@dataclass
class RefineInput:
    """Input parameters for refine API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        max_iterations_override: Override maximum iterations from config
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    max_iterations_override: int | None = None
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


@dataclass
class IterationResult:
    """Result from a single iteration."""

    iteration: int
    judge_score: float | None
    continue_iteration: bool
    materials_applied_count: int = 0
    prims_with_materials: int = 0


@dataclass
class RefineOutput(APIResult):
    """Output results from refine API."""

    iteration_count: int = 0
    final_output_path: Path | None = None
    final_judge_score: float | None = None
    termination_reason: str = "unknown"
    iteration_results: list[IterationResult] = field(default_factory=list)
    all_iteration_outputs: list[Path] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


async def arun_refine(params: RefineInput) -> RefineOutput:
    """Refine materials on USD with VLM-based iterative refinement.

    This command executes a predict-apply-judge loop repeatedly until the judge
    approves the results or maximum iterations is reached. It uses VLM to predict
    materials, applies them to USD, renders the result, and has a VLM judge evaluate
    quality by comparing against reference images.

    The configuration file must specify:
    - dataset: Path to the dataset JSONL file
    - input_usd_path: Path to the input USD file
    - output_usd_path: Path for the final output (optional)
    - iteration: Iteration settings (max_iterations, save_intermediate, etc.)
    - judge: Judge configuration (reference_images, vlm settings, etc.)

    Args:
        params: Refine input parameters

    Returns:
        RefineOutput with results or error information
    """
    logger.info("Starting material refinement via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info(f"Configuration file: {params.config}")

    if params.max_iterations_override:
        logger.info(f"Max iterations override: {params.max_iterations_override}")

    try:
        # Import workflow factory
        from material_agent.workflows.factory import (
            create_iterative_apply_workflow_from_config,
        )

        # Create workflow
        logger.info("Creating iterative apply workflow...")
        workflow = create_iterative_apply_workflow_from_config()

        # Prepare initial context with config and overrides
        initial_context: dict[str, Any] = {
            "max_iterations_override": params.max_iterations_override,
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
        else:
            initial_context["config_path"] = str(params.config)

        # Run the workflow
        logger.info("Running material refinement with iterative loop...")
        result = await workflow.arun(initial_context=initial_context)

        # Check if workflow was successful
        if result.get("iteration_count", 0) > 0:
            iteration_count = result.get("iteration_count", 0)
            iteration_results_raw = result.get("iteration_results", [])
            final_iteration = result.get("final_iteration", {})
            termination_reason = result.get("termination_reason", "unknown")
            all_outputs = result.get("all_iteration_outputs", [])
            final_output_path = result.get("final_output_path")

            # Convert iteration results to structured format
            iteration_results = [
                IterationResult(
                    iteration=item.get("iteration", 0),
                    judge_score=item.get("judge_score"),
                    continue_iteration=item.get("continue_iteration", False),
                    materials_applied_count=item.get("materials_applied_count", 0),
                    prims_with_materials=item.get("prims_with_materials", 0),
                )
                for item in iteration_results_raw
            ]

            logger.info(
                f"Material refinement completed after {iteration_count} iterations"
            )

            return RefineOutput(
                success=True,
                iteration_count=iteration_count,
                final_output_path=Path(final_output_path)
                if final_output_path
                else None,
                final_judge_score=final_iteration.get("judge_score"),
                termination_reason=termination_reason,
                iteration_results=iteration_results,
                all_iteration_outputs=[Path(p) for p in all_outputs],
                raw_result=result,
            )
        else:
            error_msg = "Material refinement workflow did not complete successfully"
            logger.error(error_msg)
            return RefineOutput(
                success=False,
                error=error_msg,
            )

    except Exception as e:
        logger.error(f"Error during material refinement: {str(e)}", exc_info=True)
        return RefineOutput(
            success=False,
            error=str(e),
        )


def run_refine(params: RefineInput) -> RefineOutput:
    """Run iterative refinement synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Refine input parameters

    Returns:
        RefineOutput with results or error information
    """
    return asyncio.run(arun_refine(params))


async def arefine(
    config: Path | dict[str, Any],
    max_iterations_override: int | None = None,
    verbose: bool = False,
) -> RefineOutput:
    """Async convenience function for refine API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        max_iterations_override: Override max iterations from config
        verbose: Enable verbose output

    Returns:
        RefineOutput with results
    """
    params = RefineInput(
        config=config,
        max_iterations_override=max_iterations_override,
        verbose=verbose,
    )
    return await arun_refine(params)


def refine(
    config: Path | dict[str, Any],
    max_iterations_override: int | None = None,
    verbose: bool = False,
) -> RefineOutput:
    """Sync convenience function for refine API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        max_iterations_override: Override max iterations from config
        verbose: Enable verbose output

    Returns:
        RefineOutput with results
    """
    return asyncio.run(arefine(config, max_iterations_override, verbose))
