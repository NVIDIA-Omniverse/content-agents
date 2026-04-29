# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from material_agent.api.types import APIResult, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluateInput:
    """Input parameters for evaluate API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        predictions_override: Optional path to override predictions from config
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    predictions_override: Path | None = None
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

        if self.predictions_override:
            self.predictions_override = Path(self.predictions_override)
            if not self.predictions_override.exists():
                raise FileNotFoundError(
                    f"Predictions file not found: {self.predictions_override}"
                )


@dataclass
class EvaluateOutput(APIResult):
    """Output results from evaluate API."""

    metrics: MetricsResult | None = None
    evaluation_path: Path | None = None
    html_report_path: Path | None = None
    raw_result: dict[str, Any] | None = None


async def arun_evaluate(params: EvaluateInput) -> EvaluateOutput:
    """Evaluate existing predictions using an LLM judge.

    This command loads an evaluation configuration file and evaluates predictions
    against ground truth using the configured LLM judge. It calculates
    metrics including Functional Correctness Score (FCS) and success rate.

    Args:
        params: Evaluate input parameters

    Returns:
        EvaluateOutput with results or error information
    """
    logger.info("Starting evaluate via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info(f"Configuration file: {params.config}")

    if params.predictions_override:
        logger.info(f"Predictions override: {params.predictions_override}")

    try:
        # Import workflow factory
        from material_agent.workflows import create_evaluation_workflow_from_config

        # Create config-driven evaluation workflow
        logger.info("Creating config-driven evaluation workflow...")
        workflow = create_evaluation_workflow_from_config()

        # Prepare initial context
        initial_context: dict[str, Any] = {
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
        else:
            initial_context["config_path"] = str(params.config)

        # Add predictions override if provided
        if params.predictions_override:
            initial_context["predictions_path"] = str(params.predictions_override)

        # Run the evaluation workflow
        logger.info("Running evaluation...")
        result = await workflow.arun(initial_context=initial_context)

        # Check if evaluation was successful
        if result.get("evaluation_complete"):
            metrics_dict = result.get("metrics", {})
            evaluation_path = result.get("evaluation_path")
            html_report_path = result.get("html_report_path")

            logger.info("Evaluation completed successfully")

            metrics = MetricsResult.from_dict(metrics_dict)

            return EvaluateOutput(
                success=True,
                metrics=metrics,
                evaluation_path=Path(evaluation_path) if evaluation_path else None,
                html_report_path=Path(html_report_path) if html_report_path else None,
                raw_result=result,
            )
        else:
            error_msg = "Evaluation workflow did not complete successfully"
            logger.error(error_msg)
            return EvaluateOutput(
                success=False,
                error=error_msg,
            )

    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}", exc_info=True)
        return EvaluateOutput(
            success=False,
            error=str(e),
        )


def run_evaluate(params: EvaluateInput) -> EvaluateOutput:
    """Evaluate existing predictions synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Evaluate input parameters

    Returns:
        EvaluateOutput with results or error information
    """
    return asyncio.run(arun_evaluate(params))


async def aevaluate(
    config: Path | dict[str, Any],
    predictions_override: Path | None = None,
    verbose: bool = False,
) -> EvaluateOutput:
    """Async convenience function for evaluate API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        predictions_override: Optional path to override predictions from config
        verbose: Enable verbose output

    Returns:
        EvaluateOutput with results
    """
    params = EvaluateInput(
        config=config,
        predictions_override=predictions_override,
        verbose=verbose,
    )
    return await arun_evaluate(params)


def evaluate(
    config: Path | dict[str, Any],
    predictions_override: Path | None = None,
    verbose: bool = False,
) -> EvaluateOutput:
    """Sync convenience function for evaluate API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        predictions_override: Optional path to override predictions from config
        verbose: Enable verbose output

    Returns:
        EvaluateOutput with results
    """
    return asyncio.run(aevaluate(config, predictions_override, verbose))
