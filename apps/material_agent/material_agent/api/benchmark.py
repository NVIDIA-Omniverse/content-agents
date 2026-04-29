# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import EventListener

from material_agent.api.types import APIResult, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkInput:
    """Input parameters for benchmark API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        dataset_override: Optional path to override dataset from config
        output_dir_override: Optional path to override output directory from config
        resume: Resume from existing predictions.jsonl
        stream_predictions: Append predictions as they are produced
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    dataset_override: Path | None = None
    output_dir_override: Path | None = None
    resume: bool = False
    stream_predictions: bool = True
    event_listener: EventListener | None = None
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        # Handle config as either Path or dict
        if isinstance(self.config, dict):
            # Config is provided as dictionary (in-memory)
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            # Config is provided as file path
            self.config = Path(self.config)
            if not self.config.exists():
                raise FileNotFoundError(f"Config file not found: {self.config}")

        if self.dataset_override:
            self.dataset_override = Path(self.dataset_override)

        if self.output_dir_override:
            self.output_dir_override = Path(self.output_dir_override)


@dataclass
class BenchmarkOutput(APIResult):
    """Output results from benchmark API."""

    metrics: MetricsResult | None = None
    evaluation_path: Path | None = None
    predictions_path: Path | None = None
    raw_result: dict[str, Any] | None = None


async def arun_benchmark(params: BenchmarkInput) -> BenchmarkOutput:
    """Run benchmark evaluation asynchronously.

    This is the core async implementation. The sync version delegates to this.

    Args:
        params: Benchmark input parameters

    Returns:
        BenchmarkOutput with results or error information
    """
    # Get or create event listener
    listener = params.event_listener
    if listener is None:
        from world_understanding.agentic.events import create_default_listener

        listener = create_default_listener(verbose=params.verbose)

    # Emit workflow started event
    listener.event(
        "workflow.started",
        {
            "workflow_type": "benchmark",
            "config_type": "dict" if isinstance(params.config, dict) else "file",
        },
    )

    listener.info("Starting benchmark via API")
    if isinstance(params.config, dict):
        listener.info("Using in-memory config dictionary")
    else:
        listener.info(f"Configuration file: {params.config}")

    if params.dataset_override:
        listener.info(f"Dataset override: {params.dataset_override}")
    if params.output_dir_override:
        listener.info(f"Output directory override: {params.output_dir_override}")

    try:
        # Import workflow factory
        from material_agent.workflows.factory import (
            create_benchmark_workflow_from_config,
        )

        # Apply defaults if using dict config
        config_to_use = params.config
        if isinstance(params.config, dict):
            from material_agent.api.defaults import get_benchmark_config_with_defaults

            config_to_use = get_benchmark_config_with_defaults(params.config)
            logger.info("Applied default values to config dictionary")

        # Build initial context
        initial_context: dict[str, Any] = {
            "dataset_override": str(params.dataset_override)
            if params.dataset_override
            else None,
            "output_dir_override": str(params.output_dir_override)
            if params.output_dir_override
            else None,
            "resume": params.resume,
            "stream_predictions": params.stream_predictions,
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(config_to_use, dict):
            initial_context["config_dict"] = config_to_use
        else:
            initial_context["config_path"] = str(config_to_use)

        # Create workflow
        listener.info("Creating config-driven benchmark workflow")
        workflow = create_benchmark_workflow_from_config()

        # Run the benchmark workflow asynchronously
        listener.info("Running benchmark workflow...")
        listener.event("workflow.executing", {"workflow_type": "benchmark"})

        result = await workflow.arun(initial_context)

        # Extract metrics from workflow result
        metrics_dict = result.get("metrics") if result else None

        if metrics_dict:
            metrics = MetricsResult.from_dict(metrics_dict)

            # Emit completion event
            listener.event(
                "workflow.completed",
                {
                    "workflow_type": "benchmark",
                    "metrics": metrics.to_dict(),
                    "evaluation_path": result.get("evaluation_path"),
                    "predictions_path": result.get("predictions_path"),
                },
            )
            listener.info("Benchmark completed successfully")

            return BenchmarkOutput(
                success=True,
                metrics=metrics,
                evaluation_path=Path(result["evaluation_path"])
                if result.get("evaluation_path")
                else None,
                predictions_path=Path(result["predictions_path"])
                if result.get("predictions_path")
                else None,
                raw_result=result,
            )
        else:
            error_msg = "Benchmark workflow completed but returned no metrics"
            listener.error(error_msg)
            listener.event(
                "workflow.failed", {"workflow_type": "benchmark", "error": error_msg}
            )
            return BenchmarkOutput(
                success=False,
                error=error_msg,
            )

    except Exception as e:
        listener.error(f"Error running benchmark: {str(e)}")
        listener.event(
            "workflow.failed", {"workflow_type": "benchmark", "error": str(e)}
        )
        return BenchmarkOutput(
            success=False,
            error=str(e),
        )


def run_benchmark(params: BenchmarkInput) -> BenchmarkOutput:
    """Run benchmark evaluation synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Benchmark input parameters

    Returns:
        BenchmarkOutput with results or error information
    """
    return asyncio.run(arun_benchmark(params))


async def abenchmark(
    config: Path | dict[str, Any],
    dataset_override: Path | None = None,
    output_dir_override: Path | None = None,
    resume: bool = False,
    stream_predictions: bool = True,
    event_listener: EventListener | None = None,
    verbose: bool = False,
) -> BenchmarkOutput:
    """Async convenience function for benchmark API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        dataset_override: Optional path to override dataset from config
        output_dir_override: Optional path to override output directory from config
        resume: Resume from existing predictions.jsonl
        stream_predictions: Append predictions as they are produced
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output

    Returns:
        BenchmarkOutput with results
    """
    params = BenchmarkInput(
        config=config,
        dataset_override=dataset_override,
        output_dir_override=output_dir_override,
        resume=resume,
        stream_predictions=stream_predictions,
        event_listener=event_listener,
        verbose=verbose,
    )
    return await arun_benchmark(params)


def benchmark(
    config: Path | dict[str, Any],
    dataset_override: Path | None = None,
    output_dir_override: Path | None = None,
    resume: bool = False,
    stream_predictions: bool = True,
    event_listener: EventListener | None = None,
    verbose: bool = False,
) -> BenchmarkOutput:
    """Sync convenience function for benchmark API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        dataset_override: Optional path to override dataset from config
        output_dir_override: Optional path to override output directory from config
        resume: Resume from existing predictions.jsonl
        stream_predictions: Append predictions as they are produced
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output

    Returns:
        BenchmarkOutput with results
    """
    return asyncio.run(
        abenchmark(
            config,
            dataset_override,
            output_dir_override,
            resume,
            stream_predictions,
            event_listener,
            verbose,
        )
    )
