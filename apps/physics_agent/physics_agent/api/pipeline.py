# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API for Physics Agent.

This module provides the programmatic API for running multi-step pipelines.

Usage patterns:

1. **Full Input class** - Maximum control:
    ```python
    from physics_agent.api import PipelineInput, run_pipeline

    params = PipelineInput(config=Path("config.yaml"))
    result = run_pipeline(params)
    ```

2. **Convenience function** - Minimal usage:
    ```python
    from physics_agent.api import pipeline

    result = pipeline(Path("config.yaml"))
    ```
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import EventListener

logger = logging.getLogger(__name__)


@dataclass
class PipelineInput:
    """Input parameters for pipeline API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        session_id: Optional session ID to reuse existing session directory
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        verbose: Enable verbose output
        event_listener: Optional event listener for progress reporting
    """

    config: Path | dict[str, Any]
    skip_steps: list[str] = field(default_factory=list)
    only_steps: list[str] = field(default_factory=list)
    session_id: str | None = None
    resume: bool = False
    dry_run: bool = False
    clean: bool = False
    verbose: bool = False
    event_listener: EventListener | None = None

    def __post_init__(self):
        """Validate inputs."""
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not self.config.exists():
                raise FileNotFoundError(f"Config file not found: {self.config}")


@dataclass
class PipelineOutput:
    """Output results from pipeline API."""

    success: bool
    error: str | None = None
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    session_id: str | None = None
    working_dir: Path | None = None
    raw_result: dict[str, Any] | None = None


async def arun_pipeline(params: PipelineInput) -> PipelineOutput:
    """Execute a multi-step physics agent pipeline asynchronously.

    This is the core async implementation. The sync version delegates to this.

    A typical pipeline includes:
    1. optimize_usd: Optimize USD for rendering (optional)
    2. build_dataset_usd: Render USD prims to images
    3. build_dataset_prepare_dataset: Prepare dataset with prompts
    4. predict: Run VLM inference

    The pipeline automatically connects outputs from one step to inputs of the next.

    Args:
        params: Pipeline input parameters

    Returns:
        PipelineOutput with results or error information
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
            "workflow_type": "pipeline",
            "config_type": "dict" if isinstance(params.config, dict) else "file",
            "skip_steps": params.skip_steps,
            "only_steps": params.only_steps,
        },
    )

    listener.info("Starting pipeline via API")
    if isinstance(params.config, dict):
        listener.info("Using in-memory config dictionary")
    else:
        listener.info(f"Configuration file: {params.config}")

    if params.skip_steps:
        listener.info(f"Skip steps: {', '.join(params.skip_steps)}")
    if params.only_steps:
        listener.info(f"Only steps: {', '.join(params.only_steps)}")
    if params.resume:
        listener.info("Resume mode enabled")
    if params.clean:
        listener.info("Clean mode enabled (will delete working dir and output files)")

    if params.dry_run:
        listener.info("Dry run mode - showing execution plan only")
        return _dry_run_pipeline(params)

    try:
        from physics_agent.workflows import create_unified_pipeline_workflow

        listener.info("Creating unified pipeline workflow")
        workflow = create_unified_pipeline_workflow()

        # Prepare initial context
        initial_context: dict[str, Any] = {
            "skip_steps": params.skip_steps,
            "only_steps": params.only_steps,
            "resume": params.resume,
            "clean": params.clean,
            "verbose": params.verbose,
            "event_listener": listener,
        }

        # Add session_id if provided
        if params.session_id:
            initial_context["session_id"] = params.session_id
            listener.info(f"Using provided session ID: {params.session_id}")

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
        else:
            initial_context["config_path"] = str(params.config)

        listener.info("Running unified pipeline workflow")
        listener.event("workflow.executing", {"workflow_type": "pipeline"})

        result = await workflow.arun(initial_context)

        # Check if workflow returned nothing
        if not result:
            error_msg = "Pipeline workflow did not complete successfully"
            listener.error(error_msg)
            listener.event(
                "workflow.failed",
                {"workflow_type": "pipeline", "error": error_msg},
            )
            return PipelineOutput(
                success=False,
                error=error_msg,
                skipped_steps=params.skip_steps,
                raw_result=result,
            )

        # Check for workflow errors even if result exists
        if result.get("error") or result.get("workflow_terminated"):
            failed_task = result.get("failed_task", "unknown")
            error_msg = result.get("error", "Pipeline failed without error message")
            listener.error(f"Pipeline failed at task '{failed_task}': {error_msg}")
            listener.event(
                "workflow.failed",
                {
                    "workflow_type": "pipeline",
                    "error": error_msg,
                    "failed_task": failed_task,
                },
            )
            # Still extract partial results if available
            pipeline_results = result.get("pipeline_results", {})
            completed_steps = list(pipeline_results.keys())
            return PipelineOutput(
                success=False,
                error=error_msg,
                step_results=pipeline_results,
                completed_steps=completed_steps,
                skipped_steps=params.skip_steps,
                session_id=result.get("session_id"),
                working_dir=Path(wd) if (wd := result.get("working_dir")) else None,
                raw_result=result,
            )

        # Pipeline succeeded - extract results
        pipeline_results = result.get("pipeline_results", {})
        completed_steps = list(pipeline_results.keys())
        session_id = result.get("session_id")
        working_dir = result.get("working_dir")

        # Emit completion event
        listener.event(
            "workflow.completed",
            {
                "workflow_type": "pipeline",
                "completed_steps": completed_steps,
            },
        )
        listener.info("Pipeline completed successfully")

        return PipelineOutput(
            success=True,
            step_results=pipeline_results,
            completed_steps=completed_steps,
            skipped_steps=params.skip_steps,
            session_id=session_id,
            working_dir=Path(working_dir) if working_dir else None,
            raw_result=result,
        )

    except Exception as e:
        listener.error(f"Error running pipeline: {str(e)}")
        listener.event(
            "workflow.failed", {"workflow_type": "pipeline", "error": str(e)}
        )
        return PipelineOutput(
            success=False,
            error=str(e),
        )


def _dry_run_pipeline(params: PipelineInput) -> PipelineOutput:
    """Perform a dry run of the pipeline.

    Args:
        params: Pipeline input parameters

    Returns:
        PipelineOutput with execution plan information
    """
    import yaml

    try:
        # Load config
        if isinstance(params.config, dict):
            pipeline_config = params.config
        else:
            with open(params.config, encoding="utf-8") as f:
                pipeline_config = yaml.safe_load(f)

        from physics_agent.api.defaults import PIPELINE_STEP_NAMES

        # Detect unified config format
        steps_section = pipeline_config.get("steps", pipeline_config)

        planned_steps = []
        skipped_steps = []

        for step in PIPELINE_STEP_NAMES:
            if step not in steps_section:
                continue

            step_config = steps_section[step]

            # Check if enabled
            enabled = (
                step_config.get("enabled") if isinstance(step_config, dict) else True
            )
            if enabled is None:
                has_config = (
                    any(k != "enabled" for k in step_config.keys())
                    if isinstance(step_config, dict)
                    else bool(step_config)
                )
                enabled = has_config
            if not enabled:
                continue

            if params.skip_steps and step in params.skip_steps:
                skipped_steps.append(step)
            elif params.only_steps and step not in params.only_steps:
                skipped_steps.append(step)
            else:
                planned_steps.append(step)

        logger.info(f"Dry run - would execute steps: {planned_steps}")
        logger.info(f"Dry run - would skip steps: {skipped_steps}")

        return PipelineOutput(
            success=True,
            completed_steps=planned_steps,
            skipped_steps=skipped_steps,
        )

    except Exception as e:
        logger.error(f"Error during dry run: {str(e)}", exc_info=True)
        return PipelineOutput(
            success=False,
            error=str(e),
        )


def run_pipeline(params: PipelineInput) -> PipelineOutput:
    """Execute a multi-step physics agent pipeline synchronously.

    This is a wrapper around the async implementation.

    Args:
        params: Pipeline input parameters

    Returns:
        PipelineOutput with results or error information
    """
    return asyncio.run(arun_pipeline(params))


async def apipeline(
    config: Path | dict[str, Any],
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    session_id: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    clean: bool = False,
    event_listener: EventListener | None = None,
    verbose: bool = False,
) -> PipelineOutput:
    """Async convenience function for pipeline API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        session_id: Optional session ID to reuse existing session directory
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output

    Returns:
        PipelineOutput with results
    """
    params = PipelineInput(
        config=config,
        skip_steps=skip_steps or [],
        only_steps=only_steps or [],
        session_id=session_id,
        resume=resume,
        dry_run=dry_run,
        clean=clean,
        verbose=verbose,
        event_listener=event_listener,
    )
    return await arun_pipeline(params)


def pipeline(
    config: Path | dict[str, Any],
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    session_id: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    clean: bool = False,
    event_listener: EventListener | None = None,
    verbose: bool = False,
) -> PipelineOutput:
    """Sync convenience function for pipeline API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        session_id: Optional session ID to reuse existing session directory
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output

    Returns:
        PipelineOutput with results
    """
    return asyncio.run(
        apipeline(
            config,
            skip_steps,
            only_steps,
            session_id,
            resume,
            dry_run,
            clean,
            event_listener,
            verbose,
        )
    )
