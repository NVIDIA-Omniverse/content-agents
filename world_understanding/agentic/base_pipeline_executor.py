# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base pipeline executor with common checkpoint/resume/clean logic.

This module provides a base class for pipeline executors that handles:
- State persistence (checkpoint/resume)
- Clean directory operations with safety checks
- Step filtering (skip/only)
- Context validation
- Common execution loop structure

Agent-specific executors inherit from this base and implement step execution logic.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)

# Get a tracer for pipeline operations
_tracer = trace.get_tracer(__name__)


class PathEncoder(json.JSONEncoder):
    """JSON encoder that handles Path objects.

    This encoder converts Path objects to strings when serializing to JSON,
    making it easier to persist pipeline state with file paths.
    """

    def default(self, obj: Any) -> Any:
        """Convert Path objects to strings."""
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class BasePipelineExecutor(Task):
    """Base class for pipeline executors with common functionality.

    This class provides reusable infrastructure for multi-step pipeline execution:
    - **Checkpoint/Resume**: Save progress after each step, resume from failures
    - **Clean Operations**: Safe directory cleanup with validation
    - **Step Filtering**: Support for skip_steps and only_steps
    - **State Tracking**: Track completed steps, failures, and outputs

    Subclasses must implement:
    - `_execute_step()`: Execute a single pipeline step
    - `_get_step_list_key()`: Return context key for step list
    - `_get_required_context_keys()`: Return required context keys
    - `_get_state_file()`: Return path to state file

    Example:
        >>> class MyExecutor(BasePipelineExecutor):
        ...     def _execute_step(self, step_name, context, object_store):
        ...         # Execute step-specific logic
        ...         return {"status": "completed"}
        ...
        ...     def _get_step_list_key(self):
        ...         return "steps_to_run"
        ...
        ...     def _get_required_context_keys(self):
        ...         return ["steps_to_run", "config"]
        ...
        ...     def _get_state_file(self, context):
        ...         return context["working_dir"] / ".pipeline_state.json"
    """

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Execute pipeline steps in sequence.

        This method provides the common execution loop:
        1. Validate required context keys
        2. Apply step filtering (skip/only)
        3. Clean directories (if requested)
        4. Initialize or load pipeline state
        5. Execute each step in sequence
        6. Save checkpoint after each step
        7. Update context with results

        Args:
            context: Workflow context with configuration
            object_store: Optional object store for workflow execution

        Returns:
            Updated context with pipeline results

        Raises:
            ValueError: If required context keys are missing or validation fails
            RuntimeError: If a pipeline step fails
        """
        # Get pipeline name for tracing
        pipeline_name = context.get("project_name", self.__class__.__name__)

        # Start the main pipeline span
        with _tracer.start_as_current_span("pipeline.run") as pipeline_span:
            # Set pipeline-level attributes
            pipeline_span.set_attribute("maa.pipeline.name", pipeline_name)
            pipeline_span.set_attribute(
                "maa.pipeline.session_id", context.get("session_id", "unknown")
            )

            # 1. Validate required context keys
            self._validate_context(context)

            # 2. Get step list and apply filtering
            step_list_key = self._get_step_list_key()
            steps = context.get(step_list_key, [])
            if not steps:
                raise ValueError(
                    f"No steps to run in pipeline ({step_list_key} is empty)"
                )

            steps = self._apply_step_filtering(steps, context)
            logger.info(f"Pipeline will execute {len(steps)} steps: {steps}")

            # Set total steps attribute after filtering
            pipeline_span.set_attribute("maa.pipeline.total_steps", len(steps))
            pipeline_span.set_attribute("maa.pipeline.steps", ",".join(steps))

            # 3. Clean directories if requested
            if context.get("clean", False):
                self._clean_directories(context)

            # 4. Initialize or load pipeline state
            resume = context.get("resume", False)
            pipeline_state = self._initialize_pipeline_state(context, resume)

            # 5. Execute each step
            state_file = self._get_state_file(context)
            self._log_pipeline_started(context, steps)

            for i, step_name in enumerate(steps, 1):
                # Skip if resuming and step already completed
                if resume and step_name in pipeline_state["completed_steps"]:
                    logger.info(
                        f"[{i}/{len(steps)}] Skipping completed step: {step_name}"
                    )
                    continue

                # Execute step with tracing
                logger.info(f"\n[{i}/{len(steps)}] Executing step: {step_name}")
                pipeline_state["current_step"] = step_name

                self._execute_step_with_tracing(
                    step_name=step_name,
                    step_index=i - 1,  # 0-based index for attributes
                    total_steps=len(steps),
                    context=context,
                    object_store=object_store,
                    pipeline_state=pipeline_state,
                    state_file=state_file,
                )

            # 6. Mark pipeline as completed
            pipeline_state["current_step"] = None
            self._save_checkpoint(pipeline_state, state_file)

            # 7. Update context with results
            self._log_pipeline_completed(context, pipeline_state)
            self._update_context_with_results(context, pipeline_state)

            # Set final pipeline status
            completed_count = len(pipeline_state["completed_steps"])
            failed_count = len(pipeline_state["failed_steps"])
            pipeline_span.set_attribute("maa.pipeline.completed_steps", completed_count)
            pipeline_span.set_attribute("maa.pipeline.failed_steps", failed_count)
            pipeline_span.set_attribute("maa.pipeline.status", "completed")

            return context

    def _execute_step_with_tracing(
        self,
        step_name: str,
        step_index: int,
        total_steps: int,
        context: dict[str, Any],
        object_store: ObjectStore | None,
        pipeline_state: dict[str, Any],
        state_file: Path,
    ) -> dict[str, Any]:
        """Execute a single pipeline step with OpenTelemetry tracing.

        Args:
            step_name: Name of the step to execute
            step_index: 0-based index of the step in the pipeline
            total_steps: Total number of steps in the pipeline
            context: Workflow context
            object_store: Optional object store
            pipeline_state: Current pipeline state dictionary
            state_file: Path to the state file for checkpointing

        Returns:
            Dictionary with step results/outputs

        Raises:
            RuntimeError: If the step fails
        """
        with _tracer.start_as_current_span(f"pipeline.step.{step_name}") as step_span:
            # Set step-level attributes
            step_span.set_attribute("maa.pipeline.step.name", step_name)
            step_span.set_attribute("maa.pipeline.step.index", step_index)
            step_span.set_attribute("maa.pipeline.step.total", total_steps)

            try:
                step_result = self._execute_step(step_name, context, object_store)

                # Track completion
                pipeline_state["completed_steps"].append(step_name)
                pipeline_state["step_outputs"][step_name] = step_result

                # Save checkpoint
                self._save_checkpoint(pipeline_state, state_file)

                logger.info(f"Step '{step_name}' completed successfully")

                # Set success attributes
                step_span.set_attribute("maa.pipeline.step.status", "completed")

                return step_result

            except Exception as e:
                # Track failure
                pipeline_state["failed_steps"].append(step_name)
                self._save_checkpoint(pipeline_state, state_file)

                logger.error(f"Step '{step_name}' failed: {e}")

                # Set failure attributes and record exception
                step_span.set_attribute("maa.pipeline.step.status", "failed")
                step_span.record_exception(e)
                step_span.set_status(Status(StatusCode.ERROR, str(e)))

                raise RuntimeError(f"Pipeline failed at step '{step_name}': {e}") from e

    # ========== Abstract Methods (must implement in subclass) ==========

    def _execute_step(
        self,
        step_name: str,
        context: dict[str, Any],
        object_store: ObjectStore | None,
    ) -> dict[str, Any]:
        """Execute a single pipeline step (agent-specific logic).

        Args:
            step_name: Name of the step to execute
            context: Workflow context
            object_store: Optional object store

        Returns:
            Dictionary with step results/outputs

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _execute_step()"
        )

    def _get_step_list_key(self) -> str:
        """Return context key for step list.

        Returns:
            Context key name (e.g., 'steps_to_run' or 'enabled_steps')

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_step_list_key()"
        )

    def _get_required_context_keys(self) -> list[str]:
        """Return list of required context keys for validation.

        Returns:
            List of required context key names

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_required_context_keys()"
        )

    def _get_state_file(self, context: dict[str, Any]) -> Path:
        """Return path to pipeline state file.

        Args:
            context: Workflow context

        Returns:
            Path to state file (typically .pipeline_state.json)

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_state_file()"
        )

    # ========== Optional Override Methods (have default implementations) ==========

    def _update_context_with_results(
        self, context: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> None:
        """Update context with pipeline results.

        Default implementation stores results in 'pipeline_results' key.
        Subclasses can override to customize result storage.

        Args:
            context: Workflow context to update
            pipeline_state: Final pipeline state
        """
        context["pipeline_results"] = pipeline_state["step_outputs"]
        context["pipeline_state"] = "completed"

    # ========== Common Methods (implemented in base class) ==========

    def _get_state_lock_file(self, state_file: Path) -> Path:
        """Get path to lock file for state file.

        Args:
            state_file: Path to state file

        Returns:
            Path to corresponding lock file
        """
        return state_file.with_suffix(".lock")

    def _validate_context(self, context: dict[str, Any]) -> None:
        """Validate that required context keys are present.

        Args:
            context: Workflow context to validate

        Raises:
            ValueError: If required context keys are missing
        """
        required_keys = self._get_required_context_keys()
        missing_keys = [key for key in required_keys if key not in context]

        if missing_keys:
            raise ValueError(
                f"Required context keys missing: {missing_keys}. "
                f"Required: {required_keys}"
            )

    def _apply_step_filtering(
        self, steps: list[str], context: dict[str, Any]
    ) -> list[str]:
        """Apply skip_steps and only_steps filtering to step list.

        Args:
            steps: Original list of steps
            context: Workflow context with optional skip_steps/only_steps

        Returns:
            Filtered list of steps
        """
        skip_steps = context.get("skip_steps", [])
        only_steps = context.get("only_steps", [])

        # Apply only_steps filter first (if provided)
        if only_steps:
            steps = [step for step in steps if step in only_steps]
            logger.debug(f"Filtered to only steps: {steps}")

        # Apply skip_steps filter
        if skip_steps:
            steps = [step for step in steps if step not in skip_steps]
            logger.debug(f"Skipped steps: {skip_steps}")

        return steps

    def _clean_directories(self, context: dict[str, Any]) -> None:
        """Clean working directory and output files with safety checks.

        This performs a clean operation to remove previous pipeline outputs.
        Includes safety checks to prevent accidental deletion of important paths.

        Args:
            context: Workflow context with working_dir

        Raises:
            ValueError: If working_dir path is unsafe to delete
        """
        working_dir = context.get("working_dir")
        if not working_dir:
            logger.warning("No working_dir in context, skipping clean operation")
            return

        working_dir_path = Path(working_dir)

        # Safety checks before deletion
        if working_dir_path == Path.home() or working_dir_path == Path("/"):
            raise ValueError(
                f"Refusing to delete potentially dangerous path: {working_dir_path}"
            )

        # Check for minimum depth from root
        if len(working_dir_path.parts) < 2:
            raise ValueError(f"Working directory path too shallow: {working_dir_path}")

        # Clean working directory
        if working_dir_path.exists():
            logger.info(f"Cleaning working directory: {working_dir_path}")
            shutil.rmtree(working_dir_path)
            logger.info("Working directory cleaned successfully")

    def _initialize_pipeline_state(
        self, context: dict[str, Any], resume: bool = False
    ) -> dict[str, Any]:
        """Initialize or load pipeline state with file locking.

        Args:
            context: Workflow context
            resume: If True, attempt to load existing state

        Returns:
            Pipeline state dictionary

        Raises:
            RuntimeError: If unable to acquire lock within timeout period
        """
        state_file = self._get_state_file(context)

        # Try to load existing state if resuming
        if resume and state_file.exists():
            lock_file = self._get_state_lock_file(state_file)

            try:
                # Timeout ensures we never block indefinitely
                with FileLock(str(lock_file), timeout=30):
                    logger.info(f"Resuming pipeline from checkpoint: {state_file}")
                    with open(state_file, encoding="utf-8") as f:
                        loaded_state: dict[str, Any] = json.load(f)

                    logger.info(
                        f"Resumed with {len(loaded_state.get('completed_steps', []))} "
                        "completed steps"
                    )
                    return loaded_state

            except Timeout as e:
                # After 30 seconds, give up gracefully
                logger.exception(
                    f"Timeout acquiring lock for {state_file}. "
                    "Another process may be reading/writing the file."
                )
                raise RuntimeError(
                    f"Failed to resume - could not acquire lock on {state_file}"
                ) from e
        # Initialize new state
        return {
            "session_id": context.get("session_id", "unknown"),
            "project_name": context.get("project_name", "unknown"),
            "completed_steps": [],
            "failed_steps": [],
            "step_outputs": {},
            "current_step": None,
        }

    def _save_checkpoint(
        self, pipeline_state: dict[str, Any], state_file: Path
    ) -> None:
        """Save pipeline state checkpoint to disk with file locking.

        Args:
            pipeline_state: Current pipeline state
            state_file: Path to state file

        Raises:
            RuntimeError: If unable to acquire lock within timeout period
        """
        state_file.parent.mkdir(parents=True, exist_ok=True)

        lock_file = self._get_state_lock_file(state_file)

        try:
            # Timeout ensures we never block indefinitely
            with FileLock(str(lock_file), timeout=30):
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(pipeline_state, f, indent=2, cls=PathEncoder)

                logger.debug(f"Checkpoint saved to: {state_file}")

        except Timeout as e:
            # After 30 seconds, give up gracefully
            logger.exception(
                f"Timeout acquiring lock for {state_file}. "
                "Another process may be accessing this session."
            )
            raise RuntimeError(
                f"Could not save checkpoint to {state_file} - lock timeout. "
                "Ensure no other pipelines are using the same session_id."
            ) from e

    def _log_pipeline_started(self, context: dict[str, Any], steps: list[str]) -> None:
        """Log pipeline start banner.

        Args:
            context: Workflow context
            steps: List of steps to execute
        """
        logger.info("\n" + "=" * 80)
        logger.info(
            f"Starting pipeline: {context.get('project_name', 'unknown')} "
            f"(session: {context.get('session_id', 'unknown')})"
        )
        logger.info(f"Steps to execute: {', '.join(steps)}")
        logger.info("=" * 80 + "\n")

    def _log_pipeline_completed(
        self, context: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> None:
        """Log pipeline completion banner.

        Args:
            context: Workflow context
            pipeline_state: Final pipeline state
        """
        completed = len(pipeline_state["completed_steps"])
        failed = len(pipeline_state["failed_steps"])

        logger.info("\n" + "=" * 80)
        logger.info(
            f"Pipeline completed: {context.get('project_name', 'unknown')} "
            f"(session: {context.get('session_id', 'unknown')})"
        )
        logger.info(f"Completed steps: {completed}")
        if failed > 0:
            logger.warning(f"Failed steps: {failed}")
        logger.info("=" * 80 + "\n")
