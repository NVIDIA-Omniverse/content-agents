# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified pipeline executor task that works with auto-wired step configs.

This executor works with step configs that have already been prepared by
UnifiedPipelineConfigTask, so it doesn't need to create temporary config files
or load configs again.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.base_pipeline_executor import BasePipelineExecutor
from world_understanding.agentic.events import get_listener

logger = logging.getLogger(__name__)


def _raise_if_cancelled(
    context: dict[str, Any], listener: Any, step_name: str | None = None
) -> None:
    """Raise ``CancelledError`` when the caller requests cancellation."""
    cancel_checker = context.get("cancel_checker")
    if not callable(cancel_checker):
        return

    if cancel_checker():
        cancelled_step = step_name or context.get("current_step") or "pipeline"
        event_payload = {
            "step_name": cancelled_step,
            "message": "Pipeline cancellation requested",
        }
        event_listener = context.get("event_listener")
        if event_listener:
            event_listener.event("step.cancelled", event_payload)
        else:
            listener.event("step.cancelled", event_payload)
        raise asyncio.CancelledError("Pipeline cancellation requested")


def _make_yaml_safe(obj: Any) -> Any:
    """Recursively convert *obj* to plain Python types safe for ``yaml.safe_dump``.

    Handles enums (StrEnum, IntEnum), Path objects, sets, and other common
    non-primitive types that ``yaml.dump`` would serialize with Python-specific
    tags (which ``yaml.safe_load`` cannot read back).
    """
    import enum

    if obj is None or isinstance(obj, bool | int | float):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, str | Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _make_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_make_yaml_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_make_yaml_safe(v) for v in sorted(obj)]
    # Fallback: convert to string representation
    return str(obj)


def _load_pipeline_state(
    working_dir: str,
    session_id: str | None,
    project_name: str | None,
    resume: bool,
) -> dict[str, Any]:
    """Load or initialise pipeline state, carrying over step_outputs for auto-wiring.

    Returns:
        A pipeline_state dict ready for use by execute_pipeline.
    """
    pipeline_state: dict[str, Any] = {
        "session_id": session_id,
        "project_name": project_name,
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }

    state_file = Path(working_dir) / ".pipeline_state.json"
    if state_file.exists():
        try:
            with open(state_file, encoding="utf-8") as f:
                saved_state = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read pipeline state file %s: %s — starting fresh",
                state_file,
                exc,
            )
            return pipeline_state

        if resume:
            logger.info("Resuming from checkpoint: %s", state_file)
            pipeline_state = saved_state

            # Verify session ID matches if present
            saved_session_id = pipeline_state.get("session_id")
            if saved_session_id and session_id and saved_session_id != session_id:
                logger.warning(
                    "Session ID mismatch! Current: %s, Saved: %s. "
                    "Continuing with current session ID.",
                    session_id,
                    saved_session_id,
                )
                pipeline_state["session_id"] = session_id

            logger.info(
                "Previously completed: %s",
                ", ".join(pipeline_state["completed_steps"]),
            )
        else:
            # Not resuming: start fresh but carry over step_outputs so that
            # downstream steps (e.g. apply) can auto-wire paths from earlier
            # steps (e.g. optimized_usd_path from optimize_usd).
            pipeline_state["step_outputs"] = saved_state.get("step_outputs", {})

    return pipeline_state


class UnifiedPipelineExecutorTask(BasePipelineExecutor):
    """Execute pipeline steps with pre-configured, auto-wired step configs.

    This executor works with the unified config system where:
    - All paths are already resolved by UnifiedPipelineConfigTask
    - Step configs are complete and ready to use
    - No additional config loading needed

    Input context keys:
        - steps_to_run: List of step names to execute
        - step_configs: Dictionary of pre-configured step configs (paths resolved)
        - path_resolver: ProjectPathResolver instance
        - working_dir: Working directory
        - materials_data: Materials data
        - resume: Optional flag to resume from checkpoint

    Output context keys:
        - pipeline_results: Dictionary of results from each step
        - pipeline_state: Final pipeline state
    """

    def __init__(self) -> None:
        """Initialize the unified pipeline executor."""
        self.name = "UnifiedPipelineExecutor"
        self.description = "Execute pipeline with auto-wired configs"

    # ========== Required Abstract Method Implementations ==========

    def _get_step_list_key(self) -> str:
        """Return context key for step list."""
        return "steps_to_run"

    def _get_required_context_keys(self) -> list[str]:
        """Return required context keys."""
        return ["steps_to_run", "step_configs"]

    def _get_state_file(self, context: dict[str, Any]) -> Path:
        """Return path to pipeline state file."""
        working_dir = context.get("working_dir", Path.cwd())
        return Path(working_dir) / ".pipeline_state.json"

    # ========== Material-Agent Specific Execution Logic ==========

    def run(
        self, context: dict[str, Any], object_store: Any | None = None
    ) -> dict[str, Any]:
        """Execute pipeline steps in sequence.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with pipeline results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        _raise_if_cancelled(context, listener)

        steps_to_run = context.get("steps_to_run", [])
        step_configs = context.get("step_configs", {})
        working_dir = context.get("working_dir", Path.cwd())
        resume = context.get("resume", False)
        clean = context.get("clean", False)
        path_resolver = context.get("path_resolver")

        if not steps_to_run:
            raise ValueError("No steps to run in pipeline")

        # Clean working directory and output file if requested
        if clean:
            import shutil

            # Clean working directory
            working_dir_path = Path(working_dir)

            # Safety checks before deletion
            if working_dir_path == Path.home() or working_dir_path == Path("/"):
                raise ValueError(
                    f"Refusing to delete potentially dangerous path: {working_dir_path}"
                )

            # Optionally check for a minimum depth from root
            if len(working_dir_path.parts) < 2:
                raise ValueError(
                    f"Working directory path too shallow: {working_dir_path}"
                )

            if working_dir_path.exists():
                logger.info("Cleaning working directory: %s", working_dir_path)
                shutil.rmtree(working_dir_path)
                logger.info("Working directory cleaned successfully")

            # Clean output files (USD and renders) if path_resolver is available
            if path_resolver and path_resolver.output_usd:
                output_file = path_resolver.output_usd
                output_dir = output_file.parent

                # Remove output USD file
                if output_file.exists():
                    logger.info("Removing output USD file: %s", output_file)
                    output_file.unlink()
                    logger.info("Output USD file removed successfully")

                # Remove flattened USD file if it exists (from render step)
                flattened_usd = output_dir / f"{output_file.stem}_flat.usd"
                if flattened_usd.exists():
                    logger.info("Removing flattened USD file: %s", flattened_usd)
                    flattened_usd.unlink()
                    logger.info("Flattened USD file removed successfully")

                # Remove renders directory if it exists
                renders_dir = output_dir / "renders"
                if renders_dir.exists() and renders_dir.is_dir():
                    logger.info("Removing renders directory: %s", renders_dir)
                    shutil.rmtree(renders_dir)
                    logger.info("Renders directory removed successfully")

        # Ensure working directory exists
        Path(working_dir).mkdir(parents=True, exist_ok=True)

        # Get session_id from context
        session_id = context.get("session_id")
        project_name = context.get("project_name")

        state_file = Path(working_dir) / ".pipeline_state.json"
        pipeline_state = _load_pipeline_state(
            working_dir, session_id, project_name, resume
        )

        # Display pipeline start with session info
        logger.info("=" * 80)
        logger.info("PIPELINE STARTING")
        logger.info("=" * 80)
        logger.info("Session ID: %s", session_id)
        logger.info("Project: %s", project_name)
        logger.info("Working Directory: %s", working_dir)
        if path_resolver:
            logger.info("Output USD: %s", path_resolver.output_usd)
        logger.info("Steps: %s", ", ".join(steps_to_run))
        logger.info("=" * 80)

        # Emit pipeline start event with session ID
        listener.event(
            "pipeline.started",
            {
                "session_id": session_id,
                "project_name": project_name,
                "working_dir": str(working_dir),
                "steps": steps_to_run,
                "completed_steps": pipeline_state.get("completed_steps", []),
            },
        )

        # Execute each step
        for i, step_name in enumerate(steps_to_run, 1):
            _raise_if_cancelled(context, listener, step_name)
            # Skip if already completed (resume mode)
            if resume and step_name in pipeline_state["completed_steps"]:
                logger.info(
                    "[%d/%d] Skipping %s (already completed)",
                    i,
                    len(steps_to_run),
                    step_name,
                )
                continue

            # Skip restore_usd when optimize_usd didn't run (nothing to restore)
            if step_name == "restore_usd" and "optimize_usd" not in pipeline_state.get(
                "step_outputs", {}
            ):
                logger.info(
                    "[%d/%d] Skipping %s (optimize_usd did not run)",
                    i,
                    len(steps_to_run),
                    step_name,
                )
                continue

            pipeline_state["current_step"] = step_name

            # Get event listener from context
            event_listener = context.get("event_listener")

            try:
                logger.info(
                    "\n[%d/%d] Executing step: %s", i, len(steps_to_run), step_name
                )

                # Emit step started event (listener will display it)
                if event_listener:
                    event_listener.event(
                        "step.started",
                        {
                            "step_name": step_name,
                            "step_index": i,
                            "total_steps": len(steps_to_run),
                        },
                    )
                else:
                    # Emit step.started event even without custom listener
                    listener.event("step.started", {"step_name": step_name})

                # Execute the step with pre-configured config
                step_config = step_configs[step_name]
                outputs = self._execute_step(
                    step_name, step_config, context, object_store, pipeline_state
                )

                # Mark step as completed
                pipeline_state["completed_steps"].append(step_name)
                pipeline_state["step_outputs"][step_name] = outputs
                if step_name in pipeline_state.get("failed_steps", []):
                    pipeline_state["failed_steps"] = [
                        s for s in pipeline_state["failed_steps"] if s != step_name
                    ]
                pipeline_state.get("step_errors", {}).pop(step_name, None)

                # Copy important stats to main context for report generation
                # Use 'is not None' to ensure 0 values are also propagated
                if step_name == "optimize_usd":
                    if outputs.get("original_prim_count") is not None:
                        context["original_prim_count"] = outputs["original_prim_count"]
                if step_name == "build_dataset_usd":
                    if outputs.get("num_prims") is not None:
                        context["num_prims"] = outputs["num_prims"]
                        # If optimize_usd didn't run, original_prim_count equals num_prims
                        # (no optimization means original == processed)
                        if context.get("original_prim_count") is None:
                            context["original_prim_count"] = outputs["num_prims"]
                    if outputs.get("num_images") is not None:
                        context["num_images"] = outputs["num_images"]

                # Save state checkpoint
                pipeline_state["current_step"] = None
                self._save_checkpoint(pipeline_state, state_file)

                logger.info("✓ Step '%s' completed successfully", step_name)

                # Emit step completed event
                if event_listener:
                    event_listener.event(
                        "step.completed",
                        {"step_name": step_name, "outputs": outputs},
                    )

            except asyncio.CancelledError:
                pipeline_state["current_step"] = None
                self._save_checkpoint(pipeline_state, state_file)
                raise

            except Exception as e:
                # If optimize_usd fails, skip it and continue with the
                # original USD rather than aborting the whole pipeline.
                if step_name == "optimize_usd":
                    logger.warning(
                        "Scene Optimizer failed — continuing pipeline "
                        "without optimization (using original USD): %s",
                        e,
                    )
                    # Save original input so downstream steps (e.g. build_dataset_usd)
                    # that were pre-wired to the optimized path can fall back correctly.
                    pipeline_state["optimize_usd_skipped_original_input"] = (
                        step_config.get("input_usd_path")
                    )
                    pipeline_state["current_step"] = None
                    self._save_checkpoint(pipeline_state, state_file)

                    if event_listener:
                        event_listener.event(
                            "step.skipped",
                            {
                                "step_name": step_name,
                                "reason": f"optimize_usd failed: {e}",
                            },
                        )
                    continue

                logger.error("✗ Step '%s' failed: %s", step_name, e, exc_info=True)
                pipeline_state["failed_steps"].append(step_name)
                pipeline_state.setdefault("step_errors", {})[step_name] = str(e)
                pipeline_state["current_step"] = None

                # Save state before failing
                self._save_checkpoint(pipeline_state, state_file)

                # Emit step failed event
                if event_listener:
                    event_listener.event(
                        "step.failed",
                        {"step_name": step_name, "error": str(e)},
                    )
                else:
                    # Fallback: Print to console if no listener
                    listener.event(
                        "step.failed", {"step_name": step_name, "error": str(e)}
                    )

                raise RuntimeError(f"Pipeline failed at step '{step_name}': {e}") from e

        # Pipeline completed successfully
        pipeline_state["current_step"] = None
        self._save_checkpoint(pipeline_state, state_file)

        # Display pipeline completion with session info
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Session ID: %s", session_id)
        logger.info("Project: %s", project_name)
        logger.info("Working Directory: %s", working_dir)
        if path_resolver:
            logger.info("Output USD: %s", path_resolver.output_usd)
            logger.info("Output Directory: %s", path_resolver.output_usd.parent)
        logger.info("Completed Steps: %s", ", ".join(pipeline_state["completed_steps"]))
        logger.info("=" * 80)
        logger.info("📁 Find your outputs in: %s/output/", working_dir)
        logger.info("=" * 80)

        # Emit success event with session ID
        listener.event(
            "pipeline.completed",
            {
                "session_id": session_id,
                "project_name": project_name,
                "working_dir": str(working_dir),
                "completed_steps": pipeline_state["completed_steps"],
                "output_usd": str(path_resolver.output_usd) if path_resolver else None,
            },
        )

        # Update context
        context["pipeline_results"] = pipeline_state["step_outputs"]
        context["pipeline_state"] = "completed"

        return context

    async def arun(
        self, context: dict[str, Any], object_store: Any | None = None
    ) -> dict[str, Any]:
        """Execute pipeline steps in sequence (async version).

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with pipeline results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        _raise_if_cancelled(context, listener)

        steps_to_run = context.get("steps_to_run", [])
        step_configs = context.get("step_configs", {})
        working_dir = context.get("working_dir", Path.cwd())
        resume = context.get("resume", False)
        clean = context.get("clean", False)
        path_resolver = context.get("path_resolver")

        if not steps_to_run:
            raise ValueError("No steps to run in pipeline")

        # Clean working directory and output file if requested
        if clean:
            import shutil

            # Clean working directory
            working_dir_path = Path(working_dir)

            # Safety checks before deletion
            if working_dir_path == Path.home() or working_dir_path == Path("/"):
                raise ValueError(
                    f"Refusing to delete potentially dangerous path: {working_dir_path}"
                )

            # Optionally check for a minimum depth from root
            if len(working_dir_path.parts) < 2:
                raise ValueError(
                    f"Working directory path too shallow: {working_dir_path}"
                )

            if working_dir_path.exists():
                logger.info("Cleaning working directory: %s", working_dir_path)
                shutil.rmtree(working_dir_path)
                logger.info("Working directory cleaned successfully")

            # Clean output files (USD and renders) if path_resolver is available
            if path_resolver and path_resolver.output_usd:
                output_file = path_resolver.output_usd
                output_dir = output_file.parent

                # Remove output USD file
                if output_file.exists():
                    logger.info("Removing output USD file: %s", output_file)
                    output_file.unlink()
                    logger.info("Output USD file removed successfully")

                # Remove flattened USD file if it exists (from render step)
                flattened_usd = output_dir / f"{output_file.stem}_flat.usd"
                if flattened_usd.exists():
                    logger.info("Removing flattened USD file: %s", flattened_usd)
                    flattened_usd.unlink()
                    logger.info("Flattened USD file removed successfully")

                # Remove renders directory if it exists
                renders_dir = output_dir / "renders"
                if renders_dir.exists() and renders_dir.is_dir():
                    logger.info("Removing renders directory: %s", renders_dir)
                    shutil.rmtree(renders_dir)
                    logger.info("Renders directory removed successfully")

        # Ensure working directory exists
        Path(working_dir).mkdir(parents=True, exist_ok=True)

        # Get session_id from context
        session_id = context.get("session_id")
        project_name = context.get("project_name")

        state_file = Path(working_dir) / ".pipeline_state.json"
        pipeline_state = _load_pipeline_state(
            working_dir, session_id, project_name, resume
        )

        # Display pipeline start with session info
        logger.info("=" * 80)
        logger.info("PIPELINE STARTING")
        logger.info("=" * 80)
        logger.info("Session ID: %s", session_id)
        logger.info("Project: %s", project_name)
        logger.info("Working Directory: %s", working_dir)
        if path_resolver:
            logger.info("Output USD: %s", path_resolver.output_usd)
        logger.info("Steps: %s", ", ".join(steps_to_run))
        logger.info("=" * 80)

        # Emit pipeline start event with session ID
        listener.event(
            "pipeline.started",
            {
                "session_id": session_id,
                "project_name": project_name,
                "working_dir": str(working_dir),
                "steps": steps_to_run,
                "completed_steps": pipeline_state.get("completed_steps", []),
            },
        )

        # Execute each step
        for i, step_name in enumerate(steps_to_run, 1):
            _raise_if_cancelled(context, listener, step_name)
            # Skip if already completed (resume mode)
            if resume and step_name in pipeline_state["completed_steps"]:
                logger.info(
                    "[%d/%d] Skipping %s (already completed)",
                    i,
                    len(steps_to_run),
                    step_name,
                )
                continue

            # Skip restore_usd when optimize_usd didn't run (nothing to restore)
            if step_name == "restore_usd" and "optimize_usd" not in pipeline_state.get(
                "step_outputs", {}
            ):
                logger.info(
                    "[%d/%d] Skipping %s (optimize_usd did not run)",
                    i,
                    len(steps_to_run),
                    step_name,
                )
                continue

            pipeline_state["current_step"] = step_name

            # Get event listener from context
            event_listener = context.get("event_listener")

            try:
                logger.info(
                    "\n[%d/%d] Executing step: %s", i, len(steps_to_run), step_name
                )

                # Emit step started event (listener will display it)
                if event_listener:
                    event_listener.event(
                        "step.started",
                        {
                            "step_name": step_name,
                            "step_index": i,
                            "total_steps": len(steps_to_run),
                        },
                    )
                else:
                    # Emit step.started event even without custom listener
                    listener.event("step.started", {"step_name": step_name})

                # Execute the step with pre-configured config (async)
                step_config = step_configs[step_name]
                outputs = await self._aexecute_step(
                    step_name, step_config, context, object_store, pipeline_state
                )

                # Mark step as completed
                pipeline_state["completed_steps"].append(step_name)
                pipeline_state["step_outputs"][step_name] = outputs
                if step_name in pipeline_state.get("failed_steps", []):
                    pipeline_state["failed_steps"] = [
                        s for s in pipeline_state["failed_steps"] if s != step_name
                    ]
                pipeline_state.get("step_errors", {}).pop(step_name, None)

                # Copy important stats to main context for report generation
                # Use 'is not None' to ensure 0 values are also propagated
                if step_name == "optimize_usd":
                    if outputs.get("original_prim_count") is not None:
                        context["original_prim_count"] = outputs["original_prim_count"]
                if step_name == "build_dataset_usd":
                    if outputs.get("num_prims") is not None:
                        context["num_prims"] = outputs["num_prims"]
                        # If optimize_usd didn't run, original_prim_count equals num_prims
                        # (no optimization means original == processed)
                        if context.get("original_prim_count") is None:
                            context["original_prim_count"] = outputs["num_prims"]
                    if outputs.get("num_images") is not None:
                        context["num_images"] = outputs["num_images"]

                # Save state checkpoint
                pipeline_state["current_step"] = None
                self._save_checkpoint(pipeline_state, state_file)

                logger.info("✓ Step '%s' completed successfully", step_name)

                # Emit step completed event
                if event_listener:
                    event_listener.event(
                        "step.completed",
                        {"step_name": step_name, "outputs": outputs},
                    )

            except asyncio.CancelledError:
                pipeline_state["current_step"] = None
                self._save_checkpoint(pipeline_state, state_file)
                raise

            except Exception as e:
                # If optimize_usd fails, skip it and continue with the
                # original USD rather than aborting the whole pipeline.
                if step_name == "optimize_usd":
                    logger.warning(
                        "Scene Optimizer failed — continuing pipeline "
                        "without optimization (using original USD): %s",
                        e,
                    )
                    # Save original input so downstream steps (e.g. build_dataset_usd)
                    # that were pre-wired to the optimized path can fall back correctly.
                    pipeline_state["optimize_usd_skipped_original_input"] = (
                        step_config.get("input_usd_path")
                    )
                    pipeline_state["current_step"] = None
                    self._save_checkpoint(pipeline_state, state_file)

                    if event_listener:
                        event_listener.event(
                            "step.skipped",
                            {
                                "step_name": step_name,
                                "reason": f"optimize_usd failed: {e}",
                            },
                        )
                    continue

                logger.error("✗ Step '%s' failed: %s", step_name, e, exc_info=True)
                pipeline_state["failed_steps"].append(step_name)
                pipeline_state.setdefault("step_errors", {})[step_name] = str(e)
                pipeline_state["current_step"] = None

                # Save state before failing
                self._save_checkpoint(pipeline_state, state_file)

                # Emit step failed event
                if event_listener:
                    event_listener.event(
                        "step.failed",
                        {"step_name": step_name, "error": str(e)},
                    )
                else:
                    # Fallback: Print to console if no listener
                    listener.event(
                        "step.failed", {"step_name": step_name, "error": str(e)}
                    )

                raise RuntimeError(f"Pipeline failed at step '{step_name}': {e}") from e

        # Pipeline completed successfully
        pipeline_state["current_step"] = None
        self._save_checkpoint(pipeline_state, state_file)

        # Display pipeline completion with session info
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Session ID: %s", session_id)
        logger.info("Project: %s", project_name)
        logger.info("Working Directory: %s", working_dir)
        if path_resolver:
            logger.info("Output USD: %s", path_resolver.output_usd)
            logger.info("Output Directory: %s", path_resolver.output_usd.parent)
        logger.info("Completed Steps: %s", ", ".join(pipeline_state["completed_steps"]))
        logger.info("=" * 80)
        logger.info("📁 Find your outputs in: %s/output/", working_dir)
        logger.info("=" * 80)

        # Emit success event with session ID
        listener.event(
            "pipeline.completed",
            {
                "session_id": session_id,
                "project_name": project_name,
                "working_dir": str(working_dir),
                "completed_steps": pipeline_state["completed_steps"],
                "output_usd": str(path_resolver.output_usd) if path_resolver else None,
            },
        )

        # Update context
        context["pipeline_results"] = pipeline_state["step_outputs"]
        context["pipeline_state"] = "completed"

        return context

    def _execute_step(  # type: ignore[override]
        self,
        step_name: str,
        step_config: dict[str, Any],
        context: dict[str, Any],
        object_store: Any,
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single pipeline step.

        Since step_config is already complete with all paths resolved,
        we just need to call the appropriate workflow with it.

        Args:
            step_name: Name of the step
            step_config: Pre-configured step configuration
            context: Workflow context
            object_store: Optional object store
            pipeline_state: Current pipeline state

        Returns:
            Dictionary with relevant outputs
        """
        # Auto-wire outputs from previous steps if needed
        step_outputs = pipeline_state.get("step_outputs", {})

        # Auto-wire fixed USD from validate_input (fix mode)
        # Precedence: optimize_usd output > validate_input fix > config default
        #  - optimize_usd always gets the fixed file (it's the next consumer)
        #  - Other steps only get the fixed file when optimize_usd didn't run
        #    (if optimize_usd ran, it already consumed the fix and its output
        #    takes over via the optimize_usd auto-wire below)
        if "validate_input" in step_outputs:
            fixed_path = step_outputs["validate_input"].get("validation_fixed_usd_path")
            if fixed_path:
                if step_name == "optimize_usd":
                    # optimize_usd is the direct consumer of the fixed input
                    step_config["input_usd_path"] = str(fixed_path)
                    logger.info(
                        "Auto-wired input_usd_path for optimize_usd "
                        "from validate_input fix: %s",
                        fixed_path,
                    )
                elif (
                    step_name
                    in ["render_preview", "identify_asset", "build_dataset_usd"]
                    and "optimize_usd" not in step_outputs
                ):
                    # Only wire directly when optimize_usd didn't run
                    step_config["usd_path"] = str(fixed_path)
                    logger.info(
                        "Auto-wired usd_path for %s from validate_input "
                        "fix (optimize_usd not in pipeline): %s",
                        step_name,
                        fixed_path,
                    )

        # Auto-wire optimized USD for steps that consume USD files
        # When optimize_usd has run, downstream steps should use optimized geometry
        # UNLESS restore_usd has run, in which case apply/refine should use original
        if step_name in [
            "render_preview",
            "identify_asset",
            "build_dataset_usd",
            "apply",
            "refine",
        ]:
            # Skip optimization auto-wiring for apply/refine if restore_usd has run
            if step_name in ["apply", "refine"] and "restore_usd" in step_outputs:
                pass  # Will be handled by restore logic below
            elif "optimize_usd" in step_outputs:
                optimized_usd_path = step_outputs["optimize_usd"].get(
                    "optimized_usd_path"
                )
                if optimized_usd_path:
                    # Determine the correct input key for this step
                    input_key = (
                        "input_usd_path"
                        if step_name in ["apply", "refine"]
                        else "usd_path"
                    )
                    logger.info(
                        "Auto-wired %s for %s from optimize_usd: %s",
                        input_key,
                        step_name,
                        optimized_usd_path,
                    )
                    step_config[input_key] = str(optimized_usd_path)
            elif "optimize_usd_skipped_original_input" in pipeline_state:
                # optimize_usd was in the pipeline but failed and was skipped.
                # The step config was pre-wired to the (non-existent) optimized path;
                # revert to the original input USD so downstream steps can proceed.
                original_usd = pipeline_state["optimize_usd_skipped_original_input"]
                if original_usd:
                    input_key = (
                        "input_usd_path"
                        if step_name in ["apply", "refine"]
                        else "usd_path"
                    )
                    logger.info(
                        "Auto-wired %s for %s to original (optimize_usd skipped): %s",
                        input_key,
                        step_name,
                        original_usd,
                    )
                    step_config[input_key] = str(original_usd)

        # Auto-wire original USD path and restored predictions for apply/refine steps
        # When restore_usd has run, use original USD and restored predictions
        if step_name in ["apply", "refine"]:
            if "restore_usd" in step_outputs:
                # Restore input_usd_path to original_usd_path
                if "optimize_usd" in step_outputs:
                    original_usd_path = step_outputs["optimize_usd"].get(
                        "original_usd_path"
                    )
                    if original_usd_path:
                        logger.info(
                            "Auto-wired input_usd_path back to original after restore_usd: %s",
                            original_usd_path,
                        )
                        step_config["input_usd_path"] = str(original_usd_path)

                # Use restored predictions
                restored_predictions_path = step_outputs["restore_usd"].get(
                    "restored_predictions_path"
                )
                if restored_predictions_path:
                    logger.info(
                        "Auto-wired predictions_path for %s from restore_usd: %s",
                        step_name,
                        restored_predictions_path,
                    )
                    step_config["predictions_path"] = str(restored_predictions_path)

        # Auto-wire VLM prompt path for refine iterative step
        # NOTE: This auto-wiring is disabled for v0.2 datasets.
        # In v0.2 format, system prompts are stored in dataset.json and
        # loaded automatically by the predict task.
        if step_name == "refine":
            # Legacy: system_prompt_file auto-wiring removed
            # (v0.2 datasets store prompts in dataset.json)

            # Auto-wire reference_images from input config to judge
            pipeline_config = context.get("pipeline_config", {})
            input_config = pipeline_config.get("input", {})
            reference_images = input_config.get("reference_images", [])

            if reference_images:
                if "judge" not in step_config:
                    step_config["judge"] = {}
                if "reference_images" not in step_config["judge"]:
                    logger.info(
                        "Auto-wired %d reference_images to %s.judge",
                        len(reference_images),
                        step_name,
                    )
                    step_config["judge"]["reference_images"] = reference_images

        if step_name == "identify_asset":
            if "render_preview" in step_outputs:
                render_preview_outputs = step_outputs["render_preview"]
                preview_paths = render_preview_outputs.get("rendered_preview_paths")
                if preview_paths and not step_config.get("rendered_preview_paths"):
                    logger.info(
                        "Auto-wired rendered_preview_paths to identify_asset: %d image(s)",
                        len(preview_paths),
                    )
                    step_config["rendered_preview_paths"] = preview_paths
                composition_images = render_preview_outputs.get("composition_images")
                if not composition_images:
                    composition_images = preview_paths
                if composition_images and not step_config.get("composition_images"):
                    step_config["composition_images"] = composition_images

            path_resolver = context.get("path_resolver")
            reference_images = []
            if path_resolver is not None:
                reference_images = [
                    str(img) for img in getattr(path_resolver, "reference_images", [])
                ]
            if reference_images and not step_config.get("reference_images"):
                step_config["reference_images"] = reference_images

        if step_name == "generate_reference_image":
            if "render_preview" in step_outputs:
                preview_paths = step_outputs["render_preview"].get(
                    "rendered_preview_paths"
                )
                if preview_paths and not step_config.get("rendered_preview_paths"):
                    logger.info(
                        "Auto-wired rendered_preview_paths to generate_reference_image: %d image(s)",
                        len(preview_paths),
                    )
                    step_config["rendered_preview_paths"] = preview_paths

            if "identify_asset" in step_outputs:
                identify_outputs = step_outputs["identify_asset"]
                if identify_outputs.get("identification") and not step_config.get(
                    "identification"
                ):
                    step_config["identification"] = identify_outputs["identification"]
                if identify_outputs.get("image_gen_prompt") and not step_config.get(
                    "image_gen_prompt"
                ):
                    step_config["image_gen_prompt"] = identify_outputs[
                        "image_gen_prompt"
                    ]

        if step_name == "build_dataset_prepare_dataset":
            generated_refs = step_outputs.get("generate_reference_image", {}).get(
                "generated_reference_image_paths",
                [],
            )
            if generated_refs:
                existing_refs = list(step_config.get("reference_images") or [])
                for ref_path in generated_refs:
                    if ref_path not in existing_refs:
                        existing_refs.append(ref_path)
                logger.info(
                    "Auto-wired %d generated reference image(s) to build_dataset_prepare_dataset",
                    len(generated_refs),
                )
                step_config["reference_images"] = existing_refs

        # Auto-wire cluster_prims: inject dataset_path and working_dir
        if step_name == "cluster_prims":
            working_dir = Path(context.get("working_dir", Path.cwd()))
            step_outputs = pipeline_state.get("step_outputs", {})

            if "dataset_path" not in step_config:
                dataset_path = None
                if "build_dataset_prepare_dataset" in step_outputs:
                    prep_out = step_outputs["build_dataset_prepare_dataset"]
                    # prefer dataset_jsonl_path (file); dataset_path is the directory
                    dataset_path = prep_out.get("dataset_jsonl_path") or prep_out.get(
                        "dataset_path"
                    )
                if not dataset_path:
                    dataset_path = working_dir / "dataset" / "dataset.jsonl"
                logger.info(
                    "Auto-wired dataset_path to cluster_prims: %s", dataset_path
                )
                step_config["dataset_path"] = str(dataset_path)

            if "working_dir" not in step_config:
                step_config["working_dir"] = str(working_dir)

        # Auto-wire predict: use representative dataset when cluster_prims ran
        if step_name == "predict":
            step_outputs = pipeline_state.get("step_outputs", {})

            if "cluster_prims" in step_outputs:
                rep_dataset = step_outputs["cluster_prims"].get(
                    "dataset_representatives_path"
                )
                if rep_dataset:
                    logger.info(
                        "Auto-wired dataset to predict from cluster_prims: %s",
                        rep_dataset,
                    )
                    step_config["dataset"] = str(rep_dataset)

        # Auto-wire expand_cluster_predictions: predictions_path and cluster_map_path
        if step_name == "expand_cluster_predictions":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            # Propagate cluster_prims_ran so the task can skip itself if needed
            if "cluster_prims_ran" not in step_config:
                cluster_prims_ran = step_outputs.get("cluster_prims", {}).get(
                    "cluster_prims_ran", True
                )
                step_config["cluster_prims_ran"] = cluster_prims_ran

            if "predictions_path" not in step_config:
                predictions_path = None
                if "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")
                if not predictions_path:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"
                logger.info(
                    "Auto-wired predictions_path to expand_cluster_predictions: %s",
                    predictions_path,
                )
                step_config["predictions_path"] = str(predictions_path)

            if "cluster_map_path" not in step_config:
                cluster_map_path = None
                if "cluster_prims" in step_outputs:
                    cluster_map_path = step_outputs["cluster_prims"].get(
                        "cluster_map_path"
                    )
                if not cluster_map_path:
                    cluster_map_path = working_dir / "clusters" / "cluster_map.jsonl"
                logger.info(
                    "Auto-wired cluster_map_path to expand_cluster_predictions: %s",
                    cluster_map_path,
                )
                step_config["cluster_map_path"] = str(cluster_map_path)

        # Auto-wire predictions and dataset from predict/benchmark step for evaluate
        if step_name == "evaluate":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            # Auto-wire predictions_path
            if "predictions_path" not in step_config:
                predictions_path = None

                # First try: Get from previous step outputs
                # (prefer harmonized > validated > raw)
                if "harmonize_predictions" in step_outputs:
                    predictions_path = step_outputs["harmonize_predictions"].get(
                        "predictions_path"
                    )
                elif "validate_predictions" in step_outputs:
                    predictions_path = step_outputs["validate_predictions"].get(
                        "predictions_path"
                    )
                elif "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")

                # Fallback: Derive from working_dir structure
                if not predictions_path:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"
                    if not predictions_path.exists():
                        logger.warning(
                            "predictions_path not found at %s - evaluate step may fail",
                            predictions_path,
                        )

                if predictions_path:
                    logger.info(
                        "Auto-wired predictions_path to evaluate: %s", predictions_path
                    )
                    step_config["predictions_path"] = str(predictions_path)

            # Auto-wire dataset_path
            if "dataset_path" not in step_config:
                dataset_path = None

                # First try: Get from previous step outputs
                if "build_dataset_prepare_dataset" in step_outputs:
                    dataset_path = step_outputs["build_dataset_prepare_dataset"].get(
                        "dataset_jsonl_path"
                    )

                # Fallback: Derive from working_dir structure
                if not dataset_path:
                    dataset_path = working_dir / "dataset" / "dataset.jsonl"
                    if not dataset_path.exists():
                        logger.warning(
                            "dataset_path not found at %s - ground truth may not be available",
                            dataset_path,
                        )

                if dataset_path:
                    logger.info("Auto-wired dataset_path to evaluate: %s", dataset_path)
                    step_config["dataset_path"] = str(dataset_path)

            # Auto-wire system_prompt_file
            if "system_prompt_file" not in step_config:
                vlm_prompt_path = None

                # First try: Get from previous step outputs
                if "build_dataset_prepare_dataset" in step_outputs:
                    vlm_prompt_path = step_outputs["build_dataset_prepare_dataset"].get(
                        "vlm_prompt_path"
                    )

                # Fallback: Derive from working_dir structure
                if not vlm_prompt_path:
                    vlm_prompt_path = working_dir / "dataset" / "vlm_system_prompt.txt"
                    if not vlm_prompt_path.exists():
                        logger.debug(
                            "system_prompt_file not found at %s - will not be included in report",
                            vlm_prompt_path,
                        )
                        vlm_prompt_path = None

                if vlm_prompt_path:
                    logger.info(
                        "Auto-wired system_prompt_file to evaluate: %s", vlm_prompt_path
                    )
                    step_config["system_prompt_file"] = str(vlm_prompt_path)

            # Auto-wire output_dir from working_dir
            if "output_dir" not in step_config:
                output_dir = working_dir / "evaluation"
                logger.info("Auto-wired output_dir to evaluate: %s", output_dir)
                step_config["output_dir"] = str(output_dir)

        # Auto-wire outputs from restore_usd/apply/refine step for render step
        if step_name == "render":
            step_outputs = pipeline_state.get("step_outputs", {})

            if "input_usd_path" not in step_config:
                usd_path = None
                source_step = None

                # restore_usd normally restores prediction paths, not a USD file.
                # Keep this fallback for older configs that may emit a USD path.
                if "restore_usd" in step_outputs:
                    usd_path = step_outputs["restore_usd"].get("restored_usd_path")
                    if usd_path:
                        source_step = "restore_usd"

                if not usd_path and "refine" in step_outputs:
                    usd_path = step_outputs["refine"].get("final_output_path")
                    if not usd_path:
                        usd_path = step_outputs["refine"].get("output_usd_path")
                    if usd_path:
                        source_step = "refine"

                if not usd_path and "apply" in step_outputs:
                    usd_path = step_outputs["apply"].get("output_usd_path")
                    if usd_path:
                        source_step = "apply"

                if usd_path and "input_usd_path" not in step_config:
                    logger.info(
                        "Auto-wired input_usd_path to render from %s: %s",
                        source_step,
                        usd_path,
                    )
                    step_config["input_usd_path"] = str(usd_path)

        # Special handling: Auto-wire harmonize_predictions with predictions path
        # Harmonize runs BEFORE validate — reads from predict/benchmark.
        if step_name == "harmonize_predictions":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            if "predictions_path" not in step_config:
                predictions_path = None
                if "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")

                if not predictions_path:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"

                logger.info(
                    "Auto-wired predictions_path to harmonize_predictions: %s",
                    predictions_path,
                )
                step_config["predictions_path"] = str(predictions_path)

            # Auto-wire optimized USD path for geometry-based grouping
            if "optimized_usd_path" not in step_config:
                if "optimize_usd" in step_outputs:
                    opt_usd = step_outputs["optimize_usd"].get("optimized_usd_path")
                    if opt_usd:
                        step_config["optimized_usd_path"] = str(opt_usd)

        # Special handling: Auto-wire validate_predictions with predictions path
        # Validate runs AFTER harmonize — prefers harmonized output.
        if step_name == "validate_predictions":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            if "predictions_path" not in step_config:
                predictions_path = None
                if "harmonize_predictions" in step_outputs:
                    predictions_path = step_outputs["harmonize_predictions"].get(
                        "predictions_path"
                    )
                elif "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")

                if not predictions_path:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"

                logger.info(
                    "Auto-wired predictions_path to validate_predictions: %s",
                    predictions_path,
                )
                step_config["predictions_path"] = str(predictions_path)

        # Special handling: Auto-wire restore_usd with required paths and metadata
        if step_name == "restore_usd":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            # Auto-wire original USD path from optimize_usd step outputs
            if "optimize_usd" in step_outputs:
                original_usd_path = step_outputs["optimize_usd"].get(
                    "original_usd_path"
                )
                if original_usd_path:
                    step_config["original_usd_path"] = str(original_usd_path)
                    logger.info(
                        "Auto-wired original_usd_path from optimize_usd: %s",
                        original_usd_path,
                    )
            else:
                # Fallback to path_resolver if optimize_usd didn't run
                path_resolver = context.get("path_resolver")
                if path_resolver and path_resolver.input_usd:
                    step_config["original_usd_path"] = str(path_resolver.input_usd)
                    logger.info(
                        "Auto-wired original_usd_path from input: %s",
                        path_resolver.input_usd,
                    )

            # Auto-wire predictions — prefer harmonized > validated > raw
            if "predictions_path" not in step_config:
                predictions_path = None
                if "harmonize_predictions" in step_outputs:
                    predictions_path = step_outputs["harmonize_predictions"].get(
                        "predictions_path"
                    )
                elif "validate_predictions" in step_outputs:
                    predictions_path = step_outputs["validate_predictions"].get(
                        "predictions_path"
                    )
                elif "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")

                if predictions_path:
                    logger.info(
                        "Auto-wired predictions_path to restore: %s", predictions_path
                    )
                    step_config["predictions_path"] = str(predictions_path)
                else:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"
                    step_config["predictions_path"] = str(predictions_path)

            # Auto-wire output predictions path
            if "output_predictions_path" not in step_config:
                output_predictions_path = (
                    working_dir / "restored" / "restored_predictions.jsonl"
                )
                logger.info(
                    "Auto-wired output_predictions_path to restore: %s",
                    output_predictions_path,
                )
                step_config["output_predictions_path"] = str(output_predictions_path)

            # Inject optimization metadata
            if "optimization_metadata" in pipeline_state:
                step_config["optimization_metadata"] = pipeline_state[
                    "optimization_metadata"
                ]
                logger.info("Injected optimization metadata into restore_usd config")
            else:
                # Try to find metadata file in standard location
                optimization_metadata_path = (
                    working_dir / "optimized" / "optimized_input.metadata.json"
                )
                if optimization_metadata_path.exists():
                    with open(optimization_metadata_path, encoding="utf-8") as f:
                        optimization_metadata = json.load(f)
                    step_config["optimization_metadata"] = optimization_metadata
                else:
                    logger.warning(
                        "No optimization metadata found at %s - restore_usd may not work correctly",
                        optimization_metadata_path,
                    )

        # Auto-wire validate_output with output USD and original USD paths
        if step_name == "validate_output":
            step_outputs = pipeline_state.get("step_outputs", {})

            # Auto-wire output USD path from apply/refine step
            if "refine" in step_outputs:
                usd_path = step_outputs["refine"].get(
                    "final_output_path"
                ) or step_outputs["refine"].get("output_usd_path")
                if usd_path:
                    step_config["input_usd_path"] = str(usd_path)
                    logger.info(
                        "Auto-wired input_usd_path to validate_output from refine: %s",
                        usd_path,
                    )
            elif "apply" in step_outputs:
                usd_path = step_outputs["apply"].get("output_usd_path")
                if usd_path:
                    step_config["input_usd_path"] = str(usd_path)
                    logger.info(
                        "Auto-wired input_usd_path to validate_output from apply: %s",
                        usd_path,
                    )

            # Auto-wire original USD path for baseline comparison
            if "original_usd_path" not in step_config:
                # Try optimize_usd (it stores the original path)
                if "optimize_usd" in step_outputs:
                    original = step_outputs["optimize_usd"].get("original_usd_path")
                    if original:
                        step_config["original_usd_path"] = str(original)
                        logger.info(
                            "Auto-wired original_usd_path to validate_output: %s",
                            original,
                        )
                else:
                    # Fall back to path_resolver's original input
                    path_resolver = context.get("path_resolver")
                    if path_resolver:
                        # Use the config's original input USD (before optimize rewrote it)
                        config = context.get("config", {})
                        input_section = config.get("input", {})
                        raw_input = input_section.get("usd_path")
                        if raw_input:
                            resolved = path_resolver.resolve_path(raw_input)
                            step_config["original_usd_path"] = str(resolved)
                            logger.info(
                                "Auto-wired original_usd_path to validate_output from config: %s",
                                resolved,
                            )

            # Inject cached baseline from validate_input (avoids re-validating input).
            # IMPORTANT: Do NOT use the cached baseline when validate_input applied
            # a fix — the cached result describes the pre-fix state, but downstream
            # steps consumed the fixed USD. Let validate_output re-validate the
            # fixed input to get an accurate baseline.
            if "validate_input" in step_outputs:
                vi_outputs = step_outputs["validate_input"]
                used_fix = bool(vi_outputs.get("validation_fixed_usd_path"))

                if used_fix:
                    # Point original_usd_path to the fixed file so
                    # validate_output re-validates it for baseline
                    fixed_path = vi_outputs["validation_fixed_usd_path"]
                    step_config["original_usd_path"] = str(fixed_path)
                    logger.info(
                        "validate_input used fix — baseline will be "
                        "re-validated from fixed input: %s",
                        fixed_path,
                    )
                else:
                    baseline_result = vi_outputs.get("validation_result")
                    if baseline_result:
                        step_config["baseline_validation"] = baseline_result
                        logger.info(
                            "Injected cached baseline from validate_input (%d issues)",
                            len(baseline_result.get("issues", [])),
                        )
        # Auto-wire cluster_prims: needs dataset_path and working_dir
        if step_name in ["cluster_prims", "expand_cluster_predictions"]:
            step_outputs = pipeline_state.get("step_outputs", {})

        if step_name == "cluster_prims":
            working_dir = Path(context.get("working_dir", Path.cwd()))
            if "dataset_path" not in step_config:
                dataset_path = None
                if "build_dataset_prepare_dataset" in step_outputs:
                    dataset_path = step_outputs["build_dataset_prepare_dataset"].get(
                        "dataset_jsonl_path"
                    )
                if not dataset_path:
                    dataset_path = str(working_dir / "dataset" / "dataset.jsonl")
                step_config["dataset_path"] = str(dataset_path)
                logger.info(
                    "Auto-wired dataset_path to cluster_prims: %s", dataset_path
                )
            if "working_dir" not in step_config:
                step_config["working_dir"] = str(working_dir)

        # Auto-wire predict: use representative dataset when clustering ran
        if step_name in ["predict", "benchmark"]:
            step_outputs = pipeline_state.get("step_outputs", {})
            if "cluster_prims" in step_outputs:
                cluster_out = step_outputs["cluster_prims"]
                if cluster_out.get("cluster_prims_ran"):
                    reps_path = cluster_out.get("dataset_representatives_path")
                    if reps_path:
                        step_config["dataset"] = str(reps_path)
                        logger.info(
                            "Auto-wired predict dataset to cluster representatives: %s",
                            reps_path,
                        )

        # Auto-wire expand_cluster_predictions
        if step_name == "expand_cluster_predictions":
            working_dir = Path(context.get("working_dir", Path.cwd()))
            if "cluster_prims" in step_outputs:
                cluster_out = step_outputs["cluster_prims"]
                if cluster_out.get("cluster_prims_ran") is False:
                    step_config["cluster_prims_ran"] = False
                else:
                    if "predictions_path" not in step_config:
                        predictions_path = None
                        if "predict" in step_outputs:
                            predictions_path = step_outputs["predict"].get(
                                "predictions_path"
                            )
                        if not predictions_path:
                            predictions_path = str(
                                working_dir / "predictions" / "predictions.jsonl"
                            )
                        step_config["predictions_path"] = str(predictions_path)
                    if "cluster_map_path" not in step_config:
                        cluster_map_path = cluster_out.get("cluster_map_path")
                        if not cluster_map_path:
                            cluster_map_path = str(
                                working_dir / "clusters" / "cluster_map.jsonl"
                            )
                        step_config["cluster_map_path"] = str(cluster_map_path)
            else:
                step_config["cluster_prims_ran"] = False

        # Auto-wire predictions_path for validate_predictions and harmonize_predictions
        if step_name in ["validate_predictions", "harmonize_predictions"]:
            step_outputs = pipeline_state.get("step_outputs", {})
            if "predictions_path" not in step_config:
                predictions_path = None
                if "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")
                if predictions_path:
                    logger.info(
                        "Auto-wired predictions_path for %s: %s",
                        step_name,
                        predictions_path,
                    )
                    step_config["predictions_path"] = str(predictions_path)

        # Create temporary config file for the step
        # This is needed because existing workflows expect config_path
        working_dir = Path(context.get("working_dir", Path.cwd()))
        temp_config_path = self._create_temp_config_file(
            step_name, step_config, working_dir
        )

        # Import workflows
        from material_agent.workflows import (
            create_apply_workflow_from_config,
            create_benchmark_workflow_from_config,
            create_cluster_prims_workflow_from_config,
            create_evaluation_workflow_from_config,
            create_expand_cluster_predictions_workflow_from_config,
            create_generate_reference_image_workflow_from_config,
            create_harmonize_predictions_workflow_from_config,
            create_identify_asset_workflow_from_config,
            create_iterative_apply_workflow_from_config,
            create_optimize_usd_workflow_from_config,
            create_pdf_vectorstore_workflow_from_config,
            create_prediction_workflow_from_config,
            create_prepare_dataset_workflow_from_config,
            create_render_preview_workflow_from_config,
            create_render_workflow_from_config,
            create_restore_usd_workflow_from_config,
            create_usd_data_preparation_workflow_from_config,
            create_validate_input_workflow_from_config,
            create_validate_output_workflow_from_config,
            create_validate_predictions_workflow_from_config,
        )

        # Map step names to workflow factories
        workflow_map = {
            "validate_input": create_validate_input_workflow_from_config,
            "optimize_usd": create_optimize_usd_workflow_from_config,
            "render_preview": create_render_preview_workflow_from_config,
            "identify_asset": create_identify_asset_workflow_from_config,
            "generate_reference_image": create_generate_reference_image_workflow_from_config,
            "build_dataset_usd": create_usd_data_preparation_workflow_from_config,
            "build_dataset_pdf_vectorstore": create_pdf_vectorstore_workflow_from_config,
            "build_dataset_prepare_dataset": create_prepare_dataset_workflow_from_config,
            "cluster_prims": create_cluster_prims_workflow_from_config,
            "predict": create_prediction_workflow_from_config,
            "expand_cluster_predictions": create_expand_cluster_predictions_workflow_from_config,
            "benchmark": create_benchmark_workflow_from_config,
            "validate_predictions": create_validate_predictions_workflow_from_config,
            "harmonize_predictions": create_harmonize_predictions_workflow_from_config,
            "evaluate": create_evaluation_workflow_from_config,
            "apply": create_apply_workflow_from_config,
            "refine": create_iterative_apply_workflow_from_config,
            "restore_usd": create_restore_usd_workflow_from_config,
            "validate_output": create_validate_output_workflow_from_config,
            "render": create_render_workflow_from_config,
        }

        if step_name not in workflow_map:
            raise ValueError(f"Unknown step: {step_name}")

        # Create workflow
        workflow = workflow_map[step_name]()

        # Prepare step context
        step_context = {"config_path": str(temp_config_path)}
        if step_name == "identify_asset":
            step_context.update(step_config)

        # Pass event listener to individual workflows if available
        if "event_listener" in context:
            step_context["event_listener"] = context["event_listener"]

        # Extract and pass report compression configuration if present
        if "report" in step_config:
            report_config = step_config["report"]
            if isinstance(report_config, dict):
                # Map report config keys to context keys
                if "image_max_size" in report_config:
                    step_context["report_image_max_size"] = report_config[
                        "image_max_size"
                    ]
                if "image_format" in report_config:
                    step_context["report_image_format"] = report_config["image_format"]
                if "image_quality" in report_config:
                    step_context["report_image_quality"] = report_config[
                        "image_quality"
                    ]

        # Pass pipeline statistics to steps that generate reports
        # These values were collected from earlier pipeline steps (optimize_usd, build_dataset_usd)
        # Use 'is not None' to ensure 0 values are also propagated
        if step_name in ["predict", "benchmark", "evaluate"]:
            # Pass original prim count (from optimize_usd step)
            if context.get("original_prim_count") is not None:
                step_context["original_prim_count"] = context["original_prim_count"]
            # Pass processed prim count and image count (from build_dataset_usd step)
            if context.get("num_prims") is not None:
                step_context["num_prims"] = context["num_prims"]
            if context.get("num_images") is not None:
                step_context["num_images"] = context["num_images"]

        # Execute workflow
        logger.debug("Running workflow for %s", step_name)
        result = workflow.run(step_context)

        if not result:
            raise RuntimeError(
                f"Step '{step_name}' did not complete successfully - workflow returned empty result"
            )

        # Check if workflow encountered errors
        if result.get("error") or result.get("workflow_terminated"):
            failed_task = result.get("failed_task", "unknown")
            error_msg = result.get("error", "Workflow terminated without error message")
            raise RuntimeError(
                f"Step '{step_name}' failed at task '{failed_task}': {error_msg}"
            )

        # Extract outputs
        outputs = self._extract_step_outputs(step_name, result)

        # Special handling: Store optimization metadata for restore_usd
        if step_name == "optimize_usd":
            if "optimization_metadata" in result:
                pipeline_state["optimization_metadata"] = result[
                    "optimization_metadata"
                ]
                logger.info("Stored optimization metadata for restore_usd step")

        return outputs

    async def _aexecute_step(
        self,
        step_name: str,
        step_config: dict[str, Any],
        context: dict[str, Any],
        object_store: Any,
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single pipeline step (async version).

        Since step_config is already complete with all paths resolved,
        we just need to call the appropriate workflow with it.

        Args:
            step_name: Name of the step
            step_config: Pre-configured step configuration
            context: Workflow context
            object_store: Optional object store
            pipeline_state: Current pipeline state

        Returns:
            Dictionary with relevant outputs
        """
        # Auto-wire outputs from previous steps if needed
        step_outputs = pipeline_state.get("step_outputs", {})

        # Auto-wire fixed USD from validate_input (fix mode)
        # Precedence: optimize_usd output > validate_input fix > config default
        #  - optimize_usd always gets the fixed file (it's the next consumer)
        #  - Other steps only get the fixed file when optimize_usd didn't run
        #    (if optimize_usd ran, it already consumed the fix and its output
        #    takes over via the optimize_usd auto-wire below)
        if "validate_input" in step_outputs:
            fixed_path = step_outputs["validate_input"].get("validation_fixed_usd_path")
            if fixed_path:
                if step_name == "optimize_usd":
                    # optimize_usd is the direct consumer of the fixed input
                    step_config["input_usd_path"] = str(fixed_path)
                    logger.info(
                        "Auto-wired input_usd_path for optimize_usd "
                        "from validate_input fix: %s",
                        fixed_path,
                    )
                elif (
                    step_name
                    in ["render_preview", "identify_asset", "build_dataset_usd"]
                    and "optimize_usd" not in step_outputs
                ):
                    # Only wire directly when optimize_usd didn't run
                    step_config["usd_path"] = str(fixed_path)
                    logger.info(
                        "Auto-wired usd_path for %s from validate_input "
                        "fix (optimize_usd not in pipeline): %s",
                        step_name,
                        fixed_path,
                    )

        # Auto-wire optimized USD for steps that consume USD files
        # When optimize_usd has run, downstream steps should use optimized geometry
        # UNLESS restore_usd has run, in which case apply/refine should use original
        if step_name in [
            "render_preview",
            "identify_asset",
            "build_dataset_usd",
            "apply",
            "refine",
        ]:
            # Skip optimization auto-wiring for apply/refine if restore_usd has run
            if step_name in ["apply", "refine"] and "restore_usd" in step_outputs:
                pass  # Will be handled by restore logic below
            elif "optimize_usd" in step_outputs:
                optimized_usd_path = step_outputs["optimize_usd"].get(
                    "optimized_usd_path"
                )
                if optimized_usd_path:
                    # Determine the correct input key for this step
                    input_key = (
                        "input_usd_path"
                        if step_name in ["apply", "refine"]
                        else "usd_path"
                    )
                    logger.info(
                        "Auto-wired %s for %s from optimize_usd: %s",
                        input_key,
                        step_name,
                        optimized_usd_path,
                    )
                    step_config[input_key] = str(optimized_usd_path)
            elif "optimize_usd_skipped_original_input" in pipeline_state:
                # optimize_usd was in the pipeline but failed and was skipped.
                # The step config was pre-wired to the (non-existent) optimized path;
                # revert to the original input USD so downstream steps can proceed.
                original_usd = pipeline_state["optimize_usd_skipped_original_input"]
                if original_usd:
                    input_key = (
                        "input_usd_path"
                        if step_name in ["apply", "refine"]
                        else "usd_path"
                    )
                    logger.info(
                        "Auto-wired %s for %s to original (optimize_usd skipped): %s",
                        input_key,
                        step_name,
                        original_usd,
                    )
                    step_config[input_key] = str(original_usd)

        # Auto-wire original USD path and restored predictions for apply/refine steps
        # When restore_usd has run, use original USD and restored predictions
        if step_name in ["apply", "refine"]:
            if "restore_usd" in step_outputs:
                # Restore input_usd_path to original_usd_path
                if "optimize_usd" in step_outputs:
                    original_usd_path = step_outputs["optimize_usd"].get(
                        "original_usd_path"
                    )
                    if original_usd_path:
                        logger.info(
                            "Auto-wired input_usd_path back to original after restore_usd: %s",
                            original_usd_path,
                        )
                        step_config["input_usd_path"] = str(original_usd_path)

                # Use restored predictions
                restored_predictions_path = step_outputs["restore_usd"].get(
                    "restored_predictions_path"
                )
                if restored_predictions_path:
                    logger.info(
                        "Auto-wired predictions_path for %s from restore_usd: %s",
                        step_name,
                        restored_predictions_path,
                    )
                    step_config["predictions_path"] = str(restored_predictions_path)

        # Auto-wire VLM prompt path for refine iterative step
        # NOTE: This auto-wiring is disabled for v0.2 datasets.
        # In v0.2 format, system prompts are stored in dataset.json and
        # loaded automatically by the predict task.
        if step_name == "refine":
            # Legacy: system_prompt_file auto-wiring removed
            # (v0.2 datasets store prompts in dataset.json)

            # Auto-wire reference_images from input config to judge
            pipeline_config = context.get("pipeline_config", {})
            input_config = pipeline_config.get("input", {})
            reference_images = input_config.get("reference_images", [])

            if reference_images:
                if "judge" not in step_config:
                    step_config["judge"] = {}
                if "reference_images" not in step_config["judge"]:
                    logger.info(
                        "Auto-wired %d reference_images to %s.judge",
                        len(reference_images),
                        step_name,
                    )
                    step_config["judge"]["reference_images"] = reference_images

        if step_name == "identify_asset":
            if "render_preview" in step_outputs:
                render_preview_outputs = step_outputs["render_preview"]
                preview_paths = render_preview_outputs.get("rendered_preview_paths")
                if preview_paths and not step_config.get("rendered_preview_paths"):
                    logger.info(
                        "Auto-wired rendered_preview_paths to identify_asset: %d image(s)",
                        len(preview_paths),
                    )
                    step_config["rendered_preview_paths"] = preview_paths
                composition_images = render_preview_outputs.get("composition_images")
                if not composition_images:
                    composition_images = preview_paths
                if composition_images and not step_config.get("composition_images"):
                    step_config["composition_images"] = composition_images

            path_resolver = context.get("path_resolver")
            reference_images = []
            if path_resolver is not None:
                reference_images = [
                    str(img) for img in getattr(path_resolver, "reference_images", [])
                ]
            if reference_images and not step_config.get("reference_images"):
                step_config["reference_images"] = reference_images

        if step_name == "generate_reference_image":
            if "render_preview" in step_outputs:
                preview_paths = step_outputs["render_preview"].get(
                    "rendered_preview_paths"
                )
                if preview_paths and not step_config.get("rendered_preview_paths"):
                    logger.info(
                        "Auto-wired rendered_preview_paths to generate_reference_image: %d image(s)",
                        len(preview_paths),
                    )
                    step_config["rendered_preview_paths"] = preview_paths

            if "identify_asset" in step_outputs:
                identify_outputs = step_outputs["identify_asset"]
                if identify_outputs.get("identification") and not step_config.get(
                    "identification"
                ):
                    step_config["identification"] = identify_outputs["identification"]
                if identify_outputs.get("image_gen_prompt") and not step_config.get(
                    "image_gen_prompt"
                ):
                    step_config["image_gen_prompt"] = identify_outputs[
                        "image_gen_prompt"
                    ]

        if step_name == "build_dataset_prepare_dataset":
            generated_refs = step_outputs.get("generate_reference_image", {}).get(
                "generated_reference_image_paths",
                [],
            )
            if generated_refs:
                existing_refs = list(step_config.get("reference_images") or [])
                for ref_path in generated_refs:
                    if ref_path not in existing_refs:
                        existing_refs.append(ref_path)
                logger.info(
                    "Auto-wired %d generated reference image(s) to build_dataset_prepare_dataset",
                    len(generated_refs),
                )
                step_config["reference_images"] = existing_refs

        # Auto-wire cluster_prims: inject dataset_path and working_dir
        if step_name == "cluster_prims":
            working_dir = Path(context.get("working_dir", Path.cwd()))
            step_outputs = pipeline_state.get("step_outputs", {})

            if "dataset_path" not in step_config:
                dataset_path = None
                if "build_dataset_prepare_dataset" in step_outputs:
                    prep_out = step_outputs["build_dataset_prepare_dataset"]
                    # prefer dataset_jsonl_path (file); dataset_path is the directory
                    dataset_path = prep_out.get("dataset_jsonl_path") or prep_out.get(
                        "dataset_path"
                    )
                if not dataset_path:
                    dataset_path = working_dir / "dataset" / "dataset.jsonl"
                logger.info(
                    "Auto-wired dataset_path to cluster_prims: %s", dataset_path
                )
                step_config["dataset_path"] = str(dataset_path)

            if "working_dir" not in step_config:
                step_config["working_dir"] = str(working_dir)

        # Auto-wire predict: use representative dataset when cluster_prims ran
        if step_name == "predict":
            step_outputs = pipeline_state.get("step_outputs", {})

            if "cluster_prims" in step_outputs:
                rep_dataset = step_outputs["cluster_prims"].get(
                    "dataset_representatives_path"
                )
                if rep_dataset:
                    logger.info(
                        "Auto-wired dataset to predict from cluster_prims: %s",
                        rep_dataset,
                    )
                    step_config["dataset"] = str(rep_dataset)

        # Auto-wire expand_cluster_predictions: predictions_path and cluster_map_path
        if step_name == "expand_cluster_predictions":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            # Propagate cluster_prims_ran so the task can skip itself if needed
            if "cluster_prims_ran" not in step_config:
                cluster_prims_ran = step_outputs.get("cluster_prims", {}).get(
                    "cluster_prims_ran", True
                )
                step_config["cluster_prims_ran"] = cluster_prims_ran

            if "predictions_path" not in step_config:
                predictions_path = None
                if "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")
                if not predictions_path:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"
                logger.info(
                    "Auto-wired predictions_path to expand_cluster_predictions: %s",
                    predictions_path,
                )
                step_config["predictions_path"] = str(predictions_path)

            if "cluster_map_path" not in step_config:
                cluster_map_path = None
                if "cluster_prims" in step_outputs:
                    cluster_map_path = step_outputs["cluster_prims"].get(
                        "cluster_map_path"
                    )
                if not cluster_map_path:
                    cluster_map_path = working_dir / "clusters" / "cluster_map.jsonl"
                logger.info(
                    "Auto-wired cluster_map_path to expand_cluster_predictions: %s",
                    cluster_map_path,
                )
                step_config["cluster_map_path"] = str(cluster_map_path)

        # Auto-wire predictions and dataset from predict/benchmark step for evaluate
        if step_name == "evaluate":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            # Auto-wire predictions_path
            if "predictions_path" not in step_config:
                predictions_path = None

                # First try: Get from previous step outputs
                # (prefer harmonized > validated > raw)
                if "harmonize_predictions" in step_outputs:
                    predictions_path = step_outputs["harmonize_predictions"].get(
                        "predictions_path"
                    )
                elif "validate_predictions" in step_outputs:
                    predictions_path = step_outputs["validate_predictions"].get(
                        "predictions_path"
                    )
                elif "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")

                # Fallback: Derive from working_dir structure
                if not predictions_path:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"
                    if not predictions_path.exists():
                        logger.warning(
                            "predictions_path not found at %s - evaluate step may fail",
                            predictions_path,
                        )

                if predictions_path:
                    logger.info(
                        "Auto-wired predictions_path to evaluate: %s", predictions_path
                    )
                    step_config["predictions_path"] = str(predictions_path)

            # Auto-wire dataset_path
            if "dataset_path" not in step_config:
                dataset_path = None

                # First try: Get from previous step outputs
                if "build_dataset_prepare_dataset" in step_outputs:
                    dataset_path = step_outputs["build_dataset_prepare_dataset"].get(
                        "dataset_jsonl_path"
                    )

                # Fallback: Derive from working_dir structure
                if not dataset_path:
                    dataset_path = working_dir / "dataset" / "dataset.jsonl"
                    if not dataset_path.exists():
                        logger.warning(
                            "dataset_path not found at %s - ground truth may not be available",
                            dataset_path,
                        )

                if dataset_path:
                    logger.info("Auto-wired dataset_path to evaluate: %s", dataset_path)
                    step_config["dataset_path"] = str(dataset_path)

            # Auto-wire system_prompt_file
            if "system_prompt_file" not in step_config:
                vlm_prompt_path = None

                # First try: Get from previous step outputs
                if "build_dataset_prepare_dataset" in step_outputs:
                    vlm_prompt_path = step_outputs["build_dataset_prepare_dataset"].get(
                        "vlm_prompt_path"
                    )

                # Fallback: Derive from working_dir structure
                if not vlm_prompt_path:
                    vlm_prompt_path = working_dir / "dataset" / "vlm_system_prompt.txt"
                    if not vlm_prompt_path.exists():
                        logger.debug(
                            "system_prompt_file not found at %s - will not be included in report",
                            vlm_prompt_path,
                        )
                        vlm_prompt_path = None

                if vlm_prompt_path:
                    logger.info(
                        "Auto-wired system_prompt_file to evaluate: %s", vlm_prompt_path
                    )
                    step_config["system_prompt_file"] = str(vlm_prompt_path)

            # Auto-wire output_dir from working_dir
            if "output_dir" not in step_config:
                output_dir = working_dir / "evaluation"
                logger.info("Auto-wired output_dir to evaluate: %s", output_dir)
                step_config["output_dir"] = str(output_dir)

        # Auto-wire outputs from restore_usd/apply/refine step for render step
        if step_name == "render":
            step_outputs = pipeline_state.get("step_outputs", {})

            if "input_usd_path" not in step_config:
                usd_path = None
                source_step = None

                # restore_usd normally restores prediction paths, not a USD file.
                # Keep this fallback for older configs that may emit a USD path.
                if "restore_usd" in step_outputs:
                    usd_path = step_outputs["restore_usd"].get("restored_usd_path")
                    if usd_path:
                        source_step = "restore_usd"

                if not usd_path and "refine" in step_outputs:
                    usd_path = step_outputs["refine"].get("final_output_path")
                    if not usd_path:
                        usd_path = step_outputs["refine"].get("output_usd_path")
                    if usd_path:
                        source_step = "refine"

                if not usd_path and "apply" in step_outputs:
                    usd_path = step_outputs["apply"].get("output_usd_path")
                    if usd_path:
                        source_step = "apply"

                if usd_path and "input_usd_path" not in step_config:
                    logger.info(
                        "Auto-wired input_usd_path to render from %s: %s",
                        source_step,
                        usd_path,
                    )
                    step_config["input_usd_path"] = str(usd_path)

        # Special handling: Auto-wire harmonize_predictions with predictions path
        # Harmonize runs BEFORE validate — reads from predict/benchmark.
        if step_name == "harmonize_predictions":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            if "predictions_path" not in step_config:
                predictions_path = None
                if "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")

                if not predictions_path:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"

                logger.info(
                    "Auto-wired predictions_path to harmonize_predictions: %s",
                    predictions_path,
                )
                step_config["predictions_path"] = str(predictions_path)

            # Auto-wire optimized USD path for geometry-based grouping
            if "optimized_usd_path" not in step_config:
                if "optimize_usd" in step_outputs:
                    opt_usd = step_outputs["optimize_usd"].get("optimized_usd_path")
                    if opt_usd:
                        step_config["optimized_usd_path"] = str(opt_usd)

        # Special handling: Auto-wire validate_predictions with predictions path
        # Validate runs AFTER harmonize — prefers harmonized output.
        if step_name == "validate_predictions":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            if "predictions_path" not in step_config:
                predictions_path = None
                if "harmonize_predictions" in step_outputs:
                    predictions_path = step_outputs["harmonize_predictions"].get(
                        "predictions_path"
                    )
                elif "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")

                if not predictions_path:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"

                logger.info(
                    "Auto-wired predictions_path to validate_predictions: %s",
                    predictions_path,
                )
                step_config["predictions_path"] = str(predictions_path)

        # Special handling: Auto-wire restore_usd with required paths and metadata
        if step_name == "restore_usd":
            step_outputs = pipeline_state.get("step_outputs", {})
            working_dir = Path(context.get("working_dir", Path.cwd()))

            # Auto-wire original USD path from optimize_usd step outputs
            if "optimize_usd" in step_outputs:
                original_usd_path = step_outputs["optimize_usd"].get(
                    "original_usd_path"
                )
                if original_usd_path:
                    step_config["original_usd_path"] = str(original_usd_path)
                    logger.info(
                        "Auto-wired original_usd_path from optimize_usd: %s",
                        original_usd_path,
                    )
            else:
                # Fallback to path_resolver if optimize_usd didn't run
                path_resolver = context.get("path_resolver")
                if path_resolver and path_resolver.input_usd:
                    step_config["original_usd_path"] = str(path_resolver.input_usd)
                    logger.info(
                        "Auto-wired original_usd_path from input: %s",
                        path_resolver.input_usd,
                    )

            # Auto-wire predictions — prefer harmonized > validated > raw
            if "predictions_path" not in step_config:
                predictions_path = None
                if "harmonize_predictions" in step_outputs:
                    predictions_path = step_outputs["harmonize_predictions"].get(
                        "predictions_path"
                    )
                elif "validate_predictions" in step_outputs:
                    predictions_path = step_outputs["validate_predictions"].get(
                        "predictions_path"
                    )
                elif "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")

                if predictions_path:
                    logger.info(
                        "Auto-wired predictions_path to restore: %s", predictions_path
                    )
                    step_config["predictions_path"] = str(predictions_path)
                else:
                    predictions_path = working_dir / "predictions" / "predictions.jsonl"
                    step_config["predictions_path"] = str(predictions_path)

            # Auto-wire output predictions path
            if "output_predictions_path" not in step_config:
                output_predictions_path = (
                    working_dir / "restored" / "restored_predictions.jsonl"
                )
                logger.info(
                    "Auto-wired output_predictions_path to restore: %s",
                    output_predictions_path,
                )
                step_config["output_predictions_path"] = str(output_predictions_path)

            # Inject optimization metadata
            if "optimization_metadata" in pipeline_state:
                step_config["optimization_metadata"] = pipeline_state[
                    "optimization_metadata"
                ]
                logger.info("Injected optimization metadata into restore_usd config")
            else:
                # Try to find metadata file in standard location
                optimization_metadata_path = (
                    working_dir / "optimized" / "optimized_input.metadata.json"
                )
                if optimization_metadata_path.exists():
                    with open(optimization_metadata_path, encoding="utf-8") as f:
                        optimization_metadata = json.load(f)
                    step_config["optimization_metadata"] = optimization_metadata
                else:
                    logger.warning(
                        "No optimization metadata found at %s - restore_usd may not work correctly",
                        optimization_metadata_path,
                    )

        # Auto-wire validate_output with output USD and original USD paths
        if step_name == "validate_output":
            step_outputs = pipeline_state.get("step_outputs", {})

            # Auto-wire output USD path from apply/refine step
            if "refine" in step_outputs:
                usd_path = step_outputs["refine"].get(
                    "final_output_path"
                ) or step_outputs["refine"].get("output_usd_path")
                if usd_path:
                    step_config["input_usd_path"] = str(usd_path)
                    logger.info(
                        "Auto-wired input_usd_path to validate_output from refine: %s",
                        usd_path,
                    )
            elif "apply" in step_outputs:
                usd_path = step_outputs["apply"].get("output_usd_path")
                if usd_path:
                    step_config["input_usd_path"] = str(usd_path)
                    logger.info(
                        "Auto-wired input_usd_path to validate_output from apply: %s",
                        usd_path,
                    )

            # Auto-wire original USD path for baseline comparison
            if "original_usd_path" not in step_config:
                # Try optimize_usd (it stores the original path)
                if "optimize_usd" in step_outputs:
                    original = step_outputs["optimize_usd"].get("original_usd_path")
                    if original:
                        step_config["original_usd_path"] = str(original)
                        logger.info(
                            "Auto-wired original_usd_path to validate_output: %s",
                            original,
                        )
                else:
                    # Fall back to path_resolver's original input
                    path_resolver = context.get("path_resolver")
                    if path_resolver:
                        # Use the config's original input USD (before optimize rewrote it)
                        config = context.get("config", {})
                        input_section = config.get("input", {})
                        raw_input = input_section.get("usd_path")
                        if raw_input:
                            resolved = path_resolver.resolve_path(raw_input)
                            step_config["original_usd_path"] = str(resolved)
                            logger.info(
                                "Auto-wired original_usd_path to validate_output from config: %s",
                                resolved,
                            )

            # Inject cached baseline from validate_input (avoids re-validating input).
            # IMPORTANT: Do NOT use the cached baseline when validate_input applied
            # a fix — the cached result describes the pre-fix state, but downstream
            # steps consumed the fixed USD. Let validate_output re-validate the
            # fixed input to get an accurate baseline.
            if "validate_input" in step_outputs:
                vi_outputs = step_outputs["validate_input"]
                used_fix = bool(vi_outputs.get("validation_fixed_usd_path"))

                if used_fix:
                    # Point original_usd_path to the fixed file so
                    # validate_output re-validates it for baseline
                    fixed_path = vi_outputs["validation_fixed_usd_path"]
                    step_config["original_usd_path"] = str(fixed_path)
                    logger.info(
                        "validate_input used fix — baseline will be "
                        "re-validated from fixed input: %s",
                        fixed_path,
                    )
                else:
                    baseline_result = vi_outputs.get("validation_result")
                    if baseline_result:
                        step_config["baseline_validation"] = baseline_result
                        logger.info(
                            "Injected cached baseline from validate_input (%d issues)",
                            len(baseline_result.get("issues", [])),
                        )
        # Auto-wire cluster_prims: needs dataset_path and working_dir
        if step_name in ["cluster_prims", "expand_cluster_predictions"]:
            step_outputs = pipeline_state.get("step_outputs", {})

        if step_name == "cluster_prims":
            working_dir = Path(context.get("working_dir", Path.cwd()))
            if "dataset_path" not in step_config:
                dataset_path = None
                if "build_dataset_prepare_dataset" in step_outputs:
                    dataset_path = step_outputs["build_dataset_prepare_dataset"].get(
                        "dataset_jsonl_path"
                    )
                if not dataset_path:
                    dataset_path = str(working_dir / "dataset" / "dataset.jsonl")
                step_config["dataset_path"] = str(dataset_path)
                logger.info(
                    "Auto-wired dataset_path to cluster_prims: %s", dataset_path
                )
            if "working_dir" not in step_config:
                step_config["working_dir"] = str(working_dir)

        # Auto-wire predict: use representative dataset when clustering ran
        if step_name in ["predict", "benchmark"]:
            step_outputs = pipeline_state.get("step_outputs", {})
            if "cluster_prims" in step_outputs:
                cluster_out = step_outputs["cluster_prims"]
                if cluster_out.get("cluster_prims_ran"):
                    reps_path = cluster_out.get("dataset_representatives_path")
                    if reps_path:
                        step_config["dataset"] = str(reps_path)
                        logger.info(
                            "Auto-wired predict dataset to cluster representatives: %s",
                            reps_path,
                        )

        # Auto-wire expand_cluster_predictions
        if step_name == "expand_cluster_predictions":
            working_dir = Path(context.get("working_dir", Path.cwd()))
            if "cluster_prims" in step_outputs:
                cluster_out = step_outputs["cluster_prims"]
                if cluster_out.get("cluster_prims_ran") is False:
                    step_config["cluster_prims_ran"] = False
                else:
                    if "predictions_path" not in step_config:
                        predictions_path = None
                        if "predict" in step_outputs:
                            predictions_path = step_outputs["predict"].get(
                                "predictions_path"
                            )
                        if not predictions_path:
                            predictions_path = str(
                                working_dir / "predictions" / "predictions.jsonl"
                            )
                        step_config["predictions_path"] = str(predictions_path)
                    if "cluster_map_path" not in step_config:
                        cluster_map_path = cluster_out.get("cluster_map_path")
                        if not cluster_map_path:
                            cluster_map_path = str(
                                working_dir / "clusters" / "cluster_map.jsonl"
                            )
                        step_config["cluster_map_path"] = str(cluster_map_path)
            else:
                step_config["cluster_prims_ran"] = False

        # Auto-wire predictions_path for validate_predictions and harmonize_predictions
        if step_name in ["validate_predictions", "harmonize_predictions"]:
            step_outputs = pipeline_state.get("step_outputs", {})
            if "predictions_path" not in step_config:
                predictions_path = None
                if "predict" in step_outputs:
                    predictions_path = step_outputs["predict"].get("predictions_path")
                elif "benchmark" in step_outputs:
                    predictions_path = step_outputs["benchmark"].get("predictions_path")
                if predictions_path:
                    logger.info(
                        "Auto-wired predictions_path for %s: %s",
                        step_name,
                        predictions_path,
                    )
                    step_config["predictions_path"] = str(predictions_path)

        # Create temporary config file for the step
        # This is needed because existing workflows expect config_path
        working_dir = Path(context.get("working_dir", Path.cwd()))
        temp_config_path = self._create_temp_config_file(
            step_name, step_config, working_dir
        )

        # Import workflows
        from material_agent.workflows import (
            create_apply_workflow_from_config,
            create_benchmark_workflow_from_config,
            create_cluster_prims_workflow_from_config,
            create_evaluation_workflow_from_config,
            create_expand_cluster_predictions_workflow_from_config,
            create_generate_reference_image_workflow_from_config,
            create_harmonize_predictions_workflow_from_config,
            create_identify_asset_workflow_from_config,
            create_iterative_apply_workflow_from_config,
            create_optimize_usd_workflow_from_config,
            create_pdf_vectorstore_workflow_from_config,
            create_prediction_workflow_from_config,
            create_prepare_dataset_workflow_from_config,
            create_render_preview_workflow_from_config,
            create_render_workflow_from_config,
            create_restore_usd_workflow_from_config,
            create_usd_data_preparation_workflow_from_config,
            create_validate_input_workflow_from_config,
            create_validate_output_workflow_from_config,
            create_validate_predictions_workflow_from_config,
        )

        # Map step names to workflow factories
        workflow_map = {
            "validate_input": create_validate_input_workflow_from_config,
            "optimize_usd": create_optimize_usd_workflow_from_config,
            "render_preview": create_render_preview_workflow_from_config,
            "identify_asset": create_identify_asset_workflow_from_config,
            "generate_reference_image": create_generate_reference_image_workflow_from_config,
            "build_dataset_usd": create_usd_data_preparation_workflow_from_config,
            "build_dataset_pdf_vectorstore": create_pdf_vectorstore_workflow_from_config,
            "build_dataset_prepare_dataset": create_prepare_dataset_workflow_from_config,
            "cluster_prims": create_cluster_prims_workflow_from_config,
            "predict": create_prediction_workflow_from_config,
            "expand_cluster_predictions": create_expand_cluster_predictions_workflow_from_config,
            "benchmark": create_benchmark_workflow_from_config,
            "validate_predictions": create_validate_predictions_workflow_from_config,
            "harmonize_predictions": create_harmonize_predictions_workflow_from_config,
            "evaluate": create_evaluation_workflow_from_config,
            "apply": create_apply_workflow_from_config,
            "refine": create_iterative_apply_workflow_from_config,
            "restore_usd": create_restore_usd_workflow_from_config,
            "validate_output": create_validate_output_workflow_from_config,
            "render": create_render_workflow_from_config,
        }

        if step_name not in workflow_map:
            raise ValueError(f"Unknown step: {step_name}")

        # Create workflow
        workflow = workflow_map[step_name]()

        # Prepare step context
        step_context = {"config_path": str(temp_config_path)}
        if step_name == "identify_asset":
            step_context.update(step_config)

        # Pass event listener to individual workflows if available
        if "event_listener" in context:
            step_context["event_listener"] = context["event_listener"]

        # Extract and pass report compression configuration if present
        if "report" in step_config:
            report_config = step_config["report"]
            if isinstance(report_config, dict):
                # Map report config keys to context keys
                if "image_max_size" in report_config:
                    step_context["report_image_max_size"] = report_config[
                        "image_max_size"
                    ]
                if "image_format" in report_config:
                    step_context["report_image_format"] = report_config["image_format"]
                if "image_quality" in report_config:
                    step_context["report_image_quality"] = report_config[
                        "image_quality"
                    ]

        # Pass pipeline statistics to steps that generate reports
        # These values were collected from earlier pipeline steps (optimize_usd, build_dataset_usd)
        # Use 'is not None' to ensure 0 values are also propagated
        if step_name in ["predict", "benchmark", "evaluate"]:
            # Pass original prim count (from optimize_usd step)
            if context.get("original_prim_count") is not None:
                step_context["original_prim_count"] = context["original_prim_count"]
            # Pass processed prim count and image count (from build_dataset_usd step)
            if context.get("num_prims") is not None:
                step_context["num_prims"] = context["num_prims"]
            if context.get("num_images") is not None:
                step_context["num_images"] = context["num_images"]

        # Execute workflow (async)
        logger.debug("Running workflow for %s", step_name)
        result = await workflow.arun(step_context)

        if not result:
            raise RuntimeError(
                f"Step '{step_name}' did not complete successfully - workflow returned empty result"
            )

        # Check if workflow encountered errors
        if result.get("error") or result.get("workflow_terminated"):
            failed_task = result.get("failed_task", "unknown")
            error_msg = result.get("error", "Workflow terminated without error message")
            raise RuntimeError(
                f"Step '{step_name}' failed at task '{failed_task}': {error_msg}"
            )

        # Extract outputs
        outputs = self._extract_step_outputs(step_name, result)

        # Special handling: Store optimization metadata for restore_usd
        if step_name == "optimize_usd":
            if "optimization_metadata" in result:
                pipeline_state["optimization_metadata"] = result[
                    "optimization_metadata"
                ]
                logger.info("Stored optimization metadata for restore_usd step")

        return outputs

    def _create_temp_config_file(
        self, step_name: str, step_config: dict[str, Any], working_dir: Path
    ) -> Path:
        """Create temporary config file for a step.

        Args:
            step_name: Name of the step
            step_config: Step configuration dictionary
            working_dir: Working directory

        Returns:
            Path to the temporary config file
        """
        import uuid

        import yaml

        temp_dir = working_dir / ".pipeline_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique ID to prevent collisions in concurrent execution
        unique_id = uuid.uuid4().hex[:8]
        temp_config_path = temp_dir / f"{step_name}_config_{unique_id}.yaml"

        # Create a copy of step_config without non-serializable objects
        # (e.g., Pydantic models, dataclass instances)
        serializable_config = {}
        for key, value in step_config.items():
            if key == "renderer" and isinstance(value, dict):
                # Copy renderer config but exclude non-serializable objects
                renderer_copy = {
                    k: v
                    for k, v in value.items()
                    if not k.startswith(
                        "_"
                    )  # Skip _unified_config, _rendering_modes_config
                }
                serializable_config[key] = renderer_copy
            else:
                serializable_config[key] = value

        # Write config to temp file using safe_dump (no Python tags).
        # _make_yaml_safe recursively converts enums, dataclasses, etc.
        # to plain Python types so safe_dump can handle them.
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                _make_yaml_safe(serializable_config),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        logger.debug("Created temp config: %s", temp_config_path)
        return temp_config_path

    def _extract_step_outputs(
        self, step_name: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract relevant outputs from step result.

        Args:
            step_name: Name of the step
            result: Step execution result

        Returns:
            Dictionary with relevant outputs
        """
        outputs = {}

        if step_name == "render_preview":
            outputs["output_dir"] = result.get("output_dir")
            outputs["rendered_preview_paths"] = result.get(
                "rendered_preview_paths",
                [],
            )
            outputs["composition_images"] = result.get("composition_images", [])

        elif step_name == "identify_asset":
            outputs["identification"] = result.get("identification")
            outputs["identification_path"] = result.get("identification_path")
            outputs["image_gen_prompt"] = result.get("image_gen_prompt")

        elif step_name == "generate_reference_image":
            outputs["output_dir"] = result.get("output_dir")
            outputs["generated_reference_image_paths"] = result.get(
                "generated_reference_image_paths",
                [],
            )

        elif step_name == "build_dataset_usd":
            outputs["output_dir"] = result.get("output_dir")
            outputs["usd_dataset_dir"] = result.get("output_dir")
            outputs["num_prims"] = result.get("num_prims", 0)
            outputs["num_images"] = result.get("num_images", 0)

        elif step_name == "build_dataset_pdf_vectorstore":
            outputs["vectorstore_dir"] = result.get("output_dir")

        elif step_name == "build_dataset_prepare_dataset":
            outputs["dataset_path"] = result.get("dataset_path")
            outputs["dataset_jsonl_path"] = result.get("dataset_jsonl_path")
            outputs["vlm_prompt_path"] = result.get("vlm_prompt_path")
            outputs["num_entries"] = result.get("num_entries", 0)

        elif step_name == "cluster_prims":
            outputs["cluster_map_path"] = result.get("cluster_map_path")
            outputs["dataset_representatives_path"] = result.get(
                "dataset_representatives_path"
            )
            outputs["cluster_prims_ran"] = result.get("cluster_prims_ran", False)
            outputs["cluster_summary_path"] = result.get("cluster_summary_path")
            outputs["cluster_report_path"] = result.get("cluster_report_path")
            outputs["cluster_total_prims"] = result.get("cluster_total_prims", 0)
            outputs["cluster_count"] = result.get("cluster_count", 0)
            outputs["cluster_representative_count"] = result.get(
                "cluster_representative_count", 0
            )
            outputs["cluster_reduction_percent"] = result.get(
                "cluster_reduction_percent", 0.0
            )
            outputs["cluster_multi_member_count"] = result.get(
                "cluster_multi_member_count", 0
            )
            outputs["cluster_singleton_count"] = result.get(
                "cluster_singleton_count", 0
            )
            outputs["cluster_max_size"] = result.get("cluster_max_size")
            outputs["cluster_capped_count"] = result.get("cluster_capped_count", 0)

        elif step_name in ["predict", "benchmark"]:
            outputs["predictions_path"] = result.get("predictions_path")
            outputs["predictions_count"] = result.get("predictions_count")

        elif step_name == "expand_cluster_predictions":
            outputs["predictions_path"] = result.get("predictions_path")

        elif step_name == "validate_predictions":
            outputs["predictions_path"] = result.get("predictions_path")
            outputs["validation_stats"] = result.get("validation_stats")

        elif step_name == "harmonize_predictions":
            outputs["predictions_path"] = result.get("predictions_path")
            outputs["harmonized_count"] = result.get("harmonized_count")
            outputs["remap"] = result.get("remap")

        elif step_name == "evaluate":
            outputs["evaluation_path"] = result.get("evaluation_path")
            outputs["html_report_path"] = result.get("html_report_path")
            outputs["metrics"] = result.get("metrics")

        elif step_name == "optimize_usd":
            outputs["optimized_usd_path"] = result.get("optimized_usd_path")
            outputs["optimization_success"] = result.get("optimization_success")
            outputs["original_usd_path"] = result.get("original_usd_path")
            outputs["original_prim_count"] = result.get("original_prim_count")
            outputs["optimization_metadata"] = result.get("optimization_metadata")

        elif step_name == "apply":
            outputs["output_usd_path"] = result.get("output_usd_path")
            outputs["materials_applied"] = result.get("materials_applied")

        elif step_name == "refine":
            # Get the final output path from the iterative workflow.
            outputs["output_usd_path"] = result.get("final_output_path")
            outputs["final_output_path"] = result.get("final_output_path")

        elif step_name == "render":
            outputs["rendered_image_paths"] = result.get("rendered_image_paths")
            outputs["rendered_image_path"] = result.get("rendered_image_path")
            outputs["flattened_usd_path"] = result.get("flattened_usd_path")

        elif step_name == "validate_input":
            outputs["validation_result"] = result.get("validation_result")
            outputs["validation_summary"] = result.get("validation_summary")
            outputs["validation_is_valid"] = result.get("validation_is_valid")
            outputs["validation_fixed_usd_path"] = result.get(
                "validation_fixed_usd_path"
            )
            outputs["validation_success"] = result.get("validation_success")
            outputs["validation_skipped"] = result.get("validation_skipped")
            outputs["validation_error"] = result.get("validation_error")

        elif step_name == "validate_output":
            outputs["validation_result"] = result.get("validation_result")
            outputs["validation_summary"] = result.get("validation_summary")
            outputs["validation_is_valid"] = result.get("validation_is_valid")
            outputs["validation_regression"] = result.get("validation_regression")
            outputs["validation_new_issues"] = result.get("validation_new_issues")
            outputs["validation_success"] = result.get("validation_success")
            outputs["validation_skipped"] = result.get("validation_skipped")
            outputs["validation_error"] = result.get("validation_error")

        elif step_name == "restore_usd":
            outputs["restored_usd_path"] = result.get("restored_usd_path")
            outputs["restored_predictions_path"] = result.get(
                "restored_predictions_path"
            )
            outputs["restore_success"] = result.get("restore_success")
            outputs["predictions_count"] = result.get("predictions_count")

        return outputs
