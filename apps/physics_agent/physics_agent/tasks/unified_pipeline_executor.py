# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified pipeline executor task for Physics Agent.

This executor works with step configs that have already been prepared by
UnifiedPipelineConfigTask, so it doesn't need to create temporary config files
or load configs again.
"""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.base_pipeline_executor import BasePipelineExecutor
from world_understanding.agentic.events import get_listener

logger = logging.getLogger(__name__)


def resolve_apply_physics_inputs(
    step_outputs: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    """Pick the predictions source and output_key for the apply_physics step.

    When optimize_usd ran, apply_physics must author onto the optimized USD,
    so it must use raw predict output keyed to optimized prim paths. The
    restored predictions remain useful as an artifact/reporting output but
    are not used for physics authoring in this topology.

    Resolution order:
      1. If `optimize_usd` ran, use `predict.predictions_path` because it
         references the optimized/deinstanced USD that apply_physics opens.
      2. If no optimization ran, prefer `restore_usd`'s
         `restored_predictions_path` when it is set, then fall back to
         `predict.predictions_path`.
      3. If no predictions are available, return `(None, ...)` and
         let the caller warn / fail downstream.

    Note that `restore_usd` records a no-op output entry when `optimize_usd`
    was disabled; the `.get("restored_predictions_path")` check correctly
    falls through to the predict fallback in that case.

    Args:
        step_outputs: Mapping of completed step name → its output dict, as
            recorded in `pipeline_state["step_outputs"]`.

    Returns:
        `(predictions_path, output_key)` where `predictions_path` is the
        absolute path to the JSONL to read or `None` if neither source is
        available, and `output_key` is the top-level dict key under which
        each prediction holds its classification (defaults to
        `"classification"` if predict didn't record one).
    """
    restored_path = (step_outputs.get("restore_usd") or {}).get(
        "restored_predictions_path"
    )
    predict_outputs = step_outputs.get("predict") or {}
    predict_path = predict_outputs.get("predictions_path")

    predictions_path: str | None = None
    if "optimize_usd" in step_outputs:
        if predict_path:
            predictions_path = str(predict_path)
    elif restored_path:
        predictions_path = str(restored_path)
    elif "predict" in step_outputs:
        if predict_path:
            predictions_path = str(predict_path)

    output_key = predict_outputs.get("output_key") or "classification"

    return predictions_path, output_key


class UnifiedPipelineExecutorTask(BasePipelineExecutor):
    """Execute pipeline steps with pre-configured, auto-wired step configs.

    This executor works with the unified config system where:
    - All paths are already resolved by UnifiedPipelineConfigTask
    - Step configs are complete and ready to use
    - No additional config loading needed

    Input context keys:
        - steps_to_run: List of step names to execute
        - step_configs: Dictionary of pre-configured step configs
        - path_resolver: ProjectPathResolver instance
        - working_dir: Working directory
        - resume: Optional flag to resume from checkpoint

    Output context keys:
        - pipeline_results: Dictionary of results from each step
        - pipeline_state: Final pipeline state
    """

    def __init__(self):
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

    # ========== Pipeline Execution Logic ==========

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Execute pipeline steps in sequence.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with pipeline results
        """
        listener = get_listener(context, logger_name=__name__)

        steps_to_run = context.get("steps_to_run", [])
        step_configs = context.get("step_configs", {})
        working_dir = context.get("working_dir", Path.cwd())
        resume = context.get("resume", False)
        clean = context.get("clean", False)

        if not steps_to_run:
            raise ValueError("No steps to run in pipeline")

        # Clean working directory if requested
        if clean:
            import shutil

            working_dir_path = Path(working_dir)

            # Safety checks
            if working_dir_path == Path.home() or working_dir_path == Path("/"):
                raise ValueError(f"Refusing to delete: {working_dir_path}")

            if len(working_dir_path.parts) < 2:
                raise ValueError(f"Path too shallow: {working_dir_path}")

            if working_dir_path.exists():
                logger.info("Cleaning working directory: %s", working_dir_path)
                shutil.rmtree(working_dir_path)

        # Ensure working directory exists
        Path(working_dir).mkdir(parents=True, exist_ok=True)

        # Get session info
        session_id = context.get("session_id")
        project_name = context.get("project_name")

        # Initialize pipeline state
        pipeline_state = {
            "session_id": session_id,
            "project_name": project_name,
            "completed_steps": [],
            "failed_steps": [],
            "step_outputs": {},
            "current_step": None,
        }

        # Load existing state if resuming
        state_file = Path(working_dir) / ".pipeline_state.json"
        if resume and state_file.exists():
            logger.info("Resuming from checkpoint: %s", state_file)
            with open(state_file, encoding="utf-8") as f:
                pipeline_state = json.load(f)
            logger.info(
                "Previously completed: %s", ", ".join(pipeline_state["completed_steps"])
            )

        # Display pipeline start
        logger.info("=" * 80)
        logger.info("PIPELINE STARTING")
        logger.info("=" * 80)
        logger.info("Session ID: %s", session_id)
        logger.info("Project: %s", project_name)
        logger.info("Working Directory: %s", working_dir)
        logger.info("Steps: %s", ", ".join(steps_to_run))
        logger.info("=" * 80)

        # Emit pipeline start event
        listener.event(
            "pipeline.started",
            {
                "session_id": session_id,
                "project_name": project_name,
                "working_dir": str(working_dir),
                "steps": steps_to_run,
            },
        )

        # Execute each step
        for i, step_name in enumerate(steps_to_run, 1):
            # Skip if already completed (resume mode)
            if resume and step_name in pipeline_state["completed_steps"]:
                logger.info(
                    "[%d/%d] Skipping %s (already completed)",
                    i,
                    len(steps_to_run),
                    step_name,
                )
                continue

            pipeline_state["current_step"] = step_name
            event_listener = context.get("event_listener")

            try:
                logger.info(
                    "\n[%d/%d] Executing step: %s", i, len(steps_to_run), step_name
                )

                if event_listener:
                    event_listener.event(
                        "step.started",
                        {
                            "step_name": step_name,
                            "step_index": i,
                            "total_steps": len(steps_to_run),
                        },
                    )

                # Execute the step
                step_config = step_configs[step_name]
                outputs = self._execute_step(
                    step_name, step_config, context, object_store, pipeline_state
                )

                # Mark step as completed
                pipeline_state["completed_steps"].append(step_name)
                pipeline_state["step_outputs"][step_name] = outputs

                # Save checkpoint
                self._save_checkpoint(pipeline_state, state_file)

                logger.info("Step '%s' completed successfully", step_name)

                if event_listener:
                    event_listener.event(
                        "step.completed",
                        {"step_name": step_name, "outputs": outputs},
                    )

            except Exception as e:
                logger.error("Step '%s' failed: %s", step_name, e, exc_info=True)
                pipeline_state["failed_steps"].append(step_name)
                pipeline_state["current_step"] = None

                self._save_checkpoint(pipeline_state, state_file)

                if event_listener:
                    event_listener.event(
                        "step.failed",
                        {"step_name": step_name, "error": str(e)},
                    )

                raise RuntimeError(f"Pipeline failed at step '{step_name}': {e}") from e

        # Pipeline completed
        pipeline_state["current_step"] = None
        self._save_checkpoint(pipeline_state, state_file)

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Session ID: %s", session_id)
        logger.info("Completed Steps: %s", ", ".join(pipeline_state["completed_steps"]))
        logger.info("=" * 80)

        listener.event(
            "pipeline.completed",
            {
                "session_id": session_id,
                "project_name": project_name,
                "completed_steps": pipeline_state["completed_steps"],
            },
        )

        context["pipeline_results"] = pipeline_state["step_outputs"]
        context["pipeline_state"] = "completed"

        return context

    def _execute_step(
        self,
        step_name: str,
        step_config: dict[str, Any],
        context: dict[str, Any],
        object_store: Any,
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single pipeline step.

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
        working_dir = Path(context.get("working_dir", Path.cwd()))

        # Auto-wire optimized USD for identify_asset and build_dataset_usd steps
        # When optimize_usd has run, use the optimized USD for downstream steps
        if step_name in ("identify_asset", "build_dataset_usd"):
            if "optimize_usd" in step_outputs:
                optimized_usd_path = step_outputs["optimize_usd"].get(
                    "optimized_usd_path"
                )
                if optimized_usd_path:
                    logger.info(
                        "Auto-wired usd_path for %s from optimize_usd: %s",
                        step_name,
                        optimized_usd_path,
                    )
                    step_config["usd_path"] = str(optimized_usd_path)

        # Auto-wire identification results into prepare_dataset step
        if step_name == "build_dataset_prepare_dataset":
            if "identify_asset" in step_outputs:
                identification = step_outputs["identify_asset"].get("identification")
                if identification:
                    asset_type = identification.get("asset_type", "")
                    asset_subtype = identification.get("asset_subtype", "")
                    if asset_type:
                        # Inject asset type into prompts context
                        prompts = step_config.get("prompts", {})
                        system_prompt = prompts.get("system", "")
                        if system_prompt and "{asset_type}" not in system_prompt:
                            # Prepend identification context
                            id_context = f"This is a {asset_type}"
                            if asset_subtype:
                                id_context += f" ({asset_subtype})"
                            id_context += ". "
                            prompts["system"] = id_context + system_prompt
                            step_config["prompts"] = prompts
                        logger.info(
                            "Auto-wired identification for %s: %s / %s",
                            step_name,
                            asset_type,
                            asset_subtype,
                        )

        # Auto-wire apply_physics inputs from prior USD/prediction steps.
        # Resolution logic lives in resolve_apply_physics_inputs so the supported
        # pipeline topologies are directly unit-testable.
        if step_name == "apply_physics":
            if "optimize_usd" in step_outputs:
                optimized_usd_path = step_outputs["optimize_usd"].get(
                    "optimized_usd_path"
                )
                if not optimized_usd_path:
                    raise ValueError(
                        "apply_physics cannot run after optimize_usd without "
                        "optimized_usd_path"
                    )
                step_config["usd_path"] = str(optimized_usd_path)
                logger.info(
                    "Auto-wired usd_path for apply_physics from optimize_usd: %s",
                    optimized_usd_path,
                )

            predictions_path, output_key = resolve_apply_physics_inputs(step_outputs)
            if predictions_path:
                step_config["predictions_path"] = predictions_path
                if "optimize_usd" in step_outputs:
                    source = "predict"
                elif (step_outputs.get("restore_usd") or {}).get(
                    "restored_predictions_path"
                ):
                    source = "restore_usd"
                else:
                    source = "predict"
                logger.info(
                    "Auto-wired predictions_path for apply_physics from %s: %s",
                    source,
                    predictions_path,
                )
            else:
                logger.warning(
                    "apply_physics step has no predictions to wire — "
                    "neither predict nor restore_usd has run yet"
                )
            step_config["output_key"] = output_key

        # Auto-wire restore_usd inputs from optimize_usd and predict steps
        if step_name == "restore_usd":
            if "optimize_usd" in step_outputs:
                opt_outputs = step_outputs["optimize_usd"]
                original_usd_path = opt_outputs.get("original_usd_path")
                optimization_metadata = opt_outputs.get("optimization_metadata")
                if original_usd_path:
                    step_config["original_usd_path"] = str(original_usd_path)
                    logger.info(
                        "Auto-wired original_usd_path for restore_usd: %s",
                        original_usd_path,
                    )
                if optimization_metadata:
                    step_config["optimization_metadata"] = optimization_metadata
                    logger.info("Auto-wired optimization_metadata for restore_usd")
            else:
                logger.warning(
                    "restore_usd requested but optimize_usd has not run - "
                    "skipping restoration (no optimization metadata available)"
                )
                return {"restore_skipped": True, "reason": "no optimization metadata"}

            if "predict" in step_outputs:
                predictions_path = step_outputs["predict"].get("predictions_path")
                if predictions_path:
                    step_config["predictions_path"] = str(predictions_path)
                    logger.info(
                        "Auto-wired predictions_path for restore_usd: %s",
                        predictions_path,
                    )

            # Set output path in working dir
            if "output_predictions_path" not in step_config:
                step_config["output_predictions_path"] = str(
                    working_dir / "restored_predictions.jsonl"
                )

        # Create temporary config file for the step (after all auto-wiring)
        temp_config_path = self._create_temp_config_file(
            step_name, step_config, working_dir
        )

        # Import workflows
        from physics_agent.workflows import (
            create_apply_physics_workflow_from_config,
            create_identify_asset_workflow_from_config,
            create_optimize_usd_workflow_from_config,
            create_prediction_workflow_from_config,
            create_prepare_dataset_workflow_from_config,
            create_restore_usd_workflow_from_config,
            create_usd_data_preparation_workflow_from_config,
        )

        # Map step names to workflow factories
        workflow_map = {
            "optimize_usd": create_optimize_usd_workflow_from_config,
            "build_dataset_usd": create_usd_data_preparation_workflow_from_config,
            "identify_asset": create_identify_asset_workflow_from_config,
            "build_dataset_prepare_dataset": create_prepare_dataset_workflow_from_config,
            "predict": create_prediction_workflow_from_config,
            "restore_usd": create_restore_usd_workflow_from_config,
            "apply_physics": create_apply_physics_workflow_from_config,
        }

        if step_name not in workflow_map:
            raise ValueError(f"Unknown step: {step_name}")

        # Create workflow
        workflow = workflow_map[step_name]()

        # Prepare step context
        step_context = {"config_path": str(temp_config_path)}

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

        # Execute workflow
        logger.debug("Running workflow for %s", step_name)
        result = workflow.run(step_context)

        if not result:
            raise RuntimeError(f"Step '{step_name}' returned empty result")

        if result.get("error") or result.get("workflow_terminated"):
            failed_task = result.get("failed_task", "unknown")
            error_msg = result.get("error", "Workflow terminated")
            raise RuntimeError(
                f"Step '{step_name}' failed at '{failed_task}': {error_msg}"
            )

        # Extract outputs
        return self._extract_step_outputs(step_name, result)

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

        unique_id = uuid.uuid4().hex[:8]
        temp_config_path = temp_dir / f"{step_name}_config_{unique_id}.yaml"

        # Create serializable config
        serializable_config = {}
        for key, value in step_config.items():
            if key == "renderer" and isinstance(value, dict):
                # Skip non-serializable objects
                renderer_copy = {
                    k: v for k, v in value.items() if not k.startswith("_")
                }
                serializable_config[key] = self._make_yaml_safe(renderer_copy)
            else:
                serializable_config[key] = self._make_yaml_safe(value)

        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                serializable_config,
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        logger.debug("Created temp config: %s", temp_config_path)
        return temp_config_path

    def _make_yaml_safe(self, value: Any) -> Any:
        """Recursively convert runtime objects into safe YAML scalars."""
        from dataclasses import asdict, is_dataclass
        from enum import Enum

        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value) and not isinstance(value, type):
            return self._make_yaml_safe(asdict(value))
        if hasattr(value, "model_dump"):
            return self._make_yaml_safe(value.model_dump())
        if isinstance(value, dict):
            return {
                self._make_yaml_safe(key): self._make_yaml_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, list | tuple | set):
            return [self._make_yaml_safe(item) for item in value]
        return value

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

        if step_name == "optimize_usd":
            outputs["optimized_usd_path"] = result.get("optimized_usd_path")
            outputs["optimization_metadata"] = result.get("optimization_metadata")
            outputs["optimization_success"] = result.get("optimization_success")
            outputs["original_usd_path"] = result.get("original_usd_path")

        elif step_name == "identify_asset":
            outputs["identification"] = result.get("identification")
            outputs["identification_path"] = result.get("identification_path")

        elif step_name == "build_dataset_usd":
            outputs["output_dir"] = result.get("output_dir")
            outputs["usd_dataset_dir"] = result.get("output_dir")

        elif step_name == "build_dataset_prepare_dataset":
            outputs["dataset_path"] = result.get("dataset_path")
            outputs["dataset_jsonl_path"] = result.get("dataset_jsonl_path")

        elif step_name == "predict":
            outputs["predictions_path"] = result.get("predictions_path")
            outputs["predictions_count"] = result.get("predictions_count")
            outputs["failed_count"] = result.get("failed_count", 0)
            outputs["token_stats"] = result.get("token_stats") or {}
            outputs["output_key"] = result.get("output_key")

        elif step_name == "restore_usd":
            outputs["restored_predictions_path"] = result.get(
                "restored_predictions_path"
            )
            outputs["restore_success"] = result.get("restore_success")
            outputs["predictions_count"] = result.get("predictions_count")
            outputs["restore_stats"] = result.get("restore_stats")

        elif step_name == "apply_physics":
            outputs["output_usd_path"] = result.get("output_usd_path")

        return outputs
