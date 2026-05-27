# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM inference task for asset classification."""

import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.token_tracking import TokenTracker, format_token_stats

from physics_agent.functions.inference import batch_classify_assets
from physics_agent.functions.mass_scale_quality import (
    build_mass_scale_quality_warnings,
)
from physics_agent.functions.prediction_schema import unwrap_output_key_payload

logger = logging.getLogger(__name__)


class VLMInferenceTask(Task):
    """Run VLM inference on dataset for asset classification.

    This task supports configurable output_key for flexible classification
    tasks (e.g., component identification, material prediction, property estimation).

    Input context keys:
        - dataset or dataset_path: Dataset to process
        - vlm: VLM instance
        - llm: LLM instance (optional, uses VLM if not provided)
        - vlm_config: VLM configuration
        - system_prompt: Base system prompt
        - output_key: Key for classification output (default: "classification")
        - allow_empty_predictions: Allow zero-entry or zero-success runs to
          produce an empty predictions file (default: False)

    Output context keys:
        - predictions: List of prediction results
        - predictions_path: Path to saved predictions file
    """

    def __init__(
        self,
        vlm: Any = None,
        llm: Any | None = None,
        system_prompt: str | None = None,
        output_key: str | None = None,
    ):
        """Initialize the VLM inference task.

        Args:
            vlm: VLM instance (None to use from context)
            llm: Optional LLM for parsing (None to use from context or VLM)
            system_prompt: Optional custom system prompt (None to use from context)
            output_key: Key for classification output (None to use from context)
        """
        self.vlm = vlm
        self.llm = llm
        self.system_prompt = system_prompt
        self.output_key = output_key
        self.name = "VLMInference"
        self.description = "Run VLM inference for asset classification"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Run batch inference on dataset.

        Args:
            context: Workflow context with dataset metadata
            object_store: Storage for dataset and predictions

        Returns:
            Updated context with inference results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Resolve parameters from constructor or context
        vlm = self.vlm if self.vlm is not None else context.get("vlm")
        llm = self.llm if self.llm is not None else context.get("llm")
        if llm is None:
            llm = vlm  # Use VLM as LLM if not provided

        # Get config values from context if not provided in constructor
        vlm_config = context.get("vlm_config", {})
        max_retries = vlm_config.get("max_retries", 3)
        system_prompt = (
            self.system_prompt
            if self.system_prompt is not None
            else context.get("system_prompt")
            or context.get("config", {}).get("system_prompt")
        )

        # Get output_key (configurable)
        output_key = (
            self.output_key
            if self.output_key is not None
            else context.get("output_key", "classification")
        )

        # Build per-invoke kwargs from provisioning
        vlm_invoke_kwargs: dict[str, Any] = dict(context.get("vlm_invoke_kwargs", {}))

        # Validate required values
        if vlm is None:
            raise ValueError("VLM not provided in constructor or context")

        allow_empty_predictions = context.get("allow_empty_predictions", False)
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )

        # Get dataset from object store or context
        if object_store and object_store.exists("dataset"):
            dataset = object_store.get("dataset")
        else:
            # Fallback: load from file if not in store
            dataset_path_str = context.get("dataset_path")
            if not dataset_path_str:
                raise ValueError(
                    "dataset_path not found in context and dataset not in object_store"
                )
            dataset_path = Path(dataset_path_str)
            with open(dataset_path, encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f]

        # Streaming and resume options
        stream_predictions = context.get("stream_predictions", True)
        resume_enabled = context.get("resume", False)
        explicit_predictions_path = bool(context.get("predictions_path"))
        stale_predictions_path: Path | None = None
        if context.get("predictions_path"):
            stale_predictions_path = Path(context["predictions_path"])
        elif context.get("output_dir"):
            stale_predictions_path = Path(context["output_dir"]) / "predictions.jsonl"
        elif context.get("dataset_path"):
            stale_predictions_path = (
                Path(context["dataset_path"]).parent / "output" / "predictions.jsonl"
            )

        if not dataset and not allow_empty_predictions:
            if (
                not resume_enabled
                and not explicit_predictions_path
                and stale_predictions_path is not None
                and stale_predictions_path.exists()
            ):
                stale_predictions_path.unlink()
                logger.info("Cleared existing predictions file")
            path_hint = (
                f" predictions_path={stale_predictions_path}"
                if stale_predictions_path is not None
                else ""
            )
            raise RuntimeError(
                "VLM inference received zero dataset entries. Refusing to "
                "produce empty prediction output; set allow_empty_predictions=true "
                "only for workflows that intentionally permit empty prediction "
                f"output.{path_hint}"
            )

        # Emit task started event
        listener.event(
            "task.started",
            {
                "task_name": "VLMInference",
                "total_entries": len(dataset),
            },
        )
        listener.info(f"Starting VLM inference for {len(dataset)} entries")
        listener.info(f"Output key: {output_key}")

        # Progress callback
        processed_count = [0]

        def on_progress(entry_id: str, response: str) -> None:
            """Log progress after processing each entry."""
            processed_count[0] += 1
            listener.event(
                "task.progress",
                {
                    "task_name": "VLMInference",
                    "current": processed_count[0],
                    "total": len(dataset),
                    "percentage": (processed_count[0] / len(dataset)) * 100
                    if dataset
                    else 0,
                    "entry_id": entry_id,
                },
            )
            if processed_count[0] % 10 == 0:
                listener.info(f"Processed {processed_count[0]}/{len(dataset)} entries")

        # Error callback
        def on_error(entry_id: str, error: str) -> None:
            """Handle errors during processing."""
            logger.error(f"Error processing {entry_id}: {error}")
            listener.error(f"Error processing {entry_id}: {error}")

        # Prediction callback
        def on_prediction(entry_id: str, result_dict: dict[str, Any]) -> None:
            """Emit event for each prediction."""
            try:
                classification = result_dict.get(output_key, "unknown")
                confidence = result_dict.get("confidence")

                listener.event(
                    "prediction.completed",
                    {
                        "entry_id": entry_id,
                        output_key: classification,
                        "confidence": confidence,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to emit prediction event for {entry_id}: {e}")

        # Resolve predictions path
        predictions_path = context.get("predictions_path")
        output_dir: Path
        if predictions_path:
            predictions_path = Path(predictions_path)
            output_dir = predictions_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir_value = context.get("output_dir")
            if output_dir_value is None:
                dataset_path_str = context.get("dataset_path")
                if not dataset_path_str:
                    raise ValueError("dataset_path not found in context")
                output_dir = Path(dataset_path_str).parent / "output"
            else:
                output_dir = Path(output_dir_value)
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = output_dir / "predictions.jsonl"

        # Clear existing predictions if not resuming
        if stream_predictions and not resume_enabled and predictions_path.exists():
            if explicit_predictions_path and predictions_path.stat().st_size > 0:
                raise RuntimeError(
                    "Refusing to overwrite existing explicit predictions_path for "
                    "a streaming prediction run. Remove the file, choose a new "
                    f"predictions_path, or set resume=true. "
                    f"predictions_path={predictions_path}"
                )
            predictions_path.unlink()
            logger.info("Cleared existing predictions file")

        # Load processed ids when resuming
        processed_ids: set[str] = set()
        if stream_predictions and resume_enabled and predictions_path.exists():
            recovered_temp_path: Path | None = None
            diagnostics_temp_path: Path | None = None
            try:
                valid_line_count = 0
                diagnostic_line_count = 0
                rewrite_recovered_predictions = False
                with (
                    tempfile.NamedTemporaryFile(
                        "w",
                        encoding="utf-8",
                        dir=predictions_path.parent,
                        prefix=f".{predictions_path.stem}.recovered.",
                        suffix=predictions_path.suffix,
                        delete=False,
                    ) as recovered_file,
                    tempfile.NamedTemporaryFile(
                        "w",
                        encoding="utf-8",
                        dir=predictions_path.parent,
                        prefix=f".{predictions_path.stem}.diagnostics.",
                        suffix=predictions_path.suffix,
                        delete=False,
                    ) as diagnostics_file,
                    open(predictions_path, encoding="utf-8") as f,
                ):
                    recovered_temp_path = Path(recovered_file.name)
                    diagnostics_temp_path = Path(diagnostics_file.name)
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            diagnostics_file.write(line)
                            diagnostic_line_count += 1
                            continue
                        if isinstance(rec, dict) and "id" in rec:
                            processed_ids.add(rec["id"])
                        recovered_file.write(line)
                        if not line.endswith("\n"):
                            recovered_file.write("\n")
                            rewrite_recovered_predictions = True
                        valid_line_count += 1
                if diagnostic_line_count or rewrite_recovered_predictions:
                    assert recovered_temp_path is not None
                    recovered_temp_path.replace(predictions_path)
                    recovered_temp_path = None
                if diagnostic_line_count:
                    diagnostics_path = predictions_path.with_name(
                        f"{predictions_path.stem}.diagnostics{predictions_path.suffix}"
                    )
                    assert diagnostics_temp_path is not None
                    with (
                        open(diagnostics_temp_path, encoding="utf-8") as src,
                        open(diagnostics_path, "a", encoding="utf-8") as dst,
                    ):
                        for line in src:
                            dst.write(line)
                    diagnostics_temp_path.unlink(missing_ok=True)
                    diagnostics_temp_path = None
                    logger.warning(
                        "Recovered %d valid resume prediction line(s) from %s; "
                        "quarantined %d malformed line(s) to %s",
                        valid_line_count,
                        predictions_path,
                        diagnostic_line_count,
                        diagnostics_path,
                    )
                else:
                    if diagnostics_temp_path is not None:
                        diagnostics_temp_path.unlink(missing_ok=True)
                        diagnostics_temp_path = None
                    if recovered_temp_path is not None:
                        recovered_temp_path.unlink(missing_ok=True)
                        recovered_temp_path = None
                logger.info(f"Resuming: {len(processed_ids)} entries already processed")
            except Exception as e:
                logger.warning(f"Failed to parse existing predictions: {e}")
                if diagnostics_temp_path is not None:
                    diagnostics_temp_path.unlink(missing_ok=True)
                if recovered_temp_path is not None:
                    recovered_temp_path.unlink(missing_ok=True)

        # Define result callback for streaming
        prediction_write_lock = threading.Lock()

        def on_result(result: dict[str, Any], entry: dict[str, Any]) -> None:
            if not stream_predictions:
                return
            try:
                if result.get("status") != "success":
                    return
                prediction_payload = unwrap_output_key_payload(
                    result.get("vlm_response"), output_key
                )
                output_entry: dict[str, Any] = {
                    "id": result["id"],
                    output_key: prediction_payload,
                }
                quality_warnings = build_mass_scale_quality_warnings(
                    output_entry, entry, output_key
                )
                if quality_warnings:
                    output_entry["quality_warnings"] = quality_warnings
                    listener.warning(
                        f"Mass/scale QA warning for {result['id']}: "
                        f"{quality_warnings[0]['message']}"
                    )
                if "images" in entry:
                    output_entry["images"] = entry["images"]
                elif "image_path" in entry:
                    output_entry["image_path"] = entry.get("image_path", "")

                with prediction_write_lock:
                    with open(predictions_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(output_entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to append streaming prediction: {e}")

        # Get optional max_workers
        max_workers = context.get("max_workers")

        # Create token tracker
        token_tracker = TokenTracker()

        # Run inference
        results = batch_classify_assets(
            vlm=vlm,
            entries=dataset,
            llm=llm,
            image_base_dir=Path(context["image_base_dir"])
            if context.get("image_base_dir")
            else None,
            system_prompt=system_prompt,
            invoke_kwargs=vlm_invoke_kwargs,
            on_progress=on_progress,
            on_error=on_error,
            processed_ids=processed_ids,
            on_result=on_result,
            on_prediction=on_prediction,
            max_workers=max_workers,
            max_retries=max_retries,
            output_key=output_key,
            token_tracker=token_tracker,
        )

        # Log token usage
        token_stats = token_tracker.get_stats()
        logger.info(f"\n{format_token_stats(token_stats)}")

        # Filter results
        predictions = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]

        listener.info(
            f"Inference complete: {len(predictions)} successful, {len(failed)} failed"
        )

        # Reload predictions from file if streaming. Empty current datasets only
        # reload during resume, where existing records are the completed work.
        if (
            stream_predictions
            and predictions_path.exists()
            and (dataset or resume_enabled)
        ):
            file_predictions = []
            diagnostic_lines = []
            with open(predictions_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        diagnostic_lines.append(line)
                        continue
                    file_predictions.append(record)
            if diagnostic_lines:
                diagnostics_path = predictions_path.with_name(
                    f"{predictions_path.stem}.diagnostics{predictions_path.suffix}"
                )
                with open(diagnostics_path, "a", encoding="utf-8") as diagnostics_file:
                    diagnostics_file.writelines(diagnostic_lines)
                raise RuntimeError(
                    "Streaming predictions file contains malformed JSON records; "
                    "wrote diagnostics to "
                    f"{diagnostics_path}. Refusing to continue. "
                    f"predictions_path={predictions_path}"
                )
            predictions = []
            for p in file_predictions:
                prediction = {
                    "id": p.get("id"),
                    "vlm_response": p.get(output_key),
                    "status": "success",
                }
                for key in ("quality_warnings", "images", "image_path", "media"):
                    if key in p:
                        prediction[key] = p[key]
                predictions.append(prediction)

        if not predictions and not allow_empty_predictions:
            entry_word = "entry" if len(dataset) == 1 else "entries"
            raise RuntimeError(
                "VLM inference produced zero successful predictions for "
                f"{len(dataset)} dataset {entry_word} "
                f"({len(failed)} failed). Refusing to continue; set "
                "allow_empty_predictions=true only for workflows that intentionally "
                f"permit empty prediction output. predictions_path={predictions_path}"
            )

        if not predictions and allow_empty_predictions:
            if (
                explicit_predictions_path
                and predictions_path.exists()
                and predictions_path.stat().st_size > 0
                and (not stream_predictions or not dataset)
            ):
                mode_label = "streaming" if stream_predictions else "non-streaming"
                raise RuntimeError(
                    "Refusing to overwrite existing explicit predictions_path for "
                    f"an empty {mode_label} prediction run. Remove the file or "
                    "choose a new predictions_path if this empty output is "
                    f"intentional. predictions_path={predictions_path}"
                )
            if stream_predictions:
                if not dataset:
                    predictions_path.write_text("", encoding="utf-8")
                elif not resume_enabled:
                    predictions_path.write_text("", encoding="utf-8")
                else:
                    predictions_path.touch()
                logger.info("Created empty predictions file at %s", predictions_path)
            else:
                # SavePredictionsTask owns non-streaming JSONL creation from the
                # object-store predictions list, including the intentional empty list.
                predictions_path.unlink(missing_ok=True)
                logger.info(
                    "Deferred empty non-streaming predictions file creation to "
                    "SavePredictionsTask at %s",
                    predictions_path,
                )

        # Store in object store
        if object_store:
            object_store.set("predictions", predictions)
            object_store.set("failed_predictions", failed)

        # Update context
        context["predictions_count"] = len(predictions)
        context["failed_count"] = len(failed)
        context["inference_complete"] = True
        context["predictions_path"] = str(predictions_path)
        context["token_stats"] = token_stats
        context["output_key"] = output_key

        return context
