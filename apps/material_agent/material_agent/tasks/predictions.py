# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for saving predictions to file."""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class SavePredictionsTask(Task):
    """Save predictions to JSONL file in dataset format."""

    def __init__(
        self, output_dir: Path | None = None, include_ground_truth: bool = False
    ):
        """Initialize the save predictions task.

        Args:
            output_dir: Directory to save predictions (None to use from context)
            include_ground_truth: Whether to include ground truth in output
        """
        self.output_dir = output_dir
        self.include_ground_truth = include_ground_truth
        self.name = "SavePredictions"
        self.description = "Save predictions to JSONL file"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Save predictions to JSONL file.

        Args:
            context: Workflow context
            object_store: Storage for predictions

        Returns:
            Updated context with output file path
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        # Resolve output directory from constructor or context
        output_dir = (
            self.output_dir
            if self.output_dir is not None
            else context.get("output_dir")
        )
        if output_dir is None:
            # Default to dataset directory if not specified
            dataset_path = context.get("dataset_path")
            if dataset_path:
                output_dir = Path(dataset_path).parent / "output"
            else:
                raise ValueError("output_dir not provided in constructor or context")

        output_dir = Path(output_dir)

        # If streaming was enabled and predictions.jsonl already exists, skip writing
        predictions_path_from_context = context.get("predictions_path")
        stream_predictions = context.get("stream_predictions", True)
        if stream_predictions and predictions_path_from_context:
            pp = Path(predictions_path_from_context)
            if pp.exists():
                # Load dataset for validation
                dataset = None
                if object_store and object_store.exists("dataset"):
                    dataset = object_store.get("dataset")
                elif context.get("dataset_path"):
                    dataset_path = Path(context["dataset_path"])
                    with open(dataset_path, encoding="utf-8") as f:
                        dataset = [json.loads(line) for line in f]

                # Validate that existing predictions match expected dataset entries
                if dataset and pp.exists():
                    existing_ids = set()
                    with open(pp, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    rec = json.loads(line)
                                    existing_ids.add(rec.get("id"))
                                except json.JSONDecodeError as e:
                                    listener.warning(
                                        f"Skipping invalid JSON line in predictions file: {e}"
                                    )
                                    continue
                    dataset_ids = {e["id"] for e in dataset if "id" in e}
                    if existing_ids != dataset_ids:
                        listener.warning(
                            f"Existing predictions incomplete: {len(existing_ids)}/{len(dataset_ids)} entries"
                        )
                        listener.warning(
                            f"Predictions incomplete: {len(existing_ids)}/{len(dataset_ids)} entries"
                        )

                # If we need to include ground truth, enrich streamed predictions
                if self.include_ground_truth and dataset:
                    gt_map = {
                        e["id"]: e.get("ground_truth") for e in dataset if "id" in e
                    }

                    # Read existing predictions, enrich, and rewrite file
                    enriched: list[dict[str, Any]] = []
                    with open(pp, encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError as e:
                                listener.warning(
                                    f"Skipping invalid JSON line in predictions file: {e}"
                                )
                                continue
                            pid = rec.get("id")
                            if pid in gt_map and gt_map[pid] is not None:
                                rec["ground_truth"] = gt_map[pid]
                            enriched.append(rec)

                    with open(pp, "w", encoding="utf-8") as f:
                        for rec in enriched:
                            f.write(json.dumps(rec) + "\n")

                    listener.info(
                        f"✓ Enriched streamed predictions with ground truth at {pp}"
                    )

                # Already present; update context and return
                context["predictions_path"] = str(pp)
                context["output_format"] = "jsonl"
                listener.info(f"✓ Predictions already present at {pp}. Skipping save.")
                return context

        # Get predictions from object store
        if object_store and object_store.exists("predictions"):
            predictions = object_store.get("predictions")
        else:
            raise ValueError("No predictions found to save")

        # Get original dataset if we need ground truth
        dataset = None
        if self.include_ground_truth:
            if object_store and object_store.exists("dataset"):
                dataset = object_store.get("dataset")
            else:
                dataset_path = Path(context["dataset_path"])
                with open(dataset_path, encoding="utf-8") as f:
                    dataset = [json.loads(line) for line in f]

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions in dataset format
        output_file = output_dir / "predictions.jsonl"
        listener.info(f"Saving predictions to {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                # Create output entry in dataset format
                output_entry = {
                    "id": pred["id"],
                    "materials": pred["vlm_response"],  # The material assignments
                }

                # Include image field(s) if present
                if "images" in pred:
                    output_entry["images"] = pred["images"]
                elif "image_path" in pred:
                    output_entry["image_path"] = pred.get("image_path", "")

                # Add ground truth if requested and available
                if self.include_ground_truth and dataset:
                    original = next((e for e in dataset if e["id"] == pred["id"]), None)
                    if original and "ground_truth" in original:
                        output_entry["ground_truth"] = original["ground_truth"]

                # Add any additional metadata from the prediction
                if "confidence" in pred:
                    output_entry["confidence"] = pred["confidence"]

                f.write(json.dumps(output_entry) + "\n")

        listener.info(f"✓ Saved {len(predictions)} predictions to {output_file}")
        listener.info(f"Saved predictions to {output_file}")

        # Update context
        context["predictions_path"] = str(output_file)
        context["output_format"] = "jsonl"

        return context
