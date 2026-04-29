# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for identifying unique materials from prediction files."""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class IdentifyUniqueMaterialsTask(Task):
    """Task to identify unique materials from prediction JSON files.

    This task loads a predictions file (JSON or JSONL format) and extracts
    all unique material names from the predictions. It handles both single
    JSON objects and JSONL format with multiple prediction entries.

    Input context keys:
        - predictions_path: Path to the predictions file (JSON or JSONL)

    Output context keys:
        - unique_materials: List of unique material names found in predictions
        - predictions_data: The loaded predictions data for potential future use
        - total_predictions: Number of prediction entries processed
    """

    def __init__(self):
        """Initialize the identify unique materials task."""
        self.name = "IdentifyUniqueMaterials"
        self.description = "Identify unique materials from prediction files"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Identify unique materials from predictions.

        Args:
            context: Workflow context containing predictions_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with unique materials
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        predictions_path = context.get("predictions_path")
        if not predictions_path:
            raise ValueError("predictions_path not provided in context")

        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        listener.info(f"Loading predictions from {predictions_path}")

        # Load predictions data
        predictions_data = self._load_predictions(predictions_path, listener)

        # Extract unique materials
        unique_materials = self._extract_unique_materials(predictions_data, listener)

        listener.info(
            f"Found {len(unique_materials)} unique materials from {len(predictions_data)} predictions"
        )

        if logger.isEnabledFor(logging.DEBUG):
            listener.debug(f"Unique materials: {unique_materials}")

        # Update context
        context["unique_materials"] = unique_materials
        context["predictions_data"] = predictions_data
        context["total_predictions"] = len(predictions_data)

        return context

    def _load_predictions(
        self, predictions_path: Path, listener
    ) -> list[dict[str, Any]]:
        """Load predictions from JSON or JSONL file.

        Args:
            predictions_path: Path to the predictions file
            listener: Event listener for logging

        Returns:
            List of prediction dictionaries

        Raises:
            ValueError: If file format is not supported or data is invalid
        """
        try:
            with open(predictions_path, encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                listener.warning("Predictions file is empty")
                return []

            # Try to parse as JSON first (single object or array)
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # Single prediction object
                    return [data]
                elif isinstance(data, list):
                    # Array of predictions
                    return data
                else:
                    raise ValueError(f"Unexpected JSON root type: {type(data)}")
            except json.JSONDecodeError:
                # Try to parse as JSONL (one JSON object per line)
                predictions = []
                for line_num, line in enumerate(content.split("\n"), 1):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    try:
                        prediction = json.loads(line)
                        predictions.append(prediction)
                    except json.JSONDecodeError as e:
                        listener.warning(
                            f"Skipping invalid JSON on line {line_num}: {e}"
                        )
                        continue

                if not predictions:
                    raise ValueError("No valid JSON objects found in file") from None

                return predictions

        except Exception as e:
            raise ValueError(
                f"Failed to load predictions from {predictions_path}: {str(e)}"
            ) from e

    def _extract_unique_materials(
        self, predictions_data: list[dict[str, Any]], listener
    ) -> list[str]:
        """Extract unique materials from prediction data.

        Args:
            predictions_data: List of prediction dictionaries
            listener: Event listener for logging

        Returns:
            List of unique material names (sorted)
        """
        unique_materials = set()

        for i, prediction in enumerate(predictions_data):
            try:
                materials = self._extract_materials_from_prediction(prediction)
                unique_materials.update(materials)

                # Also check if the prediction contains an array of sub-predictions
                # This handles cases like: {"predictions": [{"predicted_material": "Steel"}, ...]}
                if "predictions" in prediction and isinstance(
                    prediction["predictions"], list
                ):
                    for sub_prediction in prediction["predictions"]:
                        if isinstance(sub_prediction, dict):
                            sub_materials = self._extract_materials_from_prediction(
                                sub_prediction
                            )
                            unique_materials.update(sub_materials)

            except Exception as e:
                listener.warning(
                    f"Failed to extract materials from prediction {i}: {e}"
                )
                continue

        # Return sorted list for consistent ordering
        return sorted(unique_materials)

    def _extract_materials_from_prediction(
        self, prediction: dict[str, Any]
    ) -> list[str]:
        """Extract materials from a single prediction entry.

        This method handles various possible formats for material data in predictions.
        It looks for common field names that might contain material information.

        Args:
            prediction: Single prediction dictionary

        Returns:
            List of material names from this prediction
        """
        materials = []

        # Common field names that might contain material information
        material_fields = [
            "materials",
            "material",
            "predicted_material",
            "predicted_materials",
            "predictions",
            "material_predictions",
            "assigned_materials",
        ]

        for field_name in material_fields:
            if field_name in prediction:
                field_value = prediction[field_name]
                extracted = self._extract_materials_from_field(field_value)
                materials.extend(extracted)

        # Also check if the prediction itself is just a material string
        if isinstance(prediction, str):
            materials.append(prediction)

        # Remove duplicates and filter out empty strings
        materials = [m for m in set(materials) if m and isinstance(m, str)]

        return materials

    def _extract_materials_from_field(self, field_value: Any) -> list[str]:
        """Extract material names from a field value.

        Args:
            field_value: The value of a field that might contain materials

        Returns:
            List of material names extracted from the field
        """
        materials = []

        if isinstance(field_value, str):
            # Single material string
            materials.append(field_value)
        elif isinstance(field_value, list):
            # List of materials
            for item in field_value:
                if isinstance(item, str):
                    materials.append(item)
                elif isinstance(item, dict):
                    # Material might be in a dictionary structure
                    # Look for common keys
                    for key in ["name", "material", "type", "value"]:
                        if key in item and isinstance(item[key], str):
                            materials.append(item[key])
                            break
        elif isinstance(field_value, dict):
            # Materials might be in a dictionary structure
            # First check for specific material keys
            material_keys = [
                "material",
                "name",
                "type",
                "value",
                "material_name",
                "material_type",
            ]
            for key in material_keys:
                if key in field_value and isinstance(field_value[key], str):
                    materials.append(field_value[key])
                    # Don't break here - there might be multiple material keys

            # If no specific material keys found, look for lists within the dict
            if not materials:
                for _key, value in field_value.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                materials.append(item)

        return materials
