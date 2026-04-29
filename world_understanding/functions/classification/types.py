# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic classification types and dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image as PILImage


@dataclass
class ClassificationEntry:
    """Input entry for classification.

    This represents a single object to be classified, containing:
    - Unique identifier
    - Context text (description, available classes, etc.)
    - Images (file paths or PIL Images)
    - Optional ground truth label
    - Optional metadata

    Example:
        ```python
        entry = ClassificationEntry(
            id="vehicle_001",
            text="This is a vehicle. Available types: sedan, SUV, truck",
            images=["front.jpg", "side.jpg"],
            ground_truth="sedan"
        )
        ```
    """

    id: str
    text: str
    images: list[str | Path | PILImage.Image]
    ground_truth: str | None = None
    image_metadata: list[dict[str, Any]] | None = None


@dataclass
class ClassificationResult:
    """Output result from classification.

    This represents the classification result for a single object.

    The `predicted_value` field contains the classified label.
    The `output_dict` contains the full response with the parameterized key.

    Example:
        ```python
        result = ClassificationResult(
            id="vehicle_001",
            predicted_value="sedan",
            confidence=0.95,
            original_response="Based on the images, this is a sedan...",
            output_dict={"vehicle_type": "sedan", "original_response": "..."}
        )
        ```
    """

    id: str
    predicted_value: str
    confidence: float = 1.0
    original_response: str | None = None
    status: str = "success"
    error: str | None = None
    output_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationMetrics:
    """Evaluation metrics for classification.

    Contains standard classification metrics computed from predictions
    vs. ground truth.

    Example:
        ```python
        metrics = ClassificationMetrics(
            accuracy=0.85,
            precision=0.87,
            recall=0.83,
            f1_score=0.85,
            confusion_matrix={
                "sedan": {"sedan": 45, "SUV": 3, "truck": 2},
                "SUV": {"sedan": 2, "SUV": 48, "truck": 0},
            },
            total_samples=100
        )
        ```
    """

    accuracy: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    total_samples: int = 0
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
