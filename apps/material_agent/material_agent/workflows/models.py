# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for material agent workflows."""

from typing import Any

from pydantic import BaseModel, Field


class DatasetEntry(BaseModel):
    """Model for a dataset entry."""

    id: str = Field(..., description="Unique identifier for the entry")
    image_path: str = Field(..., description="Path to the image file")
    ground_truth: str | None = Field(
        None, description="Ground truth material assignment"
    )
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class Prediction(BaseModel):
    """Model for a prediction result."""

    id: str = Field(..., description="Entry identifier")
    image_path: str = Field(..., description="Path to the image file")
    materials: str = Field(..., description="Predicted material assignments")
    confidence: float | None = Field(None, description="Confidence score")
    ground_truth: str | None = Field(None, description="Ground truth if available")


class Evaluation(BaseModel):
    """Model for an evaluation result."""

    id: str = Field(..., description="Entry identifier")
    materials: str = Field(..., description="Predicted materials")
    ground_truth: str = Field(..., description="Ground truth materials")
    score: int = Field(..., ge=0, le=5, description="Judge score (0=error, 1-5=valid)")
    explanation: str = Field(..., description="Judge explanation")


class Metrics(BaseModel):
    """Model for evaluation metrics."""

    functional_correctness_score: float = Field(..., description="Average score (FCS)")
    success_rate: float = Field(..., description="Percentage of successful predictions")
    total_cases: int = Field(..., description="Total number of cases")
    valid_cases: int = Field(..., description="Number of valid evaluations")
    successful_cases: int = Field(..., description="Number of successful predictions")
    score_distribution: dict[int, int] = Field(
        ..., description="Distribution of scores"
    )
    failure_count: int = Field(..., description="Number of failed predictions")
