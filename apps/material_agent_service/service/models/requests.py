# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request models for Material Agent Service API."""

from enum import StrEnum

from pydantic import BaseModel, Field


class PipelineStep(StrEnum):
    """Available pipeline steps."""

    BUILD_DATASET = "build_dataset_usd"
    PREDICT = "predict"
    APPLY = "apply"


class PipelineRequest(BaseModel):
    """Simplified pipeline request for MVP.

    All materials, VLM, LLM configs are pre-configured on backend.
    User provides USD file, optional prompt, and optional reference images.
    """

    # Execution control
    steps: list[PipelineStep] | None = Field(
        default=None,
        description="Steps to execute. If None, runs all steps.",
        examples=[["build_dataset_usd", "predict", "apply"]],
    )

    # Rendering options
    camera_views: list[str] = Field(
        default=["+x+y+z", "-x-y-z"],
        description="Camera positions for rendering",
        examples=[["+x+y+z", "-x+y+z", "+x-y+z", "-x-y+z"]],
    )

    # User-configurable prompt
    user_prompt: str | None = Field(
        default=None,
        description="Custom user prompt for VLM. If None, uses backend default.",
        examples=[
            "Please identify the highlighted part and select the appropriate material. "
            "This is a mechanical assembly with metal and plastic components."
        ],
    )

    # Prediction batching
    prediction_batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of prims per VLM call. 1 = one prim per call (default). "
        "N > 1 = group N prims into a single VLM call for faster throughput.",
    )


class RegenerateRequest(BaseModel):
    """Request to regenerate specific steps from cache."""

    steps: list[PipelineStep] = Field(description="Steps to re-run from cache")

    # Can override user prompt for regeneration
    user_prompt: str | None = Field(
        default=None, description="Override user prompt for regeneration"
    )

    # Output format
    layer_only: bool = Field(
        default=False,
        description=(
            "Output only a material binding layer instead of a full USD. "
            "When true, preserves original scene structure."
        ),
    )
