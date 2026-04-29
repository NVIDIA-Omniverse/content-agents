# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event models for pipeline progress tracking."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StepState(StrEnum):
    """Step execution state."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressEvent(BaseModel):
    """Progress event emitted by pipeline steps.

    Events flow through the EventBus and update the canonical in-memory state.
    The web UI subscribes via SSE to receive real-time updates.
    """

    session_id: str = Field(description="Session identifier")
    step: str = Field(
        description="Step name (e.g., 'build_dataset_usd', 'predict', 'apply')"
    )
    state: StepState = Field(description="Current state of the step")

    # Progress metrics (None when state is QUEUED/COMPLETED/FAILED)
    current: int | None = Field(default=None, description="Current progress count")
    total: int | None = Field(default=None, description="Total items to process")
    percent: int | None = Field(default=None, description="Percentage complete (0-100)")

    # Human-readable message
    message: str | None = Field(default=None, description="Progress message")

    # Timestamp
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp when event was created",
    )

    # Optional extra data (stats, errors, etc.)
    extra: dict[str, Any] | None = Field(
        default=None, description="Additional step-specific data"
    )

    # Overall pipeline progress (enriched by EventBus)
    overall_percent: int | None = Field(
        default=None,
        description="Overall pipeline progress 0-100 (computed by EventBus)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "step": "predict",
                "state": "running",
                "current": 47,
                "total": 95,
                "percent": 49,
                "message": "Predicted /World/Part_47",
                "timestamp": "2025-10-18T12:34:56.789Z",
                "extra": {"prim_id": "/World/Part_47"},
            }
        }
