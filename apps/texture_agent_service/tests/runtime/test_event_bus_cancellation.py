# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the CANCELLING StepState transitions session-level state correctly.

The cancel route emits a `StepState.CANCELLING` event before triggering
asyncio task cancellation; without this transition, GET /status keeps
returning "running" even though the cancel was acknowledged.
"""

from __future__ import annotations

from pathlib import Path

from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState


async def test_cancelling_event_updates_snapshot_status() -> None:
    bus = EventBus()
    session_id = "abc123"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
            current=1,
            total=8,
            percent=12,
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.CANCELLING,
            message="Pipeline cancellation requested",
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelling"
    assert "cancelling_at" in snapshot

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="pipeline",
            state=StepState.CANCELLED,
            message="Pipeline cancelled by user",
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"


async def test_cancelling_event_does_not_overwrite_terminal_state() -> None:
    """If the worker finished naturally before the cancel route's CANCELLING
    event reaches the bus, the terminal `completed` status must win.

    Without this guard, a cancel request that races with completion would
    flip the snapshot back to `cancelling`, causing /status to lie and
    /results to reject valid output.
    """
    bus = EventBus()
    session_id = "abc123"

    # Mirror the executor's emit sequence: step RUNNING, step COMPLETED
    # (clears current_step), then a final pipeline-level COMPLETED with
    # extra={pipeline_completed: True} that lands the terminal status.
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="apply_textures",
            state=StepState.RUNNING,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="apply_textures",
            state=StepState.COMPLETED,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="apply_textures",
            state=StepState.COMPLETED,
            extra={"pipeline_completed": True},
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "completed"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="pipeline",
            state=StepState.CANCELLING,
            message="late cancel attempt",
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "completed"


async def test_cancelling_event_persists_to_session_manager() -> None:
    persisted: list[tuple[str, dict[str, str]]] = []

    class FakeSessionManager:
        def session_exists(self, session_id: str) -> bool:
            return True

        def update_session(self, session_id: str, updates: dict[str, str]) -> None:
            persisted.append((session_id, updates))

        def get_session_dir(self, session_id: str) -> Path:
            raise NotImplementedError

    bus = EventBus(session_manager=FakeSessionManager())

    await bus.emit(
        ProgressEvent(
            session_id="abc123",
            step="generate_textures",
            state=StepState.CANCELLING,
            message="Pipeline cancellation requested",
        )
    )

    assert ("abc123", {"status": "cancelling"}) in persisted
