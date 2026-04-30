# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify FAILED events carry structured ``failed_step_stats`` into the
EventBus snapshot. Without this, GET /status -- which reads the bus
snapshot first -- only carries the prose error message and the polling
client cannot see per-material failure detail until the run terminates
and the snapshot is replaced by /results.
"""

from __future__ import annotations

from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState


async def test_failed_event_persists_extra_into_snapshot() -> None:
    bus = EventBus()
    session_id = "fail-bus-001"

    failed_extra = {
        "textures_generated": 0,
        "textures_failed": 2,
        "errors": [
            {
                "material": "Aluminum_Brushed",
                "type": "RuntimeError",
                "status": 403,
                "message": "HTTP 403 Forbidden",
            },
            {
                "material": "Rubber_Black_Matte",
                "type": "RuntimeError",
                "status": 403,
                "message": "HTTP 403 Forbidden",
            },
        ],
    }

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.FAILED,
            message="2/2 texture generation requests failed",
            extra=failed_extra,
        )
    )

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["failed_step"] == "generate_textures"
    assert snapshot["failed_step_stats"] == failed_extra


async def test_failed_event_without_extra_omits_failed_step_stats() -> None:
    """Backward compat: a FAILED event with no extra (e.g. a non-task
    failure path) must not break the snapshot."""
    bus = EventBus()
    session_id = "fail-bus-002"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="render",
            state=StepState.FAILED,
            message="render failed",
        )
    )

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["failed_step"] == "render"
    assert "failed_step_stats" not in snapshot


async def test_clear_session_state_drops_snapshot() -> None:
    """``/regenerate`` calls ``clear_session_state`` so a retried run
    doesn't show stale ``failed_step`` / ``failed_step_stats`` from the
    prior attempt while the new run is pending or running."""
    bus = EventBus()
    session_id = "regen-001"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.FAILED,
            message="2/2 failed",
            extra={"textures_failed": 2, "errors": [{"material": "A"}]},
        )
    )
    assert bus.get_snapshot(session_id) is not None

    bus.clear_session_state(session_id)
    assert bus.get_snapshot(session_id) is None


async def test_clear_session_state_is_idempotent() -> None:
    """Safe to call against a session that has no snapshot yet."""
    bus = EventBus()
    bus.clear_session_state("never-had-events")
    assert bus.get_snapshot("never-had-events") is None


async def test_clear_session_state_drains_queued_sse_events() -> None:
    """A FAILED event left in the per-session queue from the prior run
    must not be replayed to a fresh SSE subscriber attaching after
    /regenerate. Without draining, ``stream_progress_events`` would
    hand the stale terminal event to the new client and close the
    stream while the retry is still pending or running."""
    bus = EventBus()
    session_id = "regen-queue-001"

    # Pre-populate the per-session queue with a FAILED event from the
    # "prior run" -- this is what stream_progress_events would deliver.
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.FAILED,
            message="2/2 failed",
            extra={"textures_failed": 2},
        )
    )
    queue = bus.get_queue(session_id)
    assert not queue.empty()

    bus.clear_session_state(session_id)

    # Same queue object is reused (stream_progress_events held a
    # reference) but it must now be drained.
    assert queue.empty()
