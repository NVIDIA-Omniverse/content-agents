"""Regression tests for EventBus progress / completion semantics.

Covers the full default service pipeline (identify_asset → render →
prepare → predict → apply_physics) so the bus's weight/completion maps
can't drift away from the steps that actually run.
"""

import pytest

from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState

DEFAULT_STEP_ORDER = (
    "identify_asset",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    "predict",
    "apply_physics",
)


def _completion_event(session_id: str, step: str) -> ProgressEvent:
    return ProgressEvent(
        session_id=session_id,
        step=step,
        state=StepState.COMPLETED,
        percent=100,
    )


def _running_event(session_id: str, step: str, percent: int) -> ProgressEvent:
    return ProgressEvent(
        session_id=session_id,
        step=step,
        state=StepState.RUNNING,
        percent=percent,
        message=f"running {step}",
    )


@pytest.mark.asyncio
async def test_status_stays_running_when_predict_completes_before_apply_physics() -> (
    None
):
    """predict completing must NOT flip the session to status=completed.

    apply_physics still has to run after predict and write scene_physics.usda.
    If the bus marked the session completed at this point, clients would try
    to download the physics USD before it existed.
    """
    bus = EventBus()
    session_id = "s1"

    for step in (
        "identify_asset",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
    ):
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] != "completed", (
        "session must not be marked completed until apply_physics runs"
    )
    assert state["overall_progress"]["percent"] < 100


@pytest.mark.asyncio
async def test_status_flips_to_completed_after_apply_physics_completes() -> None:
    """Once apply_physics completes, overall progress hits 100 and status flips."""
    bus = EventBus()
    session_id = "s2"

    for step in DEFAULT_STEP_ORDER:
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "completed"
    assert state["overall_progress"]["percent"] == 100


@pytest.mark.asyncio
async def test_identify_asset_has_display_name_and_weight() -> None:
    """Regression: identify_asset must advance overall progress off 0.

    Previously the weight map only covered build_dataset_usd / prepare /
    predict / apply_physics, so the ~15 seconds the default service
    pipeline spends in identify_asset showed percent=0 in /status.
    """
    bus = EventBus()
    session_id = "s3"

    await bus.emit(_running_event(session_id, "identify_asset", percent=50))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["current_step"] is not None
    assert state["current_step"]["display_name"] == "Identifying Asset"
    overall = state["overall_progress"]["percent"]
    assert 5 <= overall <= 10


@pytest.mark.asyncio
async def test_apply_physics_has_display_name_and_weight() -> None:
    """apply_physics must be registered in the bus's weight + display maps."""
    bus = EventBus()
    session_id = "s4"

    await bus.emit(_running_event(session_id, "apply_physics", percent=50))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["current_step"] is not None
    assert state["current_step"]["display_name"] == "Applying Physics Schemas"
    overall = state["overall_progress"]["percent"]
    assert 90 <= overall <= 100


@pytest.mark.asyncio
async def test_current_step_never_exceeds_total_steps_with_optimize_usd() -> None:
    """Enabling optimize_usd must not push current_step past total_steps.

    Regression: current_step used to be `len(completed_steps)`, so running
    the full 6-step sequence (optimize_usd + the 5 default steps) left the
    session at {"current_step": 6, "total_steps": 5, "percent": 100}.
    """
    from ...service.runtime.progress import TOTAL_VISIBLE_STEPS

    bus = EventBus()
    session_id = "s_opt"

    full_sequence = ("optimize_usd",) + DEFAULT_STEP_ORDER
    for step in full_sequence:
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "completed"
    assert state["overall_progress"]["percent"] == 100
    assert state["overall_progress"]["current_step"] <= TOTAL_VISIBLE_STEPS
    assert state["overall_progress"]["current_step"] == TOTAL_VISIBLE_STEPS


@pytest.mark.asyncio
async def test_predict_in_flight_progress_uses_weighted_range() -> None:
    """In-flight predict at percent=100 must not push overall to 100.

    Regression mirror on the bus side of the SessionManager issue: predict
    step-progress is weighted 60→90, so step_percent=100 should land overall
    at 90, leaving room for apply_physics.
    """
    bus = EventBus()
    session_id = "s_pred"

    for step in (
        "identify_asset",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
    ):
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    await bus.emit(_running_event(session_id, "predict", percent=100))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "running"
    assert state["overall_progress"]["percent"] == 90


@pytest.mark.asyncio
async def test_total_steps_matches_visible_default_pipeline() -> None:
    """total_steps in /status must match the actual default pipeline length."""
    from ...service.runtime.progress import TOTAL_VISIBLE_STEPS

    bus = EventBus()
    session_id = "s5"
    await bus.emit(_running_event(session_id, "identify_asset", percent=0))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["overall_progress"]["total_steps"] == TOTAL_VISIBLE_STEPS
    assert TOTAL_VISIBLE_STEPS == len(DEFAULT_STEP_ORDER)
