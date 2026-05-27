# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for DELETE /sessions/{session_id} API erasure."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from fastapi import HTTPException

from ...service.routers import artifacts_router, pipeline_router, sessions_router
from ...service.runtime import bus as bus_module
from ...service.runtime.events import ProgressEvent, StepState
from ...service.session.manager import SessionManager


def _wire_service_state(tmp_path: Path) -> tuple[SessionManager, bus_module.EventBus]:
    manager = SessionManager(tmp_path)
    pipeline_router.set_session_manager(manager)
    artifacts_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)

    bus_module._event_bus = None
    bus = bus_module.init_event_bus(manager)
    return manager, bus


async def _assert_session_not_found(
    call: Callable[[], object | Awaitable[object]],
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        result = call()
        if inspect.isawaitable(result):
            await result

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Session not found"


async def test_delete_session_clears_event_bus_snapshot(tmp_path: Path) -> None:
    manager, bus = _wire_service_state(tmp_path)
    session_id = "delete-clears-bus"
    manager.create_session(session_id)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
        )
    )
    assert bus.get_snapshot(session_id) is not None

    await sessions_router.delete_session(session_id)

    assert not manager.session_exists(session_id)
    assert bus.get_snapshot(session_id) is None


async def test_status_and_events_404_when_snapshot_outlives_session(
    tmp_path: Path,
) -> None:
    manager, bus = _wire_service_state(tmp_path)
    session_id = "stale-snapshot"
    manager.create_session(session_id)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
        )
    )
    manager.delete_session(session_id)

    # The pre-delete snapshot survives in memory until cleanup_session()
    # runs (here we deleted disk-only via SessionManager, bypassing it).
    # Public read routes must treat persistent storage as authoritative
    # so a stale snapshot can never serve a 200 for a deleted session.
    # Note: a fresh emit() at this point would be DROPPED by the bus's
    # disk-exists gate -- covered separately by
    # test_emit_after_disk_delete_is_dropped.
    assert bus.get_snapshot(session_id) is not None

    await _assert_session_not_found(
        lambda: pipeline_router.get_pipeline_status(session_id)
    )
    await _assert_session_not_found(
        lambda: pipeline_router.stream_progress_events(session_id)
    )


async def test_deleted_session_disappears_from_public_read_endpoints(
    tmp_path: Path,
) -> None:
    manager, _bus = _wire_service_state(tmp_path)
    session_id = "deleted-session"
    session_dir = manager.create_session(session_id)
    materials = session_dir / "cache" / "discovery" / "materials.json"
    materials.write_text("{}", encoding="utf-8")
    texture = session_dir / "cache" / "textures" / "albedo.png"
    texture.write_text("png", encoding="utf-8")
    output = session_dir / "cache" / "output" / "textured_output.usdz"
    output.write_text("usdz", encoding="utf-8")

    await sessions_router.delete_session(session_id)

    await _assert_session_not_found(lambda: sessions_router.get_session(session_id))
    await _assert_session_not_found(
        lambda: pipeline_router.get_pipeline_status(session_id)
    )
    await _assert_session_not_found(
        lambda: pipeline_router.get_pipeline_results(session_id)
    )
    await _assert_session_not_found(lambda: pipeline_router.get_event_log(session_id))
    await _assert_session_not_found(
        lambda: artifacts_router.download_materials(session_id)
    )
    await _assert_session_not_found(
        lambda: artifacts_router.download_textures_zip(session_id)
    )
    await _assert_session_not_found(
        lambda: artifacts_router.download_output(session_id)
    )


async def test_delete_terminates_attached_sse_subscriber(tmp_path: Path) -> None:
    """A pre-attached /pipeline/{sid}/events subscriber must receive a
    terminal CANCELLED event when the session is deleted.

    Pre-fix: cleanup_session() popped the queue from the bus dict, but an
    already-running SSE generator held a reference to the popped queue and
    never received another real event -- it just streamed 30s keepalive
    pings forever while a fresh queue served any new emit() calls.
    """
    manager, bus = _wire_service_state(tmp_path)
    session_id = "sse-attached"
    manager.create_session(session_id)

    # Simulate an attached subscriber: hold the queue and drain pre-existing
    # events so the next get() will block on a fresh delivery.
    subscriber_queue = bus.get_queue(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
        )
    )
    while not subscriber_queue.empty():
        subscriber_queue.get_nowait()

    await sessions_router.delete_session(session_id)

    # The subscriber's queue ref must now carry a terminal sentinel that
    # the SSE generator's close-on-cancelled branch will pick up.
    event = await asyncio.wait_for(subscriber_queue.get(), timeout=2.0)
    assert event.session_id == session_id
    assert event.state == StepState.CANCELLED
    assert event.message == "Session deleted"


async def test_sse_handler_resolves_queue_eagerly_so_sentinel_lands(
    tmp_path: Path,
) -> None:
    """The SSE handler resolves the per-session queue *before* returning
    EventSourceResponse, so cleanup_session() always sees the same queue
    the generator will read from. Without that, the lazy-execution window
    between handler return and generator first-iteration could let a
    DELETE land between session_exists() and get_queue(), leaving the
    generator on a fresh queue that no sentinel ever reaches.

    Exercises the cleanup-sentinel path: handler attaches → DELETE →
    generator picks up the cancelled sentinel and emits done.
    """
    manager, bus = _wire_service_state(tmp_path)
    session_id = "sse-eager-attach"
    manager.create_session(session_id)

    response = await pipeline_router.stream_progress_events(session_id)
    body_iterator = response.body_iterator

    await sessions_router.delete_session(session_id)

    # First chunk: cancelled sentinel pushed by cleanup_session().
    chunk = await asyncio.wait_for(body_iterator.__anext__(), timeout=2.0)
    assert chunk["event"] == "progress"
    assert "cancelled" in chunk["data"]

    # Second chunk: done event closing the stream.
    chunk = await asyncio.wait_for(body_iterator.__anext__(), timeout=2.0)
    assert chunk["event"] == "done"
    assert "cancelled" in chunk["data"]


async def test_sse_generator_emits_deleted_done_via_keepalive_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Belt-and-suspenders: when the cleanup_session sentinel never reaches
    this generator (e.g. the session was deleted with no subscriber attached
    and a new subscriber attaches afterward to a fresh queue), the keepalive
    branch re-checks session_exists() and emits a terminal done event with
    final_state="deleted".

    Patches asyncio.wait_for to raise TimeoutError immediately so we hit the
    keepalive branch on the first iteration without waiting 30 seconds.
    """
    manager, bus = _wire_service_state(tmp_path)
    session_id = "sse-keepalive-recheck"
    manager.create_session(session_id)

    response = await pipeline_router.stream_progress_events(session_id)
    body_iterator = response.body_iterator

    # Delete via SessionManager directly, bypassing cleanup_session() so no
    # sentinel is enqueued. The keepalive recheck is now the only path that
    # can close the stream.
    manager.delete_session(session_id)

    async def instant_timeout(coro: object, timeout: float) -> object:
        # Close the inner coroutine so it doesn't trigger
        # "RuntimeWarning: coroutine 'Queue.get' was never awaited".
        if hasattr(coro, "close"):
            coro.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", instant_timeout)

    # The generator should fire its keepalive branch on the first iteration,
    # see session_exists() == False, and emit a done event with final_state
    # == "deleted". Use asyncio.timeout (not wait_for, which is patched) to
    # guard against a hang if the recheck branch regresses.
    async with asyncio.timeout(2.0):
        chunk = await body_iterator.__anext__()

    assert chunk["event"] == "done"
    assert "deleted" in chunk["data"]


async def test_sse_waits_for_pipeline_completed_marker(tmp_path: Path) -> None:
    manager, bus = _wire_service_state(tmp_path)
    session_id = "sse-sync-barrier"
    manager.create_session(session_id)

    response = await pipeline_router.stream_progress_events(session_id)
    body_iterator = response.body_iterator

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="render",
            state=StepState.COMPLETED,
            percent=100,
        )
    )

    chunk = await asyncio.wait_for(body_iterator.__anext__(), timeout=2.0)
    assert chunk["event"] == "progress"
    assert "pipeline_completed" not in chunk["data"]

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="render",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True},
        )
    )

    chunk = await asyncio.wait_for(body_iterator.__anext__(), timeout=2.0)
    assert chunk["event"] == "progress"
    assert "pipeline_completed" in chunk["data"]

    chunk = await asyncio.wait_for(body_iterator.__anext__(), timeout=2.0)
    assert chunk["event"] == "done"
    assert "completed" in chunk["data"]


async def test_cleanup_session_drains_backlog_before_sentinel(
    tmp_path: Path,
) -> None:
    """Stale progress events queued before DELETE must NOT be delivered to a
    subscriber after the session is gone. cleanup_session() drains the
    queue before enqueueing the terminal sentinel so the subscriber sees
    the close immediately, not after walking historic progress for an
    already-deleted session.
    """
    manager, bus = _wire_service_state(tmp_path)
    session_id = "sse-backlog-drain"
    manager.create_session(session_id)

    subscriber_queue = bus.get_queue(session_id)
    for step in ("prepare_uvs", "discover_materials", "generate_prompts"):
        await bus.emit(
            ProgressEvent(session_id=session_id, step=step, state=StepState.RUNNING)
        )
    assert subscriber_queue.qsize() >= 3

    await sessions_router.delete_session(session_id)

    # Exactly one event remains: the terminal sentinel. The backlog was
    # drained inside cleanup_session() under the bus lock.
    assert subscriber_queue.qsize() == 1
    event = subscriber_queue.get_nowait()
    assert event.state == StepState.CANCELLED
    assert event.message == "Session deleted"


async def test_emit_after_disk_delete_is_dropped(tmp_path: Path) -> None:
    """Late worker emits for a deleted session must NOT repopulate
    EventBus state. Without this defense, the in-memory ``_state`` and
    ``_queues`` dicts leak per-deleted-session entries every time a
    worker emits a progress event after its session's disk dir is
    gone -- especially relevant for TTL cleanup, which deletes disk
    state without cancelling any active JobRegistry task.

    Public read endpoints already 404 via the session_exists() gates
    added to /pipeline/{sid}/status and /events, but those gates only
    hide the leak from clients; the bus would still accumulate state.
    """
    manager, bus = _wire_service_state(tmp_path)
    session_id = "late-emit-drop"
    manager.create_session(session_id)
    manager.delete_session(session_id)
    assert not manager.session_exists(session_id)
    assert bus.get_snapshot(session_id) is None

    # Late emit from a worker that didn't observe the deletion.
    await bus.emit(
        ProgressEvent(
            session_id=session_id, step="generate_textures", state=StepState.RUNNING
        )
    )

    # Must NOT have created snapshot or queue for the deleted session.
    assert bus.get_snapshot(session_id) is None
    assert session_id not in bus._queues


async def test_ttl_cleanup_releases_event_bus_state(tmp_path: Path) -> None:
    """Periodic TTL cleanup must release event-bus snapshot/queue for every
    session whose disk dir was reaped, not just the HTTP DELETE path. Without
    this, in-memory per-session bus state accumulates indefinitely in a long-
    running service because expired sessions never see the DELETE handler.

    This test simulates the cleanup loop in main.py: call
    cleanup_expired_sessions() and then iterate the returned IDs through
    EventBus.cleanup_session().
    """
    from datetime import UTC, datetime, timedelta

    manager, bus = _wire_service_state(tmp_path)
    expired_id = "ttl-expired"
    fresh_id = "ttl-fresh"
    manager.create_session(expired_id)
    manager.create_session(fresh_id)

    # Backdate one session's TTL so cleanup_expired_sessions removes it.
    manager.update_session(
        expired_id,
        {"ttl_expires_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat()},
    )

    # Both sessions emit events so the bus has per-session state.
    for sid in (expired_id, fresh_id):
        await bus.emit(
            ProgressEvent(
                session_id=sid, step="generate_textures", state=StepState.RUNNING
            )
        )
    assert bus.get_snapshot(expired_id) is not None
    assert bus.get_snapshot(fresh_id) is not None

    # Run the TTL sweep + bus cleanup wiring exactly as main.py's loop does.
    cleaned = manager.cleanup_expired_sessions()
    assert cleaned == [expired_id]
    for sid in cleaned:
        await bus.cleanup_session(sid)

    # Expired session: gone from disk and from the bus.
    assert not manager.session_exists(expired_id)
    assert bus.get_snapshot(expired_id) is None

    # Fresh session: untouched.
    assert manager.session_exists(fresh_id)
    assert bus.get_snapshot(fresh_id) is not None
