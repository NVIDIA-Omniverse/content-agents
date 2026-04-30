# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the outer wrapper of execute_pipeline_async persists `cancelled`
state when asyncio task cancellation lands before the cooperative checkpoint.

Without this branch, POST /cancel → task.cancel() mid-step would leave the
session pinned at "running" forever, which is the regression that nvbug
6122134 / OMPE-91539 surfaced.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from ...service.runtime import bus as bus_module
from ...service.session.manager import SessionManager
from ...service.workers import executor


class _StubSessionManager:
    """Minimal session_manager stub for execute_pipeline_async."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.updates: list[dict[str, Any]] = []

    def get_session_dir(self, session_id: str) -> Path:
        return self.session_dir

    def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        self.updates.append(updates)

    def session_exists(self, session_id: str) -> bool:
        return True

    def is_cancelled(self, session_id: str) -> bool:
        return False

    def worker_lock(self, session_id: str):
        from contextlib import nullcontext

        return nullcontext()


class BlockingTask:
    name = "GenerateTextures"

    def __init__(
        self,
        started: threading.Event,
        release: threading.Event,
        finished: threading.Event,
    ) -> None:
        self.started = started
        self.release = release
        self.finished = finished

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("blocking test task was not released")
        self.finished.set()
        return context


class FailingAfterCancelTask:
    name = "GenerateTextures"

    def __init__(
        self,
        started: threading.Event,
        release: threading.Event,
    ) -> None:
        self.started = started
        self.release = release

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("failing test task was not released")
        raise RuntimeError("step crashed during cancellation drain")


async def test_outer_wrapper_persists_cancelled_on_cancellederror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """task.cancel() mid-step raises CancelledError into the outer wrapper.

    The wrapper must persist `cancelled` to disk and emit a CANCELLED event
    so /status flips from `cancelling` (set by POST /cancel) to `cancelled`
    instead of stalling.
    """
    session_id = "abc123"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)

    # Reset the global bus singleton with our stub manager so emitted events
    # land in a fresh in-memory snapshot we can inspect.
    bus_module._event_bus = None
    bus = bus_module.init_event_bus(manager)

    # Pre-populate state as if the worker were mid-step (running).
    from ...service.runtime.events import ProgressEvent, StepState

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
            current=1,
            total=8,
        )
    )

    async def _raises_cancelled(*args: Any, **kwargs: Any) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(executor, "_execute_pipeline_inner", _raises_cancelled)

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_pipeline_async(
            session_id=session_id,
            config_dict={},
            session_manager=manager,
        )

    # Disk state was persisted as `cancelled` (synchronous, before the emit).
    assert {"status": "cancelled"} in manager.updates

    # In-memory bus snapshot also reached `cancelled` via the emitted event.
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"


async def test_outer_wrapper_does_not_persist_cancelled_on_normal_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity check: non-cancellation errors still hit the failed branch.

    Also pins the bus emit added in OMPE-91856/91861: the outer wrapper
    must emit a FAILED event in addition to persisting status="failed" to
    disk, so the in-memory snapshot agrees with disk on terminal status
    without read-side guards needing to encode "trust disk over bus".
    """
    session_id = "def456"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)

    bus_module._event_bus = None
    bus = bus_module.init_event_bus(manager)

    async def _raises_runtime(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(executor, "_execute_pipeline_inner", _raises_runtime)

    with pytest.raises(RuntimeError, match="kaboom"):
        await executor.execute_pipeline_async(
            session_id=session_id,
            config_dict={},
            session_manager=manager,
        )

    assert any(u.get("status") == "failed" for u in manager.updates)
    assert all(u.get("status") != "cancelled" for u in manager.updates)

    # Bus snapshot must also have reached "failed" via the emitted event,
    # not just disk via update_session.
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot.get("error") == "kaboom"


async def test_inner_cancellation_waits_for_threaded_step_to_stop(
    tmp_path: Path,
) -> None:
    """task.cancel() should keep the pipeline task active while the
    worker thread is still writing artifacts.
    """
    session_id = "threaded-cancel"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)

    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    blocking_task = BlockingTask(started, release, finished)

    def _blocking_factory(context: dict[str, Any], skip=None, only=None):
        return [blocking_task]

    pipeline_task = asyncio.create_task(
        executor._execute_pipeline_inner(
            session_id=session_id,
            config_dict={"input": {"usd_path": "/tmp/in.usd"}},
            session_manager=manager,
            event_bus=bus_module.get_event_bus(),
            session_dir=session_dir,
            only_steps=None,
            skip_steps=None,
            create_texture_pipeline_workflow=_blocking_factory,
        )
    )

    assert await asyncio.to_thread(started.wait, 1)

    pipeline_task.cancel()
    await asyncio.sleep(0.05)

    assert pipeline_task.done() is False
    assert finished.is_set() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pipeline_task, timeout=1)

    assert finished.is_set() is True


async def test_execute_pipeline_holds_worker_lock_for_inner_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cross-process writer lock must cover the whole pipeline body.

    DELETE and same-session restart checks rely on this lock after registry
    state has gone stale or while a cancelled task is draining. Holding it
    only around individual steps leaves gaps between steps and during final
    metadata/event writes.
    """
    session_id = "lock-scope"
    manager = SessionManager(storage_path=tmp_path / "sessions", ttl_hours=24)
    manager.create_session(session_id, config={})

    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    observed: dict[str, bool] = {}

    async def _assert_locked(
        session_id_arg: str,
        config_dict: dict[str, Any],
        session_manager: SessionManager,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert session_id_arg == session_id
        observed["during"] = await asyncio.to_thread(
            session_manager.is_worker_active, session_id_arg
        )
        session_manager.update_session(session_id_arg, {"status": "completed"})

    monkeypatch.setattr(executor, "_execute_pipeline_inner", _assert_locked)

    await executor.execute_pipeline_async(
        session_id=session_id,
        config_dict={},
        session_manager=manager,
    )

    assert observed["during"] is True
    assert await asyncio.to_thread(manager.is_worker_active, session_id) is False


async def test_repeated_cancel_keeps_worker_lock_until_thread_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "double-cancel"
    manager = SessionManager(storage_path=tmp_path / "sessions", ttl_hours=24)
    session_dir = manager.create_session(session_id, config={})

    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    blocking_task = BlockingTask(started, release, finished)

    def _blocking_factory(context: dict[str, Any], skip=None, only=None):
        return [blocking_task]

    from texture_agent.workflows import factory as workflow_factory

    monkeypatch.setattr(
        workflow_factory,
        "create_texture_pipeline_workflow",
        _blocking_factory,
    )

    pipeline_task = asyncio.create_task(
        executor.execute_pipeline_async(
            session_id=session_id,
            config_dict={"input": {"usd_path": str(session_dir / "input.usd")}},
            session_manager=manager,
        )
    )

    assert await asyncio.to_thread(started.wait, 1)

    pipeline_task.cancel()
    await asyncio.sleep(0.05)
    assert await asyncio.to_thread(manager.is_worker_active, session_id) is True
    assert pipeline_task.done() is False

    pipeline_task.cancel()
    await asyncio.sleep(0.05)
    assert await asyncio.to_thread(manager.is_worker_active, session_id) is True
    assert pipeline_task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pipeline_task, timeout=1)

    assert finished.is_set() is True
    assert await asyncio.to_thread(manager.is_worker_active, session_id) is False


async def test_cancel_drain_step_failure_is_not_reported_as_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the draining sync step fails, preserve failed-step diagnostics."""
    session_id = "cancel-drain-failure"
    manager = SessionManager(storage_path=tmp_path / "sessions", ttl_hours=24)
    session_dir = manager.create_session(session_id, config={})

    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    started = threading.Event()
    release = threading.Event()
    failing_task = FailingAfterCancelTask(started, release)

    def _failing_factory(context: dict[str, Any], skip=None, only=None):
        return [failing_task]

    from texture_agent.workflows import factory as workflow_factory

    monkeypatch.setattr(
        workflow_factory,
        "create_texture_pipeline_workflow",
        _failing_factory,
    )

    pipeline_task = asyncio.create_task(
        executor.execute_pipeline_async(
            session_id=session_id,
            config_dict={"input": {"usd_path": str(session_dir / "input.usd")}},
            session_manager=manager,
        )
    )

    assert await asyncio.to_thread(started.wait, 1)

    pipeline_task.cancel()
    release.set()
    with pytest.raises(RuntimeError, match="step crashed during cancellation drain"):
        await asyncio.wait_for(pipeline_task, timeout=1)

    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "failed"
    assert metadata["failed_step"] == "FailingAfterCancelTask"
    assert "step crashed during cancellation drain" in metadata["error"]

    snapshot = bus_module.get_event_bus().get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["failed_step"] == "FailingAfterCancelTask"
    assert await asyncio.to_thread(manager.is_worker_active, session_id) is False


async def test_cancel_drain_timeout_marks_worker_stalled_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-returning sync step must not pin registry capacity forever.

    The executor releases the worker lock after a bounded drain timeout, but
    leaves a stalled-worker marker so DELETE/TTL still treat artifacts as unsafe
    until the thread future eventually exits.
    """
    session_id = "cancel-timeout"
    manager = SessionManager(storage_path=tmp_path / "sessions", ttl_hours=24)
    session_dir = manager.create_session(session_id, config={})

    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    monkeypatch.setattr(executor.service_config, "cancel_drain_timeout_seconds", 0.05)

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    blocking_task = BlockingTask(started, release, finished)

    def _blocking_factory(context: dict[str, Any], skip=None, only=None):
        return [blocking_task]

    from texture_agent.workflows import factory as workflow_factory

    monkeypatch.setattr(
        workflow_factory,
        "create_texture_pipeline_workflow",
        _blocking_factory,
    )

    pipeline_task = asyncio.create_task(
        executor.execute_pipeline_async(
            session_id=session_id,
            config_dict={"input": {"usd_path": str(session_dir / "input.usd")}},
            session_manager=manager,
        )
    )

    assert await asyncio.to_thread(started.wait, 1)

    pipeline_task.cancel()
    with pytest.raises(RuntimeError, match="Cancellation timed out"):
        await asyncio.wait_for(pipeline_task, timeout=1)

    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "failed"
    assert "Cancellation timed out" in metadata["error"]
    assert metadata["failed_step"] == "BlockingTask"

    snapshot = bus_module.get_event_bus().get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["failed_step"] == "BlockingTask"
    assert finished.is_set() is False
    assert await asyncio.to_thread(manager.is_worker_active, session_id) is True
    assert manager.delete_session(session_id) is False

    release.set()
    assert await asyncio.to_thread(finished.wait, 1)
    await asyncio.sleep(0.05)

    assert await asyncio.to_thread(manager.is_worker_active, session_id) is False
