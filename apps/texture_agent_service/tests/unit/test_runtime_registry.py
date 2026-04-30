# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import inspect
import threading
from pathlib import Path

import pytest

from ...service.routers.pipeline_router import (
    _cancel_never_started_callback,
    _release_worker_slot_callback,
)
from ...service.runtime.bus import get_event_bus, init_event_bus
from ...service.runtime.registry import JobRegistry
from ...service.session.manager import SessionManager


async def test_cancel_timeout_keeps_draining_executor_registered() -> None:
    registry = JobRegistry(max_concurrent=1, cancel_wait_seconds=0.01)
    session_id = "draining-session"
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_step() -> None:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("blocking test step was not released")
        finished.set()

    async def job() -> None:
        loop = asyncio.get_running_loop()
        step_future = loop.run_in_executor(None, blocking_step)
        try:
            await asyncio.shield(step_future)
        except asyncio.CancelledError:
            await asyncio.shield(step_future)
            raise

    await registry.register(session_id, job())
    assert await asyncio.to_thread(started.wait, 1)

    assert await registry.cancel(session_id) is True

    task = registry.get_task(session_id)
    assert task is not None
    assert registry.is_running(session_id) is True
    assert registry.active_count == 1
    assert registry.registered_count == 1
    assert finished.is_set() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert finished.is_set() is True
    assert registry.active_count == 0
    assert registry.registered_count == 0


async def test_cancel_timeout_keeps_stubborn_task_registered() -> None:
    registry = JobRegistry(max_concurrent=1, cancel_wait_seconds=0.01)
    session_id = "stubborn-session"
    started = asyncio.Event()
    release = threading.Event()

    async def job() -> None:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            if not await asyncio.to_thread(release.wait, 5):
                raise TimeoutError("stubborn task cleanup was not released")
            raise

    await registry.register(session_id, job())
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await registry.cancel(session_id) is True

    task = registry.get_task(session_id)
    assert task is not None
    assert registry.is_running(session_id) is True
    assert registry.active_count == 1
    assert registry.registered_count == 1

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert registry.get_task(session_id) is None
    assert registry.active_count == 0
    assert registry.registered_count == 0


async def test_pre_start_cancel_releases_reservation_and_marks_cancelled(
    tmp_path: Path,
) -> None:
    registry = JobRegistry(max_concurrent=1, cancel_wait_seconds=0.01)
    manager = SessionManager(tmp_path)
    init_event_bus(manager)
    session_id = "pre-start-cancel"
    manager.create_session(session_id)
    worker_lock = manager.acquire_worker_lock(session_id, timeout=0)
    started = asyncio.Event()

    async def job() -> None:
        started.set()

    await registry.register(
        session_id,
        job(),
        on_never_started=_cancel_never_started_callback(manager, session_id),
        on_finished=_release_worker_slot_callback(manager, session_id, worker_lock),
    )

    assert manager.delete_session(session_id) is False

    assert await registry.cancel(session_id) is True
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert started.is_set() is False
    assert registry.get_task(session_id) is None
    assert registry.active_count == 0
    assert registry.registered_count == 0

    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    snapshot = get_event_bus().get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"

    with manager.worker_lock(session_id, timeout=0):
        pass


async def test_duplicate_register_closes_rejected_coroutine() -> None:
    registry = JobRegistry(max_concurrent=1, cancel_wait_seconds=0.01)
    session_id = "duplicate-session"
    release = asyncio.Event()

    async def running_job() -> None:
        await release.wait()

    async def rejected_job() -> None:
        return None

    await registry.register(session_id, running_job())
    rejected = rejected_job()

    with pytest.raises(RuntimeError, match="already running"):
        await registry.register(session_id, rejected)

    assert inspect.getcoroutinestate(rejected) == inspect.CORO_CLOSED

    release.set()
    task = registry.get_task(session_id)
    assert task is not None
    await asyncio.wait_for(task, timeout=1)
