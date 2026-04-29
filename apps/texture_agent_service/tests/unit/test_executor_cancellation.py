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
from pathlib import Path
from typing import Any

import pytest

from ...service.runtime import bus as bus_module
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
    """Sanity check: non-cancellation errors still hit the failed branch."""
    session_id = "def456"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)

    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

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
