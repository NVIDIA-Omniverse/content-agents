# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SessionManager per-session update locking.

Tests the race condition fix where concurrent update_session calls
could overwrite each other's changes (lost update problem).
"""

import asyncio
import tempfile
from uuid import uuid4

import pytest

from ...service.session.manager import SessionManager
from ...service.storage.local_store import LocalSessionStore


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def manager(tmp_path: str):
    return SessionManager(tmp_path, store=LocalSessionStore(tmp_path))


@pytest.fixture
def session_id() -> str:
    return str(uuid4())


@pytest.mark.unit
@pytest.mark.asyncio
class TestUpdateSessionLocking:
    """Test that per-session locks prevent concurrent update races."""

    async def test_concurrent_updates_no_lost_writes(
        self, manager: SessionManager, session_id: str
    ):
        """Test that concurrent update_session calls don't lose writes.

        Simulates the race between EventBus setting status="completed"
        and executor writing results, ensuring both changes are preserved.
        """
        await manager.create_session(session_id)

        # Simulate two concurrent updates (EventBus vs executor)
        async def eventbus_update():
            await manager.update_session(session_id, {"status": "completed"})

        async def executor_update():
            await manager.update_session(
                session_id,
                {
                    "status": "completed",
                    "results": {
                        "prims_processed": 42,
                        "predictions_made": 10,
                        "materials_applied": 5,
                    },
                    "duration_seconds": 60,
                },
            )

        # Run both concurrently
        await asyncio.gather(eventbus_update(), executor_update())

        # Verify metadata has results (not overwritten by eventbus_update)
        metadata = await manager.get_session_metadata(session_id)
        assert metadata is not None
        assert metadata["status"] == "completed"
        # Results must be present (the executor's write must not be lost)
        assert metadata.get("results") is not None
        assert metadata["results"]["prims_processed"] == 42

    async def test_lock_is_per_session(self, manager: SessionManager):
        """Test that locks are independent per session (no cross-session blocking)."""
        sid_a = str(uuid4())
        sid_b = str(uuid4())
        await manager.create_session(sid_a)
        await manager.create_session(sid_b)

        order = []

        async def update_a():
            await manager.update_session(sid_a, {"status": "running"})
            order.append("a")

        async def update_b():
            await manager.update_session(sid_b, {"status": "running"})
            order.append("b")

        await asyncio.gather(update_a(), update_b())

        # Both should complete (no deadlock)
        assert "a" in order
        assert "b" in order

        meta_a = await manager.get_session_metadata(sid_a)
        meta_b = await manager.get_session_metadata(sid_b)
        assert meta_a["status"] == "running"
        assert meta_b["status"] == "running"

    async def test_delete_session_cleans_up_lock(
        self, manager: SessionManager, session_id: str
    ):
        """Test that delete_session removes the per-session lock."""
        await manager.create_session(session_id)

        # Trigger lock creation
        await manager.update_session(session_id, {"status": "running"})
        assert session_id in manager._update_locks

        # Delete session
        await manager.delete_session(session_id)
        assert session_id not in manager._update_locks

    async def test_sequential_updates_preserve_all_fields(
        self, manager: SessionManager, session_id: str
    ):
        """Test that sequential updates under the lock preserve all fields."""
        await manager.create_session(session_id)

        await manager.update_session(session_id, {"status": "running"})
        await manager.update_session(session_id, {"current_step": {"name": "predict"}})
        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {"prims_processed": 5},
            },
        )

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["status"] == "completed"
        assert metadata["results"]["prims_processed"] == 5
        # current_step should still be present from the second update
        assert metadata["current_step"]["name"] == "predict"

    async def test_many_concurrent_updates_all_succeed(
        self, manager: SessionManager, session_id: str
    ):
        """Test that many concurrent updates to the same session all complete."""
        await manager.create_session(session_id)

        async def update_field(i: int):
            await manager.update_session(session_id, {f"field_{i}": i})

        # Fire 20 concurrent updates
        await asyncio.gather(*[update_field(i) for i in range(20)])

        metadata = await manager.get_session_metadata(session_id)
        # All 20 fields should be present
        for i in range(20):
            assert metadata[f"field_{i}"] == i
