# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /results endpoint wait-for-stats behavior.

Tests that the get_pipeline_results endpoint polls for results
when status is "completed" but stats haven't been persisted yet.
"""

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
class TestResultsWaitForStats:
    """Test the wait-for-stats polling logic at the session manager level.

    These tests verify the conditions that the /results endpoint checks:
    - When results are populated, they're returned immediately
    - When results are empty/all-zero, polling is needed
    """

    async def test_results_present_after_atomic_write(
        self, manager: SessionManager, session_id: str
    ):
        """Test that writing status and results together makes both available."""
        await manager.create_session(session_id)

        # Simulate executor writing status + results atomically
        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {
                    "prims_processed": 10,
                    "predictions_made": 5,
                    "materials_applied": 3,
                },
                "duration_seconds": 60,
                "completed_at": "2026-01-01T00:00:00",
            },
        )

        metadata = await manager.get_session_metadata(session_id)
        assert metadata is not None
        assert metadata["status"] == "completed"
        results = metadata.get("results", {})
        assert any(v for v in results.values() if v)
        assert results["prims_processed"] == 10

    async def test_status_completed_without_results_is_detectable(
        self, manager: SessionManager, session_id: str
    ):
        """Test that we can detect when status=completed but results are missing.

        This is the race condition state: EventBus set status="completed"
        but executor hasn't written results yet.
        """
        await manager.create_session(session_id)

        # Simulate EventBus writing just the status (without results)
        await manager.update_session(
            session_id,
            {"status": "completed"},
        )

        metadata = await manager.get_session_metadata(session_id)
        assert metadata is not None
        assert metadata["status"] == "completed"
        # Results should be missing or empty
        results = metadata.get("results") or {}
        has_stats = any(v for v in results.values() if v)
        assert not has_stats

    async def test_results_appear_after_delayed_executor_write(
        self, manager: SessionManager, session_id: str
    ):
        """Test that results become available after a subsequent update.

        Simulates the full race: EventBus writes status first,
        then executor writes results.
        """
        await manager.create_session(session_id)

        # Step 1: EventBus sets status=completed (no results)
        await manager.update_session(
            session_id,
            {"status": "completed"},
        )

        metadata = await manager.get_session_metadata(session_id)
        results = metadata.get("results") or {}
        assert not any(v for v in results.values() if v)

        # Step 2: Executor writes results (with status too, for atomicity)
        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {
                    "prims_processed": 42,
                    "predictions_made": 10,
                    "materials_applied": 5,
                },
                "duration_seconds": 90,
                "completed_at": "2026-01-01T00:00:00",
            },
        )

        # Step 3: Now results should be available
        metadata = await manager.get_session_metadata(session_id)
        results = metadata.get("results") or {}
        assert any(v for v in results.values() if v)
        assert results["prims_processed"] == 42
        assert results["predictions_made"] == 10
