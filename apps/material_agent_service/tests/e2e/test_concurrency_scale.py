# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end concurrency tests.

Validates that the semaphore correctly limits concurrent pipeline executions
and prevents resource exhaustion under load.
"""

import asyncio
import os

import pytest


@pytest.mark.e2e
@pytest.mark.concurrency
class TestConcurrencyScale:
    """Test concurrent pipeline execution with semaphore limits."""

    async def test_five_sessions_under_semaphore(self, client, _stub_executor):
        """Test that 5 sessions respect the 2-slot concurrency limit.

        This validates the real global semaphore from JobRegistry.
        """
        # Enable small delays to observe concurrency overlap
        old_delay = os.environ.get("TEST_STEP_DELAY")
        os.environ["TEST_STEP_DELAY"] = "0.05"

        try:
            # Launch 5 sessions rapidly
            async def start_session():
                usd_content = b"#usda 1.0\n"
                files = {
                    "usd_file": ("scene.usda", usd_content, "application/octet-stream")
                }
                return await client.post(
                    "/pipeline",
                    files=files,
                    data={"user_email": "test@example.com"},
                )

            posts = await asyncio.gather(*[start_session() for _ in range(5)])
            session_ids = [p.json()["session_id"] for p in posts]

            assert len(session_ids) == 5

            # Wait for all to complete
            async def wait_for_completion(session_id):
                for _ in range(400):
                    status_r = await client.get(f"/pipeline/{session_id}/status")
                    if status_r.json()["status"] == "completed":
                        return True
                    await asyncio.sleep(0.01)
                return False

            results = await asyncio.gather(
                *[wait_for_completion(sid) for sid in session_ids]
            )
            assert all(results), "All sessions should complete"

            # Verify peak concurrency never exceeded 2
            peak = _stub_executor["max_concurrency_seen"]()
            assert peak <= 2, f"Peak concurrency was {peak}, should be <= 2"

        finally:
            # Restore delay
            if old_delay is not None:
                os.environ["TEST_STEP_DELAY"] = old_delay
            else:
                os.environ.pop("TEST_STEP_DELAY", None)

    async def test_sessions_complete_successfully_under_load(self, client):
        """Test that all sessions complete successfully when run concurrently."""

        # Launch 5 sessions
        async def start_session():
            usd_content = b"#usda 1.0\n"
            files = {
                "usd_file": ("scene.usda", usd_content, "application/octet-stream")
            }
            return await client.post(
                "/pipeline", files=files, data={"user_email": "test@example.com"}
            )

        posts = await asyncio.gather(*[start_session() for _ in range(5)])
        session_ids = [p.json()["session_id"] for p in posts]

        # Wait for all to complete
        async def get_final_status(session_id):
            for _ in range(400):
                status_r = await client.get(f"/pipeline/{session_id}/status")
                if status_r.json()["status"] == "completed":
                    return status_r.json()
                await asyncio.sleep(0.01)
            return None

        statuses = await asyncio.gather(*[get_final_status(sid) for sid in session_ids])

        # All should complete successfully
        assert all(s is not None for s in statuses), "All sessions should complete"
        assert all(s["status"] == "completed" for s in statuses), (
            "All should reach completed status"
        )
        assert all(s["overall_progress"]["percent"] == 100 for s in statuses), (
            "All should reach 100%"
        )

    async def test_individual_sessions_unaffected_by_concurrency(self, client):
        """Test that running sessions don't affect each other's progress."""

        # Start 5 sessions
        async def start_session():
            usd_content = b"#usda 1.0\n"
            files = {
                "usd_file": ("scene.usda", usd_content, "application/octet-stream")
            }
            return await client.post(
                "/pipeline", files=files, data={"user_email": "test@example.com"}
            )

        posts = await asyncio.gather(*[start_session() for _ in range(5)])
        session_ids = [p.json()["session_id"] for p in posts]

        # Gather progress snapshots from all sessions periodically
        progress_over_time = {sid: [] for sid in session_ids}

        for poll_num in range(50):
            tasks = [client.get(f"/pipeline/{sid}/status") for sid in session_ids]
            responses = await asyncio.gather(*tasks)

            for sid, response in zip(session_ids, responses):
                if response.status_code == 200:
                    progress = response.json()["overall_progress"]["percent"]
                    progress_over_time[sid].append(progress)

            await asyncio.sleep(0.01)

        # Each session should show monotonic progress (or be complete)
        for sid, progresses in progress_over_time.items():
            if progresses:
                # Should be monotonically non-decreasing
                for i in range(1, len(progresses)):
                    assert progresses[i] >= progresses[i - 1], (
                        f"Session {sid} had non-monotonic progress: "
                        f"{progresses[i - 1]} -> {progresses[i]}"
                    )

    async def test_rapid_fire_creation(self, client):
        """Test that we can create many sessions in rapid succession."""

        # Create 20 sessions as fast as possible
        async def start_session(i):
            usd_content = b"#usda 1.0\n"
            files = {
                "usd_file": (f"scene_{i}.usda", usd_content, "application/octet-stream")
            }
            return await client.post(
                "/pipeline", files=files, data={"user_email": "test@example.com"}
            )

        posts = await asyncio.gather(*[start_session(i) for i in range(20)])

        # All should succeed
        assert all(p.status_code == 202 for p in posts)

        session_ids = [p.json()["session_id"] for p in posts]
        assert len(set(session_ids)) == 20, "All session IDs should be unique"

    async def test_cancellation_under_concurrency(self, client):
        """Test that cancelling one session doesn't affect others."""

        # Start 5 sessions
        async def start_session():
            usd_content = b"#usda 1.0\n"
            files = {
                "usd_file": ("scene.usda", usd_content, "application/octet-stream")
            }
            return await client.post(
                "/pipeline", files=files, data={"user_email": "test@example.com"}
            )

        posts = await asyncio.gather(*[start_session() for _ in range(5)])
        session_ids = [p.json()["session_id"] for p in posts]

        # Give them a moment to start
        await asyncio.sleep(0.1)

        # Cancel the first one
        await client.post(f"/pipeline/{session_ids[0]}/cancel")

        # Others should continue to completion
        async def get_final_status(session_id):
            for _ in range(400):
                status_r = await client.get(f"/pipeline/{session_id}/status")
                if status_r.json()["status"] in ["completed", "cancelled", "failed"]:
                    return status_r.json()
                await asyncio.sleep(0.01)
            return None

        # Check that most sessions still complete
        statuses = await asyncio.gather(
            *[get_final_status(sid) for sid in session_ids[1:]]
        )

        completed_count = sum(1 for s in statuses if s and s["status"] == "completed")
        assert completed_count >= 3, "At least 3 non-cancelled sessions should complete"

    async def test_semaphore_fairness(self, client, _stub_executor):
        """Test that semaphore allocation is fair (no starvation)."""
        # Create 5 sessions in sequence (not concurrently)
        session_ids = []

        for i in range(5):
            usd_content = b"#usda 1.0\n"
            files = {
                "usd_file": (f"scene_{i}.usda", usd_content, "application/octet-stream")
            }
            r = await client.post(
                "/pipeline", files=files, data={"user_email": "test@example.com"}
            )
            session_ids.append(r.json()["session_id"])
            # Small delay between submissions
            await asyncio.sleep(0.02)

        await asyncio.sleep(1)

        # Wait for all to complete
        async def wait_for_completion(session_id):
            for _ in range(400):
                status_r = await client.get(f"/pipeline/{session_id}/status")
                if status_r.json()["status"] == "completed":
                    return True
                await asyncio.sleep(0.01)
            return False

        results = await asyncio.gather(
            *[wait_for_completion(sid) for sid in session_ids]
        )
        assert all(results), "All sessions should eventually complete (no starvation)"

        # Verify they ran with proper concurrency
        peak = _stub_executor["max_concurrency_seen"]()
        assert peak <= 2, f"Peak concurrency was {peak}, should be <= 2"
