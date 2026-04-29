# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline cancellation.

Tests the cancel endpoint and cancellation semantics.
"""

import asyncio
import os

import pytest

from ..conftest import make_pipeline_files


@pytest.mark.api
class TestPipelineCancel:
    """Test pipeline cancellation."""

    async def test_cancel_running_pipeline(self, client):
        """Test cancelling a running pipeline."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        await asyncio.sleep(0.05)

        cancel_r = await client.post(f"/pipeline/{session_id}/cancel")

        assert cancel_r.status_code == 200
        assert cancel_r.json()["status"] == "cancelling"

    async def test_cancel_returns_400_for_completed(self, client):
        """Test that cancelling a completed pipeline returns 400."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        cancel_r = await client.post(f"/pipeline/{session_id}/cancel")

        assert cancel_r.status_code == 400
        assert "Cannot cancel" in cancel_r.json()["detail"]

    async def test_cancel_returns_404_for_nonexistent(self, client):
        """Test that cancelling nonexistent session returns 404."""
        cancel_r = await client.post(
            "/pipeline/00000000-0000-0000-0000-000000000000/cancel"
        )

        assert cancel_r.status_code == 404

    async def test_cancelled_pipeline_stops_processing(self, client):
        """Test that cancelled pipeline stops processing."""
        os.environ["TEST_STEP_DELAY"] = "0.1"

        try:
            create_r = await client.post("/pipeline", files=make_pipeline_files())
            session_id = create_r.json()["session_id"]

            await asyncio.sleep(0.15)

            await client.post(f"/pipeline/{session_id}/cancel")

            for _ in range(100):
                status_r = await client.get(f"/pipeline/{session_id}/status")
                status = status_r.json()["status"]
                if status in ["cancelled", "completed"]:
                    break
                await asyncio.sleep(0.05)

        finally:
            os.environ["TEST_STEP_DELAY"] = "0.01"
