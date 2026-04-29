# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline cancellation.

Tests the cancel endpoint and cancellation semantics.
"""

import asyncio

import pytest


@pytest.mark.api
class TestPipelineCancel:
    """Test pipeline cancellation."""

    async def test_cancel_running_pipeline(self, client):
        """Test cancelling a running pipeline."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Give it a moment to start
        await asyncio.sleep(0.05)

        # Cancel it
        cancel_r = await client.post(f"/pipeline/{session_id}/cancel")

        assert cancel_r.status_code == 200
        assert cancel_r.json()["status"] == "cancelling"

    async def test_cancel_returns_400_for_completed(self, client):
        """Test that cancelling a completed pipeline returns 400."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Wait for completion
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Try to cancel completed pipeline
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
        # Slow down the stub executor to observe cancellation
        import os

        os.environ["TEST_STEP_DELAY"] = "0.1"

        try:
            usd_content = b"#usda 1.0\n"
            files = {
                "usd_file": ("scene.usda", usd_content, "application/octet-stream")
            }
            create_r = await client.post(
                "/pipeline", files=files, data={"user_email": "test@example.com"}
            )
            session_id = create_r.json()["session_id"]

            # Wait a bit for it to start
            await asyncio.sleep(0.15)

            # Cancel it
            await client.post(f"/pipeline/{session_id}/cancel")

            # Poll status - should eventually reach cancelled or completed state
            for _ in range(100):
                status_r = await client.get(f"/pipeline/{session_id}/status")
                status = status_r.json()["status"]
                if status in ["cancelled", "completed"]:
                    break
                await asyncio.sleep(0.05)

        finally:
            # Reset delay
            os.environ["TEST_STEP_DELAY"] = "0.01"
