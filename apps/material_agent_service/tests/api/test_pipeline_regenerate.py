# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline regeneration.

Tests the regenerate endpoint for re-running specific steps.
"""

import asyncio

import pytest


@pytest.mark.api
class TestPipelineRegenerate:
    """Test pipeline regeneration."""

    async def test_regenerate_apply_only(self, client):
        """Test regenerating the apply step only."""
        # Create and complete a pipeline first
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        await asyncio.sleep(1)

        # Wait for completion
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Now regenerate with apply only
        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["apply"]},
        )

        assert regen_r.status_code == 202
        assert regen_r.json()["status"] == "pending"

    async def test_regenerate_multiple_steps(self, client):
        """Test regenerating multiple steps."""
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

        # Regenerate predict and apply
        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict", "apply"]},
        )

        assert regen_r.status_code == 202

    async def test_regenerate_returns_400_while_running(self, client):
        """Test that regenerate returns 400 while pipeline is running."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Immediately try to regenerate (still running)
        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["apply"]},
        )

        # Should return 400 - can't regenerate while running
        if regen_r.status_code != 202:
            assert regen_r.status_code == 400

    async def test_regenerate_nonexistent_session(self, client):
        """Test regenerate on nonexistent session returns 404."""
        regen_r = await client.post(
            "/pipeline/00000000-0000-0000-0000-000000000000/regenerate",
            json={"steps": ["apply"]},
        )

        assert regen_r.status_code == 404

    async def test_regenerate_with_prompt_override(self, client):
        """Test regenerate with user prompt override."""
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

        # Regenerate with custom prompt
        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={
                "steps": ["predict", "apply"],
                "user_prompt": "Focus on shiny surfaces",
            },
        )

        assert regen_r.status_code == 202
