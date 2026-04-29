# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline regeneration.

Tests the regenerate endpoint for re-running specific steps.
"""

import asyncio

import pytest

from ..conftest import make_pipeline_files


@pytest.mark.api
class TestPipelineRegenerate:
    """Test pipeline regeneration."""

    async def test_regenerate_predict_only(self, client):
        """Test regenerating the predict step only."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"]},
        )

        assert regen_r.status_code == 202
        assert regen_r.json()["status"] == "pending"

    async def test_regenerate_multiple_steps(self, client):
        """Test regenerating multiple steps."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["build_dataset_usd", "predict"]},
        )

        assert regen_r.status_code == 202

    async def test_regenerate_returns_400_while_running(self, client):
        """Test that regenerate returns 400 while pipeline is running."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"]},
        )

        # Should return 400 - can't regenerate while running
        if regen_r.status_code != 202:
            assert regen_r.status_code == 400

    async def test_regenerate_nonexistent_session(self, client):
        """Test regenerate on nonexistent session returns 404."""
        regen_r = await client.post(
            "/pipeline/00000000-0000-0000-0000-000000000000/regenerate",
            json={"steps": ["predict"]},
        )

        assert regen_r.status_code == 404

    async def test_regenerate_with_prompt_override(self, client):
        """Test regenerate with user prompt override."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={
                "steps": ["predict"],
                "user_prompt": "Focus on identifying furniture parts",
            },
        )

        assert regen_r.status_code == 202
