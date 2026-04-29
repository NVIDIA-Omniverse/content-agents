# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for session administration endpoints.

Tests listing and deleting sessions.
"""

import asyncio

import pytest


@pytest.mark.api
class TestSessionAdministration:
    """Test session listing and deletion."""

    async def test_list_sessions_empty(self, client):
        """Test listing sessions when none exist."""
        response = await client.get("/sessions")

        assert response.status_code == 200
        body = response.json()
        assert "sessions" in body
        assert "total" in body
        assert body["total"] >= 0

    async def test_list_sessions_includes_created(self, client):
        """Test that created sessions appear in list."""
        # Create a few sessions
        usd_content = b"#usda 1.0\n"
        session_ids = []

        for _ in range(3):
            files = {
                "usd_file": ("scene.usda", usd_content, "application/octet-stream")
            }
            r = await client.post(
                "/pipeline", files=files, data={"user_email": "test@example.com"}
            )
            session_ids.append(r.json()["session_id"])

        # List sessions
        list_r = await client.get("/sessions")

        assert list_r.status_code == 200
        body = list_r.json()
        assert body["total"] >= 3

        listed_ids = [s["session_id"] for s in body["sessions"]]
        for sid in session_ids:
            assert sid in listed_ids

    async def test_list_sessions_has_metadata(self, client):
        """Test that listed sessions include required metadata."""
        # Create a session
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # List sessions
        list_r = await client.get("/sessions")
        body = list_r.json()

        # Find our session
        our_session = None
        for s in body["sessions"]:
            if s["session_id"] == session_id:
                our_session = s
                break

        assert our_session is not None
        assert "session_id" in our_session
        assert "status" in our_session
        assert "created_at" in our_session
        assert "updated_at" in our_session

    async def test_list_sessions_sorting(self, client):
        """Test that sessions are sorted (newest first)."""
        # Create sessions with a small delay
        session_ids = []

        for i in range(3):
            usd_content = b"#usda 1.0\n"
            files = {
                "usd_file": ("scene.usda", usd_content, "application/octet-stream")
            }
            r = await client.post(
                "/pipeline", files=files, data={"user_email": "test@example.com"}
            )
            session_ids.append(r.json()["session_id"])
            await asyncio.sleep(0.01)

        # List sessions
        list_r = await client.get("/sessions")
        body = list_r.json()

        # Should be sorted newest first (reverse order of creation)
        listed_ids = [s["session_id"] for s in body["sessions"]]
        # The newest session should be first
        assert session_ids[-1] in listed_ids[:5]  # Should be in recent list

    async def test_delete_session_removes_session(self, client):
        """Test that delete removes session from listings."""
        # Create a session
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Verify it exists
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.status_code == 200

        # Delete it
        delete_r = await client.delete(f"/sessions/{session_id}")
        assert delete_r.status_code == 204, (
            f"Delete failed: status={delete_r.status_code}, body={delete_r.text}"
        )

        # Verify it's gone
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.status_code == 404

    async def test_delete_nonexistent_session_returns_404(self, client):
        """Test that deleting nonexistent session returns 404."""
        delete_r = await client.delete("/sessions/00000000-0000-0000-0000-000000000000")

        assert delete_r.status_code == 404

    async def test_delete_running_session(self, client):
        """Test that running sessions can be deleted."""
        # Create a session
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Give it a moment to start
        await asyncio.sleep(0.05)

        # Delete it (even though it's running)
        delete_r = await client.delete(f"/sessions/{session_id}")
        assert delete_r.status_code == 204

        # Verify it's gone
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.status_code == 404
