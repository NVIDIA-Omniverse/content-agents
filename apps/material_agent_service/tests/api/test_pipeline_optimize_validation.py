# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for optimize USD operations validation.

Tests the validation logic for Scene Optimizer operations:
- At least one operation must be enabled when optimize_usd is true
- Valid combinations of operations are accepted
"""

import pytest


@pytest.mark.api
class TestOptimizeOperationsValidation:
    """Test validation of optimize USD operations."""

    async def test_optimize_enabled_with_all_operations_succeeds(self, client):
        """Test that optimize_usd with all operations enabled is accepted."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "camera_views": "+x+y+z",
            "optimize_usd": "true",
            "enable_deinstance": "true",
            "enable_split": "true",
            "enable_deduplicate": "true",
            "user_email": "test@example.com",
        }

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body
        assert body["status"] == "pending"

    async def test_optimize_enabled_with_single_operation_succeeds(self, client):
        """Test that optimize_usd with only one operation enabled is accepted."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}

        # Test each operation individually
        test_cases = [
            {
                "enable_deinstance": "true",
                "enable_split": "false",
                "enable_deduplicate": "false",
            },
            {
                "enable_deinstance": "false",
                "enable_split": "true",
                "enable_deduplicate": "false",
            },
            {
                "enable_deinstance": "false",
                "enable_split": "false",
                "enable_deduplicate": "true",
            },
        ]

        for operation_flags in test_cases:
            data = {
                "camera_views": "+x+y+z",
                "optimize_usd": "true",
                "user_email": "test@example.com",
                **operation_flags,
            }

            response = await client.post("/pipeline", files=files, data=data)

            assert response.status_code == 202, f"Failed with flags: {operation_flags}"
            body = response.json()
            assert "session_id" in body

    async def test_optimize_enabled_with_no_operations_fails(self, client):
        """Test that optimize_usd with all operations disabled returns HTTP 400."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "camera_views": "+x+y+z",
            "optimize_usd": "true",
            "enable_deinstance": "false",
            "enable_split": "false",
            "enable_deduplicate": "false",
            "user_email": "test@example.com",
        }

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 400
        body = response.json()
        assert "at least one" in body["detail"].lower()
        assert "operation" in body["detail"].lower()

    async def test_optimize_disabled_ignores_operations(self, client):
        """Test that when optimize_usd is false, operation flags are ignored."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "camera_views": "+x+y+z",
            "optimize_usd": "false",
            # Operations flags shouldn't matter when optimize is disabled
            "enable_deinstance": "false",
            "enable_split": "false",
            "enable_deduplicate": "false",
            "user_email": "test@example.com",
        }

        response = await client.post("/pipeline", files=files, data=data)

        # Should succeed because optimize_usd is false
        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body

    async def test_optimize_enabled_defaults_to_all_operations(self, client):
        """Test that optimize_usd request succeeds when operation flags are not specified."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "camera_views": "+x+y+z",
            "optimize_usd": "true",
            # Don't specify operation flags - should default to all true
            "user_email": "test@example.com",
        }

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body

    async def test_optimize_with_two_operations_succeeds(self, client):
        """Test that optimize_usd with two operations enabled is accepted."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "camera_views": "+x+y+z",
            "optimize_usd": "true",
            "enable_deinstance": "true",
            "enable_split": "true",
            "enable_deduplicate": "false",
            "user_email": "test@example.com",
        }

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body
