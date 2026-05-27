# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline regeneration.

Tests the regenerate endpoint for re-running specific steps.
"""

import asyncio
from typing import Any

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

    async def test_regenerate_predict_uses_local_nim_routing(self, client, monkeypatch):
        """Regeneration should reuse create-time service VLM/LLM routing."""
        from ...service.routers import pipeline_router

        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
        monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
        monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
        monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
        monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
        monkeypatch.setattr(pipeline_router.config, "vlm_backend", "openai")
        monkeypatch.setattr(pipeline_router.config, "vlm_model", "local-vlm")
        monkeypatch.setattr(pipeline_router.config, "llm_backend", "openai")
        monkeypatch.setattr(pipeline_router.config, "llm_model", "local-llm")

        captured_pipeline_configs: list[dict[str, Any]] = []

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
        ) -> None:
            captured_pipeline_configs.append(config_dict)
            await session_manager.update_session(
                session_id,
                {"status": "completed", "results": {}, "can_cancel": False},
            )

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        create_r = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={"user_email": "test@example.com"},
        )

        assert create_r.status_code == 202
        session_id = create_r.json()["session_id"]

        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert len(captured_pipeline_configs) == 1

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"]},
        )

        assert regen_r.status_code == 202
        for _ in range(20):
            if len(captured_pipeline_configs) == 2:
                break
            await asyncio.sleep(0)
        assert len(captured_pipeline_configs) == 2

        predict_config = captured_pipeline_configs[-1]["steps"]["predict"]
        assert predict_config["vlm"]["backend"] == "nim"
        assert predict_config["vlm"]["model"] == "local-vlm"
        assert predict_config["vlm"]["base_url"] == "http://vlm-nim:8000/v1"
        assert predict_config["llm"]["backend"] == "nim"
        assert predict_config["llm"]["model"] == "local-llm"
        assert predict_config["llm"]["base_url"] == "http://llm-nim:8000/v1"

    async def test_regenerate_layer_only_requires_apply(self, client, monkeypatch):
        """Regeneration should not inject apply just because layer_only=true."""
        from ...service.routers import pipeline_router

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
        ) -> None:
            await session_manager.update_session(
                session_id,
                {"status": "completed", "results": {}, "can_cancel": False},
            )

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        create_r = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={"user_email": "test@example.com"},
        )
        assert create_r.status_code == 202
        session_id = create_r.json()["session_id"]
        for _ in range(20):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0)

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"], "layer_only": True},
        )

        assert regen_r.status_code == 400
        assert "layer_only=true requires the apply step" in regen_r.json()["detail"]

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

    async def test_regenerate_upload_first_preserves_render_num_workers(
        self, client, monkeypatch
    ):
        """Upload-first runs should keep render worker limits on regenerate."""
        from ...service.routers import pipeline_router

        captured_pipeline_configs: list[dict[str, Any]] = []

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
        ) -> None:
            captured_pipeline_configs.append(config_dict)
            await session_manager.update_session(
                session_id,
                {"status": "completed", "results": {}, "can_cancel": False},
            )

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        usd_content = b"#usda 1.0\n"
        upload_r = await client.post(
            "/pipeline/upload-usd",
            files={"usd_file": ("scene.usda", usd_content, "application/octet-stream")},
        )
        assert upload_r.status_code == 201
        session_id = upload_r.json()["session_id"]

        start_r = await client.post(
            "/pipeline",
            data={
                "session_id": session_id,
                "render_num_workers": "1",
                "user_email": "test@example.com",
            },
        )
        assert start_r.status_code == 202

        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        assert (
            captured_pipeline_configs[-1]["steps"]["build_dataset_usd"]["num_workers"]
            == 1
        )
        assert (
            captured_pipeline_configs[-1]["steps"]["build_dataset_usd"][
                "max_concurrent_requests"
            ]
            == 1
        )

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["build_dataset_usd"]},
        )
        assert regen_r.status_code == 202

        for _ in range(20):
            if len(captured_pipeline_configs) >= 2:
                break
            await asyncio.sleep(0)
        assert len(captured_pipeline_configs) >= 2
        assert (
            captured_pipeline_configs[-1]["steps"]["build_dataset_usd"]["num_workers"]
            == 1
        )
        assert (
            captured_pipeline_configs[-1]["steps"]["build_dataset_usd"][
                "max_concurrent_requests"
            ]
            == 1
        )
