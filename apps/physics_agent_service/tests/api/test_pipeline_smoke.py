# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for pipeline API endpoints (happy path).

Tests the core workflows:
- Create pipeline (upload USD file)
- Get status (poll for progress)
- Get results (download artifacts)
- Download endpoints
"""

import asyncio

import pytest
import yaml

from ...service.routers import pipeline_router
from ..conftest import make_pipeline_files


@pytest.mark.api
class TestPipelineCreation:
    """Test pipeline creation endpoint."""

    async def test_create_pipeline_with_usd_file(self, client):
        """Test creating a pipeline with a USD file."""
        files = make_pipeline_files()

        response = await client.post("/pipeline", files=files)

        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body
        assert body["status"] == "pending"
        assert body["message"] == "Pipeline queued for execution"

    async def test_create_pipeline_generates_session_id(self, client):
        """Test that each pipeline creation generates a unique session ID."""
        r1 = await client.post("/pipeline", files=make_pipeline_files())
        r2 = await client.post("/pipeline", files=make_pipeline_files())

        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]

        assert sid1 != sid2
        assert len(sid1) > 0
        assert len(sid2) > 0

    async def test_create_pipeline_rejects_unsupported_usd_extension(self, client):
        """Test that unsupported USD file types are rejected."""
        files = [
            ("usd_file", ("model.obj", b"v 0 0 0\n", "application/octet-stream")),
        ]

        response = await client.post("/pipeline", files=files)

        assert response.status_code == 400
        assert "Invalid USD file type" in response.json()["detail"]

    async def test_create_pipeline_requires_usd_or_session_id(self, client):
        """Test that pipeline creation requires either usd_file or session_id."""
        response = await client.post("/pipeline")

        assert response.status_code == 400

    async def test_create_pipeline_accepts_optimizer_boolean_form_values(self, client):
        """FastAPI should parse common boolean form values for optimizer flags."""
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={
                "optimize_usd": "yes",
                "enable_deinstance": "on",
                "enable_split": "1",
                "enable_deduplicate": "0",
            },
        )

        assert response.status_code == 202
        session_id = response.json()["session_id"]
        manager = pipeline_router.get_session_manager()
        config_path = manager.get_session_dir(session_id) / "input" / "config.yaml"
        pipeline_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        optimize = pipeline_config["steps"]["optimize_usd"]
        assert optimize["enabled"] is True
        assert optimize["scene_optimizer_settings"]["enable_deinstance"] is True
        assert optimize["scene_optimizer_settings"]["enable_split_meshes"] is True
        assert optimize["scene_optimizer_settings"]["enable_deduplicate"] is False
        assert pipeline_config["steps"]["restore_usd"]["enabled"] is True

        session_r = await client.get(f"/sessions/{session_id}")
        metadata_config = session_r.json()["config"]
        assert metadata_config["optimize_usd"] is True
        assert metadata_config["enable_deinstance"] is True
        assert metadata_config["enable_split"] is True
        assert metadata_config["enable_deduplicate"] is False

    async def test_create_pipeline_rejects_optimizer_with_no_operations(self, client):
        """optimize_usd=true requires at least one optimizer operation."""
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={
                "optimize_usd": "true",
                "enable_deinstance": "false",
                "enable_split": "false",
                "enable_deduplicate": "false",
            },
        )

        assert response.status_code == 400
        assert "At least one optimization operation" in response.json()["detail"]


@pytest.mark.api
class TestPipelineStatus:
    """Test pipeline status endpoint."""

    async def test_get_status_for_valid_session(self, client):
        """Test getting status for a valid session."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        status_r = await client.get(f"/pipeline/{session_id}/status")

        assert status_r.status_code == 200
        body = status_r.json()
        assert body["session_id"] == session_id
        assert "status" in body
        assert "overall_progress" in body
        assert "current_step" in body

    async def test_get_status_for_nonexistent_session(self, client):
        """Test getting status for nonexistent session returns 404."""
        response = await client.get(
            "/pipeline/00000000-0000-0000-0000-000000000000/status"
        )

        assert response.status_code == 404

    async def test_status_progress_updates(self, client):
        """Test that status progress updates as pipeline executes."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        # Poll status until completion
        previous_percent = 0

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            assert status_r.status_code == 200

            body = status_r.json()
            current_percent = body["overall_progress"]["percent"]

            if current_percent > previous_percent:
                pass
            previous_percent = current_percent

            if body["status"] == "completed":
                break

            await asyncio.sleep(0.01)

        final_status = (await client.get(f"/pipeline/{session_id}/status")).json()
        assert final_status["overall_progress"]["percent"] == 100
        assert final_status["status"] == "completed"

    async def test_status_shows_completed_steps(self, client):
        """Test that completed steps are shown in status."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        final_status = (await client.get(f"/pipeline/{session_id}/status")).json()
        assert len(final_status["completed_steps"]) == 4
        assert final_status["completed_steps"][0]["name"] == "build_dataset_usd"
        assert (
            final_status["completed_steps"][1]["name"]
            == "build_dataset_prepare_dataset"
        )
        assert final_status["completed_steps"][2]["name"] == "predict"
        assert final_status["completed_steps"][3]["name"] == "apply_physics"


@pytest.mark.api
class TestPipelineResults:
    """Test results endpoint."""

    async def test_get_results_returns_202_while_running(self, client):
        """Test that /results returns 202 while pipeline is running."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        results_r = await client.get(f"/pipeline/{session_id}/results")

        assert results_r.status_code == 202

    async def test_get_results_after_completion(self, client):
        """Test that /results returns completed results."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        results_r = await client.get(f"/pipeline/{session_id}/results")

        assert results_r.status_code == 200
        body = results_r.json()
        assert body["session_id"] == session_id
        assert body["status"] == "completed"
        assert "stats" in body
        assert "download_urls" in body

    async def test_results_have_download_urls(self, client):
        """Test that results include download URLs."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        results_r = await client.get(f"/pipeline/{session_id}/results")
        body = results_r.json()
        urls = body["download_urls"]

        assert "predictions" in urls
        assert "report" in urls
        assert "dataset" in urls
        assert urls["predictions"].startswith("/")
        assert urls["report"].startswith("/")
        assert urls["dataset"].startswith("/")


@pytest.mark.api
class TestDownloadEndpoints:
    """Test artifact download endpoints."""

    async def test_download_predictions(self, client):
        """Test downloading the predictions JSONL file."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        download_r = await client.get(f"/artifacts/{session_id}/predictions")

        assert download_r.status_code == 200
        assert download_r.headers["content-type"] == "application/x-ndjson"
        assert len(download_r.content) > 0

    async def test_download_dataset(self, client):
        """Test downloading the dataset JSONL file."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        download_r = await client.get(f"/artifacts/{session_id}/dataset")

        assert download_r.status_code == 200
        assert len(download_r.content) > 0

    async def test_download_nonexistent_session_returns_404(self, client):
        """Test that downloading from nonexistent session returns 404."""
        response = await client.get(
            "/artifacts/00000000-0000-0000-0000-000000000000/predictions"
        )

        assert response.status_code == 404

    async def test_download_incomplete_returns_404(self, client):
        """Test that downloading from incomplete pipeline returns 404."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        download_r = await client.get(f"/artifacts/{session_id}/predictions")

        assert download_r.status_code == 404
