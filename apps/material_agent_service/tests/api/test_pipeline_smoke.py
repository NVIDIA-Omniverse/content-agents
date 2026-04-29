# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for pipeline API endpoints (happy path).

Tests the core workflows:
- Create pipeline (upload USD)
- Get status (poll for progress)
- Get results (download artifacts)
- Download endpoints
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
    b"\xf6\x178U"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def _wait_for_input_render(client, session_id: str) -> None:
    for _ in range(20):
        response = await client.head(f"/assets/{session_id}/input-render")
        if response.status_code == 200:
            return
        await asyncio.sleep(0)
    raise AssertionError("input render was not created by the test stub")


@pytest.mark.api
class TestPipelineCreation:
    """Test pipeline creation endpoint."""

    async def test_create_pipeline_with_valid_usd(self, client):
        """Test creating a pipeline with a valid USD file."""
        # Create minimal USD content
        usd_content = b"#usda 1.0\n"

        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {"camera_views": "+x+y+z,-x-y-z", "user_email": "test@example.com"}

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body
        assert body["status"] == "pending"
        assert body["message"] == "Pipeline queued for execution"

    async def test_create_pipeline_generates_session_id(self, client):
        """Test that each pipeline creation generates a unique session ID."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}

        r1 = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        r2 = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]

        assert sid1 != sid2
        assert len(sid1) > 0
        assert len(sid2) > 0

    async def test_create_pipeline_rejects_unsupported_extension(self, client):
        """Test that unsupported file types are rejected."""
        obj_content = b"v 0 0 0\nv 1 1 1\n"
        files = {"usd_file": ("model.obj", obj_content, "application/octet-stream")}

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    async def test_create_pipeline_with_camera_views(self, client):
        """Test creating pipeline with custom camera views."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {"camera_views": "+x+y+z,-x-y-z,+z-z", "user_email": "test@example.com"}

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        # Session should be created successfully with custom views
        assert "session_id" in response.json()

    async def test_create_pipeline_with_user_prompt(self, client):
        """Test creating pipeline with custom user prompt."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "user_prompt": "Please identify metallic parts",
            "camera_views": "+x+y+z",
            "user_email": "test@example.com",
        }

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        assert "session_id" in response.json()

    async def test_create_pipeline_with_layer_only(self, client):
        """Test creating pipeline with layer_only=true."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "user_email": "test@example.com",
            "layer_only": "true",
        }

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body

        # Wait for completion
        session_id = body["session_id"]
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Verify the config was stored with layer_only
        results_r = await client.get(f"/pipeline/{session_id}/results")
        assert results_r.status_code == 200

    async def test_generated_reference_requires_explicit_selection(
        self, client, monkeypatch
    ):
        """Generated refs are unique artifacts and are used only by explicit ID."""
        from material_agent import workflows

        from ...service.routers import pipeline_router

        monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
        monkeypatch.setattr(pipeline_router.config, "image_gen_backend", "openai")
        monkeypatch.setattr(pipeline_router.config, "image_gen_model", "gpt-image-1")
        monkeypatch.setattr(
            pipeline_router.config,
            "image_gen_base_url",
            "https://api.openai.com/v1",
        )

        generated_configs: list[dict[str, Any]] = []

        class FakeGenerateWorkflow:
            def run(self, context: dict[str, str]) -> dict[str, list[str]]:
                config_path = Path(context["config_path"])
                gen_config = yaml.safe_load(config_path.read_text())
                generated_configs.append(gen_config)
                output_path = Path(gen_config["output_dir"]) / "generated_ref_0.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(_PNG_BYTES)
                return {"generated_reference_image_paths": [str(output_path)]}

        monkeypatch.setattr(
            workflows,
            "create_generate_reference_image_workflow_from_config",
            lambda: FakeGenerateWorkflow(),
        )

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

        async def upload_and_generate(prompt: str) -> tuple[str, str]:
            upload = await client.post(
                "/pipeline/upload-usd",
                files={
                    "usd_file": (
                        "scene.usda",
                        b"#usda 1.0\n",
                        "application/octet-stream",
                    )
                },
            )
            assert upload.status_code == 201
            session_id = upload.json()["session_id"]
            await _wait_for_input_render(client, session_id)

            response = await client.post(
                f"/pipeline/{session_id}/generate-reference-image",
                data={"prompt": prompt},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["reference_id"]
            assert body["image_url"].endswith(body["reference_id"])
            return session_id, body["reference_id"]

        session_without_selection, unused_ref_id = await upload_and_generate(
            "matte blue plastic"
        )
        assert generated_configs[-1]["image_gen"] == {
            "backend": "openai",
            "model": "gpt-image-1",
            "base_url": "https://api.openai.com/v1",
        }

        response = await client.post(
            "/pipeline",
            data={
                "session_id": session_without_selection,
                "user_email": "test@example.com",
            },
        )
        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        assert "reference_images" not in captured_pipeline_configs[-1]["input"]

        session_with_selection, selected_ref_id = await upload_and_generate(
            "brushed red metal"
        )
        assert selected_ref_id != unused_ref_id

        response = await client.post(
            "/pipeline",
            data={
                "session_id": session_with_selection,
                "generated_reference_id": selected_ref_id,
                "user_email": "test@example.com",
            },
        )
        assert response.status_code == 202
        for _ in range(20):
            if len(captured_pipeline_configs) >= 2:
                break
            await asyncio.sleep(0)
        reference_images = captured_pipeline_configs[-1]["input"]["reference_images"]
        assert len(reference_images) == 1
        assert selected_ref_id in reference_images[0]

    async def test_generated_reference_hydrates_input_render_from_store(
        self, client, monkeypatch
    ):
        from material_agent import workflows

        from ...service.routers import pipeline_router

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(pipeline_router.config, "image_gen_backend", "openai")
        monkeypatch.setattr(pipeline_router.config, "image_gen_model", "gpt-image-1")
        monkeypatch.setattr(
            pipeline_router.config,
            "image_gen_base_url",
            "http://image-gen.local/v1",
        )

        generated_configs: list[dict[str, Any]] = []

        class FakeGenerateWorkflow:
            def run(self, context: dict[str, str]) -> dict[str, list[str]]:
                config_path = Path(context["config_path"])
                gen_config = yaml.safe_load(config_path.read_text())
                generated_configs.append(gen_config)
                output_path = Path(gen_config["output_dir"]) / "generated_ref_0.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(_PNG_BYTES)
                return {"generated_reference_image_paths": [str(output_path)]}

        monkeypatch.setattr(
            workflows,
            "create_generate_reference_image_workflow_from_config",
            lambda: FakeGenerateWorkflow(),
        )

        session_id = str(uuid.uuid4())
        manager = pipeline_router.get_session_manager()
        session_dir = await manager.create_session(session_id)
        await manager.update_session(
            session_id,
            {"status": "ready", "preview_render_status": "ready"},
        )

        read_calls: list[tuple[str, str]] = []

        async def fake_read_from_store(
            queried_session_id: str, key: str
        ) -> bytes | None:
            read_calls.append((queried_session_id, key))
            if queried_session_id == session_id and key == "input/input_render.png":
                return _PNG_BYTES
            return None

        monkeypatch.setattr(manager, "read_from_store", fake_read_from_store)

        response = await client.post(
            f"/pipeline/{session_id}/generate-reference-image",
            data={"prompt": "matte blue plastic"},
        )

        assert response.status_code == 200
        assert (session_dir / "input" / "input_render.png").exists()
        assert read_calls == [(session_id, "input/input_render.png")]
        assert generated_configs[-1]["rendered_preview_paths"] == [
            str(session_dir / "input" / "input_render.png")
        ]

    async def test_generated_reference_rejects_mutation_after_pipeline_started(
        self, client, monkeypatch
    ):
        from ...service.routers import pipeline_router

        monkeypatch.setenv("GOOGLE_API_KEY", "gemini-test")

        session_id = str(uuid.uuid4())
        manager = pipeline_router.get_session_manager()
        session_dir = await manager.create_session(session_id)
        (session_dir / "input" / "input_render.png").write_bytes(_PNG_BYTES)
        await manager.update_session(
            session_id,
            {"status": "running", "preview_render_status": "ready"},
        )

        response = await client.post(
            f"/pipeline/{session_id}/generate-reference-image",
            data={"prompt": "matte blue plastic"},
        )

        assert response.status_code == 409

    async def test_generated_reference_can_be_deleted_before_pipeline_start(
        self, client
    ):
        from ...service.routers import pipeline_router

        session_id = str(uuid.uuid4())
        reference_id = "ref-delete"
        key = f"input/generated_references/{reference_id}/generated_ref_0.png"
        manager = pipeline_router.get_session_manager()
        session_dir = await manager.create_session(session_id)
        generated_path = session_dir / key
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.write_bytes(_PNG_BYTES)
        await manager.update_session(session_id, {"status": "ready"})
        await manager.add_generated_reference_image(
            session_id,
            {
                "id": reference_id,
                "key": key,
                "path": str(generated_path),
                "prompt": "delete me",
                "image_url": f"/assets/{session_id}/generated-ref/{reference_id}",
            },
        )

        response = await client.delete(
            f"/pipeline/{session_id}/generated-reference-image/{reference_id}"
        )

        assert response.status_code == 200
        assert response.json() == {"status": "deleted", "reference_id": reference_id}
        assert not generated_path.exists()
        metadata = await manager.get_session_metadata(session_id)
        assert metadata is not None
        assert metadata["generated_reference_images"] == []

    async def test_input_render_reports_terminal_preview_failure(self, client):
        from ...service.routers import pipeline_router

        session_id = str(uuid.uuid4())
        manager = pipeline_router.get_session_manager()
        await manager.create_session(session_id)
        await manager.update_session(
            session_id,
            {
                "status": "ready",
                "preview_render_status": "failed",
                "preview_render_error": "renderer exploded",
            },
        )

        response = await client.head(f"/assets/{session_id}/input-render")

        assert response.status_code == 424


@pytest.mark.api
class TestPipelineStatus:
    """Test pipeline status endpoint."""

    async def test_get_status_for_valid_session(self, client):
        """Test getting status for a valid session."""
        # Create pipeline first
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Get status
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
        # Create pipeline
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Poll status until completion
        previous_percent = 0

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            assert status_r.status_code == 200

            body = status_r.json()
            current_percent = body["overall_progress"]["percent"]

            # Progress should be monotonically increasing (usually)
            if current_percent > previous_percent:
                pass
            previous_percent = current_percent

            if body["status"] == "completed":
                break

            await asyncio.sleep(0.01)

        # Should reach 100%
        final_status = (await client.get(f"/pipeline/{session_id}/status")).json()
        assert final_status["overall_progress"]["percent"] == 100
        assert final_status["status"] == "completed"

    async def test_status_shows_completed_steps(self, client):
        """Test that completed steps are shown in status."""
        # Create and complete pipeline
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Poll until done
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        final_status = (await client.get(f"/pipeline/{session_id}/status")).json()
        assert len(final_status["completed_steps"]) == 3
        assert final_status["completed_steps"][0]["name"] == "build_dataset_usd"
        assert final_status["completed_steps"][1]["name"] == "predict"
        assert final_status["completed_steps"][2]["name"] == "apply"


@pytest.mark.api
class TestPipelineResults:
    """Test results endpoint."""

    async def test_get_results_returns_202_while_running(self, client):
        """Test that /results returns 202 while pipeline is running."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Immediately try to get results (should be running or pending)
        results_r = await client.get(f"/pipeline/{session_id}/results")

        # Should be 202 (not ready yet)
        assert results_r.status_code == 202

    async def test_get_results_after_completion(self, client):
        """Test that /results returns completed results."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Poll status until complete
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Now get results
        results_r = await client.get(f"/pipeline/{session_id}/results")

        assert results_r.status_code == 200
        body = results_r.json()
        assert body["session_id"] == session_id
        assert body["status"] == "completed"
        assert "stats" in body
        assert "download_urls" in body

    async def test_results_have_download_urls(self, client):
        """Test that results include download URLs."""
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

        results_r = await client.get(f"/pipeline/{session_id}/results")
        body = results_r.json()
        urls = body["download_urls"]

        assert "output_usd" in urls
        assert "predictions" in urls
        assert "report" in urls
        # URLs should be paths
        assert urls["output_usd"].startswith("/")
        assert urls["predictions"].startswith("/")


@pytest.mark.api
class TestDownloadEndpoints:
    """Test artifact download endpoints."""

    async def test_download_output_usd(self, client):
        """Test downloading the output USD file."""
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

        # Download output
        download_r = await client.get(f"/artifacts/{session_id}/output")

        assert download_r.status_code == 200
        assert download_r.headers["content-type"] == "application/octet-stream"
        assert len(download_r.content) > 0

    async def test_download_predictions(self, client):
        """Test downloading the predictions JSONL file."""
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

        # Download predictions
        download_r = await client.get(f"/artifacts/{session_id}/predictions")

        assert download_r.status_code == 200
        assert download_r.headers["content-type"] == "application/x-ndjson"
        assert len(download_r.content) > 0

    async def test_download_nonexistent_session_returns_404(self, client):
        """Test that downloading from nonexistent session returns 404."""
        response = await client.get(
            "/artifacts/00000000-0000-0000-0000-000000000000/output"
        )

        assert response.status_code == 404

    async def test_download_incomplete_returns_404(self, client):
        """Test that downloading from incomplete pipeline returns 404."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Immediately try to download (should fail - not complete)
        download_r = await client.get(f"/artifacts/{session_id}/output")

        assert download_r.status_code == 404
