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
import io
import json
import uuid
import zipfile
from datetime import UTC, datetime
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

_USD_WITH_DEFAULT_ROOT_PRIM = b"""#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
}
"""


def _materials_zip_bytes() -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(
            "custom/materials.yaml",
            """materials:
  library_path: "materials_libs.usda"
  entries:
    - name: Resume_Metal
      description: Brushed metal used by resume tests
""",
        )
        zf.writestr("custom/materials_libs.usda", "#usda 1.0\n")
    buffer.seek(0)
    return buffer


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

    @pytest.mark.parametrize("data", [{}, {"user_email": "   "}])
    async def test_create_pipeline_defaults_missing_user_email(
        self,
        client: Any,
        monkeypatch: pytest.MonkeyPatch,
        data: dict[str, str],
    ) -> None:
        """Test pipeline creation defaults missing or blank telemetry email."""
        from ...service.routers import pipeline_router

        captured: dict[str, str] = {}
        monkeypatch.setattr(
            pipeline_router.config,
            "default_user_email",
            "anonymous@nvidia.com",
            raising=True,
        )

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager: Any,
            user_email: str = "",
        ) -> None:
            captured["user_email"] = user_email
            await session_manager.update_session(
                session_id,
                {"status": "completed", "results": {}, "can_cancel": False},
            )

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        files = {"usd_file": ("scene.usda", b"#usda 1.0\n", "application/octet-stream")}
        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        for _ in range(20):
            if captured:
                break
            await asyncio.sleep(0)
        assert captured and "user_email" in captured
        assert captured["user_email"] == "anonymous@nvidia.com"

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

    async def test_create_pipeline_with_render_num_workers(self, client, monkeypatch):
        """Test creating pipeline with custom render worker count."""
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
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "render_num_workers": "1",
            "user_email": "test@example.com",
            "enable_prim_clustering": "true",
            "cluster_min_prims": "7",
            "cluster_embedding_backend": "nim",
            "cluster_embedding_max_workers": "2",
            "cluster_embedding_batch_size": "3",
            "cluster_max_size": "11",
            "cluster_similarity_threshold_low": "0.97",
            "cluster_similarity_threshold_medium": "0.94",
            "cluster_similarity_threshold_high": "0.88",
        }

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        session_id = response.json()["session_id"]
        metadata = await pipeline_router.get_session_manager().get_session_metadata(
            session_id
        )
        assert metadata is not None
        session_config = metadata["config"]
        assert session_config["cluster_min_prims"] == 7
        assert session_config["cluster_embedding_backend"] == "nim"
        assert (
            session_config["cluster_embedding_model"]
            == "nvidia/llama-nemotron-embed-vl-1b-v2"
        )
        assert session_config["cluster_embedding_max_workers"] == 2
        assert session_config["cluster_embedding_batch_size"] == 3
        assert session_config["cluster_max_size"] == 11
        assert session_config["cluster_similarity_threshold_low"] == 0.97
        assert session_config["cluster_similarity_threshold_medium"] == 0.94
        assert session_config["cluster_similarity_threshold_high"] == 0.88
        assert "api_key" not in json.dumps(session_config)

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

    async def test_create_pipeline_large_scene_rejects_zip_bundle(self, client):
        """Large-scene public uploads are a single USD-family stage, not ZIP."""
        files = {
            "usd_file": (
                "scene_bundle.zip",
                b"not a zip that should be accepted here",
                "application/zip",
            )
        }

        response = await client.post(
            "/pipeline",
            files=files,
            data={"user_email": "test@example.com", "large_scene": "true"},
        )

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    async def test_existing_session_reuses_uploaded_refs_and_materials(
        self, client, monkeypatch
    ):
        """Resuming an existing session should reuse session-scoped inputs."""
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

        response = await client.post(
            "/pipeline",
            files=[
                (
                    "usd_file",
                    ("scene.usda", b"#usda 1.0\n", "application/octet-stream"),
                ),
                ("reference_images", ("reference.png", _PNG_BYTES, "image/png")),
                (
                    "materials_zip",
                    ("materials.zip", _materials_zip_bytes(), "application/zip"),
                ),
            ],
            data={"user_email": "test@example.com"},
        )

        assert response.status_code == 202
        session_id = response.json()["session_id"]
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs

        response = await client.post(
            "/pipeline",
            data={
                "session_id": session_id,
                "user_email": "test@example.com",
                "scene_resume": "true",
            },
        )

        assert response.status_code == 202
        for _ in range(20):
            if len(captured_pipeline_configs) >= 2:
                break
            await asyncio.sleep(0)
        assert len(captured_pipeline_configs) >= 2

        resumed_config = captured_pipeline_configs[-1]
        reference_images = resumed_config["input"]["reference_images"]
        assert len(reference_images) == 1
        assert reference_images[0].endswith("reference_0000.png")
        assert resumed_config["materials"]["entries"][0]["name"] == "Resume_Metal"
        assert resumed_config["materials"]["library_path"].endswith(
            "custom/materials_libs.usda"
        )

    async def test_create_pipeline_large_scene_routes_scene_executor(
        self, client, monkeypatch
    ):
        """Large-scene requests should route through the scene worker."""
        from ...service.routers import pipeline_router

        captured_scene_jobs: list[dict[str, Any]] = []

        async def capture_scene_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
            scene_options: dict[str, Any] | None = None,
        ) -> None:
            captured_scene_jobs.append(
                {
                    "config": config_dict,
                    "user_email": user_email,
                    "scene_options": scene_options or {},
                }
            )
            await session_manager.update_session(
                session_id,
                {
                    "status": "completed",
                    "pipeline_type": "large_scene",
                    "results": {"pipeline_type": "large_scene"},
                    "can_cancel": False,
                },
            )

        monkeypatch.setattr(
            pipeline_router,
            "execute_scene_pipeline_async",
            capture_scene_execute,
            raising=True,
        )

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    _USD_WITH_DEFAULT_ROOT_PRIM,
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "large_scene": "true",
                "scene_workers": "2",
                "scene_assets": "AssetA,/World/AssetB",
                "scene_skip_existing": "true",
                "scene_no_render": "true",
                "scene_simulate": "true",
                "scene_simulate_mock_analyze": "true",
                "scene_fail_on_validation_error": "true",
                "vlm_max_workers": "3",
                "scene_filters": '{"include_prim_paths": ["/World"]}',
                "scene_analyze_llm": '{"backend": "nim", "model": "llama"}',
            },
        )

        assert response.status_code == 202
        assert response.json()["message"] == "Large-scene pipeline queued for execution"
        for _ in range(20):
            if captured_scene_jobs:
                break
            await asyncio.sleep(0)

        assert captured_scene_jobs
        captured = captured_scene_jobs[-1]
        assert captured["user_email"] == "test@example.com"
        assert captured["scene_options"]["assets"] == ["AssetA", "/World/AssetB"]
        assert captured["scene_options"]["max_workers"] == 2
        assert captured["scene_options"]["skip_existing"] is True
        assert captured["scene_options"]["no_render"] is True
        assert captured["scene_options"]["simulate"] is True
        assert captured["scene_options"]["simulate_mock_analyze"] is True
        assert captured["scene_options"]["fail_on_validation_error"] is True
        assert captured["scene_options"]["predict_max_workers"] == 3
        assert captured["config"]["steps"]["predict"]["max_workers"] == 3
        assert captured["config"]["scene"]["filters"] == {
            "include_prim_paths": ["/World"]
        }
        scene_llm = captured["config"]["scene"]["analyze"]["llm"]
        assert scene_llm == pipeline_router._build_service_llm_config(
            pipeline_router._resolve_pipeline_model_routing()
        )
        assert scene_llm["model"] != "llama"

    async def test_create_pipeline_large_scene_requires_default_root_prim(
        self, client, monkeypatch
    ):
        """Large-scene requests must be one composed stage with defaultPrim."""
        from ...service.routers import pipeline_router

        captured_scene_jobs: list[dict[str, Any]] = []

        async def capture_scene_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
            scene_options: dict[str, Any] | None = None,
        ) -> None:
            captured_scene_jobs.append(
                {"config": config_dict, "scene_options": scene_options or {}}
            )

        monkeypatch.setattr(
            pipeline_router,
            "execute_scene_pipeline_async",
            capture_scene_execute,
            raising=True,
        )

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "large_scene": "true",
            },
        )

        assert response.status_code == 400
        assert "default root prim" in response.json()["detail"]
        assert "collection of USD files" in response.json()["detail"]
        assert captured_scene_jobs == []

    async def test_create_pipeline_large_scene_rejects_vlm_concurrency_over_cap(
        self, client, monkeypatch
    ):
        """scene_workers * vlm_max_workers should stay under the scene cap."""
        from ...service.routers import pipeline_router

        monkeypatch.setattr(pipeline_router.config, "max_scene_vlm_concurrency", 2)

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    _USD_WITH_DEFAULT_ROOT_PRIM,
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "large_scene": "true",
                "scene_workers": "2",
                "vlm_max_workers": "2",
            },
        )

        assert response.status_code == 400
        assert "large-scene VLM concurrency is too high" in response.json()["detail"]

    async def test_create_pipeline_with_prim_clustering(self, client, monkeypatch):
        """Prim clustering request fields should reach queued pipeline config."""
        from ...service.routers import pipeline_router

        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.setattr(pipeline_router.config, "nvidia_api_key", "nvapi-test")
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

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
                "cluster_min_prims": "7",
                "cluster_embedding_backend": "nim",
                "cluster_embedding_max_workers": "2",
                "cluster_embedding_batch_size": "3",
                "cluster_max_size": "11",
                "cluster_similarity_threshold_low": "0.97",
                "cluster_similarity_threshold_medium": "0.94",
                "cluster_similarity_threshold_high": "0.88",
            },
        )

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        steps = captured_pipeline_configs[-1]["steps"]
        assert list(steps).index("cluster_prims") < list(steps).index("predict")
        cluster_config = steps["cluster_prims"]
        assert cluster_config["embedding_service"] == "nim"
        assert (
            cluster_config["embedding_model"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
        )
        assert cluster_config["min_prims_to_activate"] == 7
        assert cluster_config["max_workers"] == 2
        assert cluster_config["batch_size"] == 3
        assert cluster_config["max_cluster_size"] == 11
        thresholds = cluster_config["complexity_thresholds"]
        assert thresholds["low"][2] == 0.97
        assert thresholds["medium"][2] == 0.94
        assert thresholds["high"][2] == 0.88

    async def test_create_pipeline_rejects_invalid_cluster_similarity_threshold(
        self, client
    ):
        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
                "cluster_similarity_threshold_low": "1.1",
            },
        )

        assert response.status_code == 422

    async def test_create_pipeline_rejects_invalid_cluster_max_size(self, client):
        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
                "cluster_max_size": "0",
            },
        )

        assert response.status_code == 422

    async def test_create_pipeline_rejects_hosted_cluster_without_nvidia_key(
        self, client, monkeypatch
    ):
        from ...service.routers import pipeline_router

        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("MA_CLUSTER_EMBEDDING_API_KEY", raising=False)
        monkeypatch.setattr(pipeline_router.config, "nvidia_api_key", None)
        monkeypatch.setattr(pipeline_router.config, "cluster_embedding_api_key", None)
        monkeypatch.setattr(
            pipeline_router.config,
            "cluster_embedding_base_url",
            "https://integrate.api.nvidia.com/v1",
        )

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
            },
        )

        assert response.status_code == 400
        assert "requires NVIDIA_API_KEY" in response.json()["detail"]

    async def test_create_pipeline_leaves_prim_clustering_off_by_default(
        self, client, monkeypatch
    ):
        """Prim clustering remains opt-in for public service requests."""
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

        response = await client.post(
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

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        assert "cluster_prims" not in captured_pipeline_configs[-1]["steps"]

    async def test_create_pipeline_injects_restore_usd_before_default_apply(
        self, client, monkeypatch
    ):
        """Default optimized service runs should restore predictions before apply."""
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

        response = await client.post(
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

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        step_names = list(captured_pipeline_configs[-1]["steps"])
        assert "optimize_usd" in step_names
        assert "restore_usd" in step_names
        assert step_names.index("restore_usd") < step_names.index("apply")

    async def test_create_pipeline_does_not_inject_restore_without_optimization(
        self, client, monkeypatch
    ):
        """Disabling optimization keeps restore_usd out of the service config."""
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

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={"user_email": "test@example.com", "optimize_usd": "false"},
        )

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        step_names = list(captured_pipeline_configs[-1]["steps"])
        assert "optimize_usd" not in step_names
        assert "restore_usd" not in step_names

    async def test_create_pipeline_with_custom_nim_cluster_endpoint(
        self, client, monkeypatch
    ):
        """Custom local NIM embedding URLs should use an endpoint-scoped key."""
        from ...service.routers import pipeline_router

        monkeypatch.delenv("MA_CLUSTER_EMBEDDING_API_KEY", raising=False)
        monkeypatch.setattr(
            pipeline_router.config,
            "cluster_embedding_base_url",
            "http://embedding-nim:8000/v1",
        )
        monkeypatch.setattr(pipeline_router.config, "cluster_embedding_api_key", None)
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

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
                "cluster_embedding_backend": "nim",
                "cluster_embedding_base_url": "http://embedding-nim:8000/v1",
            },
        )

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        cluster_config = captured_pipeline_configs[-1]["steps"]["cluster_prims"]
        assert cluster_config["embedding_service"] == "nim"
        assert (
            cluster_config["embedding_model"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
        )
        assert cluster_config["base_url"] == "http://embedding-nim:8000/v1"
        assert cluster_config["api_key"] == "not-used"

    async def test_create_pipeline_rejects_untrusted_cluster_endpoint_override(
        self, client
    ):
        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
                "cluster_embedding_backend": "nim",
                "cluster_embedding_base_url": "http://169.254.169.254/v1",
            },
        )

        assert response.status_code == 400
        assert "request overrides are restricted" in response.json()["detail"]

    async def test_create_pipeline_does_not_persist_cluster_api_key(
        self, client, monkeypatch
    ):
        """Endpoint-scoped cluster API keys should stay in env, not config."""
        from ...service.routers import pipeline_router

        monkeypatch.setenv("MA_CLUSTER_EMBEDDING_API_KEY", "cluster-secret")
        monkeypatch.setattr(
            pipeline_router.config,
            "cluster_embedding_base_url",
            "http://embedding-nim:8000/v1",
        )
        monkeypatch.setattr(
            pipeline_router.config,
            "cluster_embedding_api_key",
            "cluster-secret",
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

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
                "cluster_embedding_backend": "nim",
                "cluster_embedding_base_url": "http://embedding-nim:8000/v1",
            },
        )

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        cluster_config = captured_pipeline_configs[-1]["steps"]["cluster_prims"]
        assert "api_key" not in cluster_config
        assert "cluster-secret" not in json.dumps(captured_pipeline_configs[-1])

    async def test_create_pipeline_uses_configured_cluster_model(
        self, client, monkeypatch
    ):
        """Service MA_CLUSTER_* defaults should feed request config."""
        from ...service.routers import pipeline_router

        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.setattr(pipeline_router.config, "nvidia_api_key", "nvapi-test")
        monkeypatch.setattr(pipeline_router.config, "cluster_embedding_backend", "nim")
        monkeypatch.setattr(
            pipeline_router.config,
            "cluster_embedding_model",
            "nvidia/llama-nemotron-embed-vl-1b-v2",
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

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
            },
        )

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        cluster_config = captured_pipeline_configs[-1]["steps"]["cluster_prims"]
        assert cluster_config["embedding_service"] == "nim"
        assert (
            cluster_config["embedding_model"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
        )

    async def test_create_pipeline_rejects_clustering_without_prediction(
        self, client, monkeypatch
    ):
        """Clustering needs a prediction step for representative expansion."""
        from ...service.routers import pipeline_router

        captured_pipeline_configs: list[dict[str, Any]] = []

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
        ) -> None:
            captured_pipeline_configs.append(config_dict)

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
                "steps": "apply",
            },
        )

        assert response.status_code == 400
        assert "requires predict or benchmark" in response.json()["detail"]
        assert captured_pipeline_configs == []

    async def test_create_pipeline_rejects_render_num_workers_above_service_cap(
        self, client, monkeypatch
    ):
        """Oversized render worker overrides should fail validation."""
        from ...service.routers import pipeline_router

        captured_pipeline_configs: list[dict[str, Any]] = []

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
        ) -> None:
            captured_pipeline_configs.append(config_dict)

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        data = {
            "render_num_workers": str(
                pipeline_router.config.max_render_num_workers + 1
            ),
            "user_email": "test@example.com",
        }

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 422
        assert captured_pipeline_configs == []

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

    async def test_local_nim_env_routes_vlm_and_dedicated_llm_sidecars(
        self, client, monkeypatch
    ):
        """Service env overrides should reach the queued pipeline config."""
        from ...service.routers import pipeline_router

        monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
        monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
        monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
        monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
        monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
        monkeypatch.setattr(pipeline_router.config, "vlm_backend", "nvidia_inference")
        monkeypatch.setattr(
            pipeline_router.config,
            "vlm_model",
            "gcp/google/gemini-3.1-pro-preview",
        )
        monkeypatch.setattr(pipeline_router.config, "llm_backend", "openai")
        monkeypatch.setattr(
            pipeline_router.config,
            "llm_model",
            "meta/llama-3.1-70b-instruct",
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

        response = await client.post(
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

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        predict_config = captured_pipeline_configs[-1]["steps"]["predict"]
        assert predict_config["vlm"]["backend"] == "nim"
        assert predict_config["vlm"]["base_url"] == "http://vlm-nim:8000/v1"
        assert predict_config["vlm"]["model"] == "gcp/google/gemini-3.1-pro-preview"
        assert predict_config["llm"]["backend"] == "nim"
        assert predict_config["llm"]["base_url"] == "http://llm-nim:8000/v1"
        assert predict_config["llm"]["model"] == "meta/llama-3.1-70b-instruct"

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

    async def test_create_pipeline_custom_steps_do_not_inject_apply(
        self, client, monkeypatch
    ):
        """Custom step lists should not run apply unless explicitly requested."""
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

        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "steps": "build_dataset_usd,build_dataset_prepare_dataset,predict",
            },
        )

        assert response.status_code == 202
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        assert "apply" not in captured_pipeline_configs[-1]["steps"]
        assert "restore_usd" not in captured_pipeline_configs[-1]["steps"]

    async def test_create_pipeline_layer_only_requires_apply(self, client):
        """Layer-only output mode is meaningful only when apply is enabled."""
        response = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "steps": "build_dataset_usd,build_dataset_prepare_dataset,predict",
                "layer_only": "true",
            },
        )

        assert response.status_code == 400
        assert "layer_only=true requires the apply step" in response.json()["detail"]

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
        monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", None)

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
        monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", "not-used")

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
        assert generated_configs[-1]["image_gen"]["api_key"] == "not-used"

    async def test_generated_reference_does_not_persist_api_key_in_session(
        self, client, monkeypatch
    ):
        """Service-side image-gen api_key must not be written under session_dir.

        Anything under session_dir is walked by sync_session_to_store and
        uploaded with user artifacts, so a credential serialized into
        ``.gen_ref_config.yaml`` (or any other session file) would leak.
        """
        from material_agent import workflows

        from ...service.routers import pipeline_router

        secret = "sk-do-not-leak-this-secret-token"
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(pipeline_router.config, "image_gen_backend", "openai")
        monkeypatch.setattr(pipeline_router.config, "image_gen_model", "gpt-image-1")
        monkeypatch.setattr(
            pipeline_router.config,
            "image_gen_base_url",
            "https://api.openai.com/v1",
        )
        monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", secret)

        observed_temp_paths: list[Path] = []
        observed_api_keys: list[str | None] = []

        class FakeGenerateWorkflow:
            def run(self, context: dict[str, str]) -> dict[str, list[str]]:
                config_path = Path(context["config_path"])
                observed_temp_paths.append(config_path)
                gen_config = yaml.safe_load(config_path.read_text())
                observed_api_keys.append(gen_config["image_gen"].get("api_key"))
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
        (session_dir / "input" / "input_render.png").write_bytes(_PNG_BYTES)
        await manager.update_session(
            session_id,
            {"status": "ready", "preview_render_status": "ready"},
        )

        response = await client.post(
            f"/pipeline/{session_id}/generate-reference-image",
            data={"prompt": "matte blue plastic"},
        )

        assert response.status_code == 200
        # The workflow saw the credential (route still works), but it was
        # delivered via a tempfile outside the session tree...
        assert observed_api_keys == [secret]
        assert observed_temp_paths
        for path in observed_temp_paths:
            assert session_dir not in path.parents, (
                f"temp config {path} was written under session_dir {session_dir}"
            )
        # ...and was cleaned up after the call.
        for path in observed_temp_paths:
            assert not path.exists(), f"temp config {path} was not removed"
        # No file under session_dir contains the secret.
        for file_path in session_dir.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                contents = file_path.read_bytes()
            except OSError:
                continue
            assert secret.encode() not in contents, (
                f"image-gen api_key leaked into session file {file_path}"
            )

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

    async def test_large_scene_results_advertise_final_render_without_metadata(
        self, client
    ):
        """Large-scene results should expose final-render even before render metadata."""
        from ...service.routers import pipeline_router

        manager = pipeline_router.get_session_manager()
        session_id = str(uuid.uuid4())
        await manager.create_session(session_id, config={"large_scene": True})
        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "pipeline_type": "large_scene",
                "results": {"pipeline_type": "large_scene"},
                "duration_seconds": 0,
                "completed_at": datetime.now(UTC).isoformat(),
                "scene": {},
            },
        )

        results_r = await client.get(f"/pipeline/{session_id}/results")
        assert results_r.status_code == 200
        urls = results_r.json()["download_urls"]
        assert urls["output_usd"] == f"/artifacts/{session_id}/output"
        assert urls["final_render"] == f"/artifacts/{session_id}/final-render"
        assert "predictions" not in urls
        assert "report" not in urls

    async def test_results_omit_output_url_when_apply_not_run(self, client):
        """Results should not advertise missing output artifacts."""
        create_r = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={
                "user_email": "test@example.com",
                "steps": "build_dataset_usd,build_dataset_prepare_dataset,predict",
            },
        )
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        results_r = await client.get(f"/pipeline/{session_id}/results")
        assert results_r.status_code == 200
        urls = results_r.json()["download_urls"]
        assert "output_usd" not in urls
        assert "predictions" in urls
        assert "report" in urls


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

    async def test_download_cluster_artifacts(self, client, monkeypatch):
        """Test downloading cluster artifacts when prim clustering ran."""
        from ...service.routers import pipeline_router

        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.setattr(pipeline_router.config, "nvidia_api_key", "nvapi-test")
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline",
            files=files,
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
            },
        )
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        results_r = await client.get(f"/pipeline/{session_id}/results")
        urls = results_r.json()["download_urls"]
        assert "cluster_map" in urls
        assert "cluster_report" in urls
        assert "cluster_summary" in urls
        assert "cluster_representatives" in urls

        cluster_map = await client.get(f"/artifacts/{session_id}/cluster-map")
        assert cluster_map.status_code == 200
        assert cluster_map.headers["content-type"] == "application/x-ndjson"
        assert b'"cluster_id"' in cluster_map.content

        cluster_reps = await client.get(
            f"/artifacts/{session_id}/cluster-representatives"
        )
        assert cluster_reps.status_code == 200
        assert cluster_reps.headers["content-type"] == "application/x-ndjson"
        assert b'"/p0"' in cluster_reps.content

        cluster_summary = await client.get(f"/artifacts/{session_id}/cluster-summary")
        assert cluster_summary.status_code == 200
        assert "application/json" in cluster_summary.headers["content-type"]
        assert cluster_summary.json()["cluster_count"] == 5

        cluster_report = await client.get(f"/artifacts/{session_id}/cluster-report")
        assert cluster_report.status_code == 200
        assert "text/html" in cluster_report.headers["content-type"]

    async def test_results_omit_cluster_report_url_when_report_disabled(
        self, client, monkeypatch
    ):
        """Test results do not advertise a disabled cluster report artifact."""
        from ...service.routers import pipeline_router

        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.setattr(pipeline_router.config, "nvidia_api_key", "nvapi-test")
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline",
            files=files,
            data={
                "user_email": "test@example.com",
                "enable_prim_clustering": "true",
                "cluster_report": "false",
            },
        )
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        results_r = await client.get(f"/pipeline/{session_id}/results")
        urls = results_r.json()["download_urls"]
        assert "cluster_map" in urls
        assert "cluster_summary" in urls
        assert "cluster_representatives" in urls
        assert "cluster_report" not in urls

        cluster_report = await client.get(f"/artifacts/{session_id}/cluster-report")
        assert cluster_report.status_code == 404

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
