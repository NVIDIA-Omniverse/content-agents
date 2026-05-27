# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for graphics rendering backend configuration."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from world_understanding.functions.graphics import render_remote_async, rendering
from world_understanding.functions.graphics.rendering import (
    NVCFRenderingBackend,
    RemoteRenderingBackend,
)


class TestRemoteRenderingBackendConfig:
    """Tests for remote REST renderer backend configuration precedence."""

    def test_reads_s3_env_at_instantiation(self, monkeypatch):
        monkeypatch.setenv("WU_S3_BUCKET", "runtime-bucket")
        monkeypatch.setenv("WU_S3_REGION", "eu-west-1")
        monkeypatch.setenv("WU_S3_PROFILE", "runtime-profile")
        monkeypatch.delenv("MA_RENDERING_USE_DATA_URI", raising=False)

        backend = RemoteRenderingBackend()

        assert backend.s3_bucket == "runtime-bucket"
        assert backend.s3_region == "eu-west-1"
        assert backend.s3_profile == "runtime-profile"
        assert backend.use_data_uri is True

    def test_explicit_false_uses_s3_transfer(self, monkeypatch):
        monkeypatch.setenv("MA_RENDERING_USE_DATA_URI", "true")

        backend = RemoteRenderingBackend(use_data_uri=False)

        assert backend.use_data_uri is False

    def test_legacy_nvcf_backend_aliases_remote_backend(self):
        assert NVCFRenderingBackend is RemoteRenderingBackend

    def test_explicit_s3_kwargs_override_runtime_env(self, monkeypatch):
        monkeypatch.setenv("WU_S3_BUCKET", "runtime-bucket")
        monkeypatch.setenv("WU_S3_REGION", "eu-west-1")
        monkeypatch.setenv("WU_S3_PROFILE", "runtime-profile")

        backend = RemoteRenderingBackend(
            s3_bucket="explicit-bucket",
            s3_region="us-west-2",
            s3_profile="explicit-profile",
        )

        assert backend.s3_bucket == "explicit-bucket"
        assert backend.s3_region == "us-west-2"
        assert backend.s3_profile == "explicit-profile"

    def test_falls_back_to_module_constants_when_no_env_or_kwargs(self, monkeypatch):
        monkeypatch.delenv("WU_S3_BUCKET", raising=False)
        monkeypatch.delenv("WU_S3_REGION", raising=False)
        monkeypatch.delenv("WU_S3_PROFILE", raising=False)
        monkeypatch.setattr(rendering, "WU_S3_BUCKET", "module-bucket")
        monkeypatch.setattr(rendering, "WU_S3_REGION", "ap-south-1")
        monkeypatch.setattr(rendering, "WU_S3_PROFILE", "module-profile")

        backend = RemoteRenderingBackend()

        assert backend.s3_bucket == "module-bucket"
        assert backend.s3_region == "ap-south-1"
        assert backend.s3_profile == "module-profile"

    def test_sync_render_passes_base_dir_to_remote_renderer(
        self,
        monkeypatch,
        tmp_path,
    ):
        captured: dict[str, object] = {}

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            captured["base_dir"] = kwargs.get("base_dir")
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(api_key="test")
        result = backend.render(object(), cameras=["/Camera"], base_dir=tmp_path)

        assert captured["base_dir"] == tmp_path
        assert result["successful_cameras"] == 1

    def test_sync_render_uses_global_request_limit(self, monkeypatch):
        active_requests = 0
        max_active_requests = 0
        calls = 0
        counters_lock = threading.Lock()

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            nonlocal active_requests, max_active_requests, calls
            with counters_lock:
                calls += 1
                active_requests += 1
                max_active_requests = max(max_active_requests, active_requests)
            time.sleep(0.01)
            with counters_lock:
                active_requests -= 1
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "1")
        render_remote_async._reset_global_remote_render_semaphore_for_tests()
        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(api_key="test")
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(backend.render, object(), cameras=["/Camera"])
                    for _ in range(2)
                ]
                results = [future.result() for future in futures]
        finally:
            render_remote_async._reset_global_remote_render_semaphore_for_tests()

        with counters_lock:
            assert calls == 2
            assert max_active_requests == 1
        assert [result["successful_cameras"] for result in results] == [1, 1]

    def test_url_render_uses_global_request_limit(self, monkeypatch):
        active_requests = 0
        max_active_requests = 0
        calls = 0
        counters_lock = threading.Lock()

        def fake_render_all_cameras_from_url(**kwargs: Any) -> dict[str, Any]:
            nonlocal active_requests, max_active_requests, calls
            with counters_lock:
                calls += 1
                active_requests += 1
                max_active_requests = max(max_active_requests, active_requests)
            time.sleep(0.01)
            with counters_lock:
                active_requests -= 1
            return {
                "successful_cameras": 1,
                "results": [{"images": ["image"], "status": "success"}],
            }

        monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "1")
        render_remote_async._reset_global_remote_render_semaphore_for_tests()
        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras_from_url",
            fake_render_all_cameras_from_url,
        )

        backend = RemoteRenderingBackend(api_key="test")
        config = rendering.RenderingConfig(image_width=64)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        rendering.render_from_prepared_prims,
                        backend,
                        object(),
                        ["/Camera"],
                        1,
                        ["/World/Prim"],
                        config,
                        stage_url="https://example.com/scene.usd",
                    )
                    for _ in range(2)
                ]
                results = [future.result() for future in futures]
        finally:
            render_remote_async._reset_global_remote_render_semaphore_for_tests()

        with counters_lock:
            assert calls == 2
            assert max_active_requests == 1
        assert [result["successful_cameras"] for result in results] == [1, 1]
