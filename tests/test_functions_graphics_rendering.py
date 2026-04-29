# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for graphics rendering backend configuration."""

from world_understanding.functions.graphics import rendering
from world_understanding.functions.graphics.rendering import NVCFRenderingBackend


class TestNVCFRenderingBackendConfig:
    """Tests for NVCF backend configuration precedence."""

    def test_reads_s3_env_at_instantiation(self, monkeypatch):
        monkeypatch.setenv("WU_S3_BUCKET", "runtime-bucket")
        monkeypatch.setenv("WU_S3_REGION", "eu-west-1")
        monkeypatch.setenv("WU_S3_PROFILE", "runtime-profile")

        backend = NVCFRenderingBackend()

        assert backend.s3_bucket == "runtime-bucket"
        assert backend.s3_region == "eu-west-1"
        assert backend.s3_profile == "runtime-profile"

    def test_explicit_s3_kwargs_override_runtime_env(self, monkeypatch):
        monkeypatch.setenv("WU_S3_BUCKET", "runtime-bucket")
        monkeypatch.setenv("WU_S3_REGION", "eu-west-1")
        monkeypatch.setenv("WU_S3_PROFILE", "runtime-profile")

        backend = NVCFRenderingBackend(
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

        backend = NVCFRenderingBackend()

        assert backend.s3_bucket == "module-bucket"
        assert backend.s3_region == "ap-south-1"
        assert backend.s3_profile == "module-profile"
