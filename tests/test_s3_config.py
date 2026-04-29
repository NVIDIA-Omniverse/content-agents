# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for centralized S3 configuration (world_understanding.config.s3)."""

import importlib

import pytest


class TestS3ConfigDefaults:
    """Verify default values when no env vars are set."""

    def test_default_bucket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WU_S3_BUCKET", raising=False)
        import world_understanding.config.s3 as s3_mod

        importlib.reload(s3_mod)
        assert s3_mod.WU_S3_BUCKET == ""

    def test_default_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WU_S3_REGION", raising=False)
        import world_understanding.config.s3 as s3_mod

        importlib.reload(s3_mod)
        assert s3_mod.WU_S3_REGION == "us-east-2"

    def test_default_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WU_S3_PROFILE", raising=False)
        import world_understanding.config.s3 as s3_mod

        importlib.reload(s3_mod)
        assert s3_mod.WU_S3_PROFILE == ""


class TestS3ConfigEnvOverrides:
    """Verify env var overrides work."""

    def test_bucket_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WU_S3_BUCKET", "custom-bucket")
        import world_understanding.config.s3 as s3_mod

        importlib.reload(s3_mod)
        assert s3_mod.WU_S3_BUCKET == "custom-bucket"

    def test_region_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WU_S3_REGION", "eu-west-1")
        import world_understanding.config.s3 as s3_mod

        importlib.reload(s3_mod)
        assert s3_mod.WU_S3_REGION == "eu-west-1"

    def test_profile_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WU_S3_PROFILE", "my-profile")
        import world_understanding.config.s3 as s3_mod

        importlib.reload(s3_mod)
        assert s3_mod.WU_S3_PROFILE == "my-profile"

    def test_empty_string_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty string env vars should yield the module defaults (only
        WU_S3_REGION has a non-empty default)."""
        monkeypatch.setenv("WU_S3_BUCKET", "")
        monkeypatch.setenv("WU_S3_REGION", "")
        monkeypatch.setenv("WU_S3_PROFILE", "")
        import world_understanding.config.s3 as s3_mod

        importlib.reload(s3_mod)
        assert s3_mod.WU_S3_BUCKET == ""
        assert s3_mod.WU_S3_REGION == "us-east-2"
        assert s3_mod.WU_S3_PROFILE == ""
