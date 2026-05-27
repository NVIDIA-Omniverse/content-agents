# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for should_use_data_uri utility (world_understanding.utils.data_uri)."""

import pytest

from world_understanding.utils.data_uri import should_use_data_uri


class TestShouldUseDataUri:
    """Decision hierarchy: explicit param > env var override > data URI default."""

    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MA_RENDERING_USE_DATA_URI", raising=False)
        monkeypatch.delenv("WU_S3_BUCKET", raising=False)

    def test_explicit_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("WU_S3_BUCKET", "would-use-s3-otherwise")
        assert should_use_data_uri(True) is True

    def test_explicit_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        assert should_use_data_uri(False) is False

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("MA_RENDERING_USE_DATA_URI", "true")
        assert should_use_data_uri(False) is False

    def test_env_var_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("WU_S3_BUCKET", "would-use-s3-otherwise")
        monkeypatch.setenv("MA_RENDERING_USE_DATA_URI", "true")
        assert should_use_data_uri() is True

    def test_env_var_TRUE_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("WU_S3_BUCKET", "would-use-s3-otherwise")
        monkeypatch.setenv("MA_RENDERING_USE_DATA_URI", "TRUE")
        assert should_use_data_uri() is True

    def test_env_var_trims_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("MA_RENDERING_USE_DATA_URI", " false ")
        assert should_use_data_uri() is False

    def test_env_var_false_forces_s3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("MA_RENDERING_USE_DATA_URI", "false")
        assert should_use_data_uri() is False

    def test_default_no_bucket_uses_data_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: ambient AWS credentials without a bucket route to data URI."""
        self._clean_env(monkeypatch)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKID")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET")
        assert should_use_data_uri() is True

    def test_default_with_bucket_still_uses_data_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("WU_S3_BUCKET", "my-bucket")
        assert should_use_data_uri() is True

    def test_default_empty_bucket_uses_data_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("WU_S3_BUCKET", "")
        assert should_use_data_uri() is True
