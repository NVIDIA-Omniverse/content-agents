# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Physics Agent Service configuration semantics."""

from pathlib import Path

from ...service.config import ServiceConfig


def test_has_required_api_keys_accepts_public_nim_credentials_with_sidecar_renderer(
    monkeypatch, tmp_path: Path
):
    """Public NIM + local sidecar rendering should not require NGC_API_KEY."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_accepts_physics_local_nim_sidecar(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("PA_NIM_API_KEY", "not-used")
    monkeypatch.setenv("PA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")

    config = ServiceConfig(
        vlm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_uses_effective_vlm_backend_for_nim_override(
    monkeypatch, tmp_path: Path
):
    """PA_VLM_NIM_BASE_URL makes readiness validate the effective NIM backend."""
    monkeypatch.setenv("PA_VLM_BACKEND", "openai")
    monkeypatch.setenv("PA_NIM_API_KEY", "not-used")
    monkeypatch.setenv("PA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.vlm_backend == "openai"
    assert config.has_required_api_keys is True


def test_has_required_api_keys_requires_ngc_for_authenticated_remote_renderer(
    monkeypatch, tmp_path: Path
):
    """Authenticated remote renderer endpoints still require NGC_API_KEY."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "https://ai.api.nvidia.com/v1/render")
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False
