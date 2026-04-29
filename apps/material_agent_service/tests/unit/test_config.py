# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Service configuration semantics."""

from pathlib import Path

from ...service.config import ServiceConfig


def test_has_required_api_keys_allows_local_render_with_public_nim(
    monkeypatch, tmp_path: Path
):
    """Local OVRTX rendering should not require NGC_API_KEY."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_requires_ngc_for_remote_render(
    monkeypatch, tmp_path: Path
):
    """Remote render endpoints should still require NGC_API_KEY."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("RENDER_ENDPOINT", "https://renderer.example.com")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_image_gen_configuration_defaults_to_public_backend(tmp_path: Path):
    config = ServiceConfig(session_storage_path=str(tmp_path / "sessions"))

    assert config.image_gen_backend == "gemini"
    assert config.image_gen_model is None
    assert config.image_gen_base_url is None


def test_image_gen_configuration_reads_deployment_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MA_IMAGE_GEN_BACKEND", "openai")
    monkeypatch.setenv("MA_IMAGE_GEN_MODEL", "gpt-image-1")
    monkeypatch.setenv("MA_IMAGE_GEN_BASE_URL", "http://image-gen:8000/v1")

    config = ServiceConfig(session_storage_path=str(tmp_path / "sessions"))

    assert config.image_gen_backend == "openai"
    assert config.image_gen_model == "gpt-image-1"
    assert config.image_gen_base_url == "http://image-gen:8000/v1"


def test_image_gen_ready_validates_selected_backend(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="gemini",
        session_storage_path=str(tmp_path / "sessions"),
    )
    assert config.image_gen_ready is False

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    assert config.image_gen_ready is False

    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    assert config.image_gen_ready is True

    openai_config = ServiceConfig(
        image_gen_backend="openai",
        image_gen_base_url="http://image-gen.local/v1",
        session_storage_path=str(tmp_path / "sessions-openai"),
    )
    assert openai_config.image_gen_ready is True

    openai_without_config = ServiceConfig(
        image_gen_backend="openai",
        session_storage_path=str(tmp_path / "sessions-openai-missing"),
    )
    assert openai_without_config.image_gen_ready is False
