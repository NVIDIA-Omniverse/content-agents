# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the wu image-gen CLI command credential resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from world_understanding.cli import app

runner = CliRunner()


@pytest.fixture
def tmp_output_path(tmp_path: Path) -> Path:
    return tmp_path / "out.png"


def _stub_image_model() -> MagicMock:
    """Return a fake image generation model with a model_name attribute."""
    fake = MagicMock()
    fake.model_name = "fake-model"
    fake.generate.return_value = MagicMock()
    return fake


def test_wu_image_gen_openai_local_base_url_injects_no_auth_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """``wu image-gen --backend openai --base-url http://localhost:8000/v1``
    must keep working without ``OPENAI_API_KEY`` and without ``--api-key``.

    The endpoint-aware credential resolver requires an explicit ``not-used``
    opt-in for local OpenAI-compatible endpoints (it does not silently
    forward a hosted ``OPENAI_API_KEY`` to a local URL anymore). The CLI
    must inject the placeholder so the documented locally-hosted image-gen
    flow does not regress.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    fake = _stub_image_model()
    fake.generate.return_value.save = lambda path: Path(path).write_bytes(b"\x89PNG")

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
        return_value=fake,
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "openai",
                "--base-url",
                "http://localhost:8000/v1",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert mock_create.call_count == 1
    backend_arg = mock_create.call_args.args[0]
    kwargs = mock_create.call_args.kwargs
    assert backend_arg == "openai"
    assert kwargs["base_url"] == "http://localhost:8000/v1"
    assert kwargs["api_key"] == "not-used"


def test_wu_image_gen_openai_no_key_no_base_url_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """Without ``OPENAI_API_KEY`` and without ``--base-url``, the command
    still errors with a clear message — the new placeholder injection is
    scoped to the local-base-url path."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    result = runner.invoke(
        app,
        [
            "image-gen",
            "test prompt",
            "--backend",
            "openai",
            "--output",
            str(tmp_output_path),
        ],
    )

    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_wu_image_gen_openai_custom_base_url_does_not_forward_hosted_key(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """``OPENAI_API_KEY`` must not be promoted as an explicit endpoint key
    when ``--base-url`` points at a non-provider URL. Otherwise the hosted
    key would be forwarded to an arbitrary OpenAI-compatible endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "openai",
                "--base-url",
                "https://api.openai-compatible.example/v1",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code != 0
    assert mock_create.call_count == 0


def test_wu_image_gen_nim_local_base_url_injects_no_auth_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """Local NIM image-gen flow must work without ``NVIDIA_API_KEY``."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    fake = _stub_image_model()
    fake.generate.return_value.save = lambda path: Path(path).write_bytes(b"\x89PNG")

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
        return_value=fake,
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "nim",
                "--base-url",
                "http://localhost:8000/v1",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    kwargs = mock_create.call_args.kwargs
    assert kwargs["api_key"] == "not-used"


def test_wu_image_gen_nim_custom_base_url_does_not_forward_hosted_key(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """``NVIDIA_API_KEY`` must not be promoted as an explicit endpoint key
    when ``--base-url`` points at a non-NVIDIA NIM URL."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-real-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "nim",
                "--base-url",
                "https://nim.example.com/v1",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code != 0
    assert mock_create.call_count == 0
