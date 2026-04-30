# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the VLM tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from world_understanding.tools.cv.vlm import VLMInput, VLMOutput, vlm_tool


def test_vlm_tool_accepts_gemini_api_key_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that VLM tool accepts GEMINI_API_KEY without GOOGLE_API_KEY."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    fake_vlm = object()

    with (
        patch(
            "world_understanding.tools.cv.vlm.create_vlm",
            return_value=fake_vlm,
        ) as mock_create_vlm,
        patch(
            "world_understanding.tools.cv.vlm.generate_vlm_response",
            return_value={"response": "Gemini vision response"},
        ),
    ):
        output = vlm_tool(
            VLMInput(
                prompt="Describe this image",
                images=["image.png"],
                backend="gemini",
            )
        )

    assert isinstance(output, VLMOutput)
    assert output.response == "Gemini vision response"
    assert output.backend_used == "gemini"
    assert output.images_analyzed == 1
    mock_create_vlm.assert_called_once()
    call_kwargs = mock_create_vlm.call_args[1]
    assert call_kwargs["api_key"] == "gemini-key"


def test_vlm_tool_openai_rejects_hosted_key_with_env_redirected_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENAI_BASE_URL`` redirects the OpenAI SDK to a custom endpoint.

    The VLM tool path must not silently forward the hosted ``OPENAI_API_KEY``
    to that endpoint — same protection as the config-driven model path.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")

    with patch("world_understanding.tools.cv.vlm.create_vlm") as mock_create_vlm:
        with pytest.raises(ValueError, match="API key required"):
            vlm_tool(
                VLMInput(
                    prompt="Describe this image",
                    images=["image.png"],
                    backend="openai",
                )
            )

    # No VLM client was constructed with the hosted key against the
    # env-redirected URL.
    assert mock_create_vlm.call_count == 0
