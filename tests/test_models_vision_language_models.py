# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for vision-language model construction."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from world_understanding.functions.models.vision_language_models import create_vlm


def test_create_gemini_vlm_accepts_gemini_api_key_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test direct Gemini VLM construction accepts GEMINI_API_KEY."""
    captured: dict[str, object] = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        SimpleNamespace(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI),
    )

    import world_understanding.functions.models.backends  # noqa: F401

    vlm = create_vlm("gemini")

    assert vlm.backend_name == "gemini"
    assert captured["google_api_key"] == "gemini-key"


def test_create_gemini_vlm_replaces_placeholder_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct Gemini VLM construction should not pass placeholders to LangChain."""
    captured: dict[str, object] = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "real-gemini-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        SimpleNamespace(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI),
    )

    import world_understanding.functions.models.backends  # noqa: F401

    vlm = create_vlm("gemini", api_key="YOUR_GOOGLE_API_KEY")

    assert vlm.backend_name == "gemini"
    assert captured["google_api_key"] == "real-gemini-key"


def test_create_openai_vlm_rejects_explicit_key_with_env_redirected_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENAI_BASE_URL`` redirects the OpenAI SDK; an explicit hosted
    ``OPENAI_API_KEY`` passed directly to the VLM factory must not follow
    that redirect to a non-provider endpoint without an explicit
    ``base_url`` pairing."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        create_vlm("openai", api_key="sk-real-openai-key")
