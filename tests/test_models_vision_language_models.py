# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for vision-language model construction."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from world_understanding.functions.models.vision_language_models import (
    NvidiaNIMVLM,
    create_vlm,
)


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


def test_nvidia_nim_vlm_omits_constructor_timeout_and_sets_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NIM timeout must not be serialized as a chat-completion body field."""
    captured: dict[str, object] = {}
    sync_client = SimpleNamespace(timeout=None)
    async_client = SimpleNamespace(timeout=None)

    class FakeChatNVIDIA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.max_tokens = 1024
            self._client = sync_client
            self._async_client = async_client

    monkeypatch.setitem(
        sys.modules,
        "langchain_nvidia_ai_endpoints",
        SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
    )

    vlm = NvidiaNIMVLM(
        api_key="test-key",
        model="test-model",
        timeout=42,
        base_url="https://integrate.api.nvidia.com/v1",
    )

    assert captured == {
        "model": "test-model",
        "nvidia_api_key": "test-key",
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
    assert vlm.chat_model.max_tokens is None
    assert sync_client.timeout == 42.0
    assert async_client.timeout == 42.0


def test_nvidia_nim_vlm_warns_when_timeout_cannot_be_applied(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing ChatNVIDIA client attrs should surface as a warning."""
    captured: dict[str, object] = {}

    class FakeChatNVIDIA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.max_tokens = 1024

    monkeypatch.setitem(
        sys.modules,
        "langchain_nvidia_ai_endpoints",
        SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
    )

    with caplog.at_level("WARNING"):
        vlm = NvidiaNIMVLM(
            api_key="test-key",
            model="test-model",
            timeout=42,
        )

    assert captured == {
        "model": "test-model",
        "nvidia_api_key": "test-key",
    }
    assert vlm.chat_model.max_tokens is None
    assert "NvidiaNIMVLM could not apply timeout=42.0" in caplog.text
