# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for create_chat_model_from_config helper."""

import os
from unittest.mock import patch

from world_understanding.functions.models.chat_models import (
    EchoChatModel,
    create_chat_model_from_config,
)


class TestCreateChatModelFromConfig:
    """Tests for create_chat_model_from_config."""

    def test_creates_echo_backend(self):
        """Config with echo backend creates an EchoChatModel."""
        config = {"backend": "echo", "api_key": "ignored"}
        model = create_chat_model_from_config(config)
        assert isinstance(model, EchoChatModel)

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    @patch.dict(os.environ, {}, clear=True)
    def test_creates_nim_backend_with_api_key(self, mock_nvidia):
        """Config with nim backend and explicit api_key works."""
        mock_nvidia.return_value = mock_nvidia
        config = {"backend": "nim", "api_key": "test-key", "model": "test-model"}
        result = create_chat_model_from_config(config)
        assert result is mock_nvidia

    def test_returns_none_without_api_key(self):
        """Returns None when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            config = {"backend": "nim", "model": "test-model"}
            result = create_chat_model_from_config(config)
            assert result is None

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "env-key"})
    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_uses_env_api_key(self, mock_nvidia):
        """Falls back to NVIDIA_API_KEY for public NIM."""
        mock_nvidia.return_value = mock_nvidia
        config = {"backend": "nim", "model": "test-model"}
        result = create_chat_model_from_config(config)
        assert result is mock_nvidia

    @patch.dict(os.environ, {"INFERENCE_NVIDIA_API_KEY": "env-key"}, clear=True)
    def test_nim_does_not_use_inference_api_key(self):
        """Public NIM should not piggyback on INFERENCE_NVIDIA_API_KEY."""
        config = {"backend": "nim", "model": "test-model"}
        result = create_chat_model_from_config(config)
        assert result is None

    def test_defaults_from_defaults_dict(self):
        """Defaults dict provides fallback values for backend/model/temperature."""
        defaults = {"backend": "echo", "temperature": 0.5}
        # echo backend bypasses api_key validation
        model = create_chat_model_from_config({}, defaults=defaults)
        assert isinstance(model, EchoChatModel)

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_config_overrides_defaults(self, mock_nvidia):
        """Config values take precedence over defaults."""
        mock_nvidia.return_value = mock_nvidia
        defaults = {"backend": "echo", "model": "default-model"}
        config = {"backend": "nim", "api_key": "key", "model": "override-model"}
        create_chat_model_from_config(config, defaults=defaults)
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["model"] == "override-model"

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_passes_temperature_and_max_tokens(self, mock_nvidia):
        """Temperature and max_tokens are passed through."""
        mock_nvidia.return_value = mock_nvidia
        config = {
            "backend": "nim",
            "api_key": "key",
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        create_chat_model_from_config(config)
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2048

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_passes_base_url(self, mock_nvidia):
        """base_url is passed through when provided."""
        mock_nvidia.return_value = mock_nvidia
        config = {
            "backend": "nim",
            "api_key": "key",
            "base_url": "http://localhost:8000",
        }
        create_chat_model_from_config(config)
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:8000"

    @patch.dict(os.environ, {}, clear=True)
    @patch("langchain_openai.ChatOpenAI")
    def test_openai_local_base_url_accepts_placeholder_key(self, mock_openai):
        """Local OpenAI-compatible endpoints may use the documented dummy key."""
        mock_openai.return_value = mock_openai
        config = {
            "backend": "openai",
            "model": "qwen/qwen3.5-35b-a3b",
            "base_url": "http://192.168.4.58:8001/v1",
            "api_key": "not-used",
        }

        result = create_chat_model_from_config(config)

        assert result is mock_openai
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "not-used"
        assert call_kwargs["base_url"] == "http://192.168.4.58:8001/v1"

    @patch.dict(os.environ, {}, clear=True)
    def test_openai_remote_base_url_rejects_placeholder_key(self):
        """Remote OpenAI-compatible endpoints still require a real key."""
        config = {
            "backend": "openai",
            "model": "gpt-4o",
            "base_url": "https://api.openai-compatible.example/v1",
            "api_key": "not-used",
        }

        result = create_chat_model_from_config(config)

        assert result is None

    @patch.dict(
        os.environ,
        {"MA_VLM_NIM_BASE_URL": "http://vlm-nim:8000/v1", "MA_NIM_API_KEY": "not-used"},
        clear=True,
    )
    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_forced_nim_override_drops_stale_provider_api_key(self, mock_nvidia):
        """When MA_*_NIM_BASE_URL forces backend=nim, the prior backend's
        api_key in llm_config is stale and must not be forwarded to the local
        NIM endpoint. Only the explicit MA_NIM_API_KEY (real or no-auth
        placeholder) should reach the NIM client."""
        mock_nvidia.return_value = mock_nvidia
        config = {
            "backend": "openai",
            "model": "gpt-4o",
            "api_key": "sk-real-openai-key",
            "base_url": "https://api.openai.com/v1",
        }

        create_chat_model_from_config(config)

        call_kwargs = mock_nvidia.call_args[1]
        # ChatNVIDIA receives the credential via ``nvidia_api_key``.
        assert call_kwargs["nvidia_api_key"] != "sk-real-openai-key"
        assert call_kwargs["nvidia_api_key"] == "not-used"
        assert call_kwargs["base_url"] == "http://vlm-nim:8000/v1"

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-real-openai-key",
            "OPENAI_BASE_URL": "https://api.openai-compatible.example/v1",
        },
        clear=True,
    )
    def test_openai_env_redirected_base_url_does_not_receive_hosted_key(self):
        """``OPENAI_BASE_URL`` redirects the SDK to a custom endpoint; the
        hosted ``OPENAI_API_KEY`` must not silently follow the redirect."""
        config = {"backend": "openai", "model": "gpt-4o"}

        result = create_chat_model_from_config(config)

        assert result is None

    @patch.dict(
        os.environ,
        {"MA_LLM_NIM_BASE_URL": "http://llm-nim:8000/v1"},
        clear=True,
    )
    def test_forced_nim_override_does_not_pierce_mock_config(self):
        """The runtime LLM NIM env override must not retarget a deliberately-
        mocked simulate config to a real NIM client. The override is a routing
        hint for *real* backends; mock/echo configs are an explicit opt-out
        from any external call."""
        config = {
            "backend": "mock",
            "model": "test-model",
            "api_key": "not-used",
        }

        result = create_chat_model_from_config(config)

        # The factory dispatches "mock" to the mock backend, not ChatNVIDIA.
        # We verify by checking the result is *not* a langchain NVIDIA model
        # (the mock backend returns a simple stub).
        assert type(result).__module__ != "langchain_nvidia_ai_endpoints.chat_models"

    @patch.dict(
        os.environ,
        {"MA_LLM_NIM_BASE_URL": "http://llm-nim:8000/v1"},
        clear=True,
    )
    def test_forced_nim_override_returns_none_when_no_nim_credential_available(self):
        """When the env override forces NIM but no MA_NIM_API_KEY (or local-NIM
        placeholder) is available, the prior backend's api_key must not be
        silently substituted; the call should return None instead."""
        config = {
            "backend": "openai",
            "model": "gpt-4o",
            "api_key": "sk-real-openai-key",
            "base_url": "https://api.openai.com/v1",
        }

        result = create_chat_model_from_config(config)

        assert result is None
