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
