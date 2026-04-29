# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for chat model implementations."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from world_understanding.functions.models.chat_models import (
    _DEFAULT_NIM_MODEL,
    EchoChatModel,
    create_chat_model,
    create_echo_chat_model,
    create_nim_chat_model,
)


class TestEchoChatModel:
    """Test cases for EchoChatModel."""

    def test_echo_model_creation(self):
        """Test basic echo model creation."""
        model = create_echo_chat_model()
        assert isinstance(model, EchoChatModel)
        assert model.prefix == "Echo: "

    def test_echo_model_custom_prefix(self):
        """Test echo model with custom prefix."""
        model = create_echo_chat_model(prefix="Reply: ")
        assert model.prefix == "Reply: "

    def test_echo_generates_response(self):
        """Test echo model generates expected response."""
        model = create_echo_chat_model()
        messages = [HumanMessage(content="Hello")]
        result = model._generate(messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Echo: Hello"

    def test_echo_with_system_message(self):
        """Test echo model handles system + human messages."""
        model = create_echo_chat_model()
        messages = [
            SystemMessage(content="You are a test bot."),
            HumanMessage(content="Test message"),
        ]
        result = model._generate(messages)

        # Should echo the last message (human message)
        assert result.generations[0].message.content == "Echo: Test message"

    def test_echo_stores_request(self):
        """Test echo model stores last request for testing."""
        model = create_echo_chat_model()
        messages = [HumanMessage(content="Test")]
        model._generate(messages, stop=["STOP"], extra_param="value")

        assert model.last_request is not None
        assert model.last_request["messages"] == messages
        assert model.last_request["stop"] == ["STOP"]
        assert model.last_request["extra_param"] == "value"

    def test_echo_with_dict_message(self):
        """Test echo model handles dict-format messages."""
        model = create_echo_chat_model()
        messages = [{"content": "Dict message"}]
        result = model._generate(messages)

        assert result.generations[0].message.content == "Echo: Dict message"

    def test_echo_empty_messages(self):
        """Test echo model handles empty message list."""
        model = create_echo_chat_model()
        result = model._generate([])

        assert result.generations[0].message.content == "Echo: "

    def test_echo_returns_ai_message(self):
        """Test echo model returns AIMessage."""
        model = create_echo_chat_model()
        messages = [HumanMessage(content="Test")]
        result = model._generate(messages)

        assert isinstance(result.generations[0].message, AIMessage)

    def test_echo_llm_type(self):
        """Test echo model returns correct LLM type."""
        model = create_echo_chat_model()
        assert model._llm_type == "echo"


class TestNIMChatModel:
    """Test cases for NIM chat model creation."""

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_default_params(self, mock_nvidia):
        """Test creating NIM model with defaults.

        ``streaming=False`` and ``timeout`` are not declared ctor fields on
        the installed langchain_nvidia_ai_endpoints version; passing them
        routes into ``model_kwargs``, which gets serialized as request-body
        fields. Strict NIM serving (e.g. Nemotron Nano 8B) rejects unknown
        body fields with "400 extra_forbidden". The factory therefore only
        forwards ``streaming`` when the caller explicitly requests it and
        never forwards ``timeout`` — relying on the client default instead.
        """
        mock_instance = Mock()
        mock_nvidia.return_value = mock_instance

        result = create_nim_chat_model(api_key="test_key")

        assert result == mock_instance
        mock_nvidia.assert_called_once_with(
            model=_DEFAULT_NIM_MODEL,
            nvidia_api_key="test_key",
        )

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_custom_params(self, mock_nvidia):
        """Test creating NIM model with custom parameters.

        See ``test_create_nim_default_params`` for why ``timeout`` is not
        forwarded. ``streaming=True`` *is* forwarded because the caller
        asked for it explicitly.
        """
        mock_instance = Mock()
        mock_nvidia.return_value = mock_instance

        result = create_nim_chat_model(
            api_key="test_key",
            model="custom-model",
            temperature=0.5,
            top_p=0.9,
            max_tokens=2048,
            streaming=True,
            custom_param="value",
        )

        assert result == mock_instance
        mock_nvidia.assert_called_once_with(
            model="custom-model",
            nvidia_api_key="test_key",
            streaming=True,
            temperature=0.5,
            top_p=0.9,
            max_tokens=2048,
            custom_param="value",
        )


# Tests for the perflab_azure_openai backend (internal-only endpoint)
# live in tests/internal/test_chat_models_internal.py.


class TestCreateChatModel:
    """Test cases for the unified create_chat_model function."""

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_backend(self, mock_nvidia):
        """Test creating NIM backend through unified function."""
        mock_model = Mock()
        mock_nvidia.return_value = mock_model

        result = create_chat_model(
            backend="nim", api_key="test_key", model="test-model"
        )

        assert result == mock_model

    def test_create_nim_without_key(self):
        """Test NIM backend requires API key."""
        with pytest.raises(ValueError, match="API key is required"):
            create_chat_model(backend="nim")

    def test_create_echo_backend(self):
        """Test creating echo backend through unified function."""
        result = create_chat_model(backend="echo", prefix="Test: ")

        assert isinstance(result, EchoChatModel)
        assert result.prefix == "Test: "

    def test_create_echo_ignores_api_key(self):
        """Test echo backend ignores API key parameter."""
        model = create_chat_model(
            backend="echo",
            api_key="ignored",
            temperature=0.1,  # Also ignored
        )

        assert isinstance(model, EchoChatModel)
        assert model.prefix == "Echo: "  # Default prefix

    def test_create_unknown_backend(self):
        """Test error for unknown backend."""
        with pytest.raises(ValueError, match="Unknown chat backend: unknown"):
            create_chat_model(backend="unknown")

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_with_default_model(self, mock_nvidia):
        """Test using default model when not specified."""
        mock_model = Mock()
        mock_nvidia.return_value = mock_model

        create_chat_model(backend="nim", api_key="key")

        mock_nvidia.assert_called_once()
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["model"] == _DEFAULT_NIM_MODEL

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_with_streaming(self, mock_nvidia):
        """Test enabling streaming mode."""
        mock_model = Mock()
        mock_nvidia.return_value = mock_model

        create_chat_model(backend="nim", api_key="key", streaming=True)

        mock_nvidia.assert_called_once()
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["streaming"] is True
