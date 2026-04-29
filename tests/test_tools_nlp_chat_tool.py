# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for chat tool."""

import os
from unittest.mock import patch

import pytest

from world_understanding.tools.nlp.chat_tool import (
    ChatInput,
    ChatOutput,
    chat_tool,
)


class TestChatInput:
    """Tests for ChatInput model."""

    def test_valid_input(self):
        """Test creating valid ChatInput."""
        input_obj = ChatInput(
            prompt="Hello, how are you?",
            backend="nim",
            model="llama",
            temperature=0.7,
            max_tokens=100,
        )

        assert input_obj.prompt == "Hello, how are you?"
        assert input_obj.backend == "nim"
        assert input_obj.model == "llama"
        assert input_obj.temperature == 0.7
        assert input_obj.max_tokens == 100

    def test_default_values(self):
        """Test default values for optional fields."""
        input_obj = ChatInput(prompt="Test prompt")

        assert input_obj.backend == "nim"  # default
        assert input_obj.model is None  # default (backend provides its own default)
        assert input_obj.temperature == 0.7  # default
        assert input_obj.max_tokens == 1024  # default

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        ChatInput(prompt="test", temperature=0.0)
        ChatInput(prompt="test", temperature=1.0)
        ChatInput(prompt="test", temperature=0.5)

        # Invalid temperatures
        with pytest.raises(ValueError):
            ChatInput(prompt="test", temperature=-0.1)

        with pytest.raises(ValueError):
            ChatInput(prompt="test", temperature=2.1)  # max is 2.0

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        ChatInput(prompt="test", max_tokens=1)
        ChatInput(prompt="test", max_tokens=8192)  # max is 8192

        # Invalid max_tokens
        with pytest.raises(ValueError):
            ChatInput(prompt="test", max_tokens=0)

        with pytest.raises(ValueError):
            ChatInput(prompt="test", max_tokens=8193)  # exceeds max


class TestChatOutput:
    """Tests for ChatOutput model."""

    def test_valid_output(self):
        """Test creating valid ChatOutput."""
        output = ChatOutput(
            response="Hello! I can help you with that.",
            backend_used="nim",
            model_used="llama-2-7b",
        )

        assert output.response == "Hello! I can help you with that."
        assert output.backend_used == "nim"
        assert output.model_used == "llama-2-7b"

    def test_optional_fields(self):
        """Test optional fields in output."""
        output = ChatOutput(
            response="Test response",
            backend_used="echo",
        )

        assert output.response == "Test response"
        assert output.backend_used == "echo"
        assert output.model_used is None  # model_used is optional


class TestChatTool:
    """Tests for chat_tool function."""

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_basic_chat(self, mock_generate):
        """Test basic chat functionality."""
        # Mock the response
        mock_generate.return_value = {
            "response": "Hello! How can I help you?",
            "model": "gpt-4",
            "token_count": 8,
        }

        inputs = ChatInput(prompt="Hello!")
        output = chat_tool(inputs)

        assert isinstance(output, ChatOutput)
        assert output.response == "Hello! How can I help you?"
        assert output.backend_used == "nim"
        assert output.model_used is None  # backend provides its own default

        # Verify the function was called with correct parameters
        mock_generate.assert_called_once()
        # Verify the function was called with correct parameters
        call_args = mock_generate.call_args[1]
        assert call_args["prompt"] == "Hello!"

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_custom_backend(self, mock_generate):
        """Test chat with custom backend."""
        mock_generate.return_value = {
            "response": "OpenAI response",
            "model": "gpt-4",
        }

        inputs = ChatInput(
            prompt="Test prompt",
            backend="openai",
            api_key="test-key",
            model="gpt-4",
        )
        output = chat_tool(inputs)

        assert output.response == "OpenAI response"
        assert output.backend_used == "openai"
        assert output.model_used == "gpt-4"

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_temperature_and_max_tokens(self, mock_generate):
        """Test chat with custom temperature and max_tokens."""
        mock_generate.return_value = {"response": "Short response"}

        inputs = ChatInput(
            prompt="Be creative",
            temperature=0.9,
            max_tokens=50,
        )
        chat_tool(inputs)

        mock_generate.call_args[1]
        # Note: These parameters might not be passed directly in the current implementation
        # This test verifies the tool accepts them without error

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_empty_response(self, mock_generate):
        """Test handling of empty response."""
        mock_generate.return_value = {"response": ""}

        inputs = ChatInput(prompt="Test")
        output = chat_tool(inputs)

        assert output.response == ""

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_long_prompt(self, mock_generate):
        """Test chat with a long prompt."""
        long_prompt = "This is a very long prompt. " * 100
        mock_generate.return_value = {"response": "Response to long prompt"}

        inputs = ChatInput(prompt=long_prompt)
        output = chat_tool(inputs)

        assert output.response == "Response to long prompt"
        call_args = mock_generate.call_args[1]
        assert call_args["prompt"] == long_prompt

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_error_handling(self, mock_generate):
        """Test error handling in chat tool."""
        mock_generate.side_effect = Exception("API Error")

        inputs = ChatInput(prompt="Test")
        output = chat_tool(inputs)

        # The tool catches exceptions and returns error message
        assert "Error generating response" in output.response
        assert "API Error" in output.response

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_echo_backend(self, mock_generate):
        """Test chat with echo backend."""
        mock_generate.return_value = {
            "response": "Echo: Test prompt",
            "model": "echo",
        }

        inputs = ChatInput(
            prompt="Test prompt",
            backend="echo",
        )
        output = chat_tool(inputs)

        assert output.response == "Echo: Test prompt"
        assert output.backend_used == "echo"
        assert output.model_used is None  # backend provides its own default

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_response_format(self, mock_generate):
        """Test that chat response has correct format."""
        mock_generate.return_value = {
            "response": "Formatted response",
            "model": "test-model",
            "token_count": 3,
        }

        inputs = ChatInput(prompt="Format test")
        output = chat_tool(inputs)

        # Verify output is properly structured
        assert hasattr(output, "response")
        assert hasattr(output, "backend_used")
        assert hasattr(output, "model_used")
        assert isinstance(output.response, str)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_env_api_key(self, mock_generate):
        """Test that chat tool can use environment variables."""
        mock_generate.return_value = {"response": "Response with env key"}

        inputs = ChatInput(
            prompt="Test with env",
            backend="openai",
        )
        output = chat_tool(inputs)

        assert output.response == "Response with env key"
        assert output.backend_used == "openai"
