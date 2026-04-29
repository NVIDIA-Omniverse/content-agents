# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chat tool for text generation using LLM backends."""

import logging
import os
from typing import Any

from pydantic import Field
from rich.console import Console

from world_understanding.functions.models.chat_models import (
    create_chat_model,
)
from world_understanding.functions.nlp.chat import generate_chat_response
from world_understanding.tools.base import (
    ExecutionPolicy,
    ToolInput,
    ToolOutput,
    register_tool,
)

logger = logging.getLogger(__name__)


class ChatInput(ToolInput):
    """Input for chat tool."""

    prompt: str = Field(..., description="User prompt/question")
    backend: str = Field(
        default="nim",
        description="Chat backend to use (e.g. nim, openai, anthropic, gemini, echo)",
    )
    api_key: str | None = Field(
        default=None,
        description=("API key for the backend (uses env var if not provided)"),
    )
    model: str | None = Field(
        default=None,
        description=("Model to use (backend-specific, uses default if not provided)"),
    )
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System instructions for the model",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for response generation"
    )
    max_tokens: int = Field(
        default=1024, ge=1, le=8192, description="Maximum tokens in response"
    )


class ChatOutput(ToolOutput):
    """Output for chat tool."""

    response: str = Field(..., description="Generated text response")
    backend_used: str = Field(..., description="Chat backend that was used")
    model_used: str | None = Field(
        default=None, description="Model that was used (if available)"
    )


def _display_chat_response(
    outputs: dict[str, Any], console: Console, indent: str = ""
) -> None:
    """Display chat response in a formatted way."""
    console.print(f"{indent}[bold]Chat Response:[/bold]")
    console.print(f"{indent}Backend: {outputs.get('backend_used', 'unknown')}")
    if outputs.get("model_used"):
        console.print(f"{indent}Model: {outputs['model_used']}")
    console.print(f"{indent}[bold]Response:[/bold]")
    console.print(f"{indent}{outputs.get('response', 'No response')}")


@register_tool(
    name="chat",
    version="0.1.0",
    description="Generate text responses using LLM backends",
    tags=["text", "generation", "llm", "cpu"],
    input_model=ChatInput,
    output_model=ChatOutput,
    policy=ExecutionPolicy(timeout_s=60.0),
)
def chat_tool(inputs: ChatInput) -> ChatOutput:
    """Generate text responses using various LLM backends."""
    # Get API key from environment if not provided
    api_key = inputs.api_key
    if not api_key:
        if inputs.backend == "nim":
            api_key = os.getenv("NVIDIA_API_KEY", "echo_key")
        elif inputs.backend == "perflab_azure_openai":
            api_key = os.getenv("NSTORAGE_API_KEY", "echo_key")
        elif inputs.backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "echo_key")
        elif inputs.backend == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "echo_key")
        elif inputs.backend == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY", "echo_key")
        elif inputs.backend == "nvidia_inference":
            api_key = os.getenv("INFERENCE_NVIDIA_API_KEY", "echo_key")
        else:
            api_key = "echo_key"

    # Create chat model with temperature and max_tokens
    chat_model = create_chat_model(
        backend=inputs.backend,
        api_key=api_key,
        model=inputs.model,
        temperature=inputs.temperature,
        max_tokens=inputs.max_tokens,
    )

    # Call the portable function
    try:
        response_dict = generate_chat_response(
            chat_model=chat_model,
            prompt=inputs.prompt,
            system_prompt=inputs.system_prompt,
        )

        # Extract the response string from the dict
        response_text = (
            response_dict.get("response", "")
            if isinstance(response_dict, dict)
            else str(response_dict)
        )

        return ChatOutput(
            response=response_text,
            backend_used=inputs.backend,
            model_used=inputs.model,
        )
    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        # Return error message as response
        return ChatOutput(
            response=f"Error generating response: {str(e)}",
            backend_used=inputs.backend,
            model_used=inputs.model,
        )


# Attach display function to the tool
chat_tool._display_function = _display_chat_response
