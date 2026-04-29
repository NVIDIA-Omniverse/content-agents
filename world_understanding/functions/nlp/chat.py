# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable function for text generation using LLM backends."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


def _extract_text(content: Any) -> str:
    """Extract text from LLM response content.

    Handles both plain strings and list-of-parts responses
    (e.g. thinking models that return content blocks with signatures).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        parts = [p for p in parts if p]
        return "\n".join(parts) if parts else str(content)
    return str(content)


def generate_chat_response(
    chat_model: Any, prompt: str, system_prompt: str = "You are a helpful AI assistant."
) -> dict[str, Any]:
    """
    Generate text response using provided chat model.

    Args:
        chat_model: Initialized chat model instance
        prompt: User prompt/question
        system_prompt: System instructions for the model

    Returns:
        Dict containing:
            - response: Generated text response
            - error: Error message (only if an error occurred)
    """

    # Generate response using LangChain messages
    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        response_message = chat_model.invoke(messages)
        response = _extract_text(response_message.content)
    except Exception as e:
        return {"error": f"Failed to generate response: {e}"}

    # Return results
    return {"response": response}


async def agenerate_chat_response(
    chat_model: Any, prompt: str, system_prompt: str = "You are a helpful AI assistant."
) -> dict[str, Any]:
    """Generate text response using provided chat model asynchronously.

    Args:
        chat_model: Initialized chat model instance (must support ainvoke)
        prompt: User prompt/question
        system_prompt: System instructions for the model

    Returns:
        Dict containing:
            - response: Generated text response
            - error: Error message (only if an error occurred)
    """
    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response_message = await chat_model.ainvoke(messages)
        response = _extract_text(response_message.content)
    except Exception as e:
        return {"error": f"Failed to generate response: {e}"}
    return {"response": response}
