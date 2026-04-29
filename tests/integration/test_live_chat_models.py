# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live inference tests for chat models (opt-in, provider-backed).

These tests hit real providers and are gated to avoid running by default.

Enable by setting environment variable RUN_LIVE_INFERENCE=1 and providing the
required API keys for each provider under test.
"""

import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from world_understanding.functions.models.chat_models import create_chat_model

pytestmark = pytest.mark.live_inference


RUN_LIVE = os.getenv("RUN_LIVE_INFERENCE") == "1"

LLMGATEWAY_CREDS_URL = os.getenv("LLMGATEWAY_CREDS_URL", "")


@pytest.mark.skipif(
    not RUN_LIVE or not os.getenv("NVIDIA_API_KEY"),
    reason="Live NIM test requires RUN_LIVE_INFERENCE=1 and NVIDIA_API_KEY",
)
def test_live_nim_chat_basic() -> None:
    """Smoke test NIM chat model end-to-end."""
    llm = create_chat_model(backend="nim", api_key=os.environ["NVIDIA_API_KEY"])

    response = llm.invoke(
        [
            SystemMessage(content="You are concise."),
            HumanMessage(content="Say hi in three words."),
        ],
        max_tokens=32,
    )

    print(f"\n[NIM chat] Response:\n{response.content}\n")

    assert hasattr(response, "content")
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0


@pytest.mark.skipif(
    not RUN_LIVE or not os.getenv("NSTORAGE_API_KEY"),
    reason=(
        "Live Azure OpenAI test requires RUN_LIVE_INFERENCE=1 and NSTORAGE_API_KEY"
    ),
)
def test_live_azure_openai_chat_basic() -> None:
    """Smoke test Azure OpenAI chat model end-to-end."""
    llm = create_chat_model(
        backend="perflab_azure_openai", api_key=os.environ["NSTORAGE_API_KEY"]
    )

    response = llm.invoke(
        [
            SystemMessage(content="You are concise."),
            HumanMessage(content="Name one fruit."),
        ],
        max_tokens=16,
        temperature=0.2,
    )

    print(f"\n[Azure OpenAI chat] Response:\n{response.content}\n")

    assert hasattr(response, "content")
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0


@pytest.mark.skipif(
    not RUN_LIVE or not LLMGATEWAY_CREDS_URL,
    reason="Live LLM Gateway tests require RUN_LIVE_INFERENCE=1 and LLMGATEWAY_CREDS_URL",
)
def test_live_llmgateway_aws_anthropic_thinking() -> None:
    """AWS Anthropic via LLM Gateway with thinking enabled.

    Verifies that thinking region is included between <thinking> tags
    and the final answer is present.
    """
    llm = create_chat_model(
        backend="llmgateway_aws_anthropic",
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        llmgateway={
            "cred_file_url": LLMGATEWAY_CREDS_URL,
        },
        include_thinking=True,
        thinking={"type": "enabled", "budget_tokens": 1024},
        max_tokens=1536,
    )

    messages = [
        SystemMessage(content="You are helpful and concise."),
        HumanMessage(content="What is 1+2? Answer with a single digit."),
    ]

    response = llm.invoke(messages)

    assert hasattr(response, "content")
    text = str(response.content)
    print(f"\n[LLMGateway AWS Anthropic] Full response:\n{text}\n")
    assert len(text.strip()) > 0
    # Thinking region should be included when include_thinking=True
    assert "<thinking>" in text and "</thinking>" in text
    assert "3" in text.lower()


@pytest.mark.skipif(
    not RUN_LIVE or not LLMGATEWAY_CREDS_URL,
    reason="Live LLM Gateway tests require RUN_LIVE_INFERENCE=1 and LLMGATEWAY_CREDS_URL",
)
def test_live_llmgateway_azure_openai_reasoning() -> None:
    """Azure OpenAI GPT-5 via LLM Gateway with reasoning effort.

    Uses a small max_completion_tokens to minimize cost.
    """
    llm = create_chat_model(
        backend="llmgateway_azure_openai",
        model="gpt-5",
        llmgateway={
            "cred_file_url": LLMGATEWAY_CREDS_URL,
        },
    )

    messages = [
        SystemMessage(content="You are helpful and concise."),
        HumanMessage(content="What is 1+2? Answer with a single digit."),
    ]

    # Prefer max_completion_tokens for GPT-5; pass reasoning effort if supported
    response = llm.invoke(
        messages,
        reasoning_effort="low",
        max_completion_tokens=1536,  # to accommodate reasoning tokens
    )

    assert hasattr(response, "content")
    text = str(response.content)
    print(f"\n[LLMGateway Azure GPT-5] Response:\n{text}\n")
    assert len(text.strip()) > 0
    assert "3" in text.lower()
