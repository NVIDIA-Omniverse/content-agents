# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live inference tests for Vision-Language Models (opt-in).

Uses simple synthetic images to keep requests small and deterministic-ish.
Enable by setting RUN_LIVE_INFERENCE=1 and providing provider API keys.
"""

import os

import pytest
from PIL import Image

from world_understanding.functions.models.vision_language_models import create_vlm

pytestmark = pytest.mark.live_inference


RUN_LIVE = os.getenv("RUN_LIVE_INFERENCE") == "1"

LLMGATEWAY_CREDS_URL = os.getenv("LLMGATEWAY_CREDS_URL", "")


def _make_solid_image(color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    return Image.new("RGB", (64, 64), color)


@pytest.mark.skipif(
    not RUN_LIVE or not os.getenv("NVIDIA_API_KEY"),
    reason="Live NIM VLM test requires RUN_LIVE_INFERENCE=1 and NVIDIA_API_KEY",
)
def test_live_nim_vlm_basic_scout_model() -> None:
    """NIM VLM smoke test using nvdev/meta/llama-4-scout-17b-16e-instruct."""
    vlm = create_vlm(
        backend="nim",
        api_key=os.environ["NVIDIA_API_KEY"],
        model="nvdev/meta/llama-4-scout-17b-16e-instruct",
    )
    img = _make_solid_image((0, 0, 0))

    resp = vlm.generate(
        prompt="What color is the image? Answer in one word.",
        images=[img],
        temperature=0.2,
        max_tokens=16,
    )

    print(f"\n[NIM VLM] Response:\n{resp}\n")

    assert isinstance(resp, str)
    assert len(resp.strip()) > 0
    assert "black" in resp.lower()


@pytest.mark.skipif(
    not RUN_LIVE or not os.getenv("NSTORAGE_API_KEY"),
    reason=(
        "Live Azure OpenAI VLM test requires RUN_LIVE_INFERENCE=1 and NSTORAGE_API_KEY"
    ),
)
def test_live_azure_vlm_basic_image_prompt() -> None:
    """Smoke test Azure OpenAI VLM end-to-end with a simple image."""
    vlm = create_vlm(
        backend="perflab_azure_openai",
        api_key=os.environ["NSTORAGE_API_KEY"],
        model="gpt-4o-20241120",
    )
    img = _make_solid_image((0, 0, 0))

    resp = vlm.generate(
        prompt="What color is the image? Answer in one word.",
        images=[img],
        temperature=0.2,
        max_tokens=16,
    )

    print(f"\n[Azure VLM] Response:\n{resp}\n")

    assert isinstance(resp, str)
    assert len(resp.strip()) > 0
    assert "black" in resp.lower()


@pytest.mark.skipif(
    not RUN_LIVE or not LLMGATEWAY_CREDS_URL,
    reason="Live LLM Gateway tests require RUN_LIVE_INFERENCE=1 and LLMGATEWAY_CREDS_URL",
)
def test_live_llmgateway_azure_openai_vlm_basic() -> None:
    """LLM Gateway Azure OpenAI VLM with reasoning effort."""
    vlm = create_vlm(
        backend="llmgateway_azure_openai",
        model="gpt-5",
        llmgateway={
            "cred_file_url": LLMGATEWAY_CREDS_URL,
        },
    )
    img = _make_solid_image((0, 0, 0))

    resp = vlm.generate(
        prompt="What color is the image? Answer in one word.",
        images=[img],
        reasoning_effort="low",
        max_completion_tokens=1536,  # to accommodate reasoning tokens
    )

    print(f"\n[LLMGateway Azure VLM] Response:\n{resp}\n")

    assert isinstance(resp, str)
    assert len(resp.strip()) > 0
    assert "black" in resp.lower()


@pytest.mark.skipif(
    not RUN_LIVE or not LLMGATEWAY_CREDS_URL,
    reason="Live LLM Gateway tests require RUN_LIVE_INFERENCE=1 and LLMGATEWAY_CREDS_URL",
)
def test_live_llmgateway_aws_anthropic_vlm_thinking() -> None:
    """LLM Gateway AWS Anthropic VLM with thinking included."""
    vlm = create_vlm(
        backend="llmgateway_aws_anthropic",
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        llmgateway={
            "cred_file_url": LLMGATEWAY_CREDS_URL,
        },
        include_thinking=True,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    img = _make_solid_image((0, 0, 0))

    resp = vlm.generate(
        prompt="What color is the image? Answer in one word.",
        images=[img],
        max_tokens=1536,
    )

    print(f"\n[LLMGateway AWS Anthropic VLM] Response:\n{resp}\n")

    assert isinstance(resp, str)
    text = str(resp)
    assert len(text.strip()) > 0
    assert "<thinking>" in text and "</thinking>" in text
    assert "black" in text.lower()
