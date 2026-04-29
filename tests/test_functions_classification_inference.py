# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for classification inference hard timeouts."""

import time

import pytest

from world_understanding.functions.classification.inference import (
    async_classify_object,
    classify_object,
)


class _SlowVLM:
    """Minimal fake VLM that never responds before the deadline."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        time.sleep(0.05)
        return '{"class": "late"}'


class _ParserFallbackVLM:
    """Fake VLM that needs a second text-only parser pass to return JSON."""

    last_token_usage = None

    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("images"):
            return "The best match is metal."
        return '{"class": "metal"}'

    async def agenerate(self, *args, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("images"):
            return "The best match is metal."
        return '{"class": "metal"}'


def test_classify_object_enforces_hard_timeout(monkeypatch):
    """Slow VLM calls should fail fast instead of hanging indefinitely."""
    monkeypatch.setenv("WU_VLM_GENERATE_TIMEOUT_SECONDS", "0.01")

    with pytest.raises(TimeoutError, match="VLM generate did not respond"):
        classify_object(
            vlm=_SlowVLM(),
            text="classify this",
            images=["unused.png"],
            llm=object(),
            max_retries=1,
        )


def test_classify_object_supports_vlm_parser_fallback():
    """The parser fallback should support llm=vlm without requiring .invoke()."""
    parser_vlm = _ParserFallbackVLM()

    result = classify_object(
        vlm=parser_vlm,
        text="classify this object",
        images=["unused.png"],
        llm=parser_vlm,
        max_retries=1,
    )

    assert result["class"] == "metal"
    assert len(parser_vlm.calls) == 2
    assert parser_vlm.calls[0]["images"] == ["unused.png"]
    assert parser_vlm.calls[1]["images"] is None


@pytest.mark.asyncio
async def test_async_classify_object_supports_vlm_parser_fallback():
    """Async classification should support llm=vlm without requiring .ainvoke()."""
    parser_vlm = _ParserFallbackVLM()

    result = await async_classify_object(
        vlm=parser_vlm,
        text="classify this object",
        images=["unused.png"],
        llm=parser_vlm,
        max_retries=1,
    )

    assert result["class"] == "metal"
    assert len(parser_vlm.calls) == 2
    assert parser_vlm.calls[0]["images"] == ["unused.png"]
    assert parser_vlm.calls[1]["images"] is None
