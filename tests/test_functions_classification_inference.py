# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for classification inference hard timeouts."""

import json
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


class _JsonThenPlaceholderAnswerVLM:
    """Fake VLM that emits valid JSON plus a stale answer placeholder."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return """<reasoning>
The part is a black structural shoulder component.
</reasoning>

```json
{
  "material": "Steel Painted Black"
}
```
<answer>your answer</answer>"""


class _PhysicsStructuredAnswerVLM:
    """Fake VLM that emits the physics-agent structured answer schema."""

    last_token_usage = None

    @staticmethod
    def _response():
        payload = {
            "asset_type": "light fixture",
            "component_type": "housing",
            "component_name": "lamp housing",
            "material": "metal",
            "physical_properties": {
                "density": 7850,
                "estimated_mass_kg": 0.4,
                "static_friction": 0.6,
                "dynamic_friction": 0.45,
                "restitution": 0.1,
            },
            "confidence": "high",
            "reasoning": "Visible metallic shell around the bulb.",
        }
        return f"<answer>\n{json.dumps(payload)}\n</answer>"

    def generate(self, *args, **kwargs):
        return self._response()

    async def agenerate(self, *args, **kwargs):
        return self._response()


class _StructuredAnswerWithRequestedOutputKeyVLM:
    """Fake VLM that emits both the requested output key and material metadata."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        payload = {
            "classification": "fixture housing",
            "material": "metal",
            "physical_properties": {"density": 7850},
        }
        return f"<answer>\n{json.dumps(payload)}\n</answer>"


class _UnknownSentinelVLM:
    """Fake VLM that returns an unstructured unknown sentinel."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return "__UNKNOWN__"

    async def agenerate(self, *args, **kwargs):
        return "__UNKNOWN__"


class _QuotedUnknownSentinelVLM:
    """Fake VLM that returns a quoted sentinel literal."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return '"__UNKNOWN__"'


class _AnswerUnknownSentinelVLM:
    """Fake VLM that returns the sentinel in a non-JSON answer block."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return "<answer>__UNKNOWN__</answer>"

    async def agenerate(self, *args, **kwargs):
        return "<answer>__UNKNOWN__</answer>"


class _NegatedUnknownJsonVLM:
    """Fake VLM that mentions the sentinel while choosing a concrete material."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return '{"material": "Not __UNKNOWN__, it is Steel"}'

    async def agenerate(self, *args, **kwargs):
        return '{"material": "Not __UNKNOWN__, it is Steel"}'


class _MaterialParserLLM:
    """Fake parser LLM that returns a concrete material."""

    def invoke(self, *args, **kwargs):
        class Response:
            content = '{"material": "Steel"}'

        return Response()


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


def test_classify_object_prefers_full_response_json_over_placeholder_answer():
    """A stale answer placeholder should not override valid JSON in the response."""
    result = classify_object(
        vlm=_JsonThenPlaceholderAnswerVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
    )

    assert result["material"] == "Steel Painted Black"
    assert result["original_response"].endswith("<answer>your answer</answer>")


def test_classify_object_preserves_structured_answer_json_for_custom_output_key():
    """Structured VLM answer JSON should not collapse to a single string."""
    result = classify_object(
        vlm=_PhysicsStructuredAnswerVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "metal"
    assert result["asset_type"] == "light fixture"
    assert result["component_type"] == "housing"
    assert result["component_name"] == "lamp housing"
    assert result["physical_properties"]["density"] == 7850
    assert result["confidence"] == "high"
    assert result["reasoning"] == "Visible metallic shell around the bulb."
    assert result["original_response"].startswith("<answer>")
    assert "material" not in result


def test_classify_object_preserves_material_when_output_key_already_exists():
    """A sibling material field should not overwrite the requested output key."""
    result = classify_object(
        vlm=_StructuredAnswerWithRequestedOutputKeyVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "fixture housing"
    assert result["material"] == "metal"
    assert result["physical_properties"]["density"] == 7850


def test_classify_object_preserves_unknown_sentinel_without_parser_fallback():
    """An explicit configured sentinel should not be replaced by LLM guessing."""
    result = classify_object(
        vlm=_UnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "__UNKNOWN__"
    assert "__UNKNOWN__" in result["original_response"]


def test_classify_object_canonicalizes_sentinel_answer_block():
    """Answer blocks with the exact configured sentinel should become sentinel."""
    result = classify_object(
        vlm=_AnswerUnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "__UNKNOWN__"


def test_classify_object_does_not_canonicalize_negated_sentinel_value():
    """JSON values that merely reference the sentinel should stay intact."""
    result = classify_object(
        vlm=_NegatedUnknownJsonVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "Not __UNKNOWN__, it is Steel"


def test_classify_object_preserves_configured_quoted_sentinel_literal():
    """A quoted sentinel config should not be stripped during comparison."""
    result = classify_object(
        vlm=_QuotedUnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel='"__UNKNOWN__"',
    )

    assert result["material"] == '"__UNKNOWN__"'


def test_classify_object_does_not_strip_quotes_from_configured_sentinel():
    """An unquoted VLM value should not match a quoted configured sentinel."""
    result = classify_object(
        vlm=_UnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=_MaterialParserLLM(),
        output_key="material",
        max_retries=1,
        unknown_sentinel='"__UNKNOWN__"',
    )

    assert result["material"] == "Steel"


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


@pytest.mark.asyncio
async def test_async_classify_object_preserves_structured_answer_json_for_custom_output_key():
    """Async classification should preserve structured VLM answer JSON too."""
    result = await async_classify_object(
        vlm=_PhysicsStructuredAnswerVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "metal"
    assert result["asset_type"] == "light fixture"
    assert result["component_type"] == "housing"
    assert result["component_name"] == "lamp housing"
    assert result["physical_properties"]["density"] == 7850
    assert result["confidence"] == "high"
    assert result["reasoning"] == "Visible metallic shell around the bulb."
    assert result["original_response"].startswith("<answer>")
    assert "material" not in result


@pytest.mark.asyncio
async def test_async_classify_object_preserves_unknown_sentinel_without_fallback():
    """Async single-object classification should preserve explicit sentinels too."""
    result = await async_classify_object(
        vlm=_UnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "__UNKNOWN__"
    assert "__UNKNOWN__" in result["original_response"]


@pytest.mark.asyncio
async def test_async_classify_object_canonicalizes_sentinel_answer_block():
    """Async parser should canonicalize exact sentinel answer text."""
    result = await async_classify_object(
        vlm=_AnswerUnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "__UNKNOWN__"


@pytest.mark.asyncio
async def test_async_classify_object_does_not_canonicalize_negated_sentinel_value():
    """Async JSON values that merely reference the sentinel should stay intact."""
    result = await async_classify_object(
        vlm=_NegatedUnknownJsonVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "Not __UNKNOWN__, it is Steel"
