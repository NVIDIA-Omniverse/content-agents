# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracing decorators for automatic instrumentation.

This module provides reusable decorators for adding OpenTelemetry tracing
to functions with minimal code changes. It supports both sync and async
functions and includes specialized decorators for LLM/VLM operations.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .attributes import GenAIAttributes, MAAttributes

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

MAX_ATTRIBUTE_LENGTH = 4096  # Truncate large values


def _truncate(value: Any, max_length: int = MAX_ATTRIBUTE_LENGTH) -> str:
    """Truncate value to max length for span attributes.

    Args:
        value: Value to truncate.
        max_length: Maximum string length.

    Returns:
        Truncated string representation.
    """
    s = str(value)
    if len(s) > max_length:
        return s[:max_length] + "...[truncated]"
    return s


def _safe_set_attribute(span: trace.Span, key: str, value: Any) -> None:
    """Safely set a span attribute, handling None and conversion errors.

    Args:
        span: The span to set the attribute on.
        key: Attribute key.
        value: Attribute value.
    """
    if value is None:
        return
    try:
        # OpenTelemetry accepts str, bool, int, float, and sequences of these
        if isinstance(value, str | bool | int | float):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, _truncate(value))
    except Exception as e:
        logger.debug(f"Failed to set span attribute {key}: {e}")


def traced(
    name: str | None = None,
    *,
    span_type: str = "span",
    capture_input: bool = False,
    capture_output: bool = False,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace function execution with OpenTelemetry.

    This decorator automatically creates spans for function calls, capturing
    timing information and optionally function inputs/outputs. It works with
    both synchronous and asynchronous functions.

    Args:
        name: Span name (defaults to function name).
        span_type: Type hint for the span (e.g., "generation", "span").
        capture_input: Whether to capture function arguments.
        capture_output: Whether to capture return value.
        attributes: Static attributes to add to span.

    Returns:
        Decorated function with tracing instrumentation.

    Example:
        @traced("llm.chat", capture_input=True, capture_output=True)
        async def chat_completion(...):
            ...

        @traced("process_data")
        def sync_function(data):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__name__
        tracer = trace.get_tracer(__name__)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("observation.type", span_type)
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                if attributes:
                    for k, v in attributes.items():
                        _safe_set_attribute(span, k, v)

                if capture_input:
                    span.set_attribute("input.args", _truncate(repr(args)))
                    span.set_attribute("input.kwargs", _truncate(repr(kwargs)))

                try:
                    result = await func(*args, **kwargs)
                    if capture_output:
                        span.set_attribute("output", _truncate(repr(result)))
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("observation.type", span_type)
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                if attributes:
                    for k, v in attributes.items():
                        _safe_set_attribute(span, k, v)

                if capture_input:
                    span.set_attribute("input.args", _truncate(repr(args)))
                    span.set_attribute("input.kwargs", _truncate(repr(kwargs)))

                try:
                    result = func(*args, **kwargs)
                    if capture_output:
                        span.set_attribute("output", _truncate(repr(result)))
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def traced_llm(
    name: str | None = None,
    *,
    system: str = "unknown",
    operation: str = "chat",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Specialized decorator for LLM calls with GenAI semantic conventions.

    This decorator wraps LLM function calls and automatically extracts and
    records GenAI-specific attributes like model name, temperature, and
    token usage from function parameters and return values.

    Args:
        name: Span name (defaults to "llm.{operation}").
        system: The GenAI system/provider (e.g., "openai", "anthropic", "nim").
        operation: Type of GenAI operation (e.g., "chat", "completion").

    Returns:
        Decorated function with LLM-specific tracing instrumentation.

    Example:
        @traced_llm(system="openai", operation="chat")
        def create_chat_model(service, api_key, model=None, temperature=None, ...):
            ...

        @traced_llm(system="nim", operation="chat")
        async def nim_chat_completion(prompt, model, temperature=0.7):
            ...
    """
    span_name = name or f"llm.{operation}"

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        tracer = trace.get_tracer(__name__)

        def _extract_llm_attributes(kwargs: dict[str, Any]) -> dict[str, Any]:
            """Extract LLM-specific attributes from function kwargs."""
            attrs: dict[str, Any] = {
                GenAIAttributes.SYSTEM: system,
                GenAIAttributes.OPERATION_NAME: operation,
            }

            # Extract model name from various possible parameter names
            model = kwargs.get("model") or kwargs.get("model_id")
            if model:
                attrs[GenAIAttributes.REQUEST_MODEL] = model

            # Extract temperature
            temperature = kwargs.get("temperature")
            if temperature is not None:
                attrs[GenAIAttributes.REQUEST_TEMPERATURE] = temperature

            # Extract max_tokens from various possible parameter names
            max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
            if max_tokens is not None:
                attrs[GenAIAttributes.REQUEST_MAX_TOKENS] = max_tokens

            return attrs

        def _set_llm_attributes(span: trace.Span, kwargs: dict[str, Any]) -> None:
            """Set LLM attributes on the span."""
            attrs = _extract_llm_attributes(kwargs)
            for key, value in attrs.items():
                _safe_set_attribute(span, key, value)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("observation.type", "generation")
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Set LLM-specific attributes from kwargs
                _set_llm_attributes(span, dict(kwargs))

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("observation.type", "generation")
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Set LLM-specific attributes from kwargs
                _set_llm_attributes(span, dict(kwargs))

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def traced_vlm(
    name: str | None = None,
    *,
    system: str = "unknown",
    operation: str = "generate",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Specialized decorator for VLM (Vision-Language Model) calls.

    This decorator extends traced_llm with VLM-specific attribute extraction,
    including image count tracking.

    Args:
        name: Span name (defaults to "vlm.{operation}").
        system: The GenAI system/provider (e.g., "openai", "nim", "gradio").
        operation: Type of VLM operation (e.g., "generate", "generate_with_pairs").

    Returns:
        Decorated function with VLM-specific tracing instrumentation.

    Example:
        @traced_vlm(system="azure_openai", operation="generate")
        def generate(self, prompt, images=None, temperature=None, max_tokens=None):
            ...
    """
    span_name = name or f"vlm.{operation}"

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        tracer = trace.get_tracer(__name__)

        def _extract_vlm_attributes(
            args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            """Extract VLM-specific attributes from function args/kwargs."""
            attrs: dict[str, Any] = {
                GenAIAttributes.SYSTEM: system,
                GenAIAttributes.OPERATION_NAME: operation,
            }

            # Extract model name - may be in kwargs or in self._model_name
            model = kwargs.get("model") or kwargs.get("model_id")
            if model:
                attrs[GenAIAttributes.REQUEST_MODEL] = model

            # Extract temperature
            temperature = kwargs.get("temperature")
            if temperature is not None:
                attrs[GenAIAttributes.REQUEST_TEMPERATURE] = temperature

            # Extract max_tokens
            max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
            if max_tokens is not None:
                attrs[GenAIAttributes.REQUEST_MAX_TOKENS] = max_tokens

            # Extract image count - images may be in kwargs or as positional arg
            images = kwargs.get("images")
            if images is not None:
                attrs[MAAttributes.VLM_IMAGE_COUNT] = (
                    len(images) if hasattr(images, "__len__") else 0
                )

            # Check for image_caption_pairs
            image_caption_pairs = kwargs.get("image_caption_pairs")
            if image_caption_pairs is not None:
                attrs[MAAttributes.VLM_IMAGE_COUNT] = (
                    len(image_caption_pairs)
                    if hasattr(image_caption_pairs, "__len__")
                    else 0
                )

            return attrs

        def _set_vlm_attributes(
            span: trace.Span, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            """Set VLM attributes on the span."""
            attrs = _extract_vlm_attributes(args, kwargs)
            for key, value in attrs.items():
                _safe_set_attribute(span, key, value)

            # Try to extract model name from self if it's a method call
            if args and hasattr(args[0], "_model_name"):
                _safe_set_attribute(
                    span, GenAIAttributes.REQUEST_MODEL, args[0]._model_name
                )
            if args and hasattr(args[0], "backend_name"):
                _safe_set_attribute(
                    span, MAAttributes.VLM_BACKEND, args[0].backend_name
                )

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("observation.type", "generation")
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Set VLM-specific attributes
                _set_vlm_attributes(span, args, dict(kwargs))

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("observation.type", "generation")
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Set VLM-specific attributes
                _set_vlm_attributes(span, args, dict(kwargs))

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def add_token_usage_to_span(
    span: trace.Span | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    model_name: str | None = None,
) -> None:
    """Add token usage information to a span.

    This helper function can be called after an LLM/VLM invocation to add
    token usage metrics to the current span.

    Args:
        span: The span to add attributes to, or None to get current span.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total tokens used.
        model_name: Model that generated the response.

    Example:
        with tracer.start_as_current_span("llm.chat") as span:
            response = model.invoke(messages)
            add_token_usage_to_span(
                span,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
    """
    if span is None:
        from . import get_current_span

        span = get_current_span()

    if span is None:
        return

    _safe_set_attribute(span, GenAIAttributes.USAGE_INPUT_TOKENS, input_tokens)
    _safe_set_attribute(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, output_tokens)
    _safe_set_attribute(span, GenAIAttributes.USAGE_TOTAL_TOKENS, total_tokens)
    if model_name:
        _safe_set_attribute(span, GenAIAttributes.RESPONSE_MODEL, model_name)
