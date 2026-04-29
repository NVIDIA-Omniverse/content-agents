# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base tool interface and core abstractions."""

import asyncio
import functools
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field

# Tracer for tool invocations
_tracer = trace.get_tracer(__name__)


class ToolInput(BaseModel):
    """Base class for tool inputs."""

    pass


class ToolOutput(BaseModel):
    """Base class for tool outputs."""

    pass


class ExecutionPolicy(BaseModel):
    """Execution policy for tools."""

    timeout_s: float = Field(default=60.0, description="Timeout in seconds")
    max_retries: int = Field(default=0, description="Maximum number of retries")
    device: str | None = Field(
        default=None,
        description="Device preference: cpu, cuda:0, mps, or None for auto",
    )


class ToolSpec(BaseModel):
    """Tool specification with metadata."""

    name: str = Field(..., description="Unique tool name")
    version: str = Field(..., description="Tool version")
    description: str = Field(..., description="Tool description")
    tags: list[str] = Field(default_factory=list, description="Tool tags for discovery")
    input_model: type[ToolInput] = Field(..., description="Input model class")
    output_model: type[ToolOutput] = Field(..., description="Output model class")
    policy: ExecutionPolicy = Field(
        default_factory=ExecutionPolicy, description="Execution policy"
    )


class Tool:
    """Wrapper class for registered tool functions."""

    def __init__(self, func: Callable, spec: ToolSpec):
        self.func = func
        self.spec = spec

    def run(self, inputs: ToolInput) -> ToolOutput:
        """Execute the tool function."""
        # Validate input
        if not isinstance(inputs, self.spec.input_model):
            inputs = self.spec.input_model.model_validate(inputs)

        # Execute function
        output = self.func(inputs)

        # Validate output
        if not isinstance(output, self.spec.output_model):
            output = self.spec.output_model.model_validate(output)

        return output

    async def arun(self, inputs: ToolInput) -> ToolOutput:
        """Execute the tool asynchronously.

        If the wrapped function is async, calls it directly.
        Otherwise, delegates to sync run() via asyncio.to_thread.
        """
        if asyncio.iscoroutinefunction(self.func):
            # Validate input
            if not isinstance(inputs, self.spec.input_model):
                inputs = self.spec.input_model.model_validate(inputs)
            # Call async function directly
            output = await self.func(inputs)
            # Validate output
            if not isinstance(output, self.spec.output_model):
                output = self.spec.output_model.model_validate(output)
            return output
        else:
            return await asyncio.to_thread(self.run, inputs)

    def validate_input(self, inputs: Any) -> ToolInput:
        """Validate and parse input."""
        return self.spec.input_model.model_validate(inputs)

    def validate_output(self, output: Any) -> ToolOutput:
        """Validate and parse output."""
        return self.spec.output_model.model_validate(output)

    def to_json_schema(self) -> dict:
        """Export tool specification as JSON Schema."""
        return {
            "name": self.spec.name,
            "version": self.spec.version,
            "description": self.spec.description,
            "input_schema": self.spec.input_model.model_json_schema(),
            "output_schema": self.spec.output_model.model_json_schema(),
        }

    def get_display_function(self) -> Callable | None:
        """Get the display function for this tool's outputs.

        The function should have signature: (outputs: Dict, console: Console, indent: str) -> None
        """
        # Check if the function has a display attribute set by decorator
        return getattr(self.func, "_display_function", None)


# Global tool registry
_TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    input_model: type[ToolInput] = None,
    output_model: type[ToolOutput] = None,
    tags: list[str] = None,
    policy: ExecutionPolicy = None,
):
    """
    Decorator to register a tool function with automatic tracing.

    Every tool invocation will create a span with:
    - tool name
    - tool description (truncated to 256 chars)
    - execution time (automatic via span)
    - errors (recorded as exceptions)

    Args:
        name: Unique tool identifier
        version: Semantic version string
        description: Human-readable description
        input_model: Pydantic model for inputs
        output_model: Pydantic model for outputs
        tags: Optional tags for discovery
        policy: Optional execution policy

    Returns:
        Decorated function that's registered as a tool
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name
        tool_desc = description or func.__doc__ or ""

        # Create tool specification
        spec = ToolSpec(
            name=tool_name,
            version=version,
            description=tool_desc,
            input_model=input_model or ToolInput,
            output_model=output_model or ToolOutput,
            tags=tags or [],
            policy=policy or ExecutionPolicy(),
        )

        # Create traced wrappers for both sync and async execution
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with _tracer.start_as_current_span(f"tool:{tool_name}") as span:
                span.set_attribute("maa.tool.name", tool_name)
                span.set_attribute("maa.tool.description", tool_desc[:256])

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with _tracer.start_as_current_span(f"tool:{tool_name}") as span:
                span.set_attribute("maa.tool.name", tool_name)
                span.set_attribute("maa.tool.description", tool_desc[:256])

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Select appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper

        # Preserve tool metadata on wrapper
        wrapper._tool_name = tool_name
        wrapper._tool_description = tool_desc

        # Create tool wrapper with traced function
        tool = Tool(wrapper, spec)

        # Register in global registry
        _TOOL_REGISTRY[tool_name] = tool

        # Add tool reference to wrapper for convenience
        wrapper._tool = tool

        # Return traced wrapper for direct use
        return wrapper

    return decorator


def get_tool_registry() -> dict[str, Tool]:
    """Get the global tool registry."""
    return _TOOL_REGISTRY


def get_tool(name: str) -> Tool | None:
    """Get a tool by name from the registry."""
    return _TOOL_REGISTRY.get(name)


def clear_registry():
    """Clear the tool registry (mainly for testing)."""
    _TOOL_REGISTRY.clear()
