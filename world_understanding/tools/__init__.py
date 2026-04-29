# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tools module for World Understanding."""

from . import base, cv, graphics, knowledge, nlp
from .base import (
    ExecutionPolicy,
    Tool,
    ToolInput,
    ToolOutput,
    ToolSpec,
    get_tool,
    get_tool_registry,
    register_tool,
)

# Import all tools to trigger registration
from .cv import find_similar_color_tool, get_dominant_colors_tool, vlm_tool
from .graphics import image_edit_tool
from .knowledge import extract_document_content_tool, split_document_content_tool
from .nlp import chat_tool

__all__ = [
    # Base classes and functions
    "Tool",
    "ToolInput",
    "ToolOutput",
    "ToolSpec",
    "ExecutionPolicy",
    "register_tool",
    "get_tool",
    "get_tool_registry",
    # CV Tools
    "get_dominant_colors_tool",
    "find_similar_color_tool",
    "vlm_tool",
    # NLP Tools
    "chat_tool",
    # Graphics Tools
    "image_edit_tool",
    # Knowledge Tools
    "extract_document_content_tool",
    "split_document_content_tool",
    # Module references
    "base",
    "cv",
    "graphics",
    "knowledge",
    "nlp",
]


def register_all_tools() -> dict[str, Tool]:
    """Get all registered tools and register their display functions.

    This function returns all tools that have been registered via the
    @register_tool decorator. Tools are automatically registered when
    their modules are imported.

    It also registers any display functions attached to the tools.

    Returns:
        dict: A dictionary mapping tool names to Tool instances
    """
    from world_understanding.registry import get_display_registry
    from world_understanding.registry import (
        get_tool_registry as get_registry_wrapper,
    )

    from .base import get_tool_registry as get_tools_dict

    # Initialize the registry wrapper (for side effects)
    get_registry_wrapper()
    display_registry = get_display_registry()

    # Get the actual tools dictionary from the base registry
    tools = get_tools_dict()

    # Register display functions for tools that have them
    for tool_name, tool in tools.items():
        # Check if the tool has a display function attached
        display_func = tool.get_display_function()
        if display_func:
            display_registry.register(tool_name, display_func)

    return tools
