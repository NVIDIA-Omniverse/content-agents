# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool registry implementation."""

import logging

from world_understanding.tools.base import Tool
from world_understanding.tools.base import get_tool_registry as get_base_registry

from .display_registry import get_display_registry

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for discovering and managing tools.

    This is a wrapper around the base tool registry that provides
    additional functionality like display registration and tag-based search.
    """

    def __init__(self) -> None:
        # We'll use the base registry from tools/base.py
        pass

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        base_registry = get_base_registry()
        return base_registry.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        base_registry = get_base_registry()
        return list(base_registry.keys())

    def list_by_tag(self, tag: str) -> list[str]:
        """List tools that have a specific tag."""
        base_registry = get_base_registry()
        result = []
        for name, tool in base_registry.items():
            if tag in tool.spec.tags:
                result.append(name)
        return result

    def get_json_schemas(self) -> dict[str, dict]:
        """Get JSON schemas for all registered tools."""
        base_registry = get_base_registry()
        schemas = {}
        for name, tool in base_registry.items():
            schemas[name] = tool.to_json_schema()
        return schemas

    def register_display(self, tool_name: str, display_func) -> None:
        """Register a display function for a tool."""
        display_registry = get_display_registry()
        display_registry.register(tool_name, display_func)
        logger.info(f"Registered display function for tool: {tool_name}")


# Global registry instance
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry wrapper."""
    return _registry
