# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry for tool output display formatters."""

from collections.abc import Callable
from typing import Any

from rich.console import Console

# Type alias for display functions
DisplayFunction = Callable[[dict[str, Any], Console, str], None]


class DisplayRegistry:
    """Registry for tool-specific output display functions."""

    def __init__(self) -> None:
        self._formatters: dict[str, DisplayFunction] = {}

    def register(self, tool_name: str, formatter: DisplayFunction) -> None:
        """Register a display formatter for a tool.

        Args:
            tool_name: Name of the tool
            formatter: Function that takes (outputs, console, indent) and
                      displays results
        """
        self._formatters[tool_name] = formatter

    def get_formatter(self, tool_name: str) -> DisplayFunction | None:
        """Get the display formatter for a tool."""
        return self._formatters.get(tool_name)

    def has_formatter(self, tool_name: str) -> bool:
        """Check if a tool has a registered formatter."""
        return tool_name in self._formatters

    def display(
        self,
        tool_name: str,
        outputs: dict[str, Any],
        console: Console,
        indent: str = "",
    ) -> bool:
        """Display tool outputs using registered formatter.

        Returns:
            True if a formatter was found and used, False otherwise
        """
        formatter = self.get_formatter(tool_name)
        if formatter:
            formatter(outputs, console, indent)
            return True
        return False


# Global display registry instance
_display_registry = DisplayRegistry()


def get_display_registry() -> DisplayRegistry:
    """Get the global display registry."""
    return _display_registry
