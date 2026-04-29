# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple agent implementation for direct tool execution."""

from typing import Any

from world_understanding.agentic.base import BaseAgent
from world_understanding.tools.base import get_tool_registry
from world_understanding.utils.object_store import ObjectStore


class SimpleAgent(BaseAgent):
    """
    Simple agent that executes tools directly as specified.

    This agent provides direct tool execution without routing logic,
    useful for deterministic workflows where the tool sequence is known.
    """

    def __init__(
        self,
        tools: dict[str, Any] | None = None,
        name: str = "SimpleAgent",
        description: str = "Direct tool execution agent",
    ):
        """
        Initialize the simple agent.

        Args:
            tools: Tool registry (uses global registry if None)
            name: Agent name
            description: Agent description
        """
        super().__init__(name, description)
        self.tools = tools or get_tool_registry()

    def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute a tool directly.

        The task should be in format "tool_name" or "tool_name:params_key"
        where params_key is a key in context containing the tool parameters.

        Args:
            task: Tool name or "tool_name:params_key"
            context: Workflow context
            object_store: Storage for artifacts

        Returns:
            Updated context with tool results
        """
        if context is None:
            context = {}

        # Parse task
        if ":" in task:
            tool_name, params_key = task.split(":", 1)
        else:
            tool_name = task
            params_key = f"{tool_name}_params"

        # Get tool
        tool = self.tools.get(tool_name)
        if not tool:
            context["error"] = f"Tool '{tool_name}' not found"
            return context

        # Get parameters from context
        params = context.get(params_key, {})

        try:
            # Create input object
            input_obj = tool.spec.input_model(**params)

            # Execute tool
            output = tool.run(input_obj)

            # Store results in context
            context[f"{tool_name}_output"] = output.model_dump()
            context[f"{tool_name}_success"] = True

            # Store in object store if provided
            if object_store:
                object_store.set(f"{tool_name}_output", output.model_dump())

        except Exception as e:
            context[f"{tool_name}_error"] = str(e)
            context[f"{tool_name}_success"] = False

        return context
