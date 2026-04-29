# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes for the agentic framework."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from world_understanding.utils.object_store import ObjectStore


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str = "BaseAgent", description: str = ""):
        """
        Initialize the agent.

        Args:
            name: Agent name for identification
            description: Human-readable description of agent's purpose
        """
        self.name = name
        self.description = description

    @abstractmethod
    def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute the agent's logic for the given task synchronously.

        Args:
            task: Task identifier or description
            context: Workflow context and state
            object_store: Storage for artifacts

        Returns:
            Updated context with results
        """
        pass

    async def arun(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute the agent's logic for the given task asynchronously.

        Default implementation delegates to sync run() via asyncio.to_thread.
        Subclasses can override for true async behavior.

        Args:
            task: Task identifier or description
            context: Workflow context and state
            object_store: Storage for artifacts

        Returns:
            Updated context with results
        """
        return await asyncio.to_thread(self.run, task, context, object_store)

    def can_handle(self, task: str) -> bool:
        """
        Check if this agent can handle the given task.

        Args:
            task: Task identifier or description

        Returns:
            True if agent can handle the task
        """
        # Default implementation - override in subclasses
        return True
