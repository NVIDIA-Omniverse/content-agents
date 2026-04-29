# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow orchestration for task execution."""

import asyncio
import logging
from typing import Any

from opentelemetry.trace import Status, StatusCode

from world_understanding.agentic.tasks import Task
from world_understanding.telemetry import get_tracer
from world_understanding.utils.object_store import (
    InMemoryObjectStore,
    ObjectStore,
)

logger = logging.getLogger(__name__)


class Workflow:
    """
    Workflow orchestrator that executes tasks in sequence.

    More complex patterns (parallel execution, conditional branching) can be added as needed.
    """

    def __init__(
        self,
        tasks: list[Task] | None = None,
        object_store: ObjectStore | None = None,
        name: str = "Workflow",
        description: str = "",
    ):
        """
        Initialize the workflow.

        Args:
            tasks: List of tasks to execute in order
            object_store: Storage for artifacts (creates InMemoryObjectStore if None)
            name: Workflow name
            description: Workflow description
        """
        self.tasks = tasks or []
        self.name = name
        self.description = description

        if object_store is None:
            object_store = InMemoryObjectStore()
        self.object_store = object_store

    def run(self, initial_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the workflow synchronously.

        This is a wrapper around the async implementation for backward
        compatibility.

        Args:
            initial_context: Initial context for the workflow

        Returns:
            Final context after all tasks have executed
        """
        return asyncio.run(self.arun(initial_context))

    async def arun(
        self, initial_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute the workflow asynchronously.

        This is the core implementation. The sync run() delegates to this.

        Args:
            initial_context: Initial context for the workflow

        Returns:
            Final context after all tasks have executed
        """
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span(f"workflow.{self.name}") as workflow_span:
            context = initial_context or {}
            context["workflow_name"] = self.name
            workflow_span.set_attribute("workflow.name", self.name)
            workflow_span.set_attribute("workflow.task_count", len(self.tasks))

            for i, task in enumerate(self.tasks):
                task_name = getattr(task, "name", task.__class__.__name__)
                context["current_task"] = task_name
                context["task_index"] = i

                logger.info(f"Executing task {i + 1}/{len(self.tasks)}: {task_name}")

                with tracer.start_as_current_span(f"task.{task_name}") as task_span:
                    task_span.set_attribute("task.name", task_name)
                    task_span.set_attribute("task.index", i)

                    try:
                        # Execute task asynchronously
                        context = await task.arun(context, self.object_store)

                        # Check for early termination
                        if context.get("workflow_terminated", False):
                            logger.info(
                                f"Workflow terminated early at task {task_name}"
                            )
                            break

                    except Exception as e:
                        task_span.record_exception(e)
                        task_span.set_status(Status(StatusCode.ERROR, str(e)))
                        logger.error(f"Task {task_name} failed: {e}")
                        context["error"] = str(e)
                        context["failed_task"] = task_name
                        context["workflow_terminated"] = True
                        break

            context["workflow_completed"] = not context.get(
                "workflow_terminated", False
            )
            return context

    def add_task(self, task: Task) -> None:
        """Add a task to the workflow."""
        self.tasks.append(task)

    def clear_tasks(self) -> None:
        """Clear all tasks from the workflow."""
        self.tasks.clear()
