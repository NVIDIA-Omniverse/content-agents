# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async workflow execution."""

import asyncio

import pytest

from world_understanding.agentic.tasks import CallableTask, Task
from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import InMemoryObjectStore


class SimpleTask(Task):
    """A simple test task."""

    def __init__(self, name: str, value_to_add: int):
        self.name = name
        self.value_to_add = value_to_add

    def run(self, context, object_store=None):
        """Synchronous run - should delegate to async."""
        return asyncio.run(self.arun(context, object_store))

    async def arun(self, context, object_store=None):
        """Async implementation."""
        # Simulate some async work
        await asyncio.sleep(0.01)

        # Update context
        count = context.get("count", 0)
        context["count"] = count + self.value_to_add
        context[f"{self.name}_executed"] = True

        return context


def test_workflow_sync_execution():
    """Test that sync workflow execution still works."""
    # Create workflow with tasks
    workflow = Workflow(
        tasks=[
            SimpleTask("task1", 1),
            SimpleTask("task2", 2),
            SimpleTask("task3", 3),
        ],
        name="TestWorkflow",
    )

    # Run workflow synchronously
    result = workflow.run({"count": 0})

    # Verify results
    assert result["count"] == 6  # 1 + 2 + 3
    assert result["task1_executed"] is True
    assert result["task2_executed"] is True
    assert result["task3_executed"] is True
    assert result["workflow_completed"] is True


@pytest.mark.asyncio
async def test_workflow_async_execution():
    """Test that async workflow execution works."""
    # Create workflow with tasks
    workflow = Workflow(
        tasks=[
            SimpleTask("task1", 1),
            SimpleTask("task2", 2),
            SimpleTask("task3", 3),
        ],
        name="TestWorkflow",
    )

    # Run workflow asynchronously
    result = await workflow.arun({"count": 0})

    # Verify results
    assert result["count"] == 6  # 1 + 2 + 3
    assert result["task1_executed"] is True
    assert result["task2_executed"] is True
    assert result["task3_executed"] is True
    assert result["workflow_completed"] is True


@pytest.mark.asyncio
async def test_workflow_early_termination():
    """Test that workflow early termination works in async mode."""

    class TerminatingTask(Task):
        """A task that terminates the workflow."""

        def __init__(self, name: str):
            self.name = name

        def run(self, context, object_store=None):
            return asyncio.run(self.arun(context, object_store))

        async def arun(self, context, object_store=None):
            context[f"{self.name}_executed"] = True
            context["workflow_terminated"] = True
            return context

    # Create workflow
    workflow = Workflow(
        tasks=[
            SimpleTask("task1", 1),
            TerminatingTask("terminator"),
            SimpleTask("task2", 2),  # Should not execute
        ],
        name="TestWorkflow",
    )

    # Run workflow
    result = await workflow.arun({"count": 0})

    # Verify results
    assert result["count"] == 1  # Only task1 executed
    assert result["task1_executed"] is True
    assert result["terminator_executed"] is True
    assert result.get("task2_executed") is None  # Should not have executed
    assert result["workflow_completed"] is False


@pytest.mark.asyncio
async def test_callable_task_async():
    """Test that CallableTask works asynchronously."""

    def my_function(context, object_store):
        context["called"] = True
        context["value"] = 42
        return context

    task = CallableTask(my_function, name="MyCallable")

    # Test async execution
    result = await task.arun({})

    assert result["called"] is True
    assert result["value"] == 42


def test_workflow_with_object_store():
    """Test that object store works with async workflow."""

    class StoreTask(Task):
        """A task that uses object store."""

        def __init__(self, name: str, key: str, value: str):
            self.name = name
            self.key = key
            self.value = value

        def run(self, context, object_store=None):
            return asyncio.run(self.arun(context, object_store))

        async def arun(self, context, object_store=None):
            if object_store:
                object_store.set(self.key, self.value)
                context[f"{self.name}_stored"] = True
            return context

    # Create workflow with object store
    object_store = InMemoryObjectStore()
    workflow = Workflow(
        tasks=[
            StoreTask("task1", "key1", "value1"),
            StoreTask("task2", "key2", "value2"),
        ],
        object_store=object_store,
        name="StoreWorkflow",
    )

    # Run workflow
    result = workflow.run({})

    # Verify results
    assert result["task1_stored"] is True
    assert result["task2_stored"] is True
    assert object_store.get("key1") == "value1"
    assert object_store.get("key2") == "value2"
