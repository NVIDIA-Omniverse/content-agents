# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import texture_agent.workflows.factory as factory


@dataclass
class _FakeTask:
    name: str
    description: str
    run_marker: str

    def run(self, context):
        context.setdefault("executed", []).append(self.run_marker)
        return context


def test_create_texture_pipeline_workflow_respects_order_and_filters(
    monkeypatch,
) -> None:
    monkeypatch.setattr(factory, "STEP_ORDER", ["one", "two", "three"])
    monkeypatch.setattr(
        factory,
        "_STEP_TASKS",
        {
            "one": lambda: _FakeTask("one", "first", "one"),
            "two": lambda: _FakeTask("two", "second", "two"),
            "three": lambda: _FakeTask("three", "third", "three"),
        },
    )

    tasks = factory.create_texture_pipeline_workflow(
        {"steps": {"two": {"enabled": False}}},
        skip=["three"],
    )

    assert [task.name for task in tasks] == ["one"]


def test_create_texture_pipeline_workflow_only_filter(monkeypatch) -> None:
    monkeypatch.setattr(factory, "STEP_ORDER", ["one", "two", "three"])
    monkeypatch.setattr(
        factory,
        "_STEP_TASKS",
        {
            "one": lambda: _FakeTask("one", "first", "one"),
            "two": lambda: _FakeTask("two", "second", "two"),
            "three": lambda: _FakeTask("three", "third", "three"),
        },
    )

    tasks = factory.create_texture_pipeline_workflow({}, only=["two", "three"])

    assert [task.name for task in tasks] == ["two", "three"]


def test_run_pipeline_dry_run_does_not_execute(monkeypatch) -> None:
    tasks = [_FakeTask("one", "first", "one"), _FakeTask("two", "second", "two")]
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: tasks,
    )

    context = {"executed": []}
    result = factory.run_pipeline(context, dry_run=True)

    assert result is context
    assert context["executed"] == []


def test_run_pipeline_executes_tasks_in_sequence(monkeypatch) -> None:
    tasks = [_FakeTask("one", "first", "one"), _FakeTask("two", "second", "two")]
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: tasks,
    )

    context = {"executed": []}
    result = factory.run_pipeline(context)

    assert result["executed"] == ["one", "two"]
