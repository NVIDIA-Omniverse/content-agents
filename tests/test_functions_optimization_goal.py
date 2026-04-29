# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Goal ABC."""

from typing import Any

import pytest

from world_understanding.functions.optimization import Goal


class MinGoal(Goal):
    metric_name = "loss"
    metric_direction = "minimize"
    time_budget = 1.0
    n_dims = 2
    bounds = (-1.0, 1.0)

    def evaluate(self, **context: Any) -> float:
        x = context["x"]
        return float(sum(xi**2 for xi in x))


class MaxGoal(Goal):
    metric_name = "score"
    metric_direction = "maximize"
    time_budget = 1.0
    n_dims = 2
    bounds = (-1.0, 1.0)

    def evaluate(self, **context: Any) -> float:
        x = context["x"]
        return float(-sum(xi**2 for xi in x))


def test_goal_instantiation() -> None:
    g = MinGoal()
    assert g.metric_name == "loss"
    assert g.metric_direction == "minimize"
    assert g.time_budget == 1.0
    assert g.n_dims == 2
    assert g.bounds == (-1.0, 1.0)


def test_goal_evaluate() -> None:
    g = MinGoal()
    val = g.evaluate(x=[1.0, 0.0])
    assert val == pytest.approx(1.0)


def test_goal_is_improvement_minimize() -> None:
    g = MinGoal()
    assert g.is_improvement(0.5, 1.0) is True
    assert g.is_improvement(1.5, 1.0) is False
    assert g.is_improvement(1.0, 1.0) is False


def test_goal_is_improvement_maximize() -> None:
    g = MaxGoal()
    assert g.is_improvement(1.5, 1.0) is True
    assert g.is_improvement(0.5, 1.0) is False
    assert g.is_improvement(1.0, 1.0) is False


def test_goal_abc_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        Goal()  # type: ignore[abstract]
