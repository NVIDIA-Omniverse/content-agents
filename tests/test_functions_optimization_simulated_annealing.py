# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for simulated_annealing function."""

import math
from typing import Any

import pytest

from world_understanding.functions.optimization import simulated_annealing


def rastrigin(**ctx: Any) -> float:
    x = ctx["x"]
    return float(10 * len(x) + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x))


def bowl(**ctx: Any) -> float:
    x = ctx["x"]
    return float(sum(xi**2 for xi in x))


def test_simulated_annealing_output_keys() -> None:
    result = simulated_annealing(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.5)
    assert set(result.keys()) >= {"best_value", "best_x", "n_evals", "elapsed"}


def test_simulated_annealing_output_types() -> None:
    result = simulated_annealing(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.5)
    assert isinstance(result["best_value"], float)
    assert isinstance(result["best_x"], list)
    assert isinstance(result["n_evals"], int)
    assert isinstance(result["elapsed"], float)
    assert len(result["best_x"]) == 2


def test_simulated_annealing_convergence_bowl() -> None:
    result = simulated_annealing(bowl, (-1.0, 1.0), n_dims=2, time_budget=1.0)
    assert result["best_value"] < 0.01


def test_simulated_annealing_evals_positive() -> None:
    result = simulated_annealing(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.5)
    assert result["n_evals"] >= 1


def test_simulated_annealing_elapsed_within_budget() -> None:
    result = simulated_annealing(bowl, (-1.0, 1.0), n_dims=2, time_budget=1.0)
    assert result["elapsed"] < 2.0


def test_simulated_annealing_seed_convergence() -> None:
    # SA with a fixed seed should converge to a small value on the bowl
    result = simulated_annealing(bowl, (-1.0, 1.0), n_dims=2, time_budget=1.0, seed=0)
    assert result["best_value"] < 0.01


def test_simulated_annealing_rastrigin() -> None:
    result = simulated_annealing(rastrigin, (-5.12, 5.12), n_dims=2, time_budget=1.0)
    assert result["best_value"] < 10.0


def test_simulated_annealing_zero_time_budget() -> None:
    with pytest.raises(ValueError, match="time_budget must be > 0"):
        simulated_annealing(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.0)


def test_simulated_annealing_negative_time_budget() -> None:
    with pytest.raises(ValueError, match="time_budget must be > 0"):
        simulated_annealing(bowl, (-1.0, 1.0), n_dims=2, time_budget=-1.0)


def test_simulated_annealing_invalid_n_dims() -> None:
    with pytest.raises(ValueError, match="n_dims must be > 0"):
        simulated_annealing(bowl, (-1.0, 1.0), n_dims=0, time_budget=0.5)


def test_simulated_annealing_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="bounds must have lower < upper"):
        simulated_annealing(bowl, (1.0, -1.0), n_dims=2, time_budget=0.5)


def test_simulated_annealing_invalid_temps() -> None:
    with pytest.raises(ValueError, match="temps must satisfy"):
        simulated_annealing(
            bowl,
            (-1.0, 1.0),
            n_dims=2,
            time_budget=0.5,
            temp_init=1.0,
            temp_final=2.0,
        )


def test_simulated_annealing_invalid_step_size() -> None:
    with pytest.raises(ValueError, match="step_size must be > 0"):
        simulated_annealing(
            bowl,
            (-1.0, 1.0),
            n_dims=2,
            time_budget=0.5,
            step_size=-0.1,
        )
