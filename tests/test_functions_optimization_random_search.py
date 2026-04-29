# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for random_search function."""

import math
from typing import Any

import pytest

from world_understanding.functions.optimization import random_search


def rastrigin(**ctx: Any) -> float:
    x = ctx["x"]
    return float(10 * len(x) + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x))


def bowl(**ctx: Any) -> float:
    """Simple bowl: f(x) = sum(xi^2), minimum at 0."""
    x = ctx["x"]
    return float(sum(xi**2 for xi in x))


def test_random_search_output_keys() -> None:
    result = random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.5)
    assert set(result.keys()) >= {"best_value", "best_x", "n_evals", "elapsed"}


def test_random_search_output_types() -> None:
    result = random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.5)
    assert isinstance(result["best_value"], float)
    assert isinstance(result["best_x"], list)
    assert isinstance(result["n_evals"], int)
    assert isinstance(result["elapsed"], float)
    assert len(result["best_x"]) == 2


def test_random_search_convergence_bowl() -> None:
    result = random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=1.0)
    assert result["best_value"] < 0.1


def test_random_search_evals_positive() -> None:
    result = random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.5)
    assert result["n_evals"] >= 1


def test_random_search_elapsed_within_budget() -> None:
    result = random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=1.0)
    assert result["elapsed"] < 2.0


def test_random_search_seed_reproducible() -> None:
    r1 = random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.3, seed=0)
    r2 = random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.3, seed=0)
    assert r1["best_value"] == pytest.approx(r2["best_value"])


def test_random_search_rastrigin() -> None:
    result = random_search(rastrigin, (-5.12, 5.12), n_dims=2, time_budget=1.0)
    # Random search should find something below the mean value (~20 for n=2)
    assert result["best_value"] < 15.0


def test_random_search_zero_time_budget() -> None:
    with pytest.raises(ValueError, match="time_budget must be > 0"):
        random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=0.0)


def test_random_search_negative_time_budget() -> None:
    with pytest.raises(ValueError, match="time_budget must be > 0"):
        random_search(bowl, (-1.0, 1.0), n_dims=2, time_budget=-1.0)


def test_random_search_invalid_n_dims() -> None:
    with pytest.raises(ValueError, match="n_dims must be > 0"):
        random_search(bowl, (-1.0, 1.0), n_dims=0, time_budget=0.5)


def test_random_search_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="bounds must have lower < upper"):
        random_search(bowl, (1.0, -1.0), n_dims=2, time_budget=0.5)
