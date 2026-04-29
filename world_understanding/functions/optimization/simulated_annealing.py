# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulated annealing for blackbox optimization."""

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def simulated_annealing(
    evaluate: Callable[..., float],
    bounds: tuple[float, float],
    n_dims: int,
    time_budget: float,
    temp_init: float = 5.0,
    temp_final: float = 1e-4,
    step_size: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Simulated annealing over a bounded hypercube.

    Args:
        evaluate: Callable accepting keyword argument ``x`` (numpy array) returning float.
        bounds: (lower, upper) bounds applied uniformly to all dimensions.
        n_dims: Dimensionality of the search space.
        time_budget: Wall-clock seconds to run.
        temp_init: Starting temperature.
        temp_final: Final temperature (cooling target).
        step_size: Standard deviation of Gaussian perturbation.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: best_value, best_x, n_evals, elapsed.
    """
    if time_budget <= 0:
        raise ValueError(f"time_budget must be > 0, got {time_budget}")
    if n_dims <= 0:
        raise ValueError(f"n_dims must be > 0, got {n_dims}")
    if bounds[0] >= bounds[1]:
        raise ValueError(f"bounds must have lower < upper, got {bounds}")
    if temp_final <= 0 or temp_init <= temp_final:
        raise ValueError(
            "temps must satisfy 0 < temp_final < temp_init, "
            f"got temp_init={temp_init}, temp_final={temp_final}"
        )
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")

    rng = np.random.default_rng(seed)
    t_start = time.time()

    x = rng.uniform(*bounds, n_dims)
    val = evaluate(x=x)
    best_x = x.copy()
    best_val = val
    n_evals = 1

    while True:
        elapsed = time.time() - t_start
        if elapsed >= time_budget:
            break

        # Exponential cooling schedule
        frac = elapsed / time_budget
        temp = temp_init * (temp_final / temp_init) ** frac

        candidate = np.clip(x + rng.normal(0, step_size, n_dims), *bounds)
        candidate_val = evaluate(x=candidate)
        n_evals += 1

        delta = candidate_val - val
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            x = candidate
            val = candidate_val
            if val < best_val:
                best_val = val
                best_x = x.copy()

    elapsed = time.time() - t_start
    return {
        "best_value": float(best_val),
        "best_x": best_x.tolist(),
        "n_evals": n_evals,
        "elapsed": elapsed,
    }
