# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure random search for blackbox optimization."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def random_search(
    evaluate: Callable[..., float],
    bounds: tuple[float, float],
    n_dims: int,
    time_budget: float,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Random search over a bounded hypercube (minimization).

    Samples points uniformly at random and returns the one with the
    lowest objective value.  To maximize, negate the objective
    (e.g., ``lambda **ctx: -f(**ctx)``).

    Args:
        evaluate: Callable accepting keyword argument ``x``
            (numpy array) returning a float objective value
            (lower is better).
        bounds: (lower, upper) bounds applied uniformly to all
            dimensions.
        n_dims: Dimensionality of the search space.
        time_budget: Wall-clock seconds to run.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: best_value (minimum objective found),
        best_x, n_evals, elapsed.
    """
    if time_budget <= 0:
        raise ValueError(f"time_budget must be > 0, got {time_budget}")
    if n_dims <= 0:
        raise ValueError(f"n_dims must be > 0, got {n_dims}")
    if bounds[0] >= bounds[1]:
        raise ValueError(f"bounds must have lower < upper, got {bounds}")

    rng = np.random.default_rng(seed)
    t_start = time.time()

    best_x = rng.uniform(*bounds, n_dims)
    best_val = evaluate(x=best_x)
    n_evals = 1

    while time.time() - t_start < time_budget:
        x = rng.uniform(*bounds, n_dims)
        val = evaluate(x=x)
        n_evals += 1
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
