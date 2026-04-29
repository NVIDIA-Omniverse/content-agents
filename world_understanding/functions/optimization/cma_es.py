# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for blackbox optimization."""

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def cma_es(
    evaluate: Callable[..., float],
    bounds: tuple[float, float],
    n_dims: int,
    time_budget: float,
    sigma_init: float = 2.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Full (μ_w, λ) CMA-ES with default hyperparameters from Hansen 2016.

    Args:
        evaluate: Callable accepting keyword argument ``x`` (numpy array) returning float.
        bounds: (lower, upper) bounds applied uniformly to all dimensions.
        n_dims: Dimensionality of the search space.
        time_budget: Wall-clock seconds to run.
        sigma_init: Initial step size (standard deviation).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: best_value, best_x, n_evals, elapsed, generations.
    """
    if time_budget <= 0:
        raise ValueError(f"time_budget must be > 0, got {time_budget}")
    if n_dims <= 0:
        raise ValueError(f"n_dims must be > 0, got {n_dims}")
    if bounds[0] >= bounds[1]:
        raise ValueError(f"bounds must have lower < upper, got {bounds}")

    rng = np.random.default_rng(seed)
    t_start = time.time()
    n = n_dims

    # Population size and recombination weights
    lam = 4 + int(3 * math.log(n))
    mu = lam // 2
    weights_raw = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights_raw / weights_raw.sum()
    mueff = 1.0 / (weights**2).sum()

    # Step-size control constants
    cs = (mueff + 2) / (n + mueff + 5)
    ds = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    # Covariance matrix adaptation constants
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

    # State
    xmean = rng.uniform(*bounds, n)
    sigma = sigma_init
    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    invsqrtC = np.eye(n)
    eigeneval = 0
    counteval = 0

    best_val = evaluate(x=xmean)
    best_x = xmean.copy()
    counteval += 1
    gen = 0

    while time.time() - t_start < time_budget:
        # Generate and evaluate offspring
        arz = rng.standard_normal((lam, n))
        arx = xmean + sigma * ((arz * D) @ B.T)
        arx = np.clip(arx, *bounds)

        fitvals = np.array([evaluate(x=x) for x in arx])
        counteval += lam

        # Select by fitness
        arindex = np.argsort(fitvals)
        xbest = arx[arindex[0]]
        if fitvals[arindex[0]] < best_val:
            best_val = fitvals[arindex[0]]
            best_x = xbest.copy()

        # Update mean
        xold = xmean.copy()
        xmean = weights @ arx[arindex[:mu]]

        # Step-size control via cumulative path length
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (
            xmean - xold
        ) / sigma
        hsig = np.linalg.norm(ps) / math.sqrt(
            1 - (1 - cs) ** (2 * counteval / lam)
        ) / chiN < 1.4 + 2 / (n + 1)
        sigma *= math.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, sigma_init)  # prevent explosion

        # Covariance matrix adaptation
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (
            xmean - xold
        ) / sigma
        artmp = (arx[arindex[:mu]] - xold) / sigma
        C = (
            (1 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            + cmu
            * sum(w * np.outer(av, av) for w, av in zip(weights, artmp, strict=False))
        )

        # Eigendecomposition periodically
        if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            invsqrtC = B @ np.diag(1.0 / D) @ B.T

        gen += 1

        # Restart from best if stagnated
        if sigma < 1e-6:
            sigma = sigma_init * 0.25
            xmean = best_x + rng.normal(0, 0.1, n)
            xmean = np.clip(xmean, *bounds)
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)

    elapsed = time.time() - t_start
    return {
        "best_value": float(best_val),
        "best_x": best_x.tolist(),
        "n_evals": counteval,
        "elapsed": elapsed,
        "generations": gen,
    }
