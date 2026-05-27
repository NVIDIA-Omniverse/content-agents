# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for optimizer dispatch + BoTorch availability handling."""

from __future__ import annotations

import pytest

from physics_agent.tuning import optimizers
from physics_agent.tuning.errors import BoTorchUnavailableError
from physics_agent.tuning.optimizers import (
    OPTIMIZER_AUTO,
    OPTIMIZER_BOTORCH,
    OPTIMIZER_CMA_ES,
    OPTIMIZER_RANDOM,
    SUPPORTED_OPTIMIZERS,
    get_runner,
    resolve_optimizer,
    run_cma_es_optimizer,
    run_random_optimizer,
)
from physics_agent.tuning.scenario import parse_scenario


def _scenario_2d():
    return parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "mass_scale", "min": 0.5, "max": 2.0},
                {"name": "static_friction", "min": 0.0, "max": 1.0},
            ],
        }
    )


def test_supported_optimizers_canonical_set() -> None:
    assert OPTIMIZER_AUTO in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_BOTORCH in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_RANDOM in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_CMA_ES in SUPPORTED_OPTIMIZERS


def test_resolve_random_passthrough() -> None:
    assert resolve_optimizer(OPTIMIZER_RANDOM) == OPTIMIZER_RANDOM


def test_resolve_cma_es_passthrough() -> None:
    assert resolve_optimizer(OPTIMIZER_CMA_ES) == OPTIMIZER_CMA_ES


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        resolve_optimizer("annealing")


def test_resolve_auto_when_botorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto must hard-error to BoTorchUnavailableError — no silent random fallback."""
    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    with pytest.raises(BoTorchUnavailableError) as ei:
        resolve_optimizer(OPTIMIZER_AUTO)
    msg = str(ei.value)
    # Exact install hint must be surfaced — part of the issue Acceptance Criteria.
    assert "BoTorch optimizer requires the tuning extra" in msg
    assert 'uv pip install -e "apps/physics_agent[tuning]"' in msg


def test_resolve_botorch_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    with pytest.raises(BoTorchUnavailableError):
        resolve_optimizer(OPTIMIZER_BOTORCH)


def test_resolve_auto_when_botorch_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: True)
    assert resolve_optimizer(OPTIMIZER_AUTO) == OPTIMIZER_BOTORCH


def test_random_optimizer_runs_max_trials() -> None:
    sc = _scenario_2d()
    calls: list[dict[str, float]] = []

    def evaluate(params: dict[str, float]) -> float:
        calls.append(dict(params))
        return 0.0

    run_random_optimizer(sc, evaluate, max_trials=7, seed=1)
    assert len(calls) == 7
    for params in calls:
        assert 0.5 <= params["mass_scale"] <= 2.0
        assert 0.0 <= params["static_friction"] <= 1.0


def test_random_optimizer_reproducible_for_same_seed() -> None:
    sc = _scenario_2d()
    a: list[float] = []
    b: list[float] = []
    run_random_optimizer(
        sc, lambda p: (a.append(p["mass_scale"]), 0.0)[1], max_trials=3, seed=42
    )
    run_random_optimizer(
        sc, lambda p: (b.append(p["mass_scale"]), 0.0)[1], max_trials=3, seed=42
    )
    assert a == b


def test_random_optimizer_respects_cancel_check() -> None:
    sc = _scenario_2d()
    calls: list[dict[str, float]] = []

    def evaluate(params: dict[str, float]) -> float:
        calls.append(dict(params))
        return 0.0

    cancelled = {"v": False}

    def cancel_check() -> bool:
        return cancelled["v"]

    # Cancel after the first call.
    def evaluate_then_cancel(params: dict[str, float]) -> float:
        calls.append(dict(params))
        cancelled["v"] = True
        return 0.0

    run_random_optimizer(
        sc, evaluate_then_cancel, max_trials=20, seed=7, cancel_check=cancel_check
    )
    # Exactly one trial completed before cancel was observed at the top of
    # the next iteration.
    assert len(calls) == 1


def test_cma_es_optimizer_respects_max_trials_budget() -> None:
    sc = _scenario_2d()
    calls: list[dict[str, float]] = []

    def evaluate(params: dict[str, float]) -> float:
        calls.append(dict(params))
        # Decreasing function so CMA-ES has signal to converge.
        return params["mass_scale"] ** 2 + params["static_friction"] ** 2

    run_cma_es_optimizer(sc, evaluate, max_trials=8, seed=5)
    # Allow CMA-ES to perform an initial-mean evaluation and then up to
    # max_trials total, but never more.
    assert 1 <= len(calls) <= 8


def test_get_runner_returns_correct_callable() -> None:
    assert get_runner(OPTIMIZER_RANDOM).__name__ == "run_random_optimizer"
    assert get_runner(OPTIMIZER_CMA_ES).__name__ == "run_cma_es_optimizer"
    assert get_runner(OPTIMIZER_BOTORCH).__name__ == "run_botorch_optimizer"


def test_get_runner_rejects_auto() -> None:
    # Caller should resolve `auto` first; passing it through is a programming
    # error.
    with pytest.raises(ValueError, match="No runner"):
        get_runner(OPTIMIZER_AUTO)


def test_run_botorch_when_missing_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When BoTorch is missing, run_botorch_optimizer raises the install hint.

    This is the authoritative test that proves the Acceptance Criteria:
    ``--optimizer botorch`` must NEVER silently fall back to random.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name in ("torch", "botorch") or name.startswith(
            ("torch.", "botorch.", "gpytorch")
        ):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    sc = _scenario_2d()
    with pytest.raises(BoTorchUnavailableError) as ei:
        optimizers.run_botorch_optimizer(sc, lambda p: 0.0, max_trials=3, seed=0)
    assert "BoTorch optimizer requires the tuning extra" in str(ei.value)
