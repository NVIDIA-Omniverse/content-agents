# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``physics_agent.tuning.scenarios`` dispatch + capability gate.

Covers the contract that ``OvPhysXBackend.evaluate`` depends on:

* :data:`SUPPORTED_SCENARIOS_PER_ENGINE` advertises what each engine
  supports (the runner's ``_validate_engine_supports_scenario`` reads
  this).
* :func:`resolve` lazy-imports the per-scenario evaluator module and
  raises an actionable error for unknown kinds.

Mocks the daemon so the test never spawns ovphysx.
"""

from __future__ import annotations

import pytest

from physics_agent.tuning.scenarios import (
    SUPPORTED_SCENARIOS_PER_ENGINE,
    resolve,
)


def test_capability_map_advertises_both_kinds_for_ovphysx() -> None:
    # PR #43's broken adapter advertised drop_settle only. After the
    # daemon-based rewrite both kinds are real.
    assert SUPPORTED_SCENARIOS_PER_ENGINE["ovphysx"] == (
        "drop_settle",
        "freeform",
    )
    assert SUPPORTED_SCENARIOS_PER_ENGINE["fake"] == (
        "drop_settle",
        "freeform",
    )
    # Newton (added alongside the Simulator-protocol refactor) supports
    # the same scenario set as ovphysx.
    assert SUPPORTED_SCENARIOS_PER_ENGINE["newton"] == (
        "drop_settle",
        "freeform",
    )


def test_resolve_drop_settle_returns_callable() -> None:
    fn = resolve("drop_settle")
    assert callable(fn)
    assert fn.__module__.endswith("scenarios.drop_settle")


def test_resolve_freeform_returns_callable() -> None:
    fn = resolve("freeform")
    assert callable(fn)
    assert fn.__module__.endswith("scenarios.freeform")


def test_resolve_unknown_raises_with_supported_list() -> None:
    with pytest.raises(RuntimeError, match="unknown scenario kind"):
        resolve("does_not_exist")


def test_runner_capability_gate_uses_relocated_map() -> None:
    """``_validate_engine_supports_scenario`` reads
    SUPPORTED_SCENARIOS_PER_ENGINE rather than a runner-local map."""
    from physics_agent.tuning.errors import TuningError
    from physics_agent.tuning.runner import _validate_engine_supports_scenario

    # All registered pairs → no raise
    _validate_engine_supports_scenario("ovphysx", "drop_settle")
    _validate_engine_supports_scenario("ovphysx", "freeform")
    _validate_engine_supports_scenario("fake", "freeform")
    _validate_engine_supports_scenario("newton", "drop_settle")
    _validate_engine_supports_scenario("newton", "freeform")

    # Unknown scenario name on a known engine → TuningError
    with pytest.raises(TuningError, match="does not support"):
        _validate_engine_supports_scenario("ovphysx", "rolling_ball")
    with pytest.raises(TuningError, match="does not support"):
        _validate_engine_supports_scenario("newton", "rolling_ball")
