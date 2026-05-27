# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario evaluators dispatched by ``OvPhysXBackend``.

Each scenario kind has its own module exporting an ``evaluate(...)``
callable. The :func:`resolve` helper lazy-imports the right one so
``physics_agent.tuning`` stays import-light when only the FakeBackend
or unrelated code paths are touched.

The capability gate
:data:`SUPPORTED_SCENARIOS_PER_ENGINE` lives here (next to the
dispatch) so a new scenario implementation and its capability advertisement
land as a single, atomic change. The runner and REST router both
import this dict.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

# Per-engine scenario capability map.
#
# This is the single source of truth for "which (engine, scenario_name)
# pairs the tuning runner accepts." The ``fake`` backend is purely
# parameter-driven so it tolerates any scenario kind. The ``ovphysx``
# backend dispatches to per-scenario evaluator functions in this
# package — every kind listed under ``ovphysx`` here MUST have a
# corresponding evaluator module imported via :func:`resolve`.
SUPPORTED_SCENARIOS_PER_ENGINE: dict[str, tuple[str, ...]] = {
    "fake": ("drop_settle", "freeform"),
    "ovphysx": ("drop_settle", "freeform"),
    "newton": ("drop_settle", "freeform"),
}


def resolve(scenario_name: str) -> Callable[..., dict[str, Any]]:
    """Lazy-import the per-scenario evaluator and return its
    ``evaluate`` callable.

    Args:
        scenario_name: Scenario kind. Must be a key in
            :data:`SUPPORTED_SCENARIOS_PER_ENGINE` for any engine.

    Raises:
        RuntimeError: ``scenario_name`` is not a known kind. The error
            message lists the supported kinds so CLI / REST callers
            see an actionable message.
    """
    module_name = {
        "drop_settle": "physics_agent.tuning.scenarios.drop_settle",
        "freeform": "physics_agent.tuning.scenarios.freeform",
    }.get(scenario_name)
    if module_name is None:
        all_kinds = sorted(
            {k for kinds in SUPPORTED_SCENARIOS_PER_ENGINE.values() for k in kinds}
        )
        raise RuntimeError(
            f"unknown scenario kind {scenario_name!r}; supported: {all_kinds!r}"
        )
    module = importlib.import_module(module_name)
    evaluate = getattr(module, "evaluate", None)
    # Defer-to-call surfaces a confusing ``TypeError: 'X' object is not
    # callable`` deep inside the runner; check up-front so a non-callable
    # ``evaluate`` symbol fails here with the module path attached
    # (CodeRabbit Round 11 thread #12).
    if not callable(evaluate):
        raise RuntimeError(
            f"scenario module {module_name!r} does not export a callable "
            f"evaluate() (got {type(evaluate).__name__})"
        )
    return evaluate  # type: ignore[no-any-return]


__all__ = ["SUPPORTED_SCENARIOS_PER_ENGINE", "resolve"]
