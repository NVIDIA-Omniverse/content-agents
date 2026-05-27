# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve parsed scenarios into backend-aware parameter bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .capabilities import (
    BINDING_KIND_SIMULATOR_PARAMETER,
    BINDING_KIND_USD_ATTRIBUTE,
    BINDING_KIND_USD_MASS_SCALE,
    BindingCapability,
    capabilities_for_backend,
)
from .errors import TuningError
from .types import Scenario
from .usd_inspector import UsdTuningReport, inspect_usd_for_tuning

RESOLVED_BINDINGS_EXTRA_KEY = "resolved_parameter_bindings"
RESOLUTION_REPORT_EXTRA_KEY = "parameter_binding_resolution"


def _backend_capabilities(backend: Any) -> tuple[BindingCapability, ...]:
    backend_name = str(getattr(backend, "name", type(backend).__name__))
    provider = getattr(backend, "tuning_capabilities", None)
    if not callable(provider):
        try:
            return capabilities_for_backend(backend_name)
        except ValueError as exc:
            raise TuningError(
                f"Backend {backend_name!r} does not declare tuning capabilities."
            ) from exc
    capabilities = provider()
    if not isinstance(capabilities, tuple):
        capabilities = tuple(capabilities)
    if not all(isinstance(c, BindingCapability) for c in capabilities):
        raise TuningError(
            f"Backend {getattr(backend, 'name', type(backend).__name__)!r} returned "
            "invalid tuning capabilities."
        )
    return capabilities


def _resolve_usd_binding(
    *,
    capability: BindingCapability,
    report: UsdTuningReport,
    backend_name: str,
) -> dict[str, Any] | None:
    if capability.schema is None or capability.attribute is None:
        return None
    matches = report.find(
        schema=capability.schema,
        attribute=capability.attribute,
        require_authored_value=capability.requires_authored_value,
    )
    if not matches:
        return None
    return {
        **capability.to_dict(),
        "backend": backend_name,
        "prim_paths": [m.prim_path for m in matches],
        "source": "usd_inspection",
    }


def _resolve_simulator_binding(
    *,
    capability: BindingCapability,
    backend_name: str,
) -> dict[str, Any] | None:
    if capability.simulator_parameter is None:
        return None
    return {
        **capability.to_dict(),
        "backend": backend_name,
        "source": "backend_capability",
    }


def _resolve_param_binding(
    *,
    param_name: str,
    capabilities: tuple[BindingCapability, ...],
    report: UsdTuningReport,
    backend_name: str,
) -> dict[str, Any]:
    candidates = sorted(
        (c for c in capabilities if c.param_name == param_name),
        key=lambda c: c.priority,
        reverse=True,
    )
    if not candidates:
        raise TuningError(
            f"Backend {backend_name!r} does not support tunable parameter "
            f"{param_name!r}."
        )

    for capability in candidates:
        if capability.binding_kind in {
            BINDING_KIND_USD_ATTRIBUTE,
            BINDING_KIND_USD_MASS_SCALE,
        }:
            resolved = _resolve_usd_binding(
                capability=capability,
                report=report,
                backend_name=backend_name,
            )
        elif capability.binding_kind == BINDING_KIND_SIMULATOR_PARAMETER:
            resolved = _resolve_simulator_binding(
                capability=capability,
                backend_name=backend_name,
            )
        else:
            resolved = None
        if resolved is not None:
            return resolved

    raise TuningError(
        f"Could not resolve tunable parameter {param_name!r} for backend "
        f"{backend_name!r} against {report.usd_path}. The backend advertises "
        "a capability, but the USD asset does not expose the required "
        "schema/attribute."
    )


def resolve_scenario_bindings(
    scenario: Scenario,
    *,
    physics_usd: Path | str,
    backend: Any,
) -> Scenario:
    """Return a copy of ``scenario`` with resolved parameter bindings in extra."""

    backend_name = str(getattr(backend, "name", type(backend).__name__))
    capabilities = _backend_capabilities(backend)
    report = inspect_usd_for_tuning(physics_usd)
    bindings = [
        _resolve_param_binding(
            param_name=param.name,
            capabilities=capabilities,
            report=report,
            backend_name=backend_name,
        )
        for param in scenario.params
    ]

    extra = dict(scenario.extra)
    extra[RESOLVED_BINDINGS_EXTRA_KEY] = bindings
    extra[RESOLUTION_REPORT_EXTRA_KEY] = {
        "backend": backend_name,
        "usd_path": str(report.usd_path),
        "candidate_count": len(report.candidates),
    }
    return Scenario(
        name=scenario.name,
        params=scenario.params,
        target=scenario.target,
        metric=scenario.metric,
        extra=extra,
    )


def get_resolved_bindings(scenario: Scenario) -> list[dict[str, Any]] | None:
    """Return bindings stored by :func:`resolve_scenario_bindings`, if any."""

    raw = scenario.extra.get(RESOLVED_BINDINGS_EXTRA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TuningError(
            f"Scenario extra {RESOLVED_BINDINGS_EXTRA_KEY!r} must be a list, "
            f"got {type(raw).__name__}."
        )
    bindings: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TuningError(
                f"Scenario binding {RESOLVED_BINDINGS_EXTRA_KEY}[{i}] must be "
                f"a mapping, got {type(item).__name__}."
            )
        bindings.append(dict(item))
    return bindings


__all__ = [
    "RESOLVED_BINDINGS_EXTRA_KEY",
    "RESOLUTION_REPORT_EXTRA_KEY",
    "get_resolved_bindings",
    "resolve_scenario_bindings",
]
