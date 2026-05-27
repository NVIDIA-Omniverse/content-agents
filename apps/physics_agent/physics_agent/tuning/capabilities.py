# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-owned tuning capability declarations.

Capabilities describe which numeric scenario parameters a backend knows how to
consume. They are deliberately static and local to the backend: the resolver
combines this catalog with facts from the actual USD asset before BoTorch sees
any parameter bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

BINDING_KIND_USD_ATTRIBUTE = "usd_attribute"
BINDING_KIND_USD_MASS_SCALE = "usd_mass_scale"
BINDING_KIND_SIMULATOR_PARAMETER = "simulator_parameter"
USD_PHYSICS_BACKENDS = frozenset({"fake", "ovphysx", "usd"})


@dataclass(frozen=True)
class BindingCapability:
    """One backend-supported way to apply a tunable parameter."""

    param_name: str
    concept: str
    binding_kind: str
    default_range: tuple[float, float]
    priority: int = 0
    schema: str | None = None
    attribute: str | None = None
    simulator_parameter: str | None = None
    value_mode: str = "set"
    requires_authored_value: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "param": self.param_name,
            "concept": self.concept,
            "kind": self.binding_kind,
            "default_range": list(self.default_range),
            "priority": int(self.priority),
            "value_mode": self.value_mode,
            "requires_authored_value": bool(self.requires_authored_value),
        }
        if self.schema is not None:
            payload["schema"] = self.schema
        if self.attribute is not None:
            payload["attribute"] = self.attribute
        if self.simulator_parameter is not None:
            payload["simulator_parameter"] = self.simulator_parameter
        return payload


def usd_physics_capabilities() -> tuple[BindingCapability, ...]:
    """Capabilities for backends that consume authored UsdPhysics schemas."""

    return (
        BindingCapability(
            param_name="restitution",
            concept="bounce_response",
            binding_kind=BINDING_KIND_USD_ATTRIBUTE,
            schema="UsdPhysics.MaterialAPI",
            attribute="physics:restitution",
            default_range=(0.0, 1.0),
            priority=100,
        ),
        BindingCapability(
            param_name="static_friction",
            concept="surface_grip",
            binding_kind=BINDING_KIND_USD_ATTRIBUTE,
            schema="UsdPhysics.MaterialAPI",
            attribute="physics:staticFriction",
            default_range=(0.05, 1.5),
            priority=90,
        ),
        BindingCapability(
            param_name="dynamic_friction",
            concept="surface_grip",
            binding_kind=BINDING_KIND_USD_ATTRIBUTE,
            schema="UsdPhysics.MaterialAPI",
            attribute="physics:dynamicFriction",
            default_range=(0.05, 1.5),
            priority=90,
        ),
        BindingCapability(
            param_name="mass_scale",
            concept="mass_response",
            binding_kind=BINDING_KIND_USD_MASS_SCALE,
            schema="UsdPhysics.MassAPI",
            attribute="physics:mass",
            default_range=(0.5, 2.0),
            priority=80,
            value_mode="scale_existing",
            requires_authored_value=True,
        ),
    )


def newton_mujoco_capabilities() -> tuple[BindingCapability, ...]:
    """Capabilities consumed by Newton's MuJoCo solver path.

    ``physics:restitution`` is intentionally absent: Newton's
    ``ShapeConfig.restitution`` is an XPBD-only coefficient, while this backend
    constructs ``SolverMuJoCo``. Bounce response for MuJoCo is controlled through
    contact stiffness/damping, imported from ``newton:contact_ke`` and
    ``newton:contact_kd`` onto ``model.shape_material_ke/kd``. Static friction
    is also absent because the current USD importer derives contact friction
    from ``physics:dynamicFriction``.
    """

    return (
        BindingCapability(
            param_name="contact_ke",
            concept="bounce_response",
            binding_kind=BINDING_KIND_USD_ATTRIBUTE,
            schema="UsdPhysics.CollisionAPI",
            attribute="newton:contact_ke",
            default_range=(100.0, 100000.0),
            priority=100,
        ),
        BindingCapability(
            param_name="contact_kd",
            concept="bounce_response",
            binding_kind=BINDING_KIND_USD_ATTRIBUTE,
            schema="UsdPhysics.CollisionAPI",
            attribute="newton:contact_kd",
            default_range=(0.0, 5000.0),
            priority=100,
        ),
        BindingCapability(
            param_name="dynamic_friction",
            concept="surface_grip",
            binding_kind=BINDING_KIND_USD_ATTRIBUTE,
            schema="UsdPhysics.MaterialAPI",
            attribute="physics:dynamicFriction",
            default_range=(0.05, 1.5),
            priority=90,
        ),
        BindingCapability(
            param_name="mass_scale",
            concept="mass_response",
            binding_kind=BINDING_KIND_USD_MASS_SCALE,
            schema="UsdPhysics.MassAPI",
            attribute="physics:mass",
            default_range=(0.5, 2.0),
            priority=80,
            value_mode="scale_existing",
            requires_authored_value=True,
        ),
    )


def capabilities_for_backend(backend_name: str) -> tuple[BindingCapability, ...]:
    """Fallback capability catalog for legacy structural backend objects."""

    normalized = backend_name.lower()
    if normalized == "newton":
        return newton_mujoco_capabilities()
    if normalized in USD_PHYSICS_BACKENDS:
        return usd_physics_capabilities()
    raise ValueError(
        f"Unknown backend {backend_name!r}; known backends are "
        f"{sorted(USD_PHYSICS_BACKENDS | {'newton'})}."
    )


__all__ = [
    "BINDING_KIND_SIMULATOR_PARAMETER",
    "BINDING_KIND_USD_ATTRIBUTE",
    "BINDING_KIND_USD_MASS_SCALE",
    "BindingCapability",
    "USD_PHYSICS_BACKENDS",
    "capabilities_for_backend",
    "newton_mujoco_capabilities",
    "usd_physics_capabilities",
]
