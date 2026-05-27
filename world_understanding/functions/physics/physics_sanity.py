# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic USD physics sanity inspection.

This module intentionally returns simple dataclasses so validation templates can
map findings to their final result models later without depending on the
Validation Agent contract layer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PHYSICS_EXPECTATION_TERMS = (
    "physics",
    "physical",
    "physx",
    "rigid",
    "collision",
    "collider",
    "mass",
    "density",
    "friction",
    "restitution",
    "simulate",
    "simulation",
    "gravity",
    "drop",
    "settle",
)

PHYSICS_NEGATION_TERMS = (
    "no physics",
    "without physics",
    "skip physics",
    "visual only",
    "appearance only",
)

# Entries are lowercase because candidate prim names are normalized before lookup.
GEOMETRY_CONTAINER_NAMES = {"geometry", "geom", "meshes", "mesh", "model"}

PHYSICS_VALIDATOR_TERMS = (
    "physics",
    "physx",
    "rigid",
    "collision",
    "collider",
    "mass",
    "density",
    "friction",
    "restitution",
)

PROPERTY_RANGES = {
    "static_friction": (0.0, 10.0),
    "dynamic_friction": (0.0, 10.0),
    "restitution": (0.0, 1.0),
    "density": (0.0, 50_000.0),
    "mass": (0.0, 1_000_000.0),
}

LARGE_COMPONENT_MAX_DIMENSION_M = 5.0
LARGE_COMPONENT_VOLUME_M3 = 5.0
HIGH_MASS_KG = 500.0


@dataclass
class PhysicsSanityFinding:
    """A contract-neutral physics sanity finding.

    ``severity`` is intentionally limited to ``"fail"`` or ``"warn"`` for
    later mapping into Validation Agent verdicts.
    """

    code: str
    severity: str
    message: str
    prim_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass
class PhysicsSanityResult:
    """Summary and findings from USD physics sanity inspection."""

    usd_path: str
    opened: bool
    physics_expected: bool
    summary: dict[str, Any] = field(default_factory=dict)
    findings: list[PhysicsSanityFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether inspection found no failing issues."""

        return self.opened and not any(
            finding.severity == "fail" for finding in self.findings
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["passed"] = self.passed
        return result


def infer_physics_expected(
    task_text: str | None = None,
    expect_physics: bool | None = None,
) -> bool:
    """Infer whether physics should be present.

    Explicit ``expect_physics`` takes precedence. The task text helper is a
    deliberately simple offline heuristic for early Validation Agent V1 use.
    """

    if expect_physics is not None:
        return expect_physics
    if not task_text:
        return False

    normalized = " ".join(task_text.lower().split())
    if any(term in normalized for term in PHYSICS_NEGATION_TERMS):
        return False
    return any(
        re.search(rf"\b{re.escape(term)}\b", normalized)
        for term in PHYSICS_EXPECTATION_TERMS
    )


def inspect_usd_physics(
    usd_path: str | Path,
    *,
    expect_physics: bool | None = None,
    task_text: str | None = None,
    single_asset: bool | None = None,
    asset_validator_report: dict[str, Any] | None = None,
) -> PhysicsSanityResult:
    """Inspect USD physics authoring without simulation or model dependencies.

    Args:
        usd_path: USD/USDZ file to inspect.
        expect_physics: Explicit expectation flag. If ``None``, ``task_text`` is
            used by :func:`infer_physics_expected`.
        task_text: Optional task text used for the simple physics expectation
            heuristic.
        single_asset: Controls multi-mesh rigid-body authoring checks. ``True``
            treats sibling mesh rigid bodies as parts of one logical asset,
            ``False`` allows independent sibling bodies, and ``None`` uses a
            conservative auto-detection for common geometry containers.
        asset_validator_report: Optional report from
            ``world_understanding.functions.graphics.validate_usd.validate_usd``.
            Physics-related validator issues are copied into the findings.

    Returns:
        Contract-neutral inspection result with metrics and findings.
    """

    path = Path(usd_path)
    physics_expected = infer_physics_expected(task_text, expect_physics)
    result = PhysicsSanityResult(
        usd_path=str(path),
        opened=False,
        physics_expected=physics_expected,
    )

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        result.findings.append(
            PhysicsSanityFinding(
                code="physics.usd_import_failed",
                severity="fail",
                message="USD Python bindings are not available.",
                details={"error": str(exc)},
            )
        )
        return result

    if not path.exists():
        result.findings.append(
            PhysicsSanityFinding(
                code="physics.usd_open_failed",
                severity="fail",
                message=f"USD file does not exist: {path}",
            )
        )
        return result

    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:
        result.findings.append(
            PhysicsSanityFinding(
                code="physics.usd_open_failed",
                severity="fail",
                message=f"Failed to open USD stage: {path}",
                details={"error": str(exc)},
            )
        )
        return result

    if not stage:
        result.findings.append(
            PhysicsSanityFinding(
                code="physics.usd_open_failed",
                severity="fail",
                message=f"Failed to open USD stage: {path}",
            )
        )
        return result

    result.opened = True
    findings: list[PhysicsSanityFinding] = []

    scenes: list[Any] = []
    rigid_bodies: list[Any] = []
    colliders: list[Any] = []
    meshes: list[Any] = []
    mass_api_prims: list[Any] = []
    material_api_prims: list[Any] = []

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scenes.append(prim)
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if _api_enabled(UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr()):
                rigid_bodies.append(prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if _api_enabled(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()):
                colliders.append(prim)
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api_prims.append(prim)
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            material_api_prims.append(prim)

    if physics_expected:
        _add_missing_expected_findings(
            findings=findings,
            scenes=scenes,
            rigid_bodies=rigid_bodies,
            colliders=colliders,
        )

    _add_rigid_body_collider_findings(findings, rigid_bodies, colliders)
    _add_multi_mesh_rigid_body_findings(findings, rigid_bodies, UsdGeom, single_asset)
    _add_property_range_findings(
        findings=findings,
        stage=stage,
        mass_api_prims=mass_api_prims,
        material_api_prims=material_api_prims,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
    )
    _add_asset_validator_findings(findings, asset_validator_report)

    result.summary = {
        "physics_scene_count": len(scenes),
        "rigid_body_count": len(rigid_bodies),
        "collider_count": len(colliders),
        "mesh_count": len(meshes),
        "mass_api_count": len(mass_api_prims),
        "material_api_count": len(material_api_prims),
        "physics_schema_count": (
            len(scenes)
            + len(rigid_bodies)
            + len(colliders)
            + len(mass_api_prims)
            + len(material_api_prims)
        ),
        "physics_scene_paths": _prim_paths(scenes),
        "rigid_body_paths": _prim_paths(rigid_bodies),
        "collider_paths": _prim_paths(colliders),
    }
    result.findings = findings
    return result


def _api_enabled(attr: Any) -> bool:
    """Treat an applied API as enabled unless it explicitly authors false."""

    # RigidBodyAPI and CollisionAPI enabled attrs default to true.
    value = attr.Get() if attr else None
    return value is not False


def _prim_paths(prims: list[Any]) -> list[str]:
    return [str(prim.GetPath()) for prim in prims]


def _add_missing_expected_findings(
    *,
    findings: list[PhysicsSanityFinding],
    scenes: list[Any],
    rigid_bodies: list[Any],
    colliders: list[Any],
) -> None:
    if not scenes and not rigid_bodies and not colliders:
        findings.append(
            PhysicsSanityFinding(
                code="physics.expected_but_missing",
                severity="fail",
                message="Physics was expected, but no physics scene, rigid bodies, or colliders were found.",
            )
        )
    if not scenes:
        findings.append(
            PhysicsSanityFinding(
                code="physics.no_physics_scene",
                severity="fail",
                message="Physics was expected, but no UsdPhysics.Scene prim was found.",
            )
        )
    if not rigid_bodies:
        findings.append(
            PhysicsSanityFinding(
                code="physics.no_rigid_bodies",
                severity="fail",
                message="Physics was expected, but no enabled RigidBodyAPI prims were found.",
            )
        )
    if not colliders:
        findings.append(
            PhysicsSanityFinding(
                code="physics.no_colliders",
                severity="fail",
                message="Physics was expected, but no enabled CollisionAPI prims were found.",
            )
        )


def _add_rigid_body_collider_findings(
    findings: list[PhysicsSanityFinding],
    rigid_bodies: list[Any],
    colliders: list[Any],
) -> None:
    collider_paths = [str(prim.GetPath()) for prim in colliders]
    for body in rigid_bodies:
        body_path = str(body.GetPath())
        if not any(_is_same_or_descendant(path, body_path) for path in collider_paths):
            findings.append(
                PhysicsSanityFinding(
                    code="physics.no_colliders",
                    severity="fail",
                    prim_path=body_path,
                    message="Rigid body has no enabled collider on itself or its descendants.",
                )
            )


def _add_multi_mesh_rigid_body_findings(
    findings: list[PhysicsSanityFinding],
    rigid_bodies: list[Any],
    UsdGeom: Any,
    single_asset: bool | None,
) -> None:
    mesh_bodies = [prim for prim in rigid_bodies if prim.IsA(UsdGeom.Mesh)]
    grouped: dict[str, tuple[Any, list[Any]]] = {}
    for prim in mesh_bodies:
        parent = _nearest_non_mesh_parent(prim, UsdGeom)
        parent_path = str(parent.GetPath()) if parent and parent.IsValid() else "/"
        if parent_path not in grouped:
            grouped[parent_path] = (parent, [])
        grouped[parent_path][1].append(prim)

    for parent_path, (parent, prims) in grouped.items():
        if len(prims) < 2:
            continue
        if not _should_flag_multi_mesh_group(parent, UsdGeom, single_asset):
            continue
        findings.append(
            PhysicsSanityFinding(
                code="physics.invalid_rigid_body_authoring",
                severity="fail",
                prim_path=parent_path,
                message=(
                    "Multiple sibling mesh prims carry RigidBodyAPI under the same "
                    "parent. For a single multi-mesh asset, put RigidBodyAPI on a "
                    "common Xformable ancestor and keep CollisionAPI on leaf meshes."
                ),
                details={
                    "rigid_body_meshes": _prim_paths(prims),
                    "suggested_rigid_body_prim": parent_path,
                },
            )
        )


def _nearest_non_mesh_parent(prim: Any, UsdGeom: Any) -> Any:
    parent = prim.GetParent()
    while parent and parent.IsValid() and not parent.IsPseudoRoot():
        if not parent.IsA(UsdGeom.Mesh):
            return parent
        parent = parent.GetParent()
    return parent


def _should_flag_multi_mesh_group(
    parent: Any,
    UsdGeom: Any,
    single_asset: bool | None,
) -> bool:
    if single_asset is not None:
        return single_asset
    if not parent or not parent.IsValid() or parent.IsPseudoRoot():
        return False
    if parent.GetName().lower() in GEOMETRY_CONTAINER_NAMES:
        return True
    if parent.IsA(UsdGeom.Xformable):
        # Non-container Xforms often group independent rigid bodies in a scene.
        return False
    return False


def _add_property_range_findings(
    *,
    findings: list[PhysicsSanityFinding],
    stage: Any,
    mass_api_prims: list[Any],
    material_api_prims: list[Any],
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> None:
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
    )
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))

    for prim in mass_api_prims:
        mass_api = UsdPhysics.MassAPI(prim)
        mass = _authored_number(mass_api.GetMassAttr())
        density = _authored_number(mass_api.GetDensityAttr())
        if mass is not None:
            _check_range(findings, prim, "mass", mass)
            _check_mass_scale(
                findings,
                prim,
                mass,
                bbox_cache,
                meters_per_unit,
                kilograms_per_unit,
            )
        if density is not None and not _density_is_mass_sentinel(density, mass):
            _check_range(findings, prim, "density", density)

    for prim in material_api_prims:
        material_api = UsdPhysics.MaterialAPI(prim)
        static_friction = _authored_number(material_api.GetStaticFrictionAttr())
        dynamic_friction = _authored_number(material_api.GetDynamicFrictionAttr())
        restitution = _authored_number(material_api.GetRestitutionAttr())

        if static_friction is not None:
            _check_range(findings, prim, "static_friction", static_friction)
        if dynamic_friction is not None:
            _check_range(findings, prim, "dynamic_friction", dynamic_friction)
        if restitution is not None:
            _check_range(findings, prim, "restitution", restitution)
        if (
            static_friction is not None
            and dynamic_friction is not None
            and dynamic_friction > static_friction
        ):
            findings.append(
                PhysicsSanityFinding(
                    code="physics.property_out_of_range",
                    severity="fail",
                    prim_path=str(prim.GetPath()),
                    message="Dynamic friction is greater than static friction.",
                    details={
                        "property": "dynamic_friction_vs_static_friction",
                        "value": dynamic_friction,
                        "static_friction": static_friction,
                    },
                )
            )


def _authored_number(attr: Any) -> float | None:
    if not attr or not attr.HasAuthoredValueOpinion():
        return None
    value = attr.Get()
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _density_is_mass_sentinel(density: float | None, mass: float | None) -> bool:
    # UsdPhysics / Isaac SimReady convention: ``MassAPI.density == 0`` paired
    # with a non-zero ``MassAPI.mass`` is a sentinel for "use the authored mass
    # directly". Density does not participate in mass computation in that case,
    # so the standard ``(0, MAX]`` density range check would fire a false
    # positive on every SimReady-derived asset. See #172.
    return density == 0.0 and mass is not None and mass > 0.0


def _check_range(
    findings: list[PhysicsSanityFinding],
    prim: Any,
    property_name: str,
    value: float,
) -> None:
    minimum, maximum = PROPERTY_RANGES[property_name]
    if property_name in {"density", "mass"}:
        in_range = minimum < value <= maximum
    else:
        in_range = minimum <= value <= maximum
    if in_range:
        return
    findings.append(
        PhysicsSanityFinding(
            code="physics.property_out_of_range",
            severity="fail",
            prim_path=str(prim.GetPath()),
            message=f"Physics property {property_name!r} is outside the expected range.",
            details={
                "property": property_name,
                "value": value,
                "minimum": minimum,
                "maximum": maximum,
            },
        )
    )


def _check_mass_scale(
    findings: list[PhysicsSanityFinding],
    prim: Any,
    mass: float,
    bbox_cache: Any,
    meters_per_unit: float,
    kilograms_per_unit: float,
) -> None:
    mass_kg = mass * kilograms_per_unit
    if mass_kg <= HIGH_MASS_KG:
        return

    bbox_info = _world_bbox_info(prim, bbox_cache)
    if not bbox_info:
        return

    size_m = [size * meters_per_unit for size in bbox_info["size"]]
    max_dimension = max(size_m)
    volume = size_m[0] * size_m[1] * size_m[2]
    if (
        max_dimension <= LARGE_COMPONENT_MAX_DIMENSION_M
        and volume <= LARGE_COMPONENT_VOLUME_M3
    ):
        return

    findings.append(
        PhysicsSanityFinding(
            code="physics.mass_scale_suspicious",
            severity="warn",
            prim_path=str(prim.GetPath()),
            message="Authored mass is high for a large component; verify scene units and mass estimate.",
            details={
                "mass_kg": mass_kg,
                "authored_mass": mass,
                "max_dimension_m": max_dimension,
                "volume_m3": volume,
                "meters_per_unit": meters_per_unit,
                "kilograms_per_unit": kilograms_per_unit,
                "thresholds": {
                    "high_mass_kg": HIGH_MASS_KG,
                    "large_component_max_dimension_m": LARGE_COMPONENT_MAX_DIMENSION_M,
                    "large_component_volume_m3": LARGE_COMPONENT_VOLUME_M3,
                },
            },
        )
    )


def _world_bbox_info(prim: Any, bbox_cache: Any) -> dict[str, list[float]] | None:
    try:
        bbox_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    except Exception:
        return None

    if bbox_range.IsEmpty():
        return None

    bbox_min = bbox_range.GetMin()
    bbox_max = bbox_range.GetMax()
    return {
        "min": [float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])],
        "max": [float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])],
        "size": [
            float(bbox_max[0] - bbox_min[0]),
            float(bbox_max[1] - bbox_min[1]),
            float(bbox_max[2] - bbox_min[2]),
        ],
    }


def _add_asset_validator_findings(
    findings: list[PhysicsSanityFinding],
    asset_validator_report: dict[str, Any] | None,
) -> None:
    if asset_validator_report is None:
        return

    if asset_validator_report.get("status") == "error":
        findings.append(
            PhysicsSanityFinding(
                code="physics.asset_validator_unavailable",
                severity="warn",
                message="USD validation report was provided but failed.",
                details={"error": asset_validator_report.get("error")},
            )
        )
        return

    for issue in asset_validator_report.get("issues", []):
        if not isinstance(issue, dict) or not _looks_physics_related(issue):
            continue
        severity = str(issue.get("severity") or "warning").lower()
        findings.append(
            PhysicsSanityFinding(
                code="physics.asset_validator_issue",
                severity="fail" if severity in {"failure", "fail", "error"} else "warn",
                prim_path=issue.get("at"),
                message=str(
                    issue.get("message") or "Physics-related USD validation issue."
                ),
                details={
                    "rule": issue.get("rule"),
                    "category": issue.get("category"),
                    "severity": issue.get("severity"),
                    "suggestion": issue.get("suggestion"),
                },
            )
        )


def _looks_physics_related(issue: dict[str, Any]) -> bool:
    text = " ".join(
        str(issue.get(key, "")) for key in ("rule", "category", "message", "suggestion")
    ).lower()
    return any(term in text for term in PHYSICS_VALIDATOR_TERMS)


def _is_same_or_descendant(path: str, ancestor_path: str) -> bool:
    return path == ancestor_path or path.startswith(f"{ancestor_path}/")
