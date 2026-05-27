# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial query functions for USD scenes.

Pure functions for querying USD stage geometry: bounding boxes, distances,
overlaps, material bindings, geometry stats, and composed transforms.
"""

from __future__ import annotations

import fnmatch
import logging
import math
from typing import Any

from pxr import Gf, Usd, UsdGeom, UsdShade

from world_understanding.utils.usd.prim import get_bbox_from_prim, traverse_prims

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------


def get_world_bbox(stage: Usd.Stage, prim_path: str) -> dict[str, Any] | None:
    """Get world-space bounding box for a prim as a plain dict.

    Returns None if the prim has no computable bounds (e.g. empty Xform).
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    bbox = get_bbox_from_prim(prim)
    bbox_range = bbox.ComputeAlignedRange()

    if bbox_range.IsEmpty():
        return None

    bmin = bbox_range.GetMin()
    bmax = bbox_range.GetMax()
    size = bmax - bmin
    center = (bmin + bmax) / 2.0

    return {
        "min": [bmin[0], bmin[1], bmin[2]],
        "max": [bmax[0], bmax[1], bmax[2]],
        "size": [size[0], size[1], size[2]],
        "center": [center[0], center[1], center[2]],
        "volume": size[0] * size[1] * size[2],
    }


def bbox_overlaps(
    a_min: list[float],
    a_max: list[float],
    b_min: list[float],
    b_max: list[float],
) -> bool:
    """Test axis-aligned bounding box overlap (3D)."""
    return (
        a_min[0] <= b_max[0]
        and a_max[0] >= b_min[0]
        and a_min[1] <= b_max[1]
        and a_max[1] >= b_min[1]
        and a_min[2] <= b_max[2]
        and a_max[2] >= b_min[2]
    )


def bbox_distance(
    a_min: list[float],
    a_max: list[float],
    b_min: list[float],
    b_max: list[float],
) -> float:
    """Minimum distance between two axis-aligned bounding boxes.

    Returns 0.0 if the boxes overlap.
    """
    sq_dist = 0.0
    for i in range(3):
        if a_max[i] < b_min[i]:
            sq_dist += (b_min[i] - a_max[i]) ** 2
        elif b_max[i] < a_min[i]:
            sq_dist += (a_min[i] - b_max[i]) ** 2
    return math.sqrt(sq_dist)


def point_to_bbox_distance(
    point: list[float], b_min: list[float], b_max: list[float]
) -> float:
    """Distance from a 3D point to an axis-aligned bounding box.

    Returns 0.0 if the point is inside the box.
    """
    sq_dist = 0.0
    for i in range(3):
        if point[i] < b_min[i]:
            sq_dist += (b_min[i] - point[i]) ** 2
        elif point[i] > b_max[i]:
            sq_dist += (point[i] - b_max[i]) ** 2
    return math.sqrt(sq_dist)


# ---------------------------------------------------------------------------
# Material binding
# ---------------------------------------------------------------------------


def get_bound_material_path(
    prim: Usd.Prim,
) -> str | None:
    """Get the path of the material bound to a prim, or None."""
    binding_api = UsdShade.MaterialBindingAPI(prim)
    mat, _ = binding_api.ComputeBoundMaterial()
    if mat:
        return str(mat.GetPath())
    return None


# ---------------------------------------------------------------------------
# Geometry stats
# ---------------------------------------------------------------------------


def get_geometry_stats(prim: Usd.Prim) -> dict[str, Any] | None:
    """Get geometry statistics for a mesh prim.

    Returns None if the prim is not a mesh.
    """
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return None

    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    subdiv = mesh.GetSubdivisionSchemeAttr().Get()

    return {
        "vertex_count": len(points) if points else 0,
        "face_count": len(face_counts) if face_counts else 0,
        "subdivision_scheme": str(subdiv) if subdiv else "none",
    }


# ---------------------------------------------------------------------------
# World transform
# ---------------------------------------------------------------------------


def get_world_transform(stage: Usd.Stage, prim_path: str) -> list[list[float]] | None:
    """Get the composed world-space 4x4 transform matrix for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    xform = xform_cache.GetLocalToWorldTransform(prim)
    return [[xform[r][c] for c in range(4)] for r in range(4)]


# ---------------------------------------------------------------------------
# Material binding map (scene-wide)
# ---------------------------------------------------------------------------


def get_material_binding_map(
    stage: Usd.Stage,
    start_prim: Usd.Prim | None = None,
) -> dict[str, list[str]]:
    """Build a map of material_path -> list of bound prim paths.

    Also includes a special key "(unassigned)" for prims with no material.
    Only considers geometry prims (Mesh, BasisCurves, etc.).
    """
    mat_map: dict[str, list[str]] = {}
    for prim in traverse_prims(stage):
        if start_prim and not str(prim.GetPath()).startswith(str(start_prim.GetPath())):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        mat_path = get_bound_material_path(prim) or "(unassigned)"
        mat_map.setdefault(mat_path, []).append(str(prim.GetPath()))
    return mat_map


# ---------------------------------------------------------------------------
# Core query engine
# ---------------------------------------------------------------------------


def _build_prim_result(
    prim: Usd.Prim,
    bbox_info: dict[str, Any] | None,
    distance: float | None = None,
) -> dict[str, Any]:
    """Build a standardised prim result dict."""
    result: dict[str, Any] = {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "parent": str(prim.GetParent().GetPath()) if prim.GetParent() else None,
    }
    if bbox_info:
        result["bbox_min"] = bbox_info["min"]
        result["bbox_max"] = bbox_info["max"]
        result["size"] = bbox_info["size"]
        result["center"] = bbox_info["center"]
        result["volume"] = bbox_info["volume"]
    if distance is not None:
        result["distance"] = round(distance, 6)

    mat = get_bound_material_path(prim)
    if mat:
        result["material"] = mat

    return result


def query_prims(
    stage: Usd.Stage,
    *,
    name_pattern: str | None = None,
    path_pattern: str | None = None,
    prim_type: str | None = None,
    has_material: bool | None = None,
    min_size: float | None = None,
    max_size: float | None = None,
    near: list[float] | str | None = None,
    radius: float | None = None,
    overlaps: str | None = None,
    sort_by: str = "name",
    limit: int | None = None,
    start_prim: str | None = None,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    """Query prims in a USD stage with filters.

    Args:
        stage: The USD stage to query.
        name_pattern: Glob pattern matched against prim name.
        path_pattern: Glob pattern matched against full prim path.
        prim_type: USD type name filter (e.g. "Mesh", "Xform").
        has_material: If True only prims with material, if False only without.
        min_size: Minimum bbox volume.
        max_size: Maximum bbox volume.
        near: Reference point [x,y,z] or prim path for distance queries.
        radius: Max distance from *near* target.
        overlaps: Prim path; return prims whose bbox overlaps with it.
        sort_by: "name", "size", "distance", "type".
        limit: Max number of results.
        start_prim: Scope traversal to this subtree.
        active_only: Skip inactive prims.

    Returns:
        List of prim result dicts.
    """
    # Resolve --near reference
    near_point: list[float] | None = None
    near_bbox: dict[str, Any] | None = None
    if isinstance(near, str) and near.startswith("/"):
        near_bbox = get_world_bbox(stage, near)
        if near_bbox:
            near_point = near_bbox["center"]
    elif isinstance(near, list):
        near_point = near

    # Resolve --overlaps reference bbox
    overlaps_bbox: dict[str, Any] | None = None
    if overlaps:
        overlaps_bbox = get_world_bbox(stage, overlaps)

    results: list[dict[str, Any]] = []

    for prim in traverse_prims(stage):
        prim_path = str(prim.GetPath())

        # Scope filter
        if start_prim and not prim_path.startswith(start_prim):
            continue

        # Active filter
        if active_only and not prim.IsActive():
            continue

        # Type filter
        if prim_type and prim.GetTypeName() != prim_type:
            continue

        # Name pattern
        if name_pattern and not fnmatch.fnmatch(prim.GetName(), name_pattern):
            continue

        # Path pattern
        if path_pattern and not fnmatch.fnmatch(prim_path, path_pattern):
            continue

        # Material filter
        if has_material is not None:
            mat = get_bound_material_path(prim)
            if has_material and not mat:
                continue
            if not has_material and mat:
                continue

        # Compute bbox (needed for size, near, overlaps filters)
        bbox_info: dict[str, Any] | None = None
        need_bbox = (
            min_size is not None
            or max_size is not None
            or near_point is not None
            or overlaps_bbox is not None
        )
        if need_bbox:
            bbox_info = get_world_bbox(stage, prim_path)

        # Size filters
        if min_size is not None:
            if not bbox_info or bbox_info["volume"] < min_size:
                continue
        if max_size is not None:
            if not bbox_info or bbox_info["volume"] > max_size:
                continue

        # Overlap filter
        if overlaps_bbox:
            if not bbox_info:
                continue
            # Skip self
            if overlaps == prim_path:
                continue
            if not bbox_overlaps(
                bbox_info["min"],
                bbox_info["max"],
                overlaps_bbox["min"],
                overlaps_bbox["max"],
            ):
                continue

        # Distance / radius filter
        distance: float | None = None
        if near_point is not None:
            if not bbox_info:
                bbox_info = get_world_bbox(stage, prim_path)
            if bbox_info:
                distance = point_to_bbox_distance(
                    near_point, bbox_info["min"], bbox_info["max"]
                )
            else:
                distance = None
            if radius is not None and (distance is None or distance > radius):
                continue

        # If we didn't compute bbox yet but want it for output
        if bbox_info is None:
            bbox_info = get_world_bbox(stage, prim_path)

        results.append(_build_prim_result(prim, bbox_info, distance))

    # Sort
    if sort_by == "name":
        results.sort(key=lambda r: r["path"])
    elif sort_by == "size":
        results.sort(key=lambda r: r.get("volume") or 0, reverse=True)
    elif sort_by == "distance":
        results.sort(
            key=lambda r: (
                r.get("distance") if r.get("distance") is not None else float("inf")
            )
        )
    elif sort_by == "type":
        results.sort(key=lambda r: (r["type"], r["path"]))

    if limit:
        results = results[:limit]

    return results


# ---------------------------------------------------------------------------
# Scene summary
# ---------------------------------------------------------------------------


def scene_summary(
    stage: Usd.Stage,
    start_prim: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Compute a scene summary with composition stats, spatial extents, and materials.

    Returns a dict suitable for JSON serialisation or rich text rendering.
    """
    from world_understanding.utils.usd.stage import get_stage_info

    info = get_stage_info(stage)

    type_counts: dict[str, int] = {}
    total_prims = 0
    instance_count = 0

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # Track per-prim volumes for "largest prims"
    prim_volumes: list[tuple[str, str, float]] = []  # (path, type, volume)

    for prim in traverse_prims(stage):
        prim_path = str(prim.GetPath())
        if start_prim and not prim_path.startswith(start_prim):
            continue

        total_prims += 1
        type_name = prim.GetTypeName() or "(untyped)"
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

        if prim.IsInstance():
            instance_count += 1

        # Compute volume for geometry prims
        if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Xformable):
            try:
                bbox = bbox_cache.ComputeWorldBound(prim)
                bbox_range = bbox.ComputeAlignedRange()
                if not bbox_range.IsEmpty():
                    sz = bbox_range.GetMax() - bbox_range.GetMin()
                    vol = sz[0] * sz[1] * sz[2]
                    if vol > 0:
                        prim_volumes.append((prim_path, type_name, vol))
            except Exception:
                pass

    # Scene-level bbox
    scene_bbox: dict[str, Any] | None = None
    pseudo_root = stage.GetPseudoRoot()
    try:
        root_bbox = bbox_cache.ComputeWorldBound(pseudo_root)
        root_range = root_bbox.ComputeAlignedRange()
        if not root_range.IsEmpty():
            bmin = root_range.GetMin()
            bmax = root_range.GetMax()
            sz = bmax - bmin
            scene_bbox = {
                "min": [bmin[0], bmin[1], bmin[2]],
                "max": [bmax[0], bmax[1], bmax[2]],
                "size": [sz[0], sz[1], sz[2]],
            }
    except Exception:
        pass

    # Largest prims
    prim_volumes.sort(key=lambda x: x[2], reverse=True)
    largest = [
        {"path": p, "type": t, "volume": round(v, 4)}
        for p, t, v in prim_volumes[:top_n]
    ]

    # Materials
    mat_map = get_material_binding_map(stage)
    materials_summary: list[dict[str, Any]] = []
    for mat_path, prims in sorted(
        mat_map.items(), key=lambda x: len(x[1]), reverse=True
    ):
        materials_summary.append(
            {
                "material": mat_path,
                "bound_prim_count": len(prims),
            }
        )

    return {
        "stage_info": {
            "root_layer": info.get("root_layer_path"),
            "up_axis": info.get("up_axis"),
            "meters_per_unit": info.get("meters_per_unit"),
            "start_time": info.get("start_time_code"),
            "end_time": info.get("end_time_code"),
            "fps": info.get("time_codes_per_second"),
            "default_prim": info.get("default_prim"),
        },
        "composition": {
            "total_prims": total_prims,
            "type_counts": dict(
                sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            ),
            "instance_count": instance_count,
        },
        "spatial_extents": scene_bbox,
        "largest_prims": largest,
        "materials": materials_summary,
    }


# ---------------------------------------------------------------------------
# Prim inspection
# ---------------------------------------------------------------------------


def inspect_prim(
    stage: Usd.Stage,
    prim_path: str,
    *,
    include_world_transform: bool = False,
    include_geometry: bool = False,
    include_properties: bool = False,
) -> dict[str, Any] | None:
    """Inspect a single prim in detail.

    Returns None if prim not found.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    result: dict[str, Any] = {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "active": prim.IsActive(),
    }

    # Bbox
    bbox_info = get_world_bbox(stage, prim_path)
    if bbox_info:
        result["bbox_min"] = bbox_info["min"]
        result["bbox_max"] = bbox_info["max"]
        result["size"] = bbox_info["size"]
        result["center"] = bbox_info["center"]
        result["volume"] = bbox_info["volume"]

    # Hierarchy
    parent = prim.GetParent()
    result["parent"] = str(parent.GetPath()) if parent else None

    children = prim.GetChildren()
    result["children"] = [str(c.GetPath()) for c in children]
    result["child_count"] = len(children)

    # Count all descendants
    desc_count = 0
    for _ in Usd.PrimRange(prim):
        desc_count += 1
    result["descendant_count"] = desc_count - 1  # exclude self

    # Material
    result["material"] = get_bound_material_path(prim)

    # Variants
    vsets = prim.GetVariantSets()
    variant_names = vsets.GetNames()
    if variant_names:
        result["variants"] = {
            name: vsets.GetVariantSet(name).GetVariantSelection()
            for name in variant_names
        }

    # World transform
    if include_world_transform:
        result["world_transform"] = get_world_transform(stage, prim_path)

        # Also include local transform
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            local_xform = xformable.GetLocalTransformation(Usd.TimeCode.Default())
            result["local_transform"] = [
                [local_xform[r][c] for c in range(4)] for r in range(4)
            ]

    # Geometry stats
    if include_geometry and prim.IsA(UsdGeom.Mesh):
        result["geometry"] = get_geometry_stats(prim)

    # Properties
    if include_properties:
        props: dict[str, Any] = {}
        for prop in prim.GetAuthoredProperties():
            try:
                attr = prim.GetAttribute(prop.GetName())
                if attr and attr.IsAuthored():
                    val = attr.Get()
                    # Convert USD types to JSON-serializable
                    if isinstance(val, Gf.Vec3f | Gf.Vec3d):
                        props[prop.GetName()] = list(val)
                    elif isinstance(val, Gf.Matrix4d | Gf.Matrix4f):
                        props[prop.GetName()] = [
                            [val[r][c] for c in range(4)] for r in range(4)
                        ]
                    elif isinstance(val, int | float | str | bool):
                        props[prop.GetName()] = val
                    else:
                        props[prop.GetName()] = str(val)
            except Exception:
                pass
        if props:
            result["properties"] = props

    return result
