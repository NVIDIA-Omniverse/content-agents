# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Automatic UV generation for meshes without texture coordinates.

Provides multiple UV projection modes for meshes that lack UV unwraps
(common in CAD-derived USD assets from STEP/IGES files).
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

logger = logging.getLogger(__name__)


class UVProjectionMode(str, Enum):
    """UV projection modes for meshes without UVs."""

    BOX = "box"
    """Box (triplanar) projection: projects from the axis with largest face area.
    Good for CAD parts, mechanical components, and boxy geometry."""

    PLANAR = "planar"
    """Planar projection from the dominant face direction.
    Good for flat or mostly-flat surfaces."""


def _compute_face_normals(
    points: np.ndarray, fvi: np.ndarray, fvc: np.ndarray
) -> np.ndarray:
    """Compute per-face normals from mesh topology."""
    normals = []
    idx = 0
    for count in fvc:
        if count < 3:
            normals.append(np.array([0.0, 1.0, 0.0]))
            idx += count
            continue
        v0 = points[fvi[idx]]
        v1 = points[fvi[idx + 1]]
        v2 = points[fvi[idx + 2]]
        n = np.cross(v1 - v0, v2 - v0)
        length = np.linalg.norm(n)
        if length > 0:
            n = n / length
        else:
            n = np.array([0.0, 1.0, 0.0])
        normals.append(n)
        idx += count
    return np.array(normals)


def generate_box_uvs(
    points: np.ndarray,
    fvi: np.ndarray,
    fvc: np.ndarray,
    margin: float = 0.05,
) -> np.ndarray:
    """Generate box-projected UVs for a mesh.

    Projects each face-vertex from the axis direction that best matches
    the face normal (triplanar projection, picking the dominant axis).

    Args:
        points: Vertex positions (N, 3).
        fvi: Face vertex indices.
        fvc: Face vertex counts.
        margin: UV margin from edges (default 0.05).

    Returns:
        UV array (len(fvi), 2) in [margin, 1-margin].
    """
    face_normals = _compute_face_normals(points, fvi, fvc)

    uvs = np.zeros((len(fvi), 2), dtype=np.float32)

    # Compute global bounding box for consistent scaling
    all_pts = points[fvi]
    bbox_min = all_pts.min(axis=0)
    bbox_max = all_pts.max(axis=0)
    bbox_range = bbox_max - bbox_min
    bbox_range[bbox_range == 0] = 1.0

    idx = 0
    for face_idx, count in enumerate(fvc):
        normal = face_normals[face_idx]
        abs_normal = np.abs(normal)
        dominant = np.argmax(abs_normal)

        # Choose projection axes (the two non-dominant axes)
        if dominant == 0:  # X dominant → project on YZ
            ax_u, ax_v = 1, 2
        elif dominant == 1:  # Y dominant → project on XZ
            ax_u, ax_v = 0, 2
        else:  # Z dominant → project on XY
            ax_u, ax_v = 0, 1

        for i in range(count):
            pt = points[fvi[idx + i]]
            u = (pt[ax_u] - bbox_min[ax_u]) / bbox_range[ax_u]
            v = (pt[ax_v] - bbox_min[ax_v]) / bbox_range[ax_v]
            uvs[idx + i] = [
                u * (1.0 - 2 * margin) + margin,
                v * (1.0 - 2 * margin) + margin,
            ]

        idx += count

    return uvs


def generate_planar_uvs(
    points: np.ndarray,
    fvi: np.ndarray,
    margin: float = 0.05,
) -> np.ndarray:
    """Generate planar-projected UVs from the direction of least variance.

    Args:
        points: Vertex positions (N, 3).
        fvi: Face vertex indices.
        margin: UV margin.

    Returns:
        UV array (len(fvi), 2).
    """
    face_verts = points[fvi]
    bbox_min = face_verts.min(axis=0)
    bbox_max = face_verts.max(axis=0)
    bbox_range = bbox_max - bbox_min
    bbox_range[bbox_range == 0] = 1.0

    # Pick the 2 axes with largest extent
    smallest_axis = np.argmin(bbox_range)
    axes = [i for i in range(3) if i != smallest_axis]

    normalized = (face_verts - bbox_min) / bbox_range
    uvs = normalized[:, axes]
    uvs = uvs * (1.0 - 2 * margin) + margin

    return uvs.astype(np.float32)


def generate_uvs_for_mesh(
    prim: Usd.Prim,
    mode: UVProjectionMode = UVProjectionMode.BOX,
) -> bool:
    """Generate UVs for a single mesh prim if it lacks them.

    Args:
        prim: A UsdGeom.Mesh prim.
        mode: UV projection mode.

    Returns:
        True if UVs were generated, False if UVs already existed.
    """
    api = UsdGeom.PrimvarsAPI(prim)
    st = api.GetPrimvar("st")
    if st and st.IsDefined() and st.Get() is not None:
        existing = np.array(st.Get())
        if len(existing) > 0:
            return False  # UVs already exist

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    fvc = mesh.GetFaceVertexCountsAttr().Get()

    if points is None or fvi is None or fvc is None:
        return False

    pts = np.array(points)
    indices = np.array(fvi)
    counts = np.array(fvc)

    if len(indices) == 0:
        return False

    if mode == UVProjectionMode.BOX:
        uvs = generate_box_uvs(pts, indices, counts)
    elif mode == UVProjectionMode.PLANAR:
        uvs = generate_planar_uvs(pts, indices)
    else:
        raise ValueError(f"Unknown UV projection mode: {mode}")

    vt_uvs = Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs])

    st_pv = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    st_pv.Set(vt_uvs)

    return True


def generate_uvs_for_stage(
    stage: Usd.Stage,
    mode: UVProjectionMode = UVProjectionMode.BOX,
) -> int:
    """Generate UVs for all meshes in a stage that lack them.

    Args:
        stage: USD stage to process.
        mode: UV projection mode.

    Returns:
        Number of meshes that received new UVs.
    """
    count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and not prim.IsInstanceProxy():
            if generate_uvs_for_mesh(prim, mode):
                count += 1
                logger.debug("Generated %s UVs for %s", mode.value, prim.GetPath())

    if count > 0:
        logger.info("Generated %s UVs for %d meshes", mode.value, count)
    return count


def fix_uv_interpolation(stage: Usd.Stage) -> int:
    """Fix meshes with 'constant' UV interpolation to 'faceVarying'.

    Returns:
        Number of meshes fixed.
    """
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
            continue
        api = UsdGeom.PrimvarsAPI(prim)
        st = api.GetPrimvar("st")
        if st and st.IsDefined() and st.GetInterpolation() == "constant":
            st.SetInterpolation("faceVarying")
            count += 1

    if count > 0:
        logger.info("Fixed UV interpolation on %d meshes", count)
    return count


def normalize_uvs(stage: Usd.Stage, margin: float = 0.025) -> int:
    """Normalize UV coordinates to [margin, 1-margin] for meshes with out-of-range UVs.

    Returns:
        Number of meshes normalized.
    """
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
            continue
        api = UsdGeom.PrimvarsAPI(prim)
        st = api.GetPrimvar("st")
        if not st or not st.IsDefined():
            continue
        uvs = np.array(st.Get())
        if len(uvs) == 0:
            continue

        u_min, u_max = uvs[:, 0].min(), uvs[:, 0].max()
        v_min, v_max = uvs[:, 1].min(), uvs[:, 1].max()

        # Only normalize if out of [0, 1] range
        if u_min >= 0 and u_max <= 1 and v_min >= 0 and v_max <= 1:
            continue

        u_range = u_max - u_min or 1.0
        v_range = v_max - v_min or 1.0
        uvs[:, 0] = (uvs[:, 0] - u_min) / u_range * (1 - 2 * margin) + margin
        uvs[:, 1] = (uvs[:, 1] - v_min) / v_range * (1 - 2 * margin) + margin

        vt_uvs = Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs])
        st.Set(vt_uvs)
        count += 1

    if count > 0:
        logger.info("Normalized UVs on %d meshes", count)
    return count
