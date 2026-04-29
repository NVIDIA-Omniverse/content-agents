# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD material introspection for OpenPBR materials.

Discovers materials in a USD stage, extracts their OpenPBR properties
(base_color, texture slots, roughness, metalness), and identifies which
geometry prims are bound to each material.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pxr import Usd, UsdGeom, UsdShade

logger = logging.getLogger(__name__)


@dataclass
class MaterialInfo:
    """Information about a discovered OpenPBR material in a USD stage."""

    prim_path: str
    """Prim path of the material (e.g., '/World/Looks/Steel_Carbon')."""

    name: str
    """Material prim name (e.g., 'Steel_Carbon')."""

    bound_prim_paths: list[str] = field(default_factory=list)
    """Geometry prim paths bound to this material."""

    base_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Constant base_color value (linear sRGB, 0-1)."""

    base_color_texture: str | None = None
    """Existing base_color_texture_file path, or None if empty."""

    base_metalness: float | None = None
    """Constant base_metalness value."""

    specular_roughness: float | None = None
    """Constant specular_roughness value."""

    has_existing_texture: bool = False
    """True if base_color_texture_file is set to a non-empty path."""


def _read_color3f(prim: Usd.Prim, attr_name: str) -> tuple[float, float, float] | None:
    """Read a color3f attribute from a prim."""
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    val = attr.Get()
    if val is None:
        return None
    return (float(val[0]), float(val[1]), float(val[2]))


def _read_float(prim: Usd.Prim, attr_name: str) -> float | None:
    """Read a float attribute from a prim."""
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    val = attr.Get()
    return float(val) if val is not None else None


def _read_asset_path(prim: Usd.Prim, attr_name: str) -> str | None:
    """Read an asset path attribute, returning None if empty or '@@'."""
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    val = attr.Get()
    if val is None:
        return None
    path = val.path if hasattr(val, "path") else str(val)
    if not path or path == "@@":
        return None
    return path


def _find_bound_prims(stage: Usd.Stage, material_path: str) -> list[str]:
    """Find all geometry prims bound to a given material."""
    bound: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        mat, _ = binding_api.ComputeBoundMaterial()
        if mat and str(mat.GetPath()) == material_path:
            bound.append(str(prim.GetPath()))
    return bound


def discover_materials(
    stage: Usd.Stage,
    prim_paths: list[str] | None = None,
) -> list[MaterialInfo]:
    """Discover all OpenPBR materials in a USD stage.

    Traverses the stage to find Material prims, extracts their constant
    OpenPBR properties, and identifies which geometry prims use each material.

    Args:
        stage: An open USD stage.
        prim_paths: Optional list of material prim paths to restrict to.
            If None, all materials in the stage are discovered.

    Returns:
        List of MaterialInfo for each discovered material.
    """
    materials: list[MaterialInfo] = []

    # Collect material prims -- use TraverseAll to include 'over' prims
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdShade.Material):
            continue

        mat_path = str(prim.GetPath())

        # Apply filter if specified
        if prim_paths and mat_path not in prim_paths:
            continue

        name = prim.GetName()

        # Read OpenPBR properties
        base_color = _read_color3f(prim, "inputs:base_color")
        base_color_texture = _read_asset_path(prim, "inputs:base_color_texture_file")
        base_metalness = _read_float(prim, "inputs:base_metalness")
        specular_roughness = _read_float(prim, "inputs:specular_roughness")

        # Find bound geometry prims
        bound_prims = _find_bound_prims(stage, mat_path)

        info = MaterialInfo(
            prim_path=mat_path,
            name=name,
            bound_prim_paths=bound_prims,
            base_color=base_color or (0.5, 0.5, 0.5),
            base_color_texture=base_color_texture,
            base_metalness=base_metalness,
            specular_roughness=specular_roughness,
            has_existing_texture=base_color_texture is not None,
        )
        materials.append(info)

        logger.info(
            "Discovered material: %s (base_color=%s, has_texture=%s, bound_prims=%d)",
            name,
            base_color,
            info.has_existing_texture,
            len(bound_prims),
        )

    logger.info("Discovered %d materials total", len(materials))
    return materials


@dataclass
class PrimTextureUnit:
    """One texture-generation unit: a specific prim getting a specific texture.

    In per-material mode, prim_path is empty and key equals the material name.
    In per-prim mode, each bound prim gets its own unit with a unique key.
    """

    prim_path: str
    """Geometry prim path (e.g., '/World/Rail_L'). Empty in per-material mode."""

    material_info: MaterialInfo
    """The original shared material bound to this prim."""

    key: str
    """Unique key for dict lookups (e.g., 'Aluminum_Brushed__Rail_L')."""

    prompt: str
    """Text prompt for this unit's texture."""

    opacity: float
    """Blend opacity."""

    seed: int | None = None
    """Seed for reproducibility. Different seeds per prim yield unique textures."""


def _stable_hash(s: str) -> int:
    """Deterministic hash stable across Python processes (unlike builtin hash)."""
    import hashlib

    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)


def _sanitize_prim_name(prim_path: str) -> str:
    """Extract a filesystem/USD-safe name from a prim path."""
    leaf = prim_path.rsplit("/", 1)[-1]
    return leaf.replace(" ", "_").replace("-", "_")


def expand_to_prim_units(
    materials: list[MaterialInfo],
    material_textures: dict[str, dict],
    mode: str = "per_material",
) -> list[PrimTextureUnit]:
    """Expand materials into texture generation units.

    Args:
        materials: Discovered materials with bound prim info.
        material_textures: Per-material texture specs from config.
        mode: "per_material" (one texture per material) or
              "per_prim" (unique texture per geometry prim).

    Returns:
        List of PrimTextureUnit, one per generation job.
    """
    units: list[PrimTextureUnit] = []

    for mat in materials:
        spec = material_textures.get(mat.name)
        if not spec:
            continue

        base_prompt = spec.get("prompt", "")
        base_opacity = spec.get("opacity", 0.85)

        if mode == "per_prim" and mat.bound_prim_paths:
            # One unit per bound prim
            per_prim_overrides = spec.get("per_prim", {})

            # Detect leaf name collisions within this material
            leaf_names = [_sanitize_prim_name(p) for p in mat.bound_prim_paths]
            has_collision = len(leaf_names) != len(set(leaf_names))

            for prim_path in mat.bound_prim_paths:
                leaf = _sanitize_prim_name(prim_path)

                # Use full sanitized path if leaf names collide
                if has_collision:
                    safe_name = prim_path.strip("/").replace("/", "_")
                else:
                    safe_name = leaf

                key = f"{mat.name}__{safe_name}"

                # Check for per-prim overrides (by full path or leaf name)
                override = (
                    per_prim_overrides.get(prim_path)
                    or per_prim_overrides.get(leaf)
                    or {}
                )

                prompt = override.get("prompt", base_prompt)
                opacity = override.get("opacity", base_opacity)
                seed = _stable_hash(prim_path)

                units.append(
                    PrimTextureUnit(
                        prim_path=prim_path,
                        material_info=mat,
                        key=key,
                        prompt=prompt,
                        opacity=opacity,
                        seed=seed,
                    )
                )
        else:
            # Per-material mode: one unit per material
            units.append(
                PrimTextureUnit(
                    prim_path="",
                    material_info=mat,
                    key=mat.name,
                    prompt=base_prompt,
                    opacity=base_opacity,
                )
            )

    return units


def discover_materials_from_file(
    usd_path: str | Path,
    prim_paths: list[str] | None = None,
) -> list[MaterialInfo]:
    """Convenience wrapper that opens a USD file and discovers materials.

    Args:
        usd_path: Path to the USD file.
        prim_paths: Optional material prim path filter.

    Returns:
        List of MaterialInfo.
    """
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise FileNotFoundError(f"Failed to open USD stage: {usd_path}")
    return discover_materials(stage, prim_paths)
