# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Apply PBR textures to materials in USD.

Supports per-material mode (shared texture) and per-prim mode (unique
texture per geometry prim via material cloning).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom, UsdShade
from world_understanding.agentic.tasks import Task

from texture_agent.functions.material_discovery import PrimTextureUnit
from texture_agent.tasks.blend_textures import BlendedTextures

logger = logging.getLogger(__name__)


def _clone_material(
    stage: Usd.Stage,
    source_mat_path: str,
    clone_name: str,
) -> str:
    """Clone a material prim (deep copy of entire shader subtree).

    Args:
        stage: The USD stage.
        source_mat_path: Path to the source material prim.
        clone_name: Name for the cloned material.

    Returns:
        Path to the cloned material prim.
    """
    parent_path = str(Sdf.Path(source_mat_path).GetParentPath())
    clone_path = f"{parent_path}/{clone_name}"

    layer = stage.GetRootLayer()
    Sdf.CopySpec(layer, source_mat_path, layer, clone_path)

    logger.debug("Cloned material: %s -> %s", source_mat_path, clone_path)
    return clone_path


def _set_texture_attr(
    prim: Usd.Prim,
    attr_name: str,
    texture_path: str,
) -> None:
    """Set an asset path attribute on a prim, creating if needed."""
    attr = prim.GetAttribute(attr_name)
    if attr and attr.IsValid():
        attr.Set(Sdf.AssetPath(texture_path))
    else:
        prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_path)
        )


def _set_tiledimage_file_input(
    stage: Usd.Stage,
    mat_path: str,
    shader_name: str,
    texture_path: str,
) -> None:
    """Set the concrete tiledimage shader input used by NVCF/OpenPBR."""
    shader_prim = stage.GetPrimAtPath(f"{mat_path}/{shader_name}")
    if not shader_prim.IsValid():
        logger.debug(
            "OpenPBR tiledimage shader not found: %s/%s", mat_path, shader_name
        )
        return

    if not shader_prim.IsA(UsdShade.Shader):
        logger.debug("Prim is not a UsdShade shader: %s", shader_prim.GetPath())
        return

    shader = UsdShade.Shader(shader_prim)
    file_input = shader.GetInput("file")
    if file_input:
        file_input.Set(Sdf.AssetPath(texture_path))
    else:
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_path)
        )


def _apply_pbr_textures(
    stage: Usd.Stage,
    mat_path: str,
    textures: BlendedTextures,
    working_dir: Path,
    key: str,
) -> None:
    """Apply albedo, normal, and ORM textures to a material prim."""
    prim = stage.GetPrimAtPath(mat_path)
    if not prim.IsValid():
        logger.warning("Material prim not found: %s", mat_path)
        return

    # Ensure parent Looks scope is defined for NVCF traversal
    parent = prim.GetParent()
    if parent.IsValid() and not parent.IsDefined():
        UsdGeom.Scope.Define(stage, parent.GetPath())

    # Albedo
    _set_texture_attr(prim, "inputs:base_color_texture_file", textures.albedo)
    _set_tiledimage_file_input(
        stage,
        mat_path,
        "tiledimage_base_color",
        textures.albedo,
    )

    # Normal
    if textures.normal and Path(textures.normal).exists():
        _set_texture_attr(prim, "inputs:geometry_normal_texture_file", textures.normal)
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_geometry_normal",
            textures.normal,
        )

    # ORM → unpack into roughness + metalness
    if textures.orm and Path(textures.orm).exists():
        import numpy as np
        from PIL import Image

        orm_img = Image.open(textures.orm)
        orm_arr = np.array(orm_img)
        tex_dir = working_dir / "textures"

        roughness_arr = orm_arr[:, :, 1]
        roughness_path = tex_dir / f"{key}_roughness.png"
        Image.fromarray(roughness_arr).save(str(roughness_path))
        _set_texture_attr(
            prim, "inputs:specular_roughness_texture_file", str(roughness_path)
        )
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_specular_roughness",
            str(roughness_path),
        )

        metalness_arr = orm_arr[:, :, 2]
        metalness_path = tex_dir / f"{key}_metalness.png"
        Image.fromarray(metalness_arr).save(str(metalness_path))
        _set_texture_attr(
            prim, "inputs:base_metalness_texture_file", str(metalness_path)
        )
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_base_metalness",
            str(metalness_path),
        )


class ApplyTexturesTask(Task):
    """Set PBR texture file paths on OpenPBR materials in the USD stage.

    In per-material mode: applies textures directly to shared materials.
    In per-prim mode: clones materials so each prim gets its own texture,
    then re-binds each geometry prim to its cloned material.

    Context keys read:
        usd_path (str): Input USD file path.
        blended_textures (dict[str, BlendedTextures]): From BlendTexturesTask.
        prim_texture_units (list[PrimTextureUnit]): From DiscoverMaterialsTask.
        working_dir (str): Working directory.

    Context keys written:
        output_usd_paths (list[str]): Paths to output USD files.
    """

    def __init__(self) -> None:
        self.name = "ApplyTextures"
        self.description = "Apply PBR texture maps to materials in USD"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        usd_path = context["usd_path"]
        blended: dict[str, BlendedTextures] = context.get("blended_textures", {})
        units: list[PrimTextureUnit] = context.get("prim_texture_units", [])
        working_dir = Path(context["working_dir"])

        if not blended:
            logger.info("No blended textures to apply")
            context["output_usd_paths"] = []
            return context

        out_dir = working_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_usd_path = out_dir / "textured_output.usd"

        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            raise FileNotFoundError(f"Failed to open USD stage: {usd_path}")

        # Group units by material for cloning decisions
        units_by_material: dict[str, list[PrimTextureUnit]] = defaultdict(list)
        for unit in units:
            if unit.key in blended:
                units_by_material[unit.material_info.name].append(unit)

        applied_count = 0

        for _mat_name, mat_units in units_by_material.items():
            mat = mat_units[0].material_info

            if len(mat_units) == 1 and not mat_units[0].prim_path:
                # Per-material mode (or single prim): apply directly
                unit = mat_units[0]
                _apply_pbr_textures(
                    stage,
                    mat.prim_path,
                    blended[unit.key],
                    working_dir,
                    unit.key,
                )
                logger.info("Applied textures to %s (direct)", unit.key)
                applied_count += 1

            else:
                # Per-prim mode: clone material for each prim
                for unit in mat_units:
                    clone_name = unit.key
                    clone_path = _clone_material(stage, mat.prim_path, clone_name)

                    # Apply textures to the clone
                    _apply_pbr_textures(
                        stage,
                        clone_path,
                        blended[unit.key],
                        working_dir,
                        unit.key,
                    )

                    # Re-bind the geometry prim to the cloned material
                    if unit.prim_path:
                        geom_prim = stage.GetPrimAtPath(unit.prim_path)
                        if geom_prim.IsValid():
                            binding_api = UsdShade.MaterialBindingAPI.Apply(geom_prim)
                            cloned_mat = UsdShade.Material(
                                stage.GetPrimAtPath(clone_path)
                            )
                            binding_api.Bind(cloned_mat)
                            logger.info(
                                "Applied textures to %s (cloned, bound %s)",
                                unit.key,
                                unit.prim_path,
                            )
                        else:
                            logger.warning(
                                "Prim not found for rebinding: %s",
                                unit.prim_path,
                            )

                    applied_count += 1

        stage.GetRootLayer().Export(str(output_usd_path))
        logger.info(
            "Applied PBR textures to %d units, saved to %s",
            applied_count,
            output_usd_path,
        )

        context["output_usd_paths"] = [str(output_usd_path)]
        return context
