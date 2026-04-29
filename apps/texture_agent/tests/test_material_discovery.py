# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material discovery functions."""

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from texture_agent.functions.material_discovery import (
    MaterialInfo,
    PrimTextureUnit,
    discover_materials,
    expand_to_prim_units,
)


def _create_stage_with_material(
    base_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    metalness: float = 1.0,
    roughness: float = 0.3,
    texture_file: str | None = None,
    material_name: str = "TestMaterial",
) -> Usd.Stage:
    """Create an in-memory USD stage with a sphere + OpenPBR material."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Create world
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    # Create geometry
    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    sphere.GetRadiusAttr().Set(1.0)

    # Create Looks scope and material
    UsdGeom.Scope.Define(stage, "/World/Looks")
    mat_path = f"/World/Looks/{material_name}"
    material = UsdShade.Material.Define(stage, mat_path)

    # Set OpenPBR inputs on the material prim
    mat_prim = material.GetPrim()
    mat_prim.CreateAttribute("inputs:base_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*base_color)
    )
    mat_prim.CreateAttribute("inputs:base_metalness", Sdf.ValueTypeNames.Float).Set(
        metalness
    )
    mat_prim.CreateAttribute("inputs:specular_roughness", Sdf.ValueTypeNames.Float).Set(
        roughness
    )

    tex_path = texture_file if texture_file else ""
    mat_prim.CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(tex_path))

    # Bind material to sphere
    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
    binding_api.Bind(material)

    return stage


class TestDiscoverMaterials:
    """Tests for discover_materials()."""

    def test_discovers_single_material(self) -> None:
        """Discovers a single material with correct properties."""
        stage = _create_stage_with_material(
            base_color=(0.9, 0.6, 0.5),
            metalness=1.0,
            roughness=0.15,
            material_name="Copper",
        )

        materials = discover_materials(stage)

        assert len(materials) == 1
        mat = materials[0]
        assert mat.name == "Copper"
        assert mat.prim_path == "/World/Looks/Copper"
        assert pytest.approx(mat.base_color[0], abs=0.01) == 0.9
        assert pytest.approx(mat.base_color[1], abs=0.01) == 0.6
        assert pytest.approx(mat.base_color[2], abs=0.01) == 0.5
        assert mat.base_metalness == pytest.approx(1.0)
        assert mat.specular_roughness == pytest.approx(0.15)
        assert mat.has_existing_texture is False
        assert len(mat.bound_prim_paths) == 1
        assert mat.bound_prim_paths[0] == "/World/Sphere"

    def test_detects_existing_texture(self) -> None:
        """Correctly flags materials that already have a texture file."""
        stage = _create_stage_with_material(
            texture_file="/path/to/albedo.png",
            material_name="Textured",
        )

        materials = discover_materials(stage)

        assert len(materials) == 1
        assert materials[0].has_existing_texture is True
        assert materials[0].base_color_texture == "/path/to/albedo.png"

    def test_empty_texture_is_not_existing(self) -> None:
        """Empty texture path ('') is treated as no texture."""
        stage = _create_stage_with_material(
            texture_file="",
            material_name="NoTexture",
        )

        materials = discover_materials(stage)

        assert len(materials) == 1
        assert materials[0].has_existing_texture is False
        assert materials[0].base_color_texture is None

    def test_multiple_materials(self) -> None:
        """Discovers multiple materials in one stage."""
        stage = Usd.Stage.CreateInMemory()
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

        UsdGeom.Scope.Define(stage, "/World/Looks")

        for name, color in [("Steel", (0.3, 0.3, 0.3)), ("Gold", (1.0, 0.8, 0.3))]:
            mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            mat.GetPrim().CreateAttribute(
                "inputs:base_color", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*color))
            mat.GetPrim().CreateAttribute(
                "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
            ).Set(Sdf.AssetPath(""))

        materials = discover_materials(stage)

        assert len(materials) == 2
        names = {m.name for m in materials}
        assert names == {"Steel", "Gold"}

    def test_prim_path_filter(self) -> None:
        """prim_paths filter restricts which materials are returned."""
        stage = Usd.Stage.CreateInMemory()
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

        UsdGeom.Scope.Define(stage, "/World/Looks")
        for name in ["MatA", "MatB", "MatC"]:
            mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            mat.GetPrim().CreateAttribute(
                "inputs:base_color", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(0.5, 0.5, 0.5))

        materials = discover_materials(stage, prim_paths=["/World/Looks/MatB"])

        assert len(materials) == 1
        assert materials[0].name == "MatB"

    def test_no_materials(self) -> None:
        """Returns empty list when no materials exist."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")

        materials = discover_materials(stage)

        assert materials == []


class TestExpandToPrimUnits:
    """Tests for expand_to_prim_units()."""

    def _make_material(self, name: str, bound: list[str] | None = None) -> MaterialInfo:
        return MaterialInfo(
            prim_path=f"/World/Looks/{name}",
            name=name,
            bound_prim_paths=bound or [],
            base_color=(0.5, 0.5, 0.5),
        )

    def test_per_material_mode(self) -> None:
        """Per-material mode creates one unit per material."""
        materials = [
            self._make_material("Steel", ["/World/A", "/World/B"]),
            self._make_material("Copper", ["/World/C"]),
        ]
        specs = {
            "Steel": {"prompt": "rusty steel", "opacity": 0.8},
            "Copper": {"prompt": "patina copper", "opacity": 0.7},
        }

        units = expand_to_prim_units(materials, specs, mode="per_material")

        assert len(units) == 2
        assert units[0].key == "Steel"
        assert units[0].prim_path == ""
        assert units[1].key == "Copper"

    def test_per_prim_mode(self) -> None:
        """Per-prim mode creates one unit per bound prim."""
        materials = [
            self._make_material("Steel", ["/World/Rail_L", "/World/Rail_R"]),
        ]
        specs = {"Steel": {"prompt": "rusty steel", "opacity": 0.8}}

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        assert len(units) == 2
        assert units[0].key == "Steel__Rail_L"
        assert units[0].prim_path == "/World/Rail_L"
        assert units[1].key == "Steel__Rail_R"
        assert units[1].prim_path == "/World/Rail_R"
        # Different seeds
        assert units[0].seed != units[1].seed

    def test_per_prim_with_overrides(self) -> None:
        """Per-prim overrides provide per-prim prompts."""
        materials = [
            self._make_material("Steel", ["/World/Rail_L", "/World/Rail_R"]),
        ]
        specs = {
            "Steel": {
                "prompt": "rusty steel",
                "opacity": 0.8,
                "per_prim": {
                    "/World/Rail_L": {
                        "prompt": "heavily rusted left rail",
                        "opacity": 0.95,
                    }
                },
            }
        }

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        assert len(units) == 2
        left = next(u for u in units if "Rail_L" in u.key)
        right = next(u for u in units if "Rail_R" in u.key)
        assert left.prompt == "heavily rusted left rail"
        assert left.opacity == 0.95
        assert right.prompt == "rusty steel"  # inherits from parent
        assert right.opacity == 0.8

    def test_skips_materials_without_spec(self) -> None:
        """Materials not in material_textures are skipped."""
        materials = [
            self._make_material("Steel", ["/World/A"]),
            self._make_material("Unknown", ["/World/B"]),
        ]
        specs = {"Steel": {"prompt": "rusty", "opacity": 0.8}}

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        assert len(units) == 1
        assert units[0].key == "Steel__A"

    def test_no_bound_prims_per_prim(self) -> None:
        """Material with no bound prims in per-prim mode falls back to per-material."""
        materials = [self._make_material("Steel", [])]
        specs = {"Steel": {"prompt": "rusty", "opacity": 0.8}}

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        assert len(units) == 1
        assert units[0].key == "Steel"
        assert units[0].prim_path == ""
