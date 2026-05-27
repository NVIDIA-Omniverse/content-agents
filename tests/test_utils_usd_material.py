# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD material asset discovery helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from world_understanding.utils.usd.material import (  # noqa: E402
    add_ovrtx_preview_fallbacks_for_materialx_openpbr,
    add_ovrtx_preview_fallbacks_to_stage_file,
    ensure_looks_scope,
    ensure_looks_scope_spec,
    get_local_mdl_assets,
    get_local_texture_file_assets,
    write_ovrtx_preview_fallback_overlay_for_materialx_openpbr,
)


def test_asset_discovery_prefers_usd_resolved_path_for_sublayer_assets(
    tmp_path: Path,
) -> None:
    asset_dir = tmp_path / "asset"
    texture_dir = asset_dir / "textures"
    material_dir = asset_dir / "materials"
    texture_dir.mkdir(parents=True)
    material_dir.mkdir()
    texture_path = texture_dir / "diffuse.png"
    mdl_path = material_dir / "surface.mdl"
    texture_path.write_bytes(b"not-a-real-png")
    mdl_path.write_text("// test mdl\n", encoding="utf-8")

    sublayer_path = asset_dir / "model.usda"
    sublayer_stage = Usd.Stage.CreateNew(str(sublayer_path))
    texture_shader = UsdShade.Shader.Define(sublayer_stage, "/TextureShader")
    texture_shader.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("textures/diffuse.png"))
    mdl_shader = UsdShade.Shader.Define(sublayer_stage, "/MdlShader")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("materials/surface.mdl"))
    sublayer_stage.Save()

    root_path = tmp_path / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.GetRootLayer().subLayerPaths.append("asset/model.usda")
    root_stage.Save()
    composed_stage = Usd.Stage.Open(str(root_path))
    assert composed_stage is not None

    texture_assets = get_local_texture_file_assets(
        composed_stage,
        base_dir=tmp_path,
    )
    mdl_assets = get_local_mdl_assets(composed_stage, base_dir=tmp_path)

    assert [asset["resolved_path"] for asset in texture_assets] == [
        str(texture_path.resolve())
    ]
    assert [asset["resolved_path"] for asset in mdl_assets] == [str(mdl_path.resolve())]


def test_texture_asset_discovery_skips_embedded_data_uris(tmp_path: Path) -> None:
    stage_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    texture_shader = UsdShade.Shader.Define(stage, "/TextureShader")
    data_uri = "data:image/png;base64," + ("A" * 600)
    texture_shader.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(data_uri))
    stage.Save()

    texture_assets = get_local_texture_file_assets(stage, base_dir=tmp_path)

    assert texture_assets == [
        {
            "prim_path": "/TextureShader",
            "attr_name": "inputs:file",
            "file_path": data_uri,
            "resolved_path": None,
            "is_local": False,
        }
    ]


def _create_materialx_openpbr_material(
    stage: Usd.Stage,
    material_path: str = "/World/Looks/Gold",
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()
    prim.CreateAttribute("inputs:base_color", Sdf.ValueTypeNames.Color3f).Set(
        (1.0, 0.766, 0.336),
    )
    prim.CreateAttribute("inputs:base_metalness", Sdf.ValueTypeNames.Float).Set(1.0)
    prim.CreateAttribute("inputs:specular_roughness", Sdf.ValueTypeNames.Float).Set(
        0.05,
    )
    prim.CreateAttribute("inputs:geometry_opacity", Sdf.ValueTypeNames.Float).Set(0.8)

    shader = UsdShade.Shader.Define(
        stage,
        f"{material_path}/open_pbr_surface_surfaceshader",
    )
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mtlx").ConnectToSource(shader_output)
    material.CreateSurfaceOutput()
    return material


def _connect_existing_preview_surface(material: UsdShade.Material) -> None:
    stage = material.GetPrim().GetStage()
    material_path = str(material.GetPath())
    shader = UsdShade.Shader.Define(stage, f"{material_path}/ExistingPreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader_output)


def _connected_surface_shader_id(material: UsdShade.Material) -> str | None:
    output = material.GetSurfaceOutput()
    if not output:
        return None
    sources, _ = output.GetConnectedSources()
    if not sources:
        return None
    shader = UsdShade.Shader(sources[0].source.GetPrim())
    return shader.GetIdAttr().Get()


def test_adds_ovrtx_preview_fallback_for_materialx_openpbr() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)

    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 1

    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert material.GetSurfaceOutput("mtlx").HasConnectedSource()

    shader = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Gold/OVRTXPreviewSurface"),
    )
    assert shader.GetInput("diffuseColor").Get() == (1.0, 0.766, 0.336)
    assert shader.GetInput("metallic").Get() == 1.0
    assert shader.GetInput("roughness").Get() == pytest.approx(0.05)
    assert shader.GetInput("opacity").Get() == pytest.approx(0.8)


def test_ovrtx_preview_fallback_is_idempotent_when_surface_exists() -> None:
    stage = Usd.Stage.CreateInMemory()
    _create_materialx_openpbr_material(stage)

    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 1
    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 0


def test_ovrtx_preview_fallback_disables_instanceable_material() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    material.GetPrim().SetInstanceable(True)

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    assert not material.GetPrim().IsInstanceable()
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_disables_referenced_material_instance(
    tmp_path: Path,
) -> None:
    prototype_path = tmp_path / "prototype.usda"
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    _create_materialx_openpbr_material(prototype_stage, "/Prototype/Gold")
    prototype_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    material.GetPrim().GetReferences().AddReference(
        str(prototype_path),
        "/Prototype/Gold",
    )
    material.GetPrim().SetInstanceable(True)
    assert material.GetPrim().IsInstance()

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    assert not material.GetPrim().IsInstance()
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_suppresses_materialx_when_surface_exists() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    _connect_existing_preview_surface(material)

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()
    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 0
    )


def test_ovrtx_preview_suppression_disables_instanceable_material() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    _connect_existing_preview_surface(material)
    material.GetPrim().SetInstanceable(True)

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    assert not material.GetPrim().IsInstanceable()
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_preserves_mdl_materials() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Gold/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)

    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 0
    assert _connected_surface_shader_id(material) is None


def test_ovrtx_preview_fallback_overlay_covers_sublayered_materials(
    tmp_path: Path,
) -> None:
    material_layer_path = tmp_path / "materials.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    _create_materialx_openpbr_material(material_stage)
    material_stage.GetRootLayer().Save()

    root_layer_path = tmp_path / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_layer_path))
    root_layer.subLayerPaths = [str(material_layer_path)]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_layer_path))
    assert stage is not None
    overlay_path = tmp_path / "ovrtx_material_fallbacks.usda"

    assert (
        write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            overlay_path,
        )
        == 1
    )

    combined_path = tmp_path / "combined.usda"
    combined_layer = Sdf.Layer.CreateNew(str(combined_path))
    combined_layer.subLayerPaths = [str(overlay_path), str(root_layer_path)]
    combined_layer.Save()

    combined = Usd.Stage.Open(str(combined_path))
    assert combined is not None
    material = UsdShade.Material(combined.GetPrimAtPath("/World/Looks/Gold"))
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()

    original = Usd.Stage.Open(str(material_layer_path))
    assert original is not None
    original_material = UsdShade.Material(
        original.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert _connected_surface_shader_id(original_material) is None
    assert original_material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_overlay_disables_referenced_material_instance(
    tmp_path: Path,
) -> None:
    prototype_path = tmp_path / "prototype.usda"
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    _create_materialx_openpbr_material(prototype_stage, "/Prototype/Gold")
    prototype_stage.GetRootLayer().Save()

    material_layer_path = tmp_path / "materials.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    material = UsdShade.Material.Define(material_stage, "/World/Looks/Gold")
    material.GetPrim().GetReferences().AddReference(
        str(prototype_path),
        "/Prototype/Gold",
    )
    material.GetPrim().SetInstanceable(True)
    material_stage.GetRootLayer().Save()

    root_layer_path = tmp_path / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_layer_path))
    root_layer.subLayerPaths = [str(material_layer_path)]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_layer_path))
    assert stage is not None
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Gold"))
    assert material.GetPrim().IsInstance()

    overlay_path = tmp_path / "ovrtx_material_fallbacks.usda"
    assert (
        write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            overlay_path,
        )
        == 1
    )

    combined_path = tmp_path / "combined.usda"
    combined_layer = Sdf.Layer.CreateNew(str(combined_path))
    combined_layer.subLayerPaths = [str(overlay_path), str(root_layer_path)]
    combined_layer.Save()

    combined = Usd.Stage.Open(str(combined_path))
    assert combined is not None
    combined_material = UsdShade.Material(
        combined.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert not combined_material.GetPrim().IsInstance()
    assert _connected_surface_shader_id(combined_material) == "UsdPreviewSurface"
    assert not combined_material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_overlay_suppresses_existing_surface_materialx(
    tmp_path: Path,
) -> None:
    material_layer_path = tmp_path / "materials.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    material = _create_materialx_openpbr_material(material_stage)
    _connect_existing_preview_surface(material)
    material_stage.GetRootLayer().Save()

    root_layer_path = tmp_path / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_layer_path))
    root_layer.subLayerPaths = [str(material_layer_path)]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_layer_path))
    assert stage is not None
    overlay_path = tmp_path / "ovrtx_material_fallbacks.usda"

    assert (
        write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            overlay_path,
        )
        == 1
    )

    combined_path = tmp_path / "combined.usda"
    combined_layer = Sdf.Layer.CreateNew(str(combined_path))
    combined_layer.subLayerPaths = [str(overlay_path), str(root_layer_path)]
    combined_layer.Save()

    combined = Usd.Stage.Open(str(combined_path))
    assert combined is not None
    combined_material = UsdShade.Material(
        combined.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert _connected_surface_shader_id(combined_material) == "UsdPreviewSurface"
    assert not combined_material.GetSurfaceOutput("mtlx").HasConnectedSource()

    original = Usd.Stage.Open(str(material_layer_path))
    assert original is not None
    original_material = UsdShade.Material(
        original.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert _connected_surface_shader_id(original_material) == "UsdPreviewSurface"
    assert original_material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_overlay_suppresses_referenced_surface_instance(
    tmp_path: Path,
) -> None:
    prototype_path = tmp_path / "prototype.usda"
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    prototype_material = _create_materialx_openpbr_material(
        prototype_stage,
        "/Prototype/Gold",
    )
    _connect_existing_preview_surface(prototype_material)
    prototype_stage.GetRootLayer().Save()

    material_layer_path = tmp_path / "materials.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    material = UsdShade.Material.Define(material_stage, "/World/Looks/Gold")
    material.GetPrim().GetReferences().AddReference(
        str(prototype_path),
        "/Prototype/Gold",
    )
    material.GetPrim().SetInstanceable(True)
    material_stage.GetRootLayer().Save()

    root_layer_path = tmp_path / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_layer_path))
    root_layer.subLayerPaths = [str(material_layer_path)]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_layer_path))
    assert stage is not None
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Gold"))
    assert material.GetPrim().IsInstance()
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert material.GetSurfaceOutput("mtlx").HasConnectedSource()

    overlay_path = tmp_path / "ovrtx_material_fallbacks.usda"
    assert (
        write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            overlay_path,
        )
        == 1
    )

    combined_path = tmp_path / "combined.usda"
    combined_layer = Sdf.Layer.CreateNew(str(combined_path))
    combined_layer.subLayerPaths = [str(overlay_path), str(root_layer_path)]
    combined_layer.Save()

    combined = Usd.Stage.Open(str(combined_path))
    assert combined is not None
    combined_material = UsdShade.Material(
        combined.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert not combined_material.GetPrim().IsInstance()
    assert _connected_surface_shader_id(combined_material) == "UsdPreviewSurface"
    assert not combined_material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_default_material_library_gets_ovrtx_preview_fallbacks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "apps"
        / "material_agent"
        / "data"
        / "materials"
        / "material_libs_default"
        / "materials_libs_v2.usd"
    )
    exported = tmp_path / "materials_libs_v2.usda"
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    assert stage.GetRootLayer().Export(str(exported))

    before = Usd.Stage.Open(str(exported))
    assert before is not None
    material_count = sum(1 for prim in before.Traverse() if prim.IsA(UsdShade.Material))

    assert add_ovrtx_preview_fallbacks_to_stage_file(exported) == material_count

    after = Usd.Stage.Open(str(exported))
    assert after is not None
    for prim in after.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        material = UsdShade.Material(prim)
        assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
        assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ensure_looks_scope_spec_types_existing_untyped_looks_only() -> None:
    layer = Sdf.Layer.CreateAnonymous()
    looks_spec = Sdf.CreatePrimInLayer(layer, "/Root/Looks")
    looks_spec.specifier = Sdf.SpecifierDef
    materials_spec = Sdf.CreatePrimInLayer(layer, "/Root/Materials")
    materials_spec.specifier = Sdf.SpecifierDef

    ensure_looks_scope_spec(layer, "/Root/Looks")
    ensure_looks_scope_spec(layer, "/Root/Materials")

    assert looks_spec.typeName == "Scope"
    assert not materials_spec.typeName


def test_ensure_looks_scope_spec_requires_opt_in_for_over_specs() -> None:
    layer = Sdf.Layer.CreateAnonymous()
    looks_spec = Sdf.CreatePrimInLayer(layer, "/Root/Looks")

    ensure_looks_scope_spec(layer, "/Root/Looks")
    assert not looks_spec.typeName

    ensure_looks_scope_spec(layer, "/Root/Looks", allow_over=True)
    assert looks_spec.typeName == "Scope"


def test_ensure_looks_scope_preserves_existing_looks_specifier() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")
    layer = stage.GetEditTarget().GetLayer()
    looks_spec = Sdf.CreatePrimInLayer(layer, "/Root/Looks")
    assert looks_spec.specifier == Sdf.SpecifierOver

    ensure_looks_scope(stage, "/Root/Looks/PhysMat")

    assert looks_spec.typeName == "Scope"
    assert looks_spec.specifier == Sdf.SpecifierOver


def test_ensure_looks_scope_defines_missing_looks_scope() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")

    ensure_looks_scope(stage, "/Root/Looks/PhysMat")

    looks_spec = stage.GetEditTarget().GetLayer().GetPrimAtPath("/Root/Looks")
    assert looks_spec.typeName == "Scope"
    assert looks_spec.specifier == Sdf.SpecifierDef


def test_ensure_looks_scope_types_all_looks_ancestors() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")
    layer = stage.GetEditTarget().GetLayer()
    outer_looks = Sdf.CreatePrimInLayer(layer, "/Root/Looks")
    inner_looks = Sdf.CreatePrimInLayer(layer, "/Root/Looks/Asset/Looks")

    ensure_looks_scope(stage, "/Root/Looks/Asset/Looks/PhysMat")

    assert inner_looks.typeName == "Scope"
    assert outer_looks.typeName == "Scope"
