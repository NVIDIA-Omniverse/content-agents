# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import texture_agent.tasks.apply_textures as apply_textures_task
import texture_agent.tasks.blend_textures as blend_textures_task
import texture_agent.tasks.discover_materials as discover_materials_task
import texture_agent.tasks.generate_prompts as generate_prompts_task
import texture_agent.tasks.generate_textures as generate_textures_task
import texture_agent.tasks.prepare_uvs as prepare_uvs_task
import texture_agent.tasks.render as render_task
import texture_agent.tasks.render_previews as render_previews_task
from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.functions.texture_generation import GeneratedTextures

pytest.importorskip("pxr")


def _material(name: str, **overrides) -> MaterialInfo:
    data = {
        "prim_path": f"/Root/Looks/{name}",
        "name": name,
        "bound_prim_paths": [f"/Root/{name}_Mesh"],
        "base_color": (0.4, 0.5, 0.6),
        "base_metalness": 0.3,
        "specular_roughness": 0.2,
    }
    data.update(overrides)
    return MaterialInfo(**data)


def _unit(
    name: str = "Steel",
    key: str | None = None,
    prim_path: str = "",
    prompt: str = "prompt",
    opacity: float = 0.8,
) -> PrimTextureUnit:
    material = _material(name)
    return PrimTextureUnit(
        prim_path=prim_path,
        material_info=material,
        key=key or name,
        prompt=prompt,
        opacity=opacity,
    )


def _save_png(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (8, 8), color).save(path)
    return str(path)


def test_load_cached_blended_textures_uses_complete_texture_sets(
    tmp_path: Path,
) -> None:
    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    _save_png(textures_dir / "Steel_albedo.png", (10, 20, 30))
    _save_png(textures_dir / "Steel_normal.png", (128, 128, 255))
    _save_png(textures_dir / "Steel_orm.png", (255, 80, 10))
    _save_png(textures_dir / "Copper_albedo.png", (40, 50, 60))

    cached = apply_textures_task._load_cached_blended_textures(
        tmp_path,
        [_unit(key="Steel"), _unit(key="Copper")],
    )

    assert set(cached) == {"Steel"}
    assert cached["Steel"].albedo == str(textures_dir / "Steel_albedo.png")
    assert cached["Steel"].normal == str(textures_dir / "Steel_normal.png")
    assert cached["Steel"].orm == str(textures_dir / "Steel_orm.png")


def test_discover_materials_task_persists_summary(tmp_path: Path, monkeypatch) -> None:
    task = discover_materials_task.DiscoverMaterialsTask()
    materials = [_material("Steel", bound_prim_paths=["/Root/A", "/Root/B"])]
    monkeypatch.setattr(
        discover_materials_task,
        "discover_materials_from_file",
        lambda usd_path, prim_paths=None: materials,
    )

    context = {
        "usd_path": "/tmp/input.usd",
        "prim_paths": ["/Root/Looks/Steel"],
        "working_dir": str(tmp_path),
    }
    result = task.run(context)

    assert result["discovered_materials"] == materials
    summary_path = tmp_path / "discovery" / "materials.json"
    assert summary_path.exists()
    assert "Steel" in summary_path.read_text(encoding="utf-8")


def test_generate_prompts_task_handles_empty_materials() -> None:
    task = generate_prompts_task.GeneratePromptsTask()

    context = task.run({"discovered_materials": [], "working_dir": "/tmp"})

    assert context["prim_texture_units"] == []


def test_generate_prompts_task_uses_fallback_when_llm_missing(
    tmp_path: Path, monkeypatch
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()
    materials = [_material("Steel")]
    captured = {}

    import world_understanding.functions.models.chat_models as chat_models

    monkeypatch.setattr(
        chat_models, "create_chat_model_from_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        generate_prompts_task,
        "_fallback_prompts",
        lambda needs_prompt, user_prompt, default_opacity: {
            "Steel": {"prompt": "fallback steel", "opacity": default_opacity}
        },
    )
    monkeypatch.setattr(
        generate_prompts_task,
        "expand_to_prim_units",
        lambda materials, material_textures, mode: captured.setdefault(
            "units",
            [
                PrimTextureUnit(
                    prim_path="",
                    material_info=materials[0],
                    key="Steel",
                    prompt=material_textures["Steel"]["prompt"],
                    opacity=material_textures["Steel"]["opacity"],
                )
            ],
        ),
    )

    result = task.run(
        {
            "discovered_materials": materials,
            "material_textures": {},
            "auto_prompt_config": {"user_prompt": "aged", "default_opacity": 0.65},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
        }
    )

    assert result["material_textures"]["Steel"]["prompt"] == "fallback steel"
    assert result["prim_texture_units"][0].prompt == "fallback steel"
    assert (tmp_path / "prompts" / "material_prompts.json").exists()


def test_prepare_uvs_task_leaves_input_when_no_fixes(
    monkeypatch, tmp_path: Path
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()

    class FakeLayer:
        def Export(self, _path: str) -> None:
            raise AssertionError("Export should not be called when no fixes are needed")

    class FakeStage:
        def Flatten(self):
            return object()

        def GetRootLayer(self):
            return FakeLayer()

    fake_stage = FakeStage()

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", lambda value: fake_stage)
    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", lambda stage, mode: 0
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 0)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 0)

    context = {
        "usd_path": "/tmp/original.usd",
        "working_dir": str(tmp_path),
        "texture_config": {"uv_mode": "box"},
    }

    result = task.run(context)

    assert result["usd_path"] == "/tmp/original.usd"
    assert result["uv_preparation"] == {
        "generated": 0,
        "fixed_interpolation": 0,
        "normalized": 0,
    }


def test_prepare_uvs_task_saves_prepared_copy(monkeypatch, tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    exported: list[str] = []

    class FakeLayer:
        def Export(self, path: str) -> None:
            exported.append(path)
            Path(path).write_text("#usda 1.0\n", encoding="utf-8")

    class FakeStage:
        def __init__(self, layer=None):
            self._layer = layer or object()

        def Flatten(self):
            return self._layer

        def GetRootLayer(self):
            return FakeLayer()

    flat_layer = object()

    def fake_open(value):
        if value is flat_layer:
            return FakeStage(flat_layer)
        return FakeStage(flat_layer)

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", fake_open)
    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", lambda stage, mode: 2
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 1)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 3)

    context = {
        "usd_path": "/tmp/original.usd",
        "working_dir": str(tmp_path),
        "texture_config": {"uv_mode": "box"},
    }

    result = task.run(context)

    prepared_path = tmp_path / "prepared" / "prepared_input.usd"
    assert result["usd_path"] == str(prepared_path)
    assert exported == [str(prepared_path)]
    assert result["uv_preparation"] == {
        "generated": 2,
        "fixed_interpolation": 1,
        "normalized": 3,
    }


def test_prepare_uvs_task_falls_back_from_scene_optimizer(
    monkeypatch, tmp_path: Path
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    exported: list[str] = []
    so_call = {}
    fallback_modes = []

    class FakeLayer:
        def Export(self, path: str) -> None:
            exported.append(path)
            Path(path).write_text("#usda 1.0\n", encoding="utf-8")

    class FakeStage:
        def Flatten(self):
            return object()

        def GetRootLayer(self):
            return FakeLayer()

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", lambda value: FakeStage())

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        so_call.update(
            {
                "input_path": input_path,
                "output_path": output_path,
                **kwargs,
            }
        )
        raise RuntimeError("SO package missing directory: python")

    monkeypatch.setattr(
        prepare_uvs_task, "generate_projection_uvs", fake_generate_projection_uvs
    )

    def fake_generate_uvs_for_stage(stage, mode):
        fallback_modes.append(mode)
        return 4

    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", fake_generate_uvs_for_stage
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 1)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 2)

    context = {
        "usd_path": "/tmp/original.usd",
        "working_dir": str(tmp_path),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_mode": "planar",
            "uv_projection": "cube",
            "uv_so_backend": "local",
            "uv_allow_remote_fallback": False,
        },
    }

    result = task.run(context)

    flat_path = tmp_path / "prepared" / "prepared_input_flat.usd"
    prepared_path = tmp_path / "prepared" / "prepared_input.usd"
    assert so_call["input_path"] == flat_path
    assert so_call["output_path"] == prepared_path
    assert so_call["projection_type"] == prepare_uvs_task.ProjectionType.CUBE
    assert so_call["backend"] == "local"
    assert so_call["allow_remote_fallback"] is False
    assert fallback_modes == [prepare_uvs_task.UVProjectionMode.BOX]
    assert exported == [str(flat_path), str(prepared_path)]
    assert result["usd_path"] == str(prepared_path)
    assert result["uv_preparation"] == {
        "generated": 4,
        "fixed_interpolation": 1,
        "normalized": 2,
    }


def test_generate_textures_task_reuses_existing_outputs(tmp_path: Path) -> None:
    task = generate_textures_task.GenerateTexturesTask()
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    albedo = out_dir / "Steel_albedo.png"
    normal = out_dir / "Steel_normal.png"
    orm = out_dir / "Steel_orm.png"
    _save_png(albedo, (10, 20, 30))
    _save_png(normal, (40, 50, 60))
    _save_png(orm, (70, 80, 90))

    result = task.run(
        {
            "prim_texture_units": [_unit()],
            "texture_config": {"skip_existing": True},
            "working_dir": str(tmp_path),
        }
    )

    assert result["generated_textures"]["Steel"] == GeneratedTextures(
        albedo=str(albedo),
        normal=str(normal),
        orm=str(orm),
    )


def test_generate_textures_task_unknown_backend_raises(tmp_path: Path) -> None:
    task = generate_textures_task.GenerateTexturesTask()

    with pytest.raises(ValueError, match="Unknown texture backend"):
        task.run(
            {
                "prim_texture_units": [_unit()],
                "texture_config": {"backend": "mystery"},
                "working_dir": str(tmp_path),
            }
        )


def test_localize_textures_copies_accessible_file_uri(tmp_path: Path) -> None:
    source = tmp_path / "remote_albedo.png"
    _save_png(source, (1, 2, 3))
    localized_dir = tmp_path / "localized"
    localized_dir.mkdir()

    result = generate_textures_task.GenerateTexturesTask._localize_textures(
        GeneratedTextures(
            albedo=f"file://{source}",
            normal="",
            orm="",
        ),
        key="Steel",
        out_dir=localized_dir,
        endpoint="http://service",
    )

    assert Path(result.albedo).exists()
    assert result.normal == ""
    assert result.orm == ""


def test_blend_textures_task_creates_outputs(tmp_path: Path) -> None:
    task = blend_textures_task.BlendTexturesTask()
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    albedo = _save_png(generated_dir / "Steel_albedo.png", (200, 100, 50))
    orm = _save_png(generated_dir / "Steel_orm.png", (20, 40, 60))
    unit = _unit(opacity=0.5)

    result = task.run(
        {
            "prim_texture_units": [unit],
            "generated_textures": {
                "Steel": GeneratedTextures(albedo=albedo, normal="", orm=orm)
            },
            "blend_config": {"output_size": 16},
            "working_dir": str(tmp_path),
        }
    )

    blended = result["blended_textures"]["Steel"]
    assert Path(blended.albedo).exists()
    assert Path(blended.normal).exists()
    assert Path(blended.orm).exists()


def test_apply_textures_task_applies_per_material(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    prim = material.GetPrim()
    prim.CreateAttribute("inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset)
    prim.CreateAttribute(
        "inputs:geometry_normal_texture_file", Sdf.ValueTypeNames.Asset
    )
    prim.CreateAttribute(
        "inputs:specular_roughness_texture_file", Sdf.ValueTypeNames.Asset
    )
    prim.CreateAttribute("inputs:base_metalness_texture_file", Sdf.ValueTypeNames.Asset)
    for shader_name in [
        "tiledimage_base_color",
        "tiledimage_geometry_normal",
        "tiledimage_specular_roughness",
        "tiledimage_base_metalness",
    ]:
        shader = UsdShade.Shader.Define(stage, f"/Root/Looks/Steel/{shader_name}")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset)
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal=_save_png(textures_dir / "steel_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "steel_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [_unit()],
            "working_dir": str(tmp_path),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    assert output_path.exists()
    output_stage = Usd.Stage.Open(str(output_path))
    output_prim = output_stage.GetPrimAtPath("/Root/Looks/Steel")
    assert (
        output_prim.GetAttribute("inputs:base_color_texture_file")
        .Get()
        .path.endswith("steel_albedo.png")
    )
    assert (
        output_prim.GetAttribute("inputs:geometry_normal_texture_file")
        .Get()
        .path.endswith("steel_normal.png")
    )
    assert (
        output_prim.GetAttribute("inputs:specular_roughness_texture_file")
        .Get()
        .path.endswith("Steel_roughness.png")
    )
    assert (
        output_prim.GetAttribute("inputs:base_metalness_texture_file")
        .Get()
        .path.endswith("Steel_metalness.png")
    )
    assert (
        UsdShade.Shader(
            output_stage.GetPrimAtPath("/Root/Looks/Steel/tiledimage_base_color")
        )
        .GetInput("file")
        .Get()
        .path.endswith("steel_albedo.png")
    )
    assert (
        UsdShade.Shader(
            output_stage.GetPrimAtPath("/Root/Looks/Steel/tiledimage_geometry_normal")
        )
        .GetInput("file")
        .Get()
        .path.endswith("steel_normal.png")
    )
    assert (
        UsdShade.Shader(
            output_stage.GetPrimAtPath(
                "/Root/Looks/Steel/tiledimage_specular_roughness"
            )
        )
        .GetInput("file")
        .Get()
        .path.endswith("Steel_roughness.png")
    )
    assert (
        UsdShade.Shader(
            output_stage.GetPrimAtPath("/Root/Looks/Steel/tiledimage_base_metalness")
        )
        .GetInput("file")
        .Get()
        .path.endswith("Steel_metalness.png")
    )


def test_apply_textures_task_overrides_prebaked_mdl_inputs(tmp_path: Path) -> None:
    """Reproduces NVBugs 6127229 / OMPE-91783: SimReady MDL materials have
    pre-baked Nucleus texture inputs (e.g. inputs:normalmap_texture) that the
    agent must overwrite with the freshly generated local textures, otherwise
    the output USD silently keeps the original references and renders broken
    once the bundle is downloaded outside Omniverse.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    # Mark the shader as MDL-sourced so the override path triggers.
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # Pre-bake Nucleus-hosted texture inputs (these would survive into the
    # output without the fix and produce broken refs after the service's
    # absolute → ../textures/<filename> rewrite step).
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Albedo.png")
    )
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Normal.png")
    )
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_ORM.png")
    )
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Opacity.png")
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    output_stage = Usd.Stage.Open(str(output_path))
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert (
        out_shader.GetInput("diffuse_texture").Get().path.endswith("Plastic_albedo.png")
    )
    assert (
        out_shader.GetInput("normalmap_texture")
        .Get()
        .path.endswith("Plastic_normal.png")
    )
    assert out_shader.GetInput("ORM_texture").Get().path.endswith("Plastic_orm.png")
    # Inputs we cannot map to a generated channel must be cleared (not left
    # pointing at the original Nucleus PNG), otherwise the service packager's
    # absolute → ../textures/<basename> rewrite step would create a dangling
    # reference to a file the bundle does not ship.
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_overridden"] >= 3
    assert any("opacity_texture" in entry for entry in stats["mdl_inputs_cleared"])


def test_apply_textures_task_localizes_local_unmapped_mdl_inputs(
    tmp_path: Path,
) -> None:
    """When an unmapped MDL `*_texture` input points at a *local* path that
    actually exists *inside the USD's upload directory*, the agent must copy
    that asset into the bundle textures dir and rewrite the input to that
    copy. Otherwise the service packager's `../textures/<basename>` rewrite
    would dangle on a file the bundle does not ship.
    """
    from pxr import Sdf, Usd, UsdShade

    # Local opacity map next to the input USD (relative path) and an
    # absolute-path emissive map under a sibling subdir — both within the
    # upload directory so the security gate accepts them.
    local_opacity = tmp_path / "local_opacity.png"
    _save_png(local_opacity, (10, 20, 30))
    abs_dir = tmp_path / "abs_assets"
    abs_dir.mkdir()
    abs_emissive = abs_dir / "local_emissive.png"
    _save_png(abs_emissive, (40, 50, 60))

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # One mapped input (Nucleus, must be overridden) + two unmapped local
    # paths (must be localized into the bundle textures dir).
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Normal.png")
    )
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./local_opacity.png")
    )
    shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(abs_emissive))
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    textures_dir = work_dir / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert (
        out_shader.GetInput("normalmap_texture")
        .Get()
        .path.endswith("Plastic_normal.png")
    )
    # Each preserved local input was copied into work_dir/textures/<safe_name>.
    opacity_out = Path(out_shader.GetInput("opacity_texture").Get().path)
    assert opacity_out.parent == textures_dir
    assert opacity_out.exists()
    assert opacity_out.name == "Plastic__opacity_texture.png"
    emissive_out = Path(out_shader.GetInput("emissive_color_texture").Get().path)
    assert emissive_out.parent == textures_dir
    assert emissive_out.exists()
    assert emissive_out.name == "Plastic__emissive_color_texture.png"
    # And the bytes are preserved (it's a real copy, not a placeholder).
    assert opacity_out.read_bytes() == local_opacity.read_bytes()
    assert emissive_out.read_bytes() == abs_emissive.read_bytes()

    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_cleared"] == []
    assert sorted(stats["mdl_inputs_localized"]) == sorted(
        [
            "/Root/Looks/Plastic:opacity_texture",
            "/Root/Looks/Plastic:emissive_color_texture",
        ]
    )


def test_apply_textures_task_clears_unresolvable_local_mdl_inputs(
    tmp_path: Path,
) -> None:
    """If an unmapped MDL `*_texture` input points at a local path that does
    not exist on disk (asset author's reference is already broken), the agent
    must clear it rather than ship a dangling ref into the bundle.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./does_not_exist.png")
    )
    shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("/nonexistent/abs/path.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    assert out_shader.GetInput("emissive_color_texture").Get().path == ""
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_localized"] == []
    assert sorted(stats["mdl_inputs_cleared"]) == sorted(
        [
            "/Root/Looks/Plastic:opacity_texture",
            "/Root/Looks/Plastic:emissive_color_texture",
        ]
    )


@pytest.mark.parametrize(
    "path,unbundleable",
    [
        ("omniverse://nucleus.example/T.png", True),
        ("http://example.com/T.png", True),
        ("https://example.com/T.png", True),
        ("file:///abs/T.png", True),
        ("./local.png", False),
        ("relative/local.png", False),
        ("/abs/local.png", False),
        ("C:/Users/me/T.png", False),  # Windows drive letter is not a URI scheme
        ("", False),
    ],
)
def test_is_unbundleable_asset_path_classification(path: str, unbundleable: bool):
    """Direct unit coverage for the URI-scheme classifier (Claude review nit)."""
    assert apply_textures_task._is_unbundleable_asset_path(path) is unbundleable


def test_apply_textures_task_refuses_localize_outside_usd_directory(
    tmp_path: Path,
) -> None:
    """Security regression for OMPE-91783: a malicious USD must not be able to
    use unmapped MDL `*_texture` inputs to copy host files outside the upload
    directory into the bundle textures dir, where they'd be exposed via the
    artifact download endpoint. Codex round-4 caught this CVE-class issue.
    """
    from pxr import Sdf, Usd, UsdShade

    # `outside_dir` is a sibling of the USD's directory, NOT under it. A
    # well-meaning author would never reach out here; an attacker would.
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.png"
    secret.write_bytes(b"secret-bytes")

    # Also a file with no/disallowed extension to verify the suffix gate.
    no_suffix = outside_dir / "passwd"
    no_suffix.write_bytes(b"root:x:0:0:")

    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    usd_path = upload_dir / "input.usda"

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # Both inputs target files that exist on disk but live outside the USD's
    # upload directory. The agent must refuse to localize them.
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(secret))
    )
    shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(no_suffix))
    )
    # Symlink-escape attempt: a relative path that resolves outside the
    # upload root once symlinks are followed.
    escape_link = upload_dir / "escape_link.png"
    escape_link.symlink_to(secret)
    shader.CreateInput("displacement_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./escape_link.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    textures_dir = work_dir / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(usd_path),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    # All three malicious inputs were cleared, none localized.
    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    assert out_shader.GetInput("emissive_color_texture").Get().path == ""
    assert out_shader.GetInput("displacement_texture").Get().path == ""

    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_localized"] == []
    assert sorted(stats["mdl_inputs_cleared"]) == sorted(
        [
            "/Root/Looks/Plastic:opacity_texture",
            "/Root/Looks/Plastic:emissive_color_texture",
            "/Root/Looks/Plastic:displacement_texture",
        ]
    )

    # Critical: nothing from outside_dir was copied into the bundle dir.
    bundle_files = sorted(p.name for p in textures_dir.iterdir())
    assert "secret.png" not in bundle_files
    # No copy under the namespaced naming convention either.
    assert not any("opacity_texture" in name for name in bundle_files)
    assert not any("emissive_color_texture" in name for name in bundle_files)
    assert not any("displacement_texture" in name for name in bundle_files)


def test_apply_textures_task_resolves_relative_paths_against_authoring_layer(
    tmp_path: Path,
) -> None:
    """Codex round-5 finding: composed USDs (referenced material libraries)
    author texture paths relative to *their own* layer, not the root. The
    agent must resolve each MDL `*_texture` against the layer that authored
    the value, otherwise legitimate textures from referenced material USDs
    are silently dropped.
    """
    from pxr import Sdf, Usd, UsdShade

    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    materials_dir = upload_dir / "materials"
    materials_dir.mkdir()
    # `opacity.png` lives next to the *referenced* material USD, NOT next to
    # the root entry-point USD.
    materials_opacity = materials_dir / "opacity.png"
    _save_png(materials_opacity, (10, 20, 30))

    # Referenced material library file with the MDL shader and a layer-local
    # asset path.
    materials_usd_path = materials_dir / "library.usda"
    materials_stage = Usd.Stage.CreateNew(str(materials_usd_path))
    UsdShade.Material.Define(materials_stage, "/Materials/Plastic")
    sub_shader = UsdShade.Shader.Define(materials_stage, "/Materials/Plastic/Shader")
    sub_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    sub_shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./opacity.png")
    )
    materials_stage.GetRootLayer().Save()

    # Root USD references the material.
    root_usd_path = upload_dir / "scene.usda"
    root_stage = Usd.Stage.CreateNew(str(root_usd_path))
    plastic_mat = UsdShade.Material.Define(root_stage, "/Root/Looks/Plastic")
    plastic_mat.GetPrim().GetReferences().AddReference(
        "./materials/library.usda", "/Materials/Plastic"
    )
    root_stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    textures_dir = work_dir / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(root_usd_path),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    opacity_out = Path(out_shader.GetInput("opacity_texture").Get().path)
    # Resolution must have anchored on materials_dir (the referenced layer),
    # not on upload_dir (the root layer). Either way the localized copy lands
    # in work_dir/textures/Plastic__opacity_texture.png with the original
    # bytes — proving the texture was found and copied, not silently lost.
    assert opacity_out.parent == textures_dir
    assert opacity_out.exists()
    assert opacity_out.read_bytes() == materials_opacity.read_bytes()
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_localized"] == ["/Root/Looks/Plastic:opacity_texture"]
    assert stats["mdl_inputs_cleared"] == []


def test_apply_textures_task_clears_non_png_local_mdl_inputs(tmp_path: Path) -> None:
    """The service packager and textures-artifact ZIP only handle
    case-sensitive ``.png``. Localizing a `.jpg` (or `.tif`/`.exr`/etc.)
    would create an inconsistent bundle — the file lands in cache/textures
    but the packager won't rewrite it and the ZIP glob won't include it.
    Codex round-5 medium finding: drop non-PNG suffixes at the localizer.
    """
    from pxr import Sdf, Usd, UsdShade

    local_jpg = tmp_path / "local_opacity.jpg"
    local_jpg.write_bytes(b"\xff\xd8\xff\xe0...not really jpg but suffix matters")

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./local_opacity.jpg")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    assert result["apply_textures_stats"]["mdl_inputs_localized"] == []
    assert result["apply_textures_stats"]["mdl_inputs_cleared"] == [
        "/Root/Looks/Plastic:opacity_texture"
    ]
    # Critical: the .jpg is NOT in the bundle textures dir.
    bundle_files = sorted(p.name for p in (work_dir / "textures").iterdir())
    assert not any(name.endswith(".jpg") for name in bundle_files)


def test_apply_textures_task_handles_string_typed_mdl_texture_inputs(
    tmp_path: Path,
) -> None:
    """Codex round-6/7 findings: an MDL shader can legally author
    ``inputs:*_texture`` as `string` or `token` (not `asset`). The previous
    fix skipped non-asset inputs entirely — but that left Nucleus URLs in
    string-typed inputs untouched, which the service packager later rewrites
    into broken `../textures/<basename>` refs (same bug, different surface).
    The agent must process string/token-typed texture inputs the same way as
    asset-typed: override mapped channels, clear unbundleable URI refs.
    Writes must use the authored type so we never crash the pipeline.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # Mapped channel authored as String + holding a Nucleus URL: the pre-fix
    # code skipped this and shipped a broken bundle ref. Must now be
    # overridden with the freshly generated local texture, written back as
    # a String (not silently coerced to Asset).
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        "omniverse://nucleus.example/T_Plastic_Albedo.png"
    )
    # Unmapped channel authored as Token + holding a Nucleus URL: must be
    # cleared in its native token type.
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Token).Set(
        "omniverse://nucleus.example/T_Plastic_Opacity.png"
    )
    # Asset-typed mapped channel mixed in: business as usual.
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Normal.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # String-typed mapped channel: overridden, type preserved, no Nucleus URL.
    diffuse = out_shader.GetInput("diffuse_texture")
    assert diffuse.GetTypeName() == Sdf.ValueTypeNames.String
    diffuse_val = diffuse.Get()
    assert diffuse_val.endswith("Plastic_albedo.png")
    assert "omniverse://" not in diffuse_val
    # Token-typed unmapped channel: cleared, type preserved.
    opacity = out_shader.GetInput("opacity_texture")
    assert opacity.GetTypeName() == Sdf.ValueTypeNames.Token
    assert opacity.Get() == ""
    # Asset-typed mapped channel: business as usual.
    normal = out_shader.GetInput("normalmap_texture")
    assert normal.GetTypeName() == Sdf.ValueTypeNames.Asset
    assert normal.Get().path.endswith("Plastic_normal.png")

    stats = result["apply_textures_stats"]
    # Both diffuse_texture (String) and normalmap_texture (Asset) overridden.
    assert stats["mdl_inputs_overridden"] == 2
    assert stats["mdl_inputs_cleared"] == ["/Root/Looks/Plastic:opacity_texture"]


def test_apply_textures_task_clears_string_typed_orm_texture(tmp_path: Path) -> None:
    """Codex round-9 finding: a string/token-typed mapped MDL input where
    the generated PNG has no parallel Asset-typed dep on the Material
    (today: only the packed ORM channel — `roughness`/`metalness` are
    written separately as OpenPBR Asset attrs, but `orm` itself is not)
    must be cleared, not overridden. Otherwise the service packager
    rewrites the path but USDZ packaging never bundles the file, leaving
    a dangling reference in the downloaded bundle.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # String-typed ORM_texture: must be cleared, not rewritten to the
    # generated packed-ORM path.
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.String).Set(
        "omniverse://nucleus.example/T_Plastic_ORM.png"
    )
    # String-typed diffuse_texture (channel='albedo' is USDZ-bundled via
    # the OpenPBR-side Asset attr): must still override.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        "omniverse://nucleus.example/T_Plastic_Albedo.png"
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # ORM_texture cleared (channel='orm' not in _USDZ_BUNDLED_CHANNELS for
    # string/token).
    orm = out_shader.GetInput("ORM_texture")
    assert orm.GetTypeName() == Sdf.ValueTypeNames.String
    assert orm.Get() == ""
    # diffuse_texture overridden (channel='albedo' is bundled via OpenPBR).
    diffuse = out_shader.GetInput("diffuse_texture")
    assert diffuse.GetTypeName() == Sdf.ValueTypeNames.String
    assert diffuse.Get().endswith("Plastic_albedo.png")
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_overridden"] == 1
    assert stats["mdl_inputs_cleared"] == ["/Root/Looks/Plastic:ORM_texture"]


def test_apply_textures_task_clears_string_typed_local_unmapped_inputs(
    tmp_path: Path,
) -> None:
    """Codex round-8 finding: string/token-typed unmapped MDL `*_texture`
    inputs cannot be safely localized. USDZ packaging only follows
    `Sdf.AssetPath` deps, so a localized PNG referenced only by a string
    input would not be bundled into the downloaded `.usdz`. Clear instead
    of localizing — the MDL falls back to its constant default, which is
    bundle-self-consistent.
    """
    from pxr import Sdf, Usd, UsdShade

    # Real local PNG inside the upload root that *would* pass the
    # security gate in the asset-typed code path.
    local_opacity = tmp_path / "local_opacity.png"
    _save_png(local_opacity, (10, 20, 30))

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # String-typed unmapped local: must be cleared (not localized).
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.String).Set(
        "./local_opacity.png"
    )

    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    textures_dir = work_dir / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    opacity = out_shader.GetInput("opacity_texture")
    assert opacity.GetTypeName() == Sdf.ValueTypeNames.String
    assert opacity.Get() == ""
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_localized"] == []
    assert stats["mdl_inputs_cleared"] == ["/Root/Looks/Plastic:opacity_texture"]
    # Critical: the local opacity PNG was NOT copied into the bundle dir
    # under any name — string/token-typed inputs do not localize.
    bundle_files = sorted(p.name for p in textures_dir.iterdir())
    assert not any("opacity_texture" in name for name in bundle_files)


def test_apply_textures_task_skips_unsupported_mdl_input_types(tmp_path: Path) -> None:
    """Defense in depth: types outside the supported set
    (``Asset``/``String``/``Token``) — e.g. ``AssetArray``, numeric types
    named ``*_texture`` — must be left untouched and not crash. We'd rather
    skip a rare schema than emit a corrupted value or abort the step.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.AssetArray).Set(
        [Sdf.AssetPath("./pretend.png")]
    )
    # Real asset-typed input alongside, to prove the loop continues.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    # Must not raise.
    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # AssetArray input untouched.
    arr = out_shader.GetInput("opacity_texture").Get()
    assert len(arr) == 1
    assert arr[0].path == "./pretend.png"
    # Real asset-typed input was overridden.
    assert (
        out_shader.GetInput("diffuse_texture").Get().path.endswith("Plastic_albedo.png")
    )
    assert result["apply_textures_stats"]["mdl_inputs_overridden"] == 1


def test_apply_textures_task_handles_nul_byte_in_asset_path(tmp_path: Path) -> None:
    """Claude round-5 nit: ``Path('foo\\x00.png').resolve()`` raises
    ``ValueError``, not ``OSError``. A malicious USD with a NUL byte in an
    MDL `*_texture` asset path must not crash apply_textures — it should
    just clear the input.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./poison\x00.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    # Must not raise.
    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    assert result["apply_textures_stats"]["mdl_inputs_cleared"] == [
        "/Root/Looks/Plastic:opacity_texture"
    ]


def test_render_material_previews_task_skips_missing_template(tmp_path: Path) -> None:
    task = render_previews_task.RenderMaterialPreviewsTask()

    result = task.run(
        {
            "discovered_materials": [_material("Steel")],
            "usd_path": "/tmp/input.usd",
            "render_preview_config": {"template_scene": str(tmp_path / "missing.usd")},
            "working_dir": str(tmp_path),
        }
    )

    assert result["material_previews"] == {}


def test_render_material_previews_task_saves_preview(
    tmp_path: Path, monkeypatch
) -> None:
    task = render_previews_task.RenderMaterialPreviewsTask()
    template = tmp_path / "template.usd"
    template.write_text("#usda 1.0\n", encoding="utf-8")

    monkeypatch.setattr(
        task, "_compose_preview_stage", lambda *args, **kwargs: object()
    )
    import world_understanding.functions.graphics.render_nvcf as render_nvcf

    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: [{"images": [Image.new("RGB", (4, 4), (1, 2, 3))]}],
    )

    result = task.run(
        {
            "discovered_materials": [_material("Steel")],
            "usd_path": "/tmp/input.usd",
            "render_preview_config": {"template_scene": str(template)},
            "working_dir": str(tmp_path),
        }
    )

    preview_path = Path(result["material_previews"]["Steel"])
    assert preview_path.exists()


def test_render_output_task_handles_empty_outputs(tmp_path: Path) -> None:
    task = render_task.RenderOutputTask()

    result = task.run({"output_usd_paths": [], "working_dir": str(tmp_path)})

    assert result["rendered_image_paths"] == []


def test_render_output_task_saves_images(tmp_path: Path, monkeypatch) -> None:
    from pxr import Usd

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.DefinePrim("/Root")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_nvcf as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: [{"images": [Image.new("RGB", (4, 4), (4, 5, 6))]}],
    )

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert Path(result["rendered_image_paths"][0]).exists()
