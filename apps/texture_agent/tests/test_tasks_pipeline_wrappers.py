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
