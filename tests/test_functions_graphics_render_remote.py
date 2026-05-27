# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for remote render response parsing, including V2-to-V1 conversion."""

import base64
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

import world_understanding.functions.graphics.render_remote as render_remote
from world_understanding.functions.graphics.render_remote import (
    RenderingStatus,
    _bundle_stage_with_local_assets,
    _convert_v2_sensor,
    _convert_v2_to_v1,
    _http_error_detail,
    _is_local_composition_asset_path,
    _is_v2_response,
    _resolve_export_asset_path,
    _stage_has_local_composition_arcs,
    export_stage_to_s3,
    render_all_cameras,
    render_single_camera_from_url,
    save_render_results,
)


def test_legacy_render_nvcf_module_aliases_remote_helpers() -> None:
    """Old NVCF module imports should keep working during the rename window."""
    from world_understanding.functions.graphics.render_nvcf import (
        RenderingStatus as LegacyRenderingStatus,
    )
    from world_understanding.functions.graphics.render_nvcf import (
        _is_v2_response as legacy_is_v2_response,
    )

    assert LegacyRenderingStatus is RenderingStatus
    assert legacy_is_v2_response({"rendered_data": {}, "total_cameras": 0}) is True


def test_export_stage_to_s3_encodes_asset_bundle_as_data_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data URI mode must preserve local asset bundles instead of S3 fallback."""
    from pxr import Usd

    def fake_bundle(
        stage: object,
        temp_dir: Path,
        base_dir: object | None = None,
        has_local_composition_arcs: bool | None = None,
    ) -> tuple[Path, bool]:
        zip_path = temp_dir / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            zf.writestr("textures/albedo.png", b"not-a-real-png")
        return zip_path, True

    def fail_upload(*args: object, **kwargs: object) -> None:
        raise AssertionError("data URI bundle path must not upload to S3")

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote._bundle_stage_with_local_assets",
        fake_bundle,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.upload_file_to_s3",
        fail_upload,
    )

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Root", "Xform")

    asset_url, s3_uri = export_stage_to_s3(stage, use_data_uri=True)

    assert s3_uri is None
    assert asset_url.startswith("data:application/zip;name=bundle.zip;base64,")
    payload = base64.b64decode(asset_url.split(",", 1)[1])
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert "stage.usda" in zf.namelist()
        assert "textures/albedo.png" in zf.namelist()


def test_render_all_cameras_passes_base_dir_to_stage_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_export_stage_to_s3(**kwargs: object) -> tuple[str, str | None]:
        captured["base_dir"] = kwargs.get("base_dir")
        return "data:model/vnd.usd;base64,ZmFrZQ==", None

    def fake_render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        return {
            "camera": kwargs["camera"],
            "images": [],
            "frame_count": 1,
            "status": RenderingStatus.success,
        }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
        fake_export_stage_to_s3,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.render_single_camera_from_url",
        fake_render_single_camera_from_url,
    )

    result = render_all_cameras(
        stage=object(),
        cameras=["/Camera"],
        base_dir=tmp_path,
        max_workers=1,
    )

    assert captured["base_dir"] == tmp_path
    assert result["successful_cameras"] == 1


def test_render_all_cameras_returns_structured_failure_on_export_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_export_stage_to_s3(**kwargs: object) -> tuple[str, str | None]:
        raise RuntimeError("Remote REST rendering requires a flattened stage")

    def fail_render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        raise AssertionError("rendering should not start when stage export fails")

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
        fail_export_stage_to_s3,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.render_single_camera_from_url",
        fail_render_single_camera_from_url,
    )

    result = render_all_cameras(
        stage=object(),
        cameras=["/CameraA", "/CameraB"],
        max_workers=1,
    )

    assert result["total_cameras"] == 2
    assert result["successful_cameras"] == 0
    assert result["failed_cameras"] == 2
    assert [item["camera"] for item in result["results"]] == [
        "/CameraA",
        "/CameraB",
    ]
    assert {item["status"] for item in result["results"]} == {RenderingStatus.exception}
    assert all(
        "requires a flattened stage" in item["error"] for item in result["results"]
    )
    assert {item["error_type"] for item in result["results"]} == {"RuntimeError"}


def test_http_error_detail_extracts_blank_render_error() -> None:
    response = requests.Response()
    response.status_code = 422
    response._content = (
        b'{"detail":{"status":"blank_render",'
        b'"error":"1/1 OVRTX render frames are blank or near-blank."}}'
    )
    response.headers["Content-Type"] = "application/json"

    assert _http_error_detail(response) == (
        "1/1 OVRTX render frames are blank or near-blank."
    )


def test_render_single_camera_preserves_blank_render_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 422
    response._content = (
        b'{"detail":{"status":"blank_render",'
        b'"error":"1/1 OVRTX render frames are blank or near-blank.",'
        b'"warnings":["1/1 frames blank"],'
        b'"blank_render_frames":[{"frame":0,"camera":"/Camera"}]}}'
    )
    response.headers["Content-Type"] = "application/json"
    response.url = "http://renderer/render"
    response.reason = "Unprocessable Entity"

    def fake_post(*args: object, **kwargs: object) -> requests.Response:
        return response

    monkeypatch.setattr(requests, "post", fake_post)

    result = render_single_camera_from_url(
        "data:model/vnd.usd;base64,ZmFrZQ==",
        "/Camera",
        api_key="test-key",
        base_url="http://renderer",
        max_retries=0,
    )

    assert result["status"] == RenderingStatus.blank_render
    assert result["warnings"] == ["1/1 frames blank"]
    assert result["blank_render_frames"] == [{"frame": 0, "camera": "/Camera"}]


def test_local_composition_asset_path_detection_uses_shared_uri_semantics() -> None:
    assert _is_local_composition_asset_path("./geometry.usda")
    assert _is_local_composition_asset_path("/tmp/geometry.usda")
    assert _is_local_composition_asset_path("C:/assets/geometry.usda")
    assert _is_local_composition_asset_path(r"C:\assets\geometry.usda")
    assert _is_local_composition_asset_path("file:/tmp/geometry.usda")

    assert not _is_local_composition_asset_path("")
    assert _is_local_composition_asset_path("anon:000002")
    assert not _is_local_composition_asset_path("https://example.com/geometry.usda")
    assert not _is_local_composition_asset_path("s3://bucket/geometry.usda")
    assert not _is_local_composition_asset_path("omniverse://server/geometry.usda")
    assert not _is_local_composition_asset_path(
        "data:application/octet-stream;base64,AA"
    )


def test_local_composition_arc_guard_ignores_deleted_list_ops() -> None:
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/DeleteReference", "Xform")
    prim_spec = stage.GetRootLayer().GetPrimAtPath("/World/DeleteReference")
    prim_spec.referenceList.deletedItems = [Sdf.Reference("./deleted.usda")]
    prim_spec.payloadList.deletedItems = [Sdf.Payload("./deleted_payload.usda")]

    assert not _stage_has_local_composition_arcs(stage)


def test_local_composition_arc_guard_rejects_anonymous_arcs() -> None:
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Reference", "Xform")
    prim_spec = stage.GetRootLayer().GetPrimAtPath("/World/Reference")
    prim_spec.referenceList.addedItems = [Sdf.Reference("anon:000002:layer.usda")]

    assert _stage_has_local_composition_arcs(stage)


def test_resolve_export_asset_path_falls_back_when_resolve_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_path = tmp_path / "textures" / "bad.png"

    def raise_os_error(self: Path, strict: bool = False) -> Path:
        raise OSError("bad path")

    monkeypatch.setattr(Path, "resolve", raise_os_error)

    assert _resolve_export_asset_path("textures/bad.png", tmp_path) == str(
        expected_path
    )


def test_bundle_stage_rewrites_relative_mdl_source_asset(tmp_path: Path) -> None:
    """Bundled stages must point MDL source assets at copied bundle paths."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "textures").mkdir()
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )
    (asset_root / "textures" / "albedo.png").write_bytes(b"not-a-real-png")

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.CreateIdAttr("mdl:OmniPBR")
    shader.GetPrim().CreateAttribute(
        "info:implementationSource",
        Sdf.ValueTypeNames.Token,
    ).Set("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    shader.GetPrim().CreateAttribute(
        "inputs:diffuse_texture",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./textures/albedo.png"))
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "mdl_materials/OmniPBR/OmniPBR.mdl" in zf.namelist()
        assert "textures/albedo.png" in zf.namelist()
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text
    assert "@./materials/OmniPBR/OmniPBR.mdl@" not in stage_text
    assert "@textures/albedo.png@" in stage_text


def test_bundle_stage_rewrites_texture_paths_by_resolved_asset_path(
    tmp_path: Path,
) -> None:
    """Textures with the same basename must not collapse to one bundle path."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    mat_a = asset_root / "mat_a"
    mat_b = asset_root / "mat_b"
    mat_a.mkdir(parents=True)
    mat_b.mkdir()
    (mat_a / "diffuse.png").write_bytes(b"texture-a")
    (mat_b / "diffuse.png").write_bytes(b"texture-b")

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    shader_a = UsdShade.Shader.Define(stage, "/World/Looks/MatA/Shader")
    shader_a.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./mat_a/diffuse.png"))
    shader_b = UsdShade.Shader.Define(stage, "/World/Looks/MatB/Shader")
    shader_b.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./mat_b/diffuse.png"))
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "textures/diffuse.png" in zf.namelist()
        assert "textures/diffuse_1.png" in zf.namelist()
        stage_text = zf.read("stage.usda").decode("utf-8")

    layer = Sdf.Layer.CreateAnonymous("bundled-stage.usda")
    assert layer.ImportFromString(stage_text)
    shader_a_spec = layer.GetPrimAtPath("/World/Looks/MatA/Shader")
    shader_b_spec = layer.GetPrimAtPath("/World/Looks/MatB/Shader")

    assert shader_a_spec.attributes["inputs:file"].default == Sdf.AssetPath(
        "textures/diffuse.png"
    )
    assert shader_b_spec.attributes["inputs:file"].default == Sdf.AssetPath(
        "textures/diffuse_1.png"
    )


def test_remote_stage_export_rejects_unflattened_local_composition(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    reference_path = tmp_path / "geometry.usda"
    reference_stage = Usd.Stage.CreateNew(str(reference_path))
    reference_stage.DefinePrim("/ReferencedRoot", "Xform")
    reference_stage.Save()

    root_path = tmp_path / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.DefinePrim("/World/Reference", "Xform").GetReferences().AddReference(
        "./geometry.usda",
        "/ReferencedRoot",
    )
    root_stage.Save()

    with pytest.raises(RuntimeError, match="requires a flattened stage"):
        export_stage_to_s3(
            root_stage,
            use_data_uri=True,
            bundle_mdl_assets=False,
        )


def test_export_stage_to_s3_adds_ovrtx_preview_fallback_to_usd_only_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    material.GetPrim().CreateAttribute(
        "inputs:base_color",
        Sdf.ValueTypeNames.Color3f,
    ).Set((1.0, 0.766, 0.336))

    shader = UsdShade.Shader.Define(
        stage,
        "/World/Looks/Gold/open_pbr_surface_surfaceshader",
    )
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )

    captured: dict[str, object] = {}

    def fake_export_stage_and_get_url(**kwargs: object) -> tuple[str, str | None]:
        exported_stage = Usd.Stage.Open(str(kwargs["stage_path"]))
        assert exported_stage is not None
        exported_material = UsdShade.Material(
            exported_stage.GetPrimAtPath("/World/Looks/Gold"),
        )
        surface = exported_material.GetSurfaceOutput()
        sources, _ = surface.GetConnectedSources()
        preview_shader = UsdShade.Shader(sources[0].source.GetPrim())
        captured["shader_id"] = preview_shader.GetIdAttr().Get()
        captured["mtlx_connected"] = exported_material.GetSurfaceOutput(
            "mtlx",
        ).HasConnectedSource()
        return "data:model/vnd.usd;base64,AA==", None

    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        fake_export_stage_and_get_url,
    )

    asset_url, s3_uri = export_stage_to_s3(stage, use_data_uri=True)

    assert asset_url == "data:model/vnd.usd;base64,AA=="
    assert s3_uri is None
    assert captured == {"shader_id": "UsdPreviewSurface", "mtlx_connected": False}


def test_export_stage_to_s3_strips_mdl_when_preview_surface_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    preview_shader = UsdShade.Shader.Define(stage, "/World/Looks/Gold/Preview")
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Gold/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniSurface")
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        mdl_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )

    captured: dict[str, object] = {}

    def fake_export_stage_and_get_url(**kwargs: object) -> tuple[str, str | None]:
        exported_stage = Usd.Stage.Open(str(kwargs["stage_path"]))
        assert exported_stage is not None
        exported_material_prim = exported_stage.GetPrimAtPath("/World/Looks/Gold")
        captured["has_mdl_output"] = any(
            prop.GetName().startswith("outputs:mdl:")
            for prop in exported_material_prim.GetProperties()
        )
        return "data:model/vnd.usd;base64,AA==", None

    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        fake_export_stage_and_get_url,
    )

    asset_url, s3_uri = export_stage_to_s3(stage, use_data_uri=True)

    assert asset_url == "data:model/vnd.usd;base64,AA=="
    assert s3_uri is None
    assert captured == {"has_mdl_output": False}


def test_bundle_stage_prefers_preview_surface_over_mdl_output(
    tmp_path: Path,
) -> None:
    """Remote bundles should use preview output when both preview and MDL exist."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")

    preview_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Preview")
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    preview_output = preview_shader.CreateOutput(
        "surface",
        Sdf.ValueTypeNames.Token,
    )
    material.CreateSurfaceOutput().ConnectToSource(preview_output)

    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "outputs:surface.connect" in stage_text
    assert "outputs:mdl:surface" not in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text


def test_bundle_stage_keeps_mdl_only_surface_output(tmp_path: Path) -> None:
    """MDL-only materials should not lose their only material output."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "outputs:mdl:surface" in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text


def test_bundle_stage_keeps_mdl_output_when_universal_surface_is_not_preview(
    tmp_path: Path,
) -> None:
    """Only UsdPreviewSurface fallback materials should have MDL outputs stripped."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")

    materialx_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/MaterialX")
    materialx_shader.CreateIdAttr("ND_standard_surface_surfaceshader")
    materialx_output = materialx_shader.CreateOutput(
        "surface",
        Sdf.ValueTypeNames.Token,
    )
    material.CreateSurfaceOutput().ConnectToSource(materialx_output)

    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "outputs:surface.connect" in stage_text
    assert "outputs:mdl:surface" in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text


def test_save_render_results_preserves_instance_segmentation_ids(
    tmp_path: Path,
) -> None:
    result = {
        "sensors": {
            "instance_id_segmentation": {
                0: np.array([[1, 256]], dtype=np.uint32),
            }
        }
    }

    stats = save_render_results(
        result,
        tmp_path,
        file_name="seg",
        image_width=2,
        image_height=1,
        save_npy=True,
    )

    assert stats["success_count"] == 1
    saved = np.load(tmp_path / "seg_f0000_instance_id_segmentation.npy")
    assert saved.dtype == np.uint32
    assert int(saved.max()) == 256
    assert (tmp_path / "seg_f0000_instance_id_segmentation.png").exists()


class TestIsV2Response:
    """Tests for V2 response detection."""

    def test_detects_v2_response(self):
        result = {
            "total_cameras": 1,
            "total_frames": 1,
            "rendered_data": {"Camera": {}},
        }
        assert _is_v2_response(result) is True

    def test_rejects_v1_response(self):
        result = {
            "images": {"0": {}},
            "status": "success",
        }
        assert _is_v2_response(result) is False

    def test_rejects_empty_dict(self):
        assert _is_v2_response({}) is False

    def test_rejects_partial_v2(self):
        # Has rendered_data but not total_cameras
        assert _is_v2_response({"rendered_data": {}}) is False


class TestConvertV2Sensor:
    """Tests for V2 sensor data conversion."""

    def test_converts_uint8_rgb(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0, 0] = [255, 0, 0, 255]
        sensor_obj = {
            "type": "array",
            "data": base64.b64encode(arr.tobytes()).decode(),
            "shape": [4, 4, 4],
            "dtype": "uint8",
        }
        result = _convert_v2_sensor(sensor_obj)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4, 4)
        assert result[0, 0, 0] == 255

    def test_returns_string_when_no_shape(self):
        sensor_obj = {"data": "abc123"}
        result = _convert_v2_sensor(sensor_obj)
        assert result == "abc123"

    def test_returns_empty_string_when_no_data(self):
        sensor_obj = {"shape": [4, 4]}
        result = _convert_v2_sensor(sensor_obj)
        assert result == ""


class TestConvertV2ToV1:
    """Tests for V2→V1 full response conversion."""

    def _make_v2_response(
        self, width: int = 4, height: int = 4, n_cameras: int = 1
    ) -> dict:
        """Create a minimal V2 response with an RGB image."""
        rendered_data = {}
        for i in range(n_cameras):
            arr = np.full((height, width, 4), 128, dtype=np.uint8)
            cam_name = f"Camera{i}"
            rendered_data[cam_name] = {
                "0": {
                    "rgb": {
                        "type": "array",
                        "data": base64.b64encode(arr.tobytes()).decode(),
                        "shape": [height, width, 4],
                        "dtype": "uint8",
                    }
                }
            }
        return {
            "total_cameras": n_cameras,
            "total_frames": 1,
            "rendered_data": rendered_data,
        }

    def test_v1_has_status_success(self):
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        assert v1["status"] == RenderingStatus.success

    def test_v1_has_images_key(self):
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        assert "images" in v1
        assert "0" in v1["images"]

    def test_v1_frame_camera_nesting(self):
        """V1 format nests frame→camera (opposite of V2 camera→frame)."""
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        frame_data = v1["images"]["0"]
        assert "Camera0" in frame_data

    def test_v1_rgb_converted_to_base64_png(self):
        """V2 raw array data should become a base64 PNG in V1 'images' key."""
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        camera_data = v1["images"]["0"]["Camera0"]
        assert "images" in camera_data  # rgb → images
        # Should be valid base64 PNG
        png_bytes = base64.b64decode(camera_data["images"])
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (4, 4)

    def test_multi_camera_response(self):
        v2 = self._make_v2_response(n_cameras=3)
        v1 = _convert_v2_to_v1(v2)
        frame_data = v1["images"]["0"]
        assert len(frame_data) == 3
        for i in range(3):
            assert f"Camera{i}" in frame_data

    def test_sensor_name_mapping(self):
        """V2 sensor names should be mapped to V1 equivalents."""
        arr = np.zeros((4, 4), dtype=np.float32)
        v2 = {
            "total_cameras": 1,
            "total_frames": 1,
            "rendered_data": {
                "Camera": {
                    "0": {
                        "rgb": {
                            "type": "array",
                            "data": base64.b64encode(
                                np.zeros((4, 4, 4), dtype=np.uint8).tobytes()
                            ).decode(),
                            "shape": [4, 4, 4],
                            "dtype": "uint8",
                        },
                        "distance_to_image_plane": {
                            "type": "array",
                            "data": base64.b64encode(arr.tobytes()).decode(),
                            "shape": [4, 4],
                            "dtype": "float32",
                        },
                        "instance_segmentation": {
                            "type": "array",
                            "data": base64.b64encode(
                                np.zeros((4, 4), dtype=np.uint32).tobytes()
                            ).decode(),
                            "shape": [4, 4],
                            "dtype": "uint32",
                        },
                    }
                }
            },
        }
        v1 = _convert_v2_to_v1(v2)
        camera_data = v1["images"]["0"]["Camera"]
        assert "images" in camera_data  # rgb → images
        assert "linear_depth" in camera_data  # distance_to_image_plane → linear_depth
        assert (
            "instance_id_segmentation" in camera_data
        )  # instance_segmentation → instance_id_segmentation

    def test_empty_rendered_data(self):
        v2 = {"total_cameras": 0, "total_frames": 0, "rendered_data": {}}
        v1 = _convert_v2_to_v1(v2)
        assert v1["status"] == RenderingStatus.success
        assert v1["images"] == {}
