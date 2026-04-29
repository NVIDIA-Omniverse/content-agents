# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for material_agent.manifest runtime paths."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import material_agent.manifest as manifest
from material_agent.manifest import GenerateManifestInput


class FakeImage:
    def save(self, path: str) -> None:
        Path(path).write_text("png")


class FakeAssetPath:
    def __init__(self, path: str, resolvedPath: str = "") -> None:
        self.path = path
        self.resolvedPath = resolvedPath


class FakeAttrSpec:
    def __init__(self, default: object) -> None:
        self.default = default


class FakePrimSpec:
    def __init__(
        self,
        attributes: dict[str, FakeAttrSpec],
        children: list[FakePrimSpec] | None = None,
    ) -> None:
        self.attributes = attributes
        self.nameChildren = children or []


class FakeLayer:
    def __init__(self, root_prims: list[FakePrimSpec]) -> None:
        self.rootPrims = root_prims
        self.saved = False

    def Save(self) -> None:
        self.saved = True


class FakeStageAttr:
    def __init__(
        self,
        value: object,
        *,
        valid: bool = True,
        type_name: str = "SdfAssetPath",
    ) -> None:
        self._value = value
        self._valid = valid
        self._type_name = type_name

    def IsValid(self) -> bool:
        return self._valid

    def Get(self) -> object:
        return self._value

    def GetTypeName(self) -> object:
        return SimpleNamespace(type=SimpleNamespace(typeName=self._type_name))


class FakeTraversePrim:
    def __init__(
        self,
        *,
        mdl_attr: FakeStageAttr | None = None,
        attrs: list[FakeStageAttr] | None = None,
        is_shader: bool = True,
    ) -> None:
        self._mdl_attr = mdl_attr
        self._attrs = attrs or []
        self._is_shader = is_shader

    def IsA(self, cls: object) -> bool:
        return self._is_shader

    def GetAttribute(self, name: str) -> FakeStageAttr | None:
        if name == "info:mdl:sourceAsset":
            return self._mdl_attr
        return None

    def GetAttributes(self) -> list[FakeStageAttr]:
        return self._attrs


class FakeExportRootLayer:
    def Export(self, path: str) -> None:
        Path(path).write_text("#usda 1.0\n")


class FakeBundleStage:
    def __init__(self, prims: list[FakeTraversePrim]) -> None:
        self._prims = prims
        self._root_layer = FakeExportRootLayer()

    def GetRootLayer(self) -> FakeExportRootLayer:
        return self._root_layer

    def Traverse(self) -> list[FakeTraversePrim]:
        return self._prims


class FakeReferences:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def AddReference(self, assetPath: str, primPath: str) -> None:
        self.calls.append({"assetPath": assetPath, "primPath": primPath})


class FakeDefinedPrim:
    def __init__(self) -> None:
        self.references = FakeReferences()

    def GetReferences(self) -> FakeReferences:
        return self.references


class FakeSpherePrim:
    def __init__(self, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid


class FakeComposeRootLayer:
    def __init__(self) -> None:
        self.subLayerPaths: list[str] = []


class FakeComposeStage:
    def __init__(self, *, sphere_valid: bool = True, flat_layer: object = None) -> None:
        self.root_layer = FakeComposeRootLayer()
        self.sphere_valid = sphere_valid
        self.flat_layer = flat_layer if flat_layer is not None else object()
        self.define_calls: list[tuple[object, str, FakeDefinedPrim]] = []

    def GetRootLayer(self) -> FakeComposeRootLayer:
        return self.root_layer

    def DefinePrim(self, path: object, prim_type: str) -> FakeDefinedPrim:
        prim = FakeDefinedPrim()
        self.define_calls.append((path, prim_type, prim))
        return prim

    def GetPrimAtPath(self, path: str) -> FakeSpherePrim:
        return FakeSpherePrim(self.sphere_valid)

    def Flatten(self) -> object:
        return self.flat_layer


class FakeBindingAPI:
    def __init__(self) -> None:
        self.bound: list[object] = []

    def Bind(self, material: object) -> None:
        self.bound.append(material)


def _install_render_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    render_result: dict[str, object],
    delete_side_effect: Exception | None = None,
    uploaded_uri: str = "s3://bucket/key",
) -> dict[str, object]:
    calls: dict[str, object] = {}

    def upload_file_to_s3(**kwargs: object) -> str:
        calls["upload"] = kwargs
        return uploaded_uri

    def delete_s3_path(uri: str, profile_name: str) -> None:
        calls["delete"] = {"uri": uri, "profile_name": profile_name}
        if delete_side_effect is not None:
            raise delete_side_effect

    def render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        calls["render"] = kwargs
        return render_result

    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.graphics.render_nvcf",
        SimpleNamespace(
            RenderingStatus=SimpleNamespace(success="success"),
            render_single_camera_from_url=render_single_camera_from_url,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.utils.nvcf_utils",
        SimpleNamespace(
            s3_uri_to_https_url=lambda uri, region: f"https://{region}/{uri}"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.utils.s3_utils",
        SimpleNamespace(
            upload_file_to_s3=upload_file_to_s3,
            delete_s3_path=delete_s3_path,
        ),
    )
    return calls


def test_render_one_thumbnail_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "thumb.png"
    calls = _install_render_modules(
        monkeypatch,
        render_result={"status": "success", "images": [FakeImage()]},
    )
    monkeypatch.setattr(manifest, "_compose_thumbnail_stage", lambda *args: "stage")

    def fake_bundle(stage: object, bundle_dir: Path) -> Path:
        zip_path = bundle_dir.parent / "bundle.zip"
        zip_path.write_text("zip")
        return zip_path

    monkeypatch.setattr(manifest, "_bundle_stage_flat", fake_bundle)

    prim_path, saved_path, error = manifest._render_one_thumbnail(
        template_path=tmp_path / "template.usd",
        usd_file=tmp_path / "materials.usd",
        material_prim_path="/World/Looks/Gold",
        output_path=output_path,
        image_size=128,
    )

    assert prim_path == "/World/Looks/Gold"
    assert saved_path == output_path
    assert error is None
    assert output_path.exists()
    assert calls["upload"]["profile_name"] == manifest._S3_PROFILE
    assert calls["delete"]["uri"] == "s3://bucket/key"
    assert calls["render"]["camera"] == manifest._TEMPLATE_CAMERA
    assert calls["render"]["image_width"] == 128


def test_compose_thumbnail_stage_binds_material_and_flattens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    template = tmp_path / "template.usd"
    template.write_text("template")
    usd_file = tmp_path / "materials.usd"
    usd_file.write_text("usd")

    fake_stage = FakeComposeStage(flat_layer="flat-layer")
    binding_api = FakeBindingAPI()
    flat_stage = object()
    convert = Mock()

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Sdf=SimpleNamespace(Path=lambda value: value),
            Usd=SimpleNamespace(
                Stage=SimpleNamespace(
                    CreateInMemory=lambda: fake_stage,
                    Open=lambda layer: flat_stage,
                )
            ),
            UsdShade=SimpleNamespace(
                MaterialBindingAPI=SimpleNamespace(Apply=lambda prim: binding_api),
                Material=lambda prim: ("material", prim),
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.utils.usd.material",
        SimpleNamespace(convert_custom_mdl_to_builtin=convert),
    )

    result = manifest._compose_thumbnail_stage(template, usd_file, "/World/Looks/Gold")

    assert result is flat_stage
    assert fake_stage.root_layer.subLayerPaths == [str(template.resolve())]
    assert fake_stage.define_calls[0][0] == "/Materials/Gold"
    assert fake_stage.define_calls[0][1] == "Material"
    assert fake_stage.define_calls[0][2].references.calls == [
        {"assetPath": str(usd_file.resolve()), "primPath": "/World/Looks/Gold"}
    ]
    assert binding_api.bound == [("material", fake_stage.define_calls[0][2])]
    convert.assert_called_once_with(flat_stage)


def test_compose_thumbnail_stage_raises_when_sphere_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_stage = FakeComposeStage(sphere_valid=False)
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Sdf=SimpleNamespace(Path=lambda value: value),
            Usd=SimpleNamespace(
                Stage=SimpleNamespace(
                    CreateInMemory=lambda: fake_stage,
                    Open=lambda layer: object(),
                )
            ),
            UsdShade=SimpleNamespace(
                MaterialBindingAPI=SimpleNamespace(Apply=lambda prim: FakeBindingAPI()),
                Material=lambda prim: ("material", prim),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="Sphere prim not found"):
        manifest._compose_thumbnail_stage(
            tmp_path / "template.usd",
            tmp_path / "materials.usd",
            "/World/Looks/Gold",
        )


@pytest.mark.parametrize(
    ("render_result", "expected_error"),
    [
        ({"status": "failed", "error": "render boom"}, "Render status: failed"),
        ({"status": "success", "images": []}, "No images returned from NVCF"),
    ],
)
def test_render_one_thumbnail_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    render_result: dict[str, object],
    expected_error: str,
) -> None:
    _install_render_modules(
        monkeypatch,
        render_result=render_result,
        delete_side_effect=RuntimeError("cleanup boom"),
    )
    monkeypatch.setattr(manifest, "_compose_thumbnail_stage", lambda *args: "stage")

    def fake_bundle(stage: object, bundle_dir: Path) -> Path:
        zip_path = bundle_dir.parent / "bundle.zip"
        zip_path.write_text("zip")
        return zip_path

    monkeypatch.setattr(manifest, "_bundle_stage_flat", fake_bundle)

    prim_path, saved_path, error = manifest._render_one_thumbnail(
        template_path=tmp_path / "template.usd",
        usd_file=tmp_path / "materials.usd",
        material_prim_path="/World/Looks/Gold",
        output_path=tmp_path / "thumb.png",
        image_size=64,
    )

    assert prim_path == "/World/Looks/Gold"
    assert saved_path is None
    assert expected_error in error


def test_render_thumbnails_handles_missing_template_and_partial_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert (
        manifest.render_thumbnails(
            usd_file=tmp_path / "materials.usd",
            prim_paths=["/World/Looks/A"],
            output_dir=tmp_path / "out",
            image_size=64,
            skip_existing=False,
            template_path=tmp_path / "missing_template.usd",
            max_workers=2,
        )
        == {}
    )

    template = tmp_path / "template.usd"
    template.write_text("template")
    output_dir = tmp_path / "out"
    existing = output_dir / "thumbs" / "64x64" / "Already.png"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("png")

    def fake_render(
        template_path: Path,
        usd_file: Path,
        material_prim_path: str,
        output_path: Path,
        image_size: int,
    ) -> tuple[str, Path | None, str | None]:
        if material_prim_path.endswith("Good"):
            output_path.write_text("png")
            return material_prim_path, output_path, None
        return material_prim_path, None, "render failed"

    monkeypatch.setattr(manifest, "_render_one_thumbnail", fake_render)

    results = manifest.render_thumbnails(
        usd_file=tmp_path / "materials.usd",
        prim_paths=["/World/Looks/Already", "/World/Looks/Good", "/World/Looks/Bad"],
        output_dir=output_dir,
        image_size=64,
        skip_existing=True,
        template_path=template,
        max_workers=2,
    )

    assert results["/World/Looks/Already"] == existing
    assert results["/World/Looks/Good"].exists()
    assert "/World/Looks/Bad" not in results


def test_render_thumbnails_returns_early_when_all_outputs_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    template = tmp_path / "template.usd"
    template.write_text("template")
    output_dir = tmp_path / "out"
    existing = output_dir / "thumbs" / "64x64"
    existing.mkdir(parents=True)
    gold = existing / "Gold.png"
    silver = existing / "Silver.png"
    gold.write_text("png")
    silver.write_text("png")
    monkeypatch.setattr(
        manifest,
        "_render_one_thumbnail",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not render")),
    )

    results = manifest.render_thumbnails(
        usd_file=tmp_path / "materials.usd",
        prim_paths=["/World/Looks/Gold", "/World/Looks/Silver"],
        output_dir=output_dir,
        image_size=64,
        skip_existing=True,
        template_path=template,
        max_workers=2,
    )

    assert results == {
        "/World/Looks/Gold": gold,
        "/World/Looks/Silver": silver,
    }


def test_bundle_stage_flat_collects_assets_and_rewrites_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    materials_dir = tmp_path / "materials"
    materials_dir.mkdir()
    mdl_path = materials_dir / "Base.mdl"
    mdl_path.write_text("mdl 1.0;\n")
    colocated_texture = materials_dir / "base_color.png"
    colocated_texture.write_text("png")
    texture_subdir = materials_dir / "Base"
    texture_subdir.mkdir()
    (texture_subdir / "normal.png").write_text("png")
    outside_texture = tmp_path / "outside.png"
    outside_texture.write_text("png")

    bundle_dir = tmp_path / "bundle"
    layer = FakeLayer(
        [
            FakePrimSpec(
                {
                    "mdl": FakeAttrSpec(FakeAssetPath(str(mdl_path))),
                    "inside": FakeAttrSpec(FakeAssetPath(str(colocated_texture))),
                    "outside": FakeAttrSpec(FakeAssetPath(str(outside_texture))),
                    "relative": FakeAttrSpec(FakeAssetPath("relative.png")),
                },
                children=[
                    FakePrimSpec(
                        {
                            "child": FakeAttrSpec(
                                FakeAssetPath(str(colocated_texture))
                            ),
                        }
                    )
                ],
            )
        ]
    )
    stage = FakeBundleStage(
        [
            FakeTraversePrim(
                mdl_attr=FakeStageAttr(FakeAssetPath(str(mdl_path), str(mdl_path))),
                attrs=[
                    FakeStageAttr(FakeAssetPath(str(mdl_path), str(mdl_path))),
                    FakeStageAttr(
                        FakeAssetPath(str(colocated_texture), str(colocated_texture))
                    ),
                    FakeStageAttr(
                        FakeAssetPath(str(outside_texture), str(outside_texture))
                    ),
                    FakeStageAttr(FakeAssetPath("relative.png", "")),
                ],
            ),
            FakeTraversePrim(is_shader=False),
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Sdf=SimpleNamespace(
                AssetPath=FakeAssetPath,
                Layer=SimpleNamespace(FindOrOpen=lambda path: layer),
            ),
            UsdShade=SimpleNamespace(Shader=object()),
        ),
    )

    zip_path = manifest._bundle_stage_flat(stage, bundle_dir)

    assert zip_path.exists()
    assert (bundle_dir / "Base.mdl").exists()
    assert (bundle_dir / "Base" / "normal.png").exists()
    assert (bundle_dir / "base_color.png").exists()
    assert (bundle_dir / "outside.png").exists()
    assert layer.saved is True
    root_attrs = layer.rootPrims[0].attributes
    assert root_attrs["mdl"].default.path == "Base.mdl"
    assert root_attrs["inside"].default.path == "base_color.png"
    assert root_attrs["outside"].default.path == "outside.png"
    assert root_attrs["relative"].default.path == "relative.png"
    child_attrs = layer.rootPrims[0].nameChildren[0].attributes
    assert child_attrs["child"].default.path == "base_color.png"


def test_generate_descriptions_handles_api_keys_and_partial_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: dict[str, object] = {}

    def create_vlm(**kwargs: object) -> object:
        created.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.models.vision_language_models",
        SimpleNamespace(create_vlm=create_vlm),
    )
    monkeypatch.setenv("INFERENCE_NVIDIA_API_KEY", "secret")

    def fake_describe(
        vlm: object, pp: str, thumb_path: Path
    ) -> tuple[str, str | None, str | None]:
        if pp.endswith("Gold"):
            return pp, "Gold is bright.", None
        return pp, None, "bad response"

    monkeypatch.setattr(manifest, "_describe_one", fake_describe)

    descriptions = manifest.generate_descriptions(
        thumbnails={
            "/World/Looks/Gold": tmp_path / "gold.png",
            "/World/Looks/Bad": tmp_path / "bad.png",
        },
        vlm_backend="nvidia_inference",
        vlm_model="fake-model",
        max_workers=2,
    )

    assert descriptions == {"/World/Looks/Gold": "Gold is bright."}
    assert created == {
        "backend": "nvidia_inference",
        "model": "fake-model",
        "api_key": "secret",
    }


def test_describe_one_handles_success_error_and_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def success(**kwargs: object) -> dict[str, object]:
        return {"response": '"Gold is bright and reflective."'}

    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.cv.vlm",
        SimpleNamespace(generate_vlm_response=success),
    )
    assert manifest._describe_one(
        object(), "/World/Looks/Gold", tmp_path / "gold.png"
    ) == (
        "/World/Looks/Gold",
        "Gold is bright and reflective.",
        None,
    )

    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.cv.vlm",
        SimpleNamespace(
            generate_vlm_response=lambda **kwargs: {"error": "bad response"}
        ),
    )
    assert manifest._describe_one(
        object(), "/World/Looks/Gold", tmp_path / "gold.png"
    ) == (
        "/World/Looks/Gold",
        None,
        "bad response",
    )

    def explode(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("vlm boom")

    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.cv.vlm",
        SimpleNamespace(generate_vlm_response=explode),
    )
    assert manifest._describe_one(
        object(), "/World/Looks/Gold", tmp_path / "gold.png"
    ) == (
        "/World/Looks/Gold",
        None,
        "vlm boom",
    )


def test_generate_descriptions_raises_without_required_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.models.vision_language_models",
        SimpleNamespace(create_vlm=lambda **kwargs: object()),
    )
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY not set for nim backend"):
        manifest.generate_descriptions(
            thumbnails={"/World/Looks/Gold": tmp_path / "gold.png"},
            vlm_backend="nim",
            vlm_model=None,
        )


def test_collect_mdl_deps_and_find_mdl_root_handle_read_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.mdl"
    broken.write_text("mdl 1.0;\n")
    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == broken:
            raise OSError("unreadable")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text, raising=False)
    assert manifest._collect_mdl_deps(broken) == {broken}
    assert manifest._find_mdl_root([broken]) == broken.parent


def test_find_mdl_root_falls_back_to_first_root_on_commonpath_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "a" / "A.mdl"
    second = tmp_path / "b" / "B.mdl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("mdl 1.0;\n")
    second.write_text("mdl 1.0;\n")

    monkeypatch.setattr(
        manifest.os.path,
        "commonpath",
        lambda paths: (_ for _ in ()).throw(ValueError("boom")),
    )

    assert manifest._find_mdl_root([first, second]) == first.parent


def test_run_generate_manifest_success_and_skip_descriptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usd_file = tmp_path / "materials.usd"
    usd_file.write_text("usd")
    template = tmp_path / "template.usd"
    template.write_text("template")
    yaml_path = tmp_path / "out" / "materials.yaml"

    monkeypatch.setattr(
        manifest,
        "discover_materials",
        lambda path: ["/World/Looks/Gold", "/World/Looks/Silver"],
    )
    monkeypatch.setattr(
        manifest,
        "render_thumbnails",
        lambda **kwargs: {"/World/Looks/Gold": tmp_path / "gold.png"},
    )
    descriptions = Mock(return_value={"/World/Looks/Gold": "Gold is bright."})
    monkeypatch.setattr(manifest, "generate_descriptions", descriptions)
    monkeypatch.setattr(manifest, "write_materials_yaml", lambda **kwargs: yaml_path)

    result = manifest.run_generate_manifest(
        GenerateManifestInput(
            usd_file=usd_file,
            output_dir=tmp_path / "out",
            template=template,
            image_size=128,
            max_workers=3,
            vlm_workers=4,
        )
    )

    assert result.success is True
    assert result.yaml_path == yaml_path
    assert result.materials_count == 2
    assert result.thumbnails_count == 1
    assert result.descriptions_count == 1
    descriptions.assert_called_once()

    descriptions.reset_mock()
    result = manifest.run_generate_manifest(
        GenerateManifestInput(
            usd_file=usd_file,
            output_dir=tmp_path / "out_skip",
            template=template,
            skip_descriptions=True,
        )
    )

    assert result.success is True
    assert result.descriptions_count == 0
    descriptions.assert_not_called()
