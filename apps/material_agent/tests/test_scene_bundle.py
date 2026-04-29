# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material_agent.scene.bundle."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from material_agent.scene.bundle import create_bundle


class FakeAssetPath:
    def __init__(self, path: str, resolvedPath: str = "") -> None:
        self.path = path
        self.resolvedPath = resolvedPath


class FakeAttr:
    def __init__(self, name: str, value: object) -> None:
        self._name = name
        self._value = value

    def HasValue(self) -> bool:
        return True

    def Get(self) -> object:
        return self._value

    def GetName(self) -> str:
        return self._name


class FakePrim:
    def __init__(self, attrs: list[FakeAttr]) -> None:
        self._attrs = attrs

    def IsA(self, cls: object) -> bool:
        return True

    def GetAttributes(self) -> list[FakeAttr]:
        return self._attrs

    def GetPath(self) -> str:
        return "/Root/Shader"


class FakeFlatLayer:
    def __init__(self, text: str) -> None:
        self._text = text

    def Export(self, path: str) -> None:
        Path(path).write_text(self._text)


class FakeMetaRootLayer:
    def Save(self) -> None:
        return None


class FakeExportRootLayer:
    def __init__(self, contents: str) -> None:
        self._contents = contents

    def Export(self, path: str) -> None:
        Path(path).write_text(self._contents)


class FakeOriginalStage:
    def __init__(self, layer_text: str) -> None:
        self._layer_text = layer_text

    def Flatten(self) -> FakeFlatLayer:
        return FakeFlatLayer(self._layer_text)


class FakeOpenedStage:
    def __init__(
        self,
        *,
        root_layer: object | None = None,
        prims: list[FakePrim] | None = None,
    ) -> None:
        self._root_layer = root_layer or FakeMetaRootLayer()
        self._prims = prims or []

    def GetRootLayer(self) -> object:
        return self._root_layer

    def Traverse(self) -> list[FakePrim]:
        return self._prims


def _install_fake_pxr(
    monkeypatch: pytest.MonkeyPatch,
    *,
    composed_scene_path: Path,
    usda_path: Path,
    final_path: Path,
    layer_text: str,
    verify_prims: list[FakePrim],
    open_failure_path: Path | None = None,
    flat_stage_missing: bool = False,
    verify_stage_missing: bool = False,
) -> None:
    open_counts: dict[str, int] = {}

    original_stage = FakeOriginalStage(layer_text)
    flat_stage = FakeOpenedStage(root_layer=FakeMetaRootLayer())
    export_stage = FakeOpenedStage(root_layer=FakeExportRootLayer("converted"))
    verify_stage = FakeOpenedStage(prims=verify_prims)

    def stage_open(path: str) -> object | None:
        path_obj = Path(path)
        if open_failure_path is not None and path_obj == open_failure_path:
            return None
        if path_obj == composed_scene_path:
            return original_stage
        if path_obj == usda_path:
            count = open_counts.get("usda", 0)
            open_counts["usda"] = count + 1
            if count == 0:
                return None if flat_stage_missing else flat_stage
            if final_path.suffix == ".usdc":
                return export_stage
            return None if verify_stage_missing else verify_stage
        if path_obj == final_path:
            return None if verify_stage_missing else verify_stage
        return None

    fake_usd = SimpleNamespace(Stage=SimpleNamespace(Open=stage_open))
    fake_usd_geom = SimpleNamespace(
        GetStageUpAxis=lambda stage: "Y",
        GetStageMetersPerUnit=lambda stage: 1.0,
        SetStageUpAxis=lambda stage, axis: None,
        SetStageMetersPerUnit=lambda stage, value: None,
    )
    fake_usd_shade = SimpleNamespace(Shader=object())
    fake_sdf = SimpleNamespace(AssetPath=FakeAssetPath)
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Sdf=fake_sdf,
            Usd=fake_usd,
            UsdGeom=fake_usd_geom,
            UsdShade=fake_usd_shade,
        ),
    )


def test_create_bundle_writes_usdc_and_counts_missing_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    composed_scene = tmp_path / "composed_scene.usd"
    composed_scene.write_text("usd")
    material_library_dir = tmp_path / "library"
    library_dir = material_library_dir / "Library"
    library_dir.mkdir(parents=True)
    (library_dir / "metal.mdl").write_text("mdl")
    (material_library_dir / "root_texture.png").write_text("png")

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "stale.txt").write_text("stale")

    usda_path = bundle_dir / "scene_flat.usda"
    usdc_path = bundle_dir / "scene_flat.usdc"
    layer_text = (
        f"{material_library_dir.resolve()}/Library/metal.mdl\n"
        f"{material_library_dir.resolve()}/root_texture.png\n"
    )
    verify_prims = [
        FakePrim(
            [
                FakeAttr(
                    "sourceAsset",
                    FakeAssetPath(
                        "./Library/metal.mdl",
                        str(bundle_dir / "Library" / "metal.mdl"),
                    ),
                ),
                FakeAttr("missing", FakeAssetPath("./missing.png", "")),
                FakeAttr("remote", FakeAssetPath("https://example.com/asset.png", "")),
            ]
        )
    ]
    _install_fake_pxr(
        monkeypatch,
        composed_scene_path=composed_scene,
        usda_path=usda_path,
        final_path=usdc_path,
        layer_text=layer_text,
        verify_prims=verify_prims,
    )

    result = create_bundle(
        composed_scene_path=composed_scene,
        material_library_dir=material_library_dir,
        bundle_dir=bundle_dir,
        output_format=".usdc",
    )

    assert result["usd_file"] == usdc_path
    assert result["verified_paths"] == 2
    assert result["missing_paths"] == 1
    assert result["library_files"] == 1
    assert usdc_path.exists()
    assert not usda_path.exists()
    assert (bundle_dir / "Library" / "metal.mdl").exists()
    assert not (bundle_dir / "stale.txt").exists()


def test_create_bundle_keeps_usda_and_handles_missing_library_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    composed_scene = tmp_path / "composed_scene.usd"
    composed_scene.write_text("usd")
    material_library_dir = tmp_path / "library"
    material_library_dir.mkdir()

    bundle_dir = tmp_path / "bundle"
    usda_path = bundle_dir / "scene_flat.usda"
    verify_prims = [FakePrim([])]
    _install_fake_pxr(
        monkeypatch,
        composed_scene_path=composed_scene,
        usda_path=usda_path,
        final_path=usda_path,
        layer_text="plain text",
        verify_prims=verify_prims,
    )

    result = create_bundle(
        composed_scene_path=composed_scene,
        material_library_dir=material_library_dir,
        bundle_dir=bundle_dir,
        output_format=".usda",
    )

    assert result["usd_file"] == usda_path
    assert result["library_files"] == 0
    assert result["verified_paths"] == 0
    assert result["missing_paths"] == 0
    assert usda_path.exists()


def test_create_bundle_raises_when_original_stage_cannot_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    composed_scene = tmp_path / "missing_stage.usd"
    composed_scene.write_text("usd")
    material_library_dir = tmp_path / "library"
    material_library_dir.mkdir()
    bundle_dir = tmp_path / "bundle"
    usda_path = bundle_dir / "scene_flat.usda"

    _install_fake_pxr(
        monkeypatch,
        composed_scene_path=composed_scene,
        usda_path=usda_path,
        final_path=usda_path,
        layer_text="unused",
        verify_prims=[],
        open_failure_path=composed_scene,
    )

    with pytest.raises(RuntimeError, match="Failed to open USD stage"):
        create_bundle(
            composed_scene_path=composed_scene,
            material_library_dir=material_library_dir,
            bundle_dir=bundle_dir,
        )


def test_create_bundle_raises_when_flattened_or_verify_stage_cannot_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    composed_scene = tmp_path / "composed_scene.usd"
    composed_scene.write_text("usd")
    material_library_dir = tmp_path / "library"
    material_library_dir.mkdir()
    bundle_dir = tmp_path / "bundle"
    usda_path = bundle_dir / "scene_flat.usda"

    _install_fake_pxr(
        monkeypatch,
        composed_scene_path=composed_scene,
        usda_path=usda_path,
        final_path=usda_path,
        layer_text="flat",
        verify_prims=[],
        flat_stage_missing=True,
    )
    with pytest.raises(RuntimeError, match="Failed to open flattened USD stage"):
        create_bundle(
            composed_scene_path=composed_scene,
            material_library_dir=material_library_dir,
            bundle_dir=bundle_dir,
            output_format=".usda",
        )

    _install_fake_pxr(
        monkeypatch,
        composed_scene_path=composed_scene,
        usda_path=usda_path,
        final_path=usda_path,
        layer_text="flat",
        verify_prims=[],
        verify_stage_missing=True,
    )
    with pytest.raises(RuntimeError, match="Failed to open bundled USD stage"):
        create_bundle(
            composed_scene_path=composed_scene,
            material_library_dir=material_library_dir,
            bundle_dir=bundle_dir,
            output_format=".usda",
        )
