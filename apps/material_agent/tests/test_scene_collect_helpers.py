# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for scene.collect helper functions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom

from material_agent.scene.collect import (
    _build_cascaded_payload_map,
    _extract_material_name,
    _fill_prediction_gaps,
    _find_predictions_path,
    _load_material_library,
    _load_payload_predictions,
    _merge_predictions,
    _path_to_filename,
)
from material_agent.scene.manifest import PayloadGroup, SceneManifest, SubAsset


def _write_jsonl(path: Path, lines: list[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + "\n")
            else:
                f.write(json.dumps(line) + "\n")
    return path


def _create_layer(path: Path, *, sublayers: list[str] | None = None) -> Path:
    layer = Sdf.Layer.CreateNew(str(path))
    if sublayers is not None:
        layer.subLayerPaths = sublayers
    layer.Save()
    return path


def test_find_predictions_path_fallback_order(tmp_path: Path) -> None:
    working_dir = tmp_path / "asset"
    restored = _write_jsonl(
        working_dir / "restored" / "restored_predictions.jsonl",
        [{"id": "/Root/A", "materials": "Steel"}],
    )
    manifest_path = _write_jsonl(
        tmp_path / "manifest_predictions.jsonl",
        [{"id": "/Root/B", "materials": "Copper"}],
    )
    raw = _write_jsonl(
        working_dir / "predictions" / "predictions.jsonl",
        [{"id": "/Root/C", "materials": "Plastic"}],
    )

    asset = SubAsset(
        id="a1",
        name="Asset",
        prim_path="/Root/Asset",
        working_dir=str(working_dir),
        predictions_path=str(manifest_path),
        status="completed",
    )
    assert _find_predictions_path(asset) == restored

    restored.unlink()
    assert _find_predictions_path(asset) == manifest_path

    manifest_path.unlink()
    assert _find_predictions_path(asset) == raw


def test_extract_material_name_supports_supported_prediction_shapes() -> None:
    assert _extract_material_name({"materials": {"material": "Steel"}}) == "Steel"
    assert _extract_material_name({"materials": "Copper"}) == "Copper"
    assert _extract_material_name({"material": "Plastic"}) == "Plastic"
    assert _extract_material_name({"materials": {"other": "x"}}) is None
    assert _extract_material_name({"materials": 123}) is None


def test_merge_predictions_prefers_restored_predictions_and_infers_parent(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "asset_a"
    _write_jsonl(
        working_dir / "predictions" / "predictions.jsonl",
        [{"id": "/Root/Asset/Mesh/Diffuse_0", "materials": "Wrong"}],
    )
    _write_jsonl(
        working_dir / "restored" / "restored_predictions.jsonl",
        [
            {
                "id": "/Root/Asset/Mesh/Diffuse_0",
                "materials": {"material": "Steel"},
            },
            {
                "id": "/Root/Asset/Mesh/Diffuse_1",
                "materials": {"material": "Steel"},
            },
            "{not-json",
        ],
    )
    payload_predictions = _write_jsonl(
        tmp_path / "payload_predictions.jsonl",
        [{"id": "/Root/Payload/Mesh", "materials": "Plastic"}],
    )

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="asset-a",
                name="AssetA",
                prim_path="/Root/Asset",
                working_dir=str(working_dir),
                predictions_path=str(tmp_path / "unused.jsonl"),
                status="completed",
            ),
            SubAsset(
                id="asset-b",
                name="Skipped",
                prim_path="/Root/Skipped",
                status="pending",
            ),
        ],
        payload_groups=[
            PayloadGroup(
                id="payload-a",
                group_name="PayloadA",
                payload_file=str(tmp_path / "payload_a.usda"),
                predictions_path=str(payload_predictions),
                status="completed",
            )
        ],
    )

    merged = _merge_predictions(manifest)

    assert merged["/Root/Asset/Mesh/Diffuse_0"] == "Steel"
    assert merged["/Root/Asset/Mesh/Diffuse_1"] == "Steel"
    assert merged["/Root/Asset/Mesh"] == "Steel"
    assert merged["/Root/Payload/Mesh"] == "Plastic"
    assert "/Root/Skipped" not in merged


def test_fill_prediction_gaps_uses_sibling_majority_then_asset_dominant(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/Asset")
    UsdGeom.Xform.Define(stage, "/Root/Asset/Group")
    UsdGeom.Mesh.Define(stage, "/Root/Asset/Group/MeshA")
    UsdGeom.Mesh.Define(stage, "/Root/Asset/Group/MeshB")
    UsdGeom.Mesh.Define(stage, "/Root/Asset/Group/MeshC")
    UsdGeom.Xform.Define(stage, "/Root/Asset/Other")
    UsdGeom.Mesh.Define(stage, "/Root/Asset/Other/MeshD")
    stage.Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="asset-a",
                name="Asset",
                prim_path="/Root/Asset",
                status="completed",
            )
        ]
    )
    prim_to_material = {
        "/Root/Asset/Group/MeshA": "Steel",
        "/Root/Asset/Group/MeshB": "Steel",
    }

    filled = _fill_prediction_gaps(scene_path, prim_to_material, manifest)

    assert filled["/Root/Asset/Group/MeshC"] == "Steel"
    assert filled["/Root/Asset/Other/MeshD"] == "Steel"


def test_load_material_library_resolves_relative_paths_and_bindings(
    tmp_path: Path,
) -> None:
    library_dir = tmp_path / "materials"
    library_dir.mkdir()
    library_usd = _create_layer(library_dir / "library.usda")
    yaml_path = tmp_path / "materials.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "library_path: materials/library.usda",
                "entries:",
                "  - name: Steel",
                "    binding: /World/Looks/Steel",
                "  - name: Incomplete",
                "  - binding: /World/Looks/MissingName",
            ]
        ),
        encoding="utf-8",
    )

    resolved_library, name_to_prim = _load_material_library(yaml_path)

    assert resolved_library == library_usd.resolve()
    assert name_to_prim == {"Steel": "/World/Looks/Steel"}


def test_load_material_library_supports_nested_schema_and_prim_path(
    tmp_path: Path,
) -> None:
    library_usd = _create_layer(tmp_path / "library.usda")
    yaml_path = tmp_path / "materials.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "materials:",
                "  library_path: library.usda",
                "  entries:",
                "    - name: Steel",
                "      prim_path: /World/Looks/Steel",
                "    - name: Copper",
                "      binding: /World/Looks/Copper",
                "    - not-a-dict",
            ]
        ),
        encoding="utf-8",
    )

    resolved_library, name_to_prim = _load_material_library(yaml_path)

    assert resolved_library == library_usd.resolve()
    assert name_to_prim == {
        "Steel": "/World/Looks/Steel",
        "Copper": "/World/Looks/Copper",
    }


def test_load_payload_predictions_prefers_explicit_path_and_ignores_invalid_json(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "payload_work"
    _write_jsonl(
        working_dir / "predictions" / "predictions.jsonl",
        [{"id": "/Root/RawMesh", "materials": "Wrong"}],
    )
    explicit = _write_jsonl(
        tmp_path / "payload_predictions.jsonl",
        [
            {"id": "/Root/MeshA", "materials": {"material": "Steel"}},
            "{broken-json",
            {"id": "/Root/MeshB", "material": "Plastic"},
        ],
    )

    payload = PayloadGroup(
        id="pg",
        group_name="Payload",
        payload_file=str(tmp_path / "payload.usda"),
        working_dir=str(working_dir),
        predictions_path=str(explicit),
        status="completed",
    )

    assert _load_payload_predictions(payload) == {
        "/Root/MeshA": "Steel",
        "/Root/MeshB": "Plastic",
    }


def test_build_cascaded_payload_map_rewrites_parent_outputs_bottom_up(
    tmp_path: Path,
) -> None:
    child_orig = _create_layer(tmp_path / "child_orig.usda")
    child_output = _create_layer(
        tmp_path / "child_output.usda",
        sublayers=[str(child_orig.resolve())],
    )
    parent_orig = _create_layer(
        tmp_path / "parent_orig.usda",
        sublayers=[str(child_orig.resolve())],
    )
    parent_output = _create_layer(
        tmp_path / "parent_output.usda",
        sublayers=[str(parent_orig.resolve())],
    )
    modified_input = _create_layer(tmp_path / "optimized_input.usda")
    orphan_orig = _create_layer(tmp_path / "orphan_orig.usda")

    manifest = SceneManifest(
        payload_groups=[
            PayloadGroup(
                id="child",
                group_name="child",
                payload_file=str(child_orig),
                output_usd_path=str(child_output),
                depth=0,
                status="completed",
            ),
            PayloadGroup(
                id="parent",
                group_name="parent",
                payload_file=str(parent_orig),
                output_usd_path=str(parent_output),
                depth=1,
                status="completed",
            ),
            PayloadGroup(
                id="no-output",
                group_name="orphan",
                payload_file=str(orphan_orig),
                modified_input_path=str(modified_input),
                depth=2,
                status="completed",
            ),
        ]
    )

    calls: list[tuple[str, dict[str, str]]] = []

    def fake_rewrite_arcs_in_layer(layer, cascaded_map, resolve_from):
        calls.append((Path(resolve_from).name, dict(cascaded_map)))
        if Path(resolve_from) == child_orig:
            return 0

        child_abs = str(child_orig.resolve())
        layer.subLayerPaths = [cascaded_map[child_abs]]
        return 1

    cascaded = _build_cascaded_payload_map(
        manifest=manifest,
        output_dir=tmp_path / "out",
        rewrite_arcs_in_layer=fake_rewrite_arcs_in_layer,
        shutil=shutil,
    )

    child_abs = str(child_orig.resolve())
    parent_abs = str(parent_orig.resolve())
    orphan_abs = str(orphan_orig.resolve())
    parent_copy = Path(cascaded[parent_abs])
    parent_base = tmp_path / "out" / "payload_copies" / "parent_base.usd"

    assert cascaded[child_abs] == str(child_output)
    assert cascaded[parent_abs] == str(parent_copy)
    assert cascaded[orphan_abs] == str(modified_input)
    assert calls[0] == ("child_orig.usda", {})
    assert calls[1][0] == "parent_orig.usda"
    assert calls[1][1][child_abs] == str(child_output)

    parent_base_layer = Sdf.Layer.FindOrOpen(str(parent_base))
    assert parent_base_layer is not None
    assert parent_base_layer.subLayerPaths == [str(child_output)]

    parent_copy_layer = Sdf.Layer.FindOrOpen(str(parent_copy))
    assert parent_copy_layer is not None
    assert parent_copy_layer.subLayerPaths == [str(parent_base.resolve())]


def test_path_to_filename_normalizes_prim_paths() -> None:
    assert _path_to_filename("/World/Foo/Bar") == "world_foo_bar"
