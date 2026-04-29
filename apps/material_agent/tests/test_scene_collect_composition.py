# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused composition tests for material_agent.scene.collect."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from pxr import Sdf, Usd, UsdGeom, UsdShade

from material_agent.scene.collect import (
    _compose_prototype_payloads,
    _copy_materials_from_library,
    _process_payload_groups,
    _rewrite_scene_payload_arcs,
    apply_and_compose,
    compose_material_layers,
)
from material_agent.scene.manifest import (
    InstanceGroup,
    PayloadGroup,
    SceneManifest,
    SubAsset,
)


def _write_jsonl(path: Path, lines: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def _create_library(tmp_path: Path, material_name: str = "Steel") -> tuple[Path, Path]:
    library_dir = tmp_path / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    library_usd = library_dir / "materials.usda"
    stage = Usd.Stage.CreateNew(str(library_usd))
    root = UsdGeom.Scope.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdShade.Material.Define(stage, f"/World/Looks/{material_name}")
    stage.GetRootLayer().Save()

    yaml_path = tmp_path / "materials.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"library_path: {library_usd.parent.name}/{library_usd.name}",
                "entries:",
                f"  - name: {material_name}",
                f"    binding: /World/Looks/{material_name}",
            ]
        ),
        encoding="utf-8",
    )
    return library_usd, yaml_path


def _binding_targets(layer: Sdf.Layer, prim_path: str) -> list[str]:
    spec = layer.GetPrimAtPath(prim_path)
    if not spec:
        return []
    rel = spec.relationships.get("material:binding")
    if not rel:
        return []
    return [str(t) for t in rel.targetPathList.explicitItems]


def _author_binding(layer: Sdf.Layer, prim_path: str, material_path: str) -> None:
    prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
    prim_spec.specifier = Sdf.SpecifierOver
    prim_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"]),
    )
    rel = Sdf.RelationshipSpec(prim_spec, "material:binding")
    rel.targetPathList.explicitItems = [Sdf.Path(material_path)]


def _make_scene_with_members(scene_path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    for member in ["RepMember", "DupeMember"]:
        UsdGeom.Xform.Define(stage, f"/Root/{member}")
        UsdGeom.Mesh.Define(stage, f"/Root/{member}/Mesh")
    stage.GetRootLayer().Save()

    # Author GeomSubset specs directly on the root layer so binding-over helper
    # sees them when propagating to duplicate members.
    layer = Sdf.Layer.FindOrOpen(str(scene_path))
    for member in ["RepMember", "DupeMember"]:
        subset = Sdf.CreatePrimInLayer(layer, f"/Root/{member}/Mesh/Diffuse_0")
        subset.typeName = "GeomSubset"
    layer.Save()
    return scene_path


def test_apply_and_compose_copies_materials_and_propagates_instance_bindings(
    tmp_path: Path,
) -> None:
    scene_path = _make_scene_with_members(tmp_path / "scene.usda")
    _library_usd, library_yaml = _create_library(tmp_path)

    predictions = _write_jsonl(
        tmp_path / "rep_work" / "predictions" / "predictions.jsonl",
        [{"id": "/Root/RepMember/Mesh", "materials": {"material": "Steel"}}],
    )
    rep = SubAsset(
        id="rep",
        name="Representative",
        prim_path="/Root/RepMember",
        working_dir=str(predictions.parent.parent),
        status="completed",
    )
    dupe = SubAsset(
        id="dupe",
        name="Duplicate",
        prim_path="/Root/DupeMember",
        status="pending",
        instance_group="dup_group",
    )
    manifest = SceneManifest(
        sub_assets=[rep, dupe],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id="rep",
                member_paths=["/Root/RepMember", "/Root/DupeMember"],
            )
        ],
    )

    output_usd = tmp_path / "output" / "composed.usda"
    result = apply_and_compose(scene_path, manifest, output_usd, library_yaml)

    assert result == output_usd
    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert layer is not None
    assert layer.defaultPrim == "Root"
    assert layer.GetPrimAtPath("/Root/Looks/Steel") is not None
    assert _binding_targets(layer, "/Root/RepMember/Mesh") == ["/Root/Looks/Steel"]
    assert _binding_targets(layer, "/Root/DupeMember/Mesh") == ["/Root/Looks/Steel"]
    assert _binding_targets(layer, "/Root/DupeMember/Mesh/Diffuse_0") == [
        "/Root/Looks/Steel"
    ]


def test_copy_materials_from_library_remaps_asset_paths(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    library_usd = library_dir / "materials.usda"
    layer = Sdf.Layer.CreateNew(str(library_usd))
    layer.defaultPrim = "World"
    shader_spec = Sdf.CreatePrimInLayer(layer, "/World/Looks/Steel/Shader")
    attr = Sdf.AttributeSpec(shader_spec, "diffuse_texture", Sdf.ValueTypeNames.Asset)
    attr.default = Sdf.AssetPath("textures/base.png")
    arr_attr = Sdf.AttributeSpec(
        shader_spec, "extra_textures", Sdf.ValueTypeNames.AssetArray
    )
    arr_attr.default = Sdf.AssetPathArray(
        [Sdf.AssetPath("textures/a.png"), Sdf.AssetPath("/absolute/keep.png")]
    )
    layer.Save()

    target_layer = Sdf.Layer.CreateAnonymous()
    output_usd = tmp_path / "output" / "composed.usda"
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    remap = _copy_materials_from_library(
        target_layer,
        library_usd,
        {"Steel": "/World/Looks/Steel", "Missing": "/World/Looks/Missing"},
        output_usd,
        scene_default_prim="Root",
    )

    shader = target_layer.GetPrimAtPath("/Root/Looks/Steel/Shader")
    assert shader is not None
    expected_rel = os.path.relpath(
        (library_dir / "textures" / "base.png").resolve(), output_usd.parent.resolve()
    ).replace("\\", "/")
    assert remap == {"/World/Looks/Steel": "/Root/Looks/Steel"}
    assert shader.attributes["diffuse_texture"].default.path == expected_rel
    arr_paths = [p.path for p in shader.attributes["extra_textures"].default]
    assert arr_paths[0].endswith("library/textures/a.png")
    assert arr_paths[1] == "/absolute/keep.png"


def test_process_payload_groups_creates_scoped_layers_and_payload_arcs(
    tmp_path: Path,
) -> None:
    _library_usd, library_yaml = _create_library(tmp_path)
    payload_path = tmp_path / "payload.usda"
    stage = Usd.Stage.CreateNew(str(payload_path))
    root = UsdGeom.Xform.Define(stage, "/Payload")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Mesh.Define(stage, "/Payload/Mesh")
    stage.GetRootLayer().Save()

    predictions = _write_jsonl(
        tmp_path / "payload_work" / "predictions" / "predictions.jsonl",
        [{"id": "/Payload/Mesh", "materials": "Steel"}],
    )
    payload_group = PayloadGroup(
        id="payload-1",
        group_name="payload_a",
        payload_file=str(payload_path),
        predictions_path=str(predictions),
        instance_paths=["/Root/InstanceA"],
        status="completed",
    )
    manifest = SceneManifest(payload_groups=[payload_group])

    composed_layer = Sdf.Layer.CreateAnonymous()
    output_usd = tmp_path / "output" / "scene.usda"
    arcs = _process_payload_groups(
        manifest,
        composed_layer,
        output_usd,
        library_yaml,
        {"Steel": "/World/Looks/Steel"},
    )

    assert arcs == 1
    assert payload_group.material_layer_path is not None
    payload_layer = Sdf.Layer.FindOrOpen(payload_group.material_layer_path)
    assert payload_layer is not None
    assert payload_layer.defaultPrim == "Payload"
    assert payload_layer.GetPrimAtPath("/Payload/Looks/Steel") is not None
    assert _binding_targets(payload_layer, "/Payload/Mesh") == ["/Payload/Looks/Steel"]
    instance_spec = composed_layer.GetPrimAtPath("/Root/InstanceA")
    assert instance_spec is not None
    assert len(instance_spec.payloadList.prependedItems) == 1
    assert instance_spec.payloadList.prependedItems[0].assetPath.endswith(
        "payload_layers/payload_a.usd"
    )


def test_compose_prototype_payloads_remaps_bindings_to_prototype_source(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/Root/Prototypes/PayloadProto")
    UsdGeom.Mesh.Define(stage, "/Root/Prototypes/PayloadProto/Mesh")
    stage.OverridePrim("/Root/Instances/PayloadInst")
    stage.GetPrimAtPath(
        "/Root/Instances/PayloadInst"
    ).GetReferences().AddInternalReference(Sdf.Path("/Root/Prototypes/PayloadProto"))
    stage.GetRootLayer().Save()

    manifest = SceneManifest(
        payload_groups=[
            PayloadGroup(
                id="pg",
                group_name="payload_proto",
                payload_file=str(tmp_path / "payload.usda"),
                instance_paths=["/Root/Instances/PayloadInst"],
                status="completed",
            )
        ]
    )
    composed_layer = Sdf.Layer.CreateAnonymous()

    written = _compose_prototype_payloads(
        scene_path,
        manifest,
        {"/Root/Instances/PayloadInst/Mesh": "Steel"},
        {"Steel": "/Root/Looks/Steel"},
        composed_layer,
    )

    assert written == 1
    assert _binding_targets(composed_layer, "/Root/Prototypes/PayloadProto/Mesh") == [
        "/Root/Looks/Steel"
    ]


def test_rewrite_scene_payload_arcs_copies_sublayers(tmp_path: Path) -> None:
    sublayer = tmp_path / "scene_sub.usda"
    Sdf.Layer.CreateNew(str(sublayer)).Save()
    scene_path = tmp_path / "scene.usda"
    layer = Sdf.Layer.CreateNew(str(scene_path))
    layer.subLayerPaths = [os.path.relpath(sublayer, scene_path.parent)]
    layer.Save()

    with (
        patch(
            "material_agent.scene.collect._build_cascaded_payload_map",
            return_value={
                str((tmp_path / "payload.usda").resolve()): str(
                    (tmp_path / "updated.usda").resolve()
                )
            },
        ),
        patch(
            "material_agent.scene.payload_dag_utils.rewrite_arcs_in_layer",
            return_value=1,
        ) as mock_rewrite,
    ):
        result = _rewrite_scene_payload_arcs(
            scene_path,
            SceneManifest(),
            tmp_path / "output",
        )

    assert len(result) == 1
    assert Path(result[0]).exists()
    assert Path(result[0]).parent.name == "scene_layers"
    mock_rewrite.assert_called_once()
    assert mock_rewrite.call_args.kwargs["resolve_from"] == str(sublayer.resolve())


def test_compose_material_layers_strips_sublayers_and_propagates_instances(
    tmp_path: Path,
) -> None:
    scene_path = _make_scene_with_members(tmp_path / "scene.usda")
    rep_layer_path = tmp_path / "rep_output.usda"
    rep_layer = Sdf.Layer.CreateNew(str(rep_layer_path))
    rep_layer.defaultPrim = "Root"
    rep_layer.subLayerPaths = [str((tmp_path / "rep_geometry.usda").resolve())]
    Sdf.CreatePrimInLayer(rep_layer, "/Root/Looks/Steel").specifier = Sdf.SpecifierDef
    _author_binding(rep_layer, "/Root/RepMember/Mesh", "/Root/Looks/Steel")
    rep_layer.Save()
    Sdf.Layer.CreateNew(str(tmp_path / "rep_geometry.usda")).Save()

    rep = SubAsset(
        id="rep",
        name="Representative",
        prim_path="/Root/RepMember",
        material_layer_path=str(rep_layer_path),
        status="completed",
    )
    dupe = SubAsset(
        id="dupe",
        name="Duplicate",
        prim_path="/Root/DupeMember",
        status="pending",
    )
    manifest = SceneManifest(
        sub_assets=[rep, dupe],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id="rep",
                member_paths=["/Root/RepMember", "/Root/DupeMember"],
            )
        ],
    )

    output_usd = tmp_path / "composed" / "scene_composed.usda"
    result = compose_material_layers(scene_path, manifest, output_usd)

    assert result == output_usd
    output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert output_layer is not None
    assert len(output_layer.subLayerPaths) == 3
    assert output_layer.subLayerPaths[-1] == str(scene_path.resolve())

    stripped_layer = Path(output_layer.subLayerPaths[0])
    propagation_layer = Path(output_layer.subLayerPaths[1])
    assert stripped_layer.exists()
    assert propagation_layer.exists()

    stripped = Sdf.Layer.FindOrOpen(str(stripped_layer))
    propagated = Sdf.Layer.FindOrOpen(str(propagation_layer))
    assert stripped is not None and not stripped.subLayerPaths
    assert propagated is not None
    assert _binding_targets(propagated, "/Root/DupeMember/Mesh") == [
        "/Root/Looks/Steel"
    ]
