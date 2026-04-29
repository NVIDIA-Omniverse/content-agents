# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Usd, UsdGeom

from material_agent.scene import analyze as analyze_module
from material_agent.scene import llm_refine as llm_refine_module
from material_agent.scene.analyze import (
    _build_payload_dag,
    _collect_payload_paths_from_node,
    _count_payload_meshes,
    _detect_payload_groups,
    _detect_structural_duplicates,
    analyze_scene,
)
from material_agent.scene.llm_refine import (
    _build_children_list,
    _build_split_context,
    _format_children_list,
)
from material_agent.scene.manifest import PayloadGroup, SubAsset


def _make_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def test_build_children_list_and_format(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path / "scene.usda")
    asset = UsdGeom.Xform.Define(stage, "/World/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    child_a = UsdGeom.Xform.Define(stage, "/World/Asset/A")
    mesh_a = UsdGeom.Mesh.Define(stage, "/World/Asset/A/Mesh")
    mesh_a.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    child_b = UsdGeom.Xform.Define(stage, "/World/Asset/B")
    mesh_b = UsdGeom.Mesh.Define(stage, "/World/Asset/B/Mesh")
    mesh_b.CreatePointsAttr([(0, 0, 0), (1, 1, 0)])
    stage.GetRootLayer().Save()

    children = _build_children_list(stage, "/World/Asset")
    formatted = _format_children_list(children)

    assert [child["name"] for child in children] == [
        child_a.GetPrim().GetName(),
        child_b.GetPrim().GetName(),
    ]
    assert [child["mesh_count"] for child in children] == [1, 1]
    assert [child["vertex_count"] for child in children] == [3, 2]
    assert "A: 1 meshes, 3 vertices" in formatted
    assert "B: 1 meshes, 2 vertices" in formatted


def test_detect_structural_duplicates_and_count_payload_meshes(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path / "scene.usda")
    for parent_name in ("A", "B"):
        parent = UsdGeom.Xform.Define(stage, f"/World/{parent_name}")
        stage.SetDefaultPrim(parent.GetPrim())
        UsdGeom.Mesh.Define(stage, f"/World/{parent_name}/Mesh")
    container = UsdGeom.Xform.Define(stage, "/World/C")
    stage.SetDefaultPrim(container.GetPrim())
    nested = UsdGeom.Xform.Define(stage, "/World/C/Nested")
    UsdGeom.Mesh.Define(stage, f"{nested.GetPath()}/Mesh")
    stage.GetRootLayer().Save()

    sub_assets = [
        SubAsset(id="a", name="A", prim_path="/World/A"),
        SubAsset(id="b", name="B", prim_path="/World/B"),
        SubAsset(id="c", name="C", prim_path="/World/C"),
        SubAsset(
            id="skip",
            name="Skip",
            prim_path="/World/DoesNotExist",
            instance_group="native_group",
        ),
    ]

    updated_assets, groups = _detect_structural_duplicates(stage, sub_assets)

    assert updated_assets[0].instance_group is None
    assert updated_assets[1].instance_group == "structural_A"
    assert len(groups) == 1
    assert groups[0].representative_id == "a"
    assert groups[0].member_paths == ["/World/B"]

    payload_path = tmp_path / "payload.usda"
    payload_stage = _make_stage(payload_path)
    UsdGeom.Mesh.Define(payload_stage, "/World/Mesh")
    payload_stage.GetRootLayer().Save()
    empty_path = tmp_path / "empty.usda"
    empty_stage = _make_stage(empty_path)
    empty_stage.GetRootLayer().Save()

    assert _count_payload_meshes(str(payload_path)) == 1
    assert _count_payload_meshes(str(empty_path)) == 0


def test_build_payload_dag_and_detect_payload_groups(
    monkeypatch, tmp_path: Path
) -> None:
    payload_a = tmp_path / "Payload A.usda"
    payload_b = tmp_path / "nested.usda"
    for path in (payload_a, payload_b):
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "material_agent.scene.payload_dag_utils.build_dag",
        lambda roots: {
            str(payload_a.resolve()): {str(payload_b.resolve())},
            str(payload_b.resolve()): set(),
        },
    )
    monkeypatch.setattr(
        "material_agent.scene.payload_dag_utils.compute_depths",
        lambda adj: {
            str(payload_a.resolve()): 1,
            str(payload_b.resolve()): 0,
        },
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._count_payload_meshes",
        lambda payload_file: 0 if Path(payload_file).name == "nested.usda" else 3,
    )

    built = _build_payload_dag(
        [
            PayloadGroup(
                id="payload_payload_a",
                group_name="payload_a",
                payload_file=str(payload_a.resolve()),
                instance_count=2,
                instance_paths=["/World/A"],
            )
        ]
    )

    nested_group = next(
        pg for pg in built if pg.payload_file == str(payload_b.resolve())
    )
    assert nested_group.depth == 0
    assert nested_group.status == "skipped"
    assert nested_group.parent_payload_files == [str(payload_a.resolve())]

    class FakePrim:
        def __init__(self, path: str, is_instance: bool, marker: str) -> None:
            self._path = path
            self._is_instance = is_instance
            self._marker = marker

        def IsInstance(self) -> bool:
            return self._is_instance

        def GetPath(self):
            return self._path

        def GetPrimIndex(self):
            return SimpleNamespace(rootNode=self._marker)

    class FakeStage:
        def Traverse(self):
            return [
                FakePrim("/World/A", True, "a"),
                FakePrim("/World/B", True, "b"),
                FakePrim("/World/C", False, "skip"),
            ]

    monkeypatch.setattr(
        "material_agent.scene.analyze._collect_payload_paths_from_node",
        lambda node, scene_dir: (
            [str(payload_a.resolve())] if node == "a" else [str(payload_b.resolve())]
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._build_payload_dag", lambda groups: groups
    )

    groups = _detect_payload_groups(FakeStage(), tmp_path / "scene.usda")

    assert len(groups) == 2
    groups_by_name = {group.group_name: group for group in groups}
    assert groups_by_name["nested"].status == "skipped"
    assert groups_by_name["payload_a"].instance_paths == ["/World/A"]


def test_collect_payload_paths_from_node(monkeypatch, tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.usda"
    payload_path.write_text("", encoding="utf-8")
    child_payload_path = tmp_path / "child_payload.usda"
    child_payload_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "pxr.Pcp.ArcTypePayload",
        "payload",
        raising=False,
    )

    node = SimpleNamespace(
        arcType="other",
        layerStack=SimpleNamespace(layers=[]),
        children=[
            SimpleNamespace(
                arcType="payload",
                layerStack=SimpleNamespace(
                    layers=[SimpleNamespace(realPath=str(payload_path))]
                ),
                children=[],
            ),
            SimpleNamespace(
                arcType="other",
                layerStack=SimpleNamespace(layers=[]),
                children=[
                    SimpleNamespace(
                        arcType="payload",
                        layerStack=SimpleNamespace(
                            layers=[SimpleNamespace(realPath=str(child_payload_path))]
                        ),
                        children=[],
                    )
                ],
            ),
        ],
    )

    collected = _collect_payload_paths_from_node(node, tmp_path)
    assert collected == [
        str(payload_path.resolve()),
        str(child_payload_path.resolve()),
    ]


def test_refine_objects_with_llm_handles_auto_and_llm_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children_map = {
        "/keep-small": [
            {"name": "a", "path": "/keep-small/a", "mesh_count": 10, "vertex_count": 1},
            {"name": "b", "path": "/keep-small/b", "mesh_count": 10, "vertex_count": 1},
        ],
        "/auto-descend": [
            {
                "name": "descended",
                "path": "/auto-descend/child",
                "mesh_count": 120,
                "vertex_count": 8,
            }
        ],
        "/auto-descend/child": [],
        "/auto-split": [
            {
                "name": "one",
                "path": "/auto-split/one",
                "mesh_count": 150,
                "vertex_count": 5,
            },
            {
                "name": "two",
                "path": "/auto-split/two",
                "mesh_count": 160,
                "vertex_count": 5,
            },
            {
                "name": "leaf",
                "path": "/auto-split/leaf",
                "mesh_count": 1,
                "vertex_count": 1,
            },
        ],
        "/auto-split/one": [],
        "/auto-split/two": [],
        "/llm-split": [
            {
                "name": "left",
                "path": "/llm-split/left",
                "mesh_count": 130,
                "vertex_count": 5,
            },
            {
                "name": "right",
                "path": "/llm-split/right",
                "mesh_count": 140,
                "vertex_count": 5,
            },
        ],
        "/llm-split/left": [],
        "/llm-split/right": [],
        "/llm-keep": [
            {
                "name": "left",
                "path": "/llm-keep/left",
                "mesh_count": 120,
                "vertex_count": 5,
            },
            {
                "name": "right",
                "path": "/llm-keep/right",
                "mesh_count": 125,
                "vertex_count": 5,
            },
        ],
        "/parse-fail": [
            {
                "name": "left",
                "path": "/parse-fail/left",
                "mesh_count": 120,
                "vertex_count": 5,
            },
            {
                "name": "right",
                "path": "/parse-fail/right",
                "mesh_count": 125,
                "vertex_count": 5,
            },
        ],
    }

    monkeypatch.setattr(
        "material_agent.scene.llm_refine._build_children_list",
        lambda stage, prim_path: children_map.get(prim_path, []),
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.get_subtree_geometry_stats",
        lambda stage, path, skip_geometry=False: {
            "mesh_count": next(
                child["mesh_count"]
                for values in children_map.values()
                for child in values
                if child["path"] == path
            ),
            "vertex_count": 42,
            "face_count": 7,
            "prim_type_breakdown": {"Mesh": 1},
        },
    )

    responses = iter(
        [
            SimpleNamespace(content='{"action": "split", "reason": "modular"}'),
            SimpleNamespace(content='{"action": "keep", "reason": "single object"}'),
            SimpleNamespace(content="not json"),
        ]
    )
    monkeypatch.setattr(
        "world_understanding.functions.models.chat_models.create_chat_model_from_config",
        lambda config, defaults=None: SimpleNamespace(
            invoke=lambda messages: next(responses)
        ),
    )
    monkeypatch.setattr(
        "world_understanding.utils.llm_parsing.extract_json_from_llm_response",
        lambda content, expected_keys=None: (
            json.loads(content) if content.startswith("{") else None
        ),
    )

    objects = [
        {
            "id": "obj_001",
            "name": "keep-small",
            "path": "/keep-small",
            "mesh_count": 50,
            "vertex_count": 1,
        },
        {
            "id": "obj_002",
            "name": "auto-descend",
            "path": "/auto-descend",
            "mesh_count": 200,
            "vertex_count": 1,
        },
        {
            "id": "obj_003",
            "name": "auto-split",
            "path": "/auto-split",
            "mesh_count": 220,
            "vertex_count": 1,
        },
        {
            "id": "obj_004",
            "name": "llm-split",
            "path": "/llm-split",
            "mesh_count": 210,
            "vertex_count": 1,
        },
        {
            "id": "obj_005",
            "name": "llm-keep",
            "path": "/llm-keep",
            "mesh_count": 210,
            "vertex_count": 1,
        },
        {
            "id": "obj_006",
            "name": "parse-fail",
            "path": "/parse-fail",
            "mesh_count": 210,
            "vertex_count": 1,
        },
    ]

    refined, instance_groups = llm_refine_module.refine_objects_with_llm(
        stage=object(),
        objects=objects,
        instance_groups=[{"group_name": "native"}],
        llm_config={"backend": "mock", "model": "mock"},
        auto_split_threshold=3,
        min_mesh_for_review=100,
    )

    refined_paths = {obj["path"] for obj in refined}
    assert instance_groups == [{"group_name": "native"}]
    assert "/keep-small" in refined_paths
    assert "/auto-descend" not in refined_paths
    assert "/auto-descend/child" in refined_paths
    assert "/auto-split" not in refined_paths
    assert "/auto-split/one" in refined_paths
    assert "/llm-split" not in refined_paths
    assert "/llm-split/left" in refined_paths
    assert "/llm-keep" in refined_paths
    assert "/parse-fail" in refined_paths

    descended = next(obj for obj in refined if obj["path"] == "/auto-descend/child")
    assert descended["split_context"] is None
    split_child = next(obj for obj in refined if obj["path"] == "/llm-split/left")
    assert split_child["split_context"]["parent_name"] == "llm-split"
    assert split_child["split_context"]["sibling_names"] == ["left", "right"]

    skipped_refine = llm_refine_module.refine_objects_with_llm(
        stage=object(),
        objects=objects,
        instance_groups=[],
        llm_config={"backend": "mock", "model": "mock"},
        auto_split_threshold=3,
        min_mesh_for_review=100,
    )
    assert skipped_refine[0]


def test_split_context_and_analyze_scene_main(monkeypatch, tmp_path: Path) -> None:
    parent_context = _build_split_context(
        {"name": "Parent", "split_context": {"ancestors": ["Root"]}},
        "Child",
        ["Sibling"],
    )
    assert parent_context == {
        "parent_name": "Parent",
        "sibling_names": ["Sibling"],
        "ancestors": ["Root", "Parent"],
    }

    fake_stage = object()
    monkeypatch.setattr("pxr.Usd.Stage.Open", lambda path: fake_stage)
    monkeypatch.setattr(
        "world_understanding.utils.usd.composition.collect_composition_arcs",
        lambda stage: {
            "sublayer_count": 1,
            "reference_count": 2,
            "unique_sub_usd_count": 3,
        },
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.collect_mesh_geometry_stats",
        lambda stage, skip_geometry=False: {
            "total_prims": 20,
            "total_meshes": 10,
            "total_vertices": 100,
        },
    )

    objects = [
        {
            "id": "obj_keep",
            "name": "Keep",
            "path": "/World/Keep",
            "mesh_count": 10,
            "vertex_count": 20,
        },
        {
            "id": "obj_skip",
            "name": "Skip",
            "path": "/World/Skip",
            "mesh_count": 10,
            "vertex_count": 20,
        },
        {
            "id": "obj_small",
            "name": "Small",
            "path": "/World/Small",
            "mesh_count": 1,
            "vertex_count": 5,
        },
        {
            "id": "obj_child",
            "name": "Child",
            "path": "/World/Parent/Child",
            "mesh_count": 12,
            "vertex_count": 24,
            "instance_group": None,
        },
    ]
    instance_groups_raw = [
        {
            "group_name": "native_group",
            "source_file": "/tmp/source.usd",
            "instance_count": 2,
            "member_paths": ["/World/Parent"],
        }
    ]
    monkeypatch.setattr(
        "world_understanding.functions.graphics.usd_scene_analysis.detect_objects",
        lambda *args, **kwargs: (objects, instance_groups_raw),
    )
    monkeypatch.setattr(
        "material_agent.scene.llm_refine.refine_objects_with_llm",
        lambda **kwargs: (objects, instance_groups_raw),
    )

    def fake_detect_dupes(stage, sub_assets):
        for sub_asset in sub_assets:
            if sub_asset.id == "obj_child":
                sub_asset.instance_group = "structural_dup"
        return sub_assets, []

    monkeypatch.setattr(
        "material_agent.scene.analyze._detect_structural_duplicates", fake_detect_dupes
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._detect_payload_groups",
        lambda stage, scene_usd_path: [
            PayloadGroup(
                id="payload_one",
                group_name="payload_one",
                payload_file="/tmp/payload_one.usd",
                instance_count=2,
                instance_paths=["/World/InstanceA"],
            )
        ],
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._extract_large_payload_representatives",
        lambda payload_groups, scene_usd_path: payload_groups.__setitem__(
            0,
            PayloadGroup(
                **{
                    **payload_groups[0].__dict__,
                    "representative_path": "/tmp/payload_one_representative.usd",
                }
            ),
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._detect_prototype_groups",
        lambda stage, scene_usd_path: [
            PayloadGroup(
                id="proto_one",
                group_name="proto_one",
                payload_file="/tmp/proto_one.usd",
                instance_count=1,
                instance_paths=["/World/Proto"],
            )
        ],
    )

    manifest = analyze_scene(
        tmp_path / "scene.usda",
        filters={
            "include_paths": ["/World"],
            "exclude_paths": ["/World/Skip"],
            "min_mesh_count": 5,
            "detect_structural_duplicates": True,
        },
        llm_config={"backend": "mock", "model": "mock"},
    )

    assert [sa.id for sa in manifest.sub_assets] == ["obj_keep", "obj_child"]
    assert manifest.sub_assets[1].instance_group is None
    assert manifest.instance_groups[0].representative_id == "obj_child"
    assert manifest.analysis["total_objects_detected"] == 4
    assert manifest.analysis["total_objects_after_filter"] == 2
    assert manifest.analysis["total_payload_groups"] == 2
    assert manifest.payload_groups[0].group_name == "payload_one"
    assert manifest.payload_groups[1].group_name == "proto_one"
