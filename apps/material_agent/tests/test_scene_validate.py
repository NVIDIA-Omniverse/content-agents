# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material_agent.scene.validate module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

from material_agent.scene.validate import (
    AssetReport,
    PayloadReport,
    SceneReport,
    check_hierarchy_match,
    check_instances,
    check_layer_sublayers,
    check_stage_bindings,
    check_topology_match,
    count_layer_bindings,
    format_asset_report,
    validate_asset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEFAULT_MESHES = ["/World/MeshA", "/World/MeshB"]


def _make_simple_stage(path: Path, meshes: list[str] | None = None) -> Usd.Stage:
    """Create a minimal USD stage with meshes under /World."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    for m in _DEFAULT_MESHES if meshes is None else meshes:
        UsdGeom.Mesh.Define(stage, m)
    stage.GetRootLayer().Save()
    return stage


def _bind_material(stage: Usd.Stage, mesh_path: str, mat_path: str) -> None:
    """Create a material at *mat_path* and bind it to the mesh."""
    mat = UsdShade.Material.Define(stage, mat_path)
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------


class TestAssetReport:
    def test_ok_when_no_errors(self) -> None:
        r = AssetReport(name="test")
        assert r.ok is True

    def test_not_ok_when_errors(self) -> None:
        r = AssetReport(name="test", errors=["something broke"])
        assert r.ok is False

    def test_defaults(self) -> None:
        r = AssetReport(name="a")
        assert r.status == "unknown"
        assert r.bindings_in_layer == 0
        assert r.topology_match is True
        assert r.hierarchy_match is True


class TestPayloadReport:
    def test_ok_property(self) -> None:
        r = PayloadReport(name="p")
        assert r.ok is True
        r.errors.append("err")
        assert r.ok is False


class TestSceneReport:
    def test_defaults(self) -> None:
        r = SceneReport()
        assert r.assets == []
        assert r.payloads == []
        assert r.total_bindings == 0
        assert r.composed_scene_path == ""


# ---------------------------------------------------------------------------
# count_layer_bindings
# ---------------------------------------------------------------------------


class TestCountLayerBindings:
    def test_empty_layer(self, tmp_path: Path) -> None:
        stage = Usd.Stage.CreateNew(str(tmp_path / "empty.usda"))
        stage.GetRootLayer().Save()
        layer = Sdf.Layer.FindOrOpen(str(tmp_path / "empty.usda"))
        bindings, mat_defs, deinstanced = count_layer_bindings(layer)
        assert bindings == 0
        assert mat_defs == 0
        assert deinstanced == 0

    def test_counts_bindings_and_materials(self, tmp_path: Path) -> None:
        p = tmp_path / "bound.usda"
        stage = _make_simple_stage(p, ["/World/MeshA", "/World/MeshB"])
        _bind_material(stage, "/World/MeshA", "/World/Looks/MatA")
        _bind_material(stage, "/World/MeshB", "/World/Looks/MatB")
        stage.GetRootLayer().Save()

        layer = Sdf.Layer.FindOrOpen(str(p))
        bindings, mat_defs, deinstanced = count_layer_bindings(layer)
        assert bindings == 2
        assert mat_defs == 2
        assert deinstanced == 0

    def test_counts_deinstanced(self, tmp_path: Path) -> None:
        p = tmp_path / "deinst.usda"
        stage = _make_simple_stage(p, ["/World/MeshA"])
        prim = stage.GetPrimAtPath("/World/MeshA")
        prim.SetInstanceable(False)
        stage.GetRootLayer().Save()

        layer = Sdf.Layer.FindOrOpen(str(p))
        _, _, deinstanced = count_layer_bindings(layer)
        assert deinstanced == 1


# ---------------------------------------------------------------------------
# check_layer_sublayers
# ---------------------------------------------------------------------------


class TestCheckLayerSublayers:
    def test_no_sublayers(self, tmp_path: Path) -> None:
        p = tmp_path / "no_sub.usda"
        stage = Usd.Stage.CreateNew(str(p))
        stage.GetRootLayer().Save()
        layer = Sdf.Layer.FindOrOpen(str(p))
        has_opt, subs = check_layer_sublayers(layer)
        assert has_opt is False
        assert subs == []

    def test_detects_optimized_sublayer(self, tmp_path: Path) -> None:
        # Create a sublayer file with "optimized" in the name
        sub_path = tmp_path / "optimized_mesh.usda"
        sub_stage = Usd.Stage.CreateNew(str(sub_path))
        sub_stage.GetRootLayer().Save()

        p = tmp_path / "parent.usda"
        stage = Usd.Stage.CreateNew(str(p))
        # Use relative path so the check is based on filename
        stage.GetRootLayer().subLayerPaths.append("./optimized_mesh.usda")
        stage.GetRootLayer().Save()

        layer = Sdf.Layer.FindOrOpen(str(p))
        has_opt, subs = check_layer_sublayers(layer)
        assert has_opt is True
        assert len(subs) == 1

    def test_non_optimized_sublayer(self, tmp_path: Path) -> None:
        sub_path = tmp_path / "original.usda"
        sub_stage = Usd.Stage.CreateNew(str(sub_path))
        sub_stage.GetRootLayer().Save()

        p = tmp_path / "parent.usda"
        stage = Usd.Stage.CreateNew(str(p))
        # Use a relative path to avoid "optimized" appearing in absolute path
        stage.GetRootLayer().subLayerPaths.append("./original.usda")
        stage.GetRootLayer().Save()

        layer = Sdf.Layer.FindOrOpen(str(p))
        has_opt, subs = check_layer_sublayers(layer)
        assert has_opt is False
        assert len(subs) == 1


# ---------------------------------------------------------------------------
# check_stage_bindings
# ---------------------------------------------------------------------------


class TestCheckStageBindings:
    def test_empty_stage(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.usda"
        stage = Usd.Stage.CreateNew(str(p))
        stage.GetRootLayer().Save()
        our, old, none_ = check_stage_bindings(stage)
        assert our == 0
        assert old == 0
        assert none_ == 0

    def test_our_bindings(self, tmp_path: Path) -> None:
        p = tmp_path / "our.usda"
        stage = _make_simple_stage(p)
        _bind_material(stage, "/World/MeshA", "/World/Looks/Mat1")
        stage.GetRootLayer().Save()
        stage2 = Usd.Stage.Open(str(p))
        our, old, none_ = check_stage_bindings(stage2)
        assert our == 1
        assert old == 0
        assert none_ == 1  # MeshB has no binding

    def test_old_bindings(self, tmp_path: Path) -> None:
        p = tmp_path / "old.usda"
        stage = _make_simple_stage(p, ["/World/MeshA"])
        # Bind to a path that does NOT contain /Looks/
        _bind_material(stage, "/World/MeshA", "/World/Materials/OldMat")
        stage.GetRootLayer().Save()
        stage2 = Usd.Stage.Open(str(p))
        our, old, none_ = check_stage_bindings(stage2)
        assert our == 0
        assert old == 1
        assert none_ == 0

    def test_unbound_mesh(self, tmp_path: Path) -> None:
        p = tmp_path / "none.usda"
        stage = _make_simple_stage(p, ["/World/MeshA"])
        stage.GetRootLayer().Save()
        stage2 = Usd.Stage.Open(str(p))
        our, old, none_ = check_stage_bindings(stage2)
        assert our == 0
        assert old == 0
        assert none_ == 1

    def test_geomsubset_counted(self, tmp_path: Path) -> None:
        p = tmp_path / "subset.usda"
        stage = _make_simple_stage(p, ["/World/MeshA"])
        # Add a GeomSubset under MeshA
        subset = UsdGeom.Subset.Define(stage, "/World/MeshA/Subset0")
        subset.CreateElementTypeAttr("face")
        subset.CreateIndicesAttr([0])
        _bind_material(stage, "/World/MeshA/Subset0", "/World/Looks/SubMat")
        stage.GetRootLayer().Save()
        stage2 = Usd.Stage.Open(str(p))
        our, old, none_ = check_stage_bindings(stage2)
        # MeshA itself has no binding -> none; Subset0 -> our
        assert our == 1
        assert none_ == 1


# ---------------------------------------------------------------------------
# check_topology_match
# ---------------------------------------------------------------------------


class TestCheckTopologyMatch:
    def test_matching_topology(self, tmp_path: Path) -> None:
        p1 = tmp_path / "in.usda"
        p2 = tmp_path / "out.usda"
        _make_simple_stage(p1, ["/World/A", "/World/B"])
        _make_simple_stage(p2, ["/World/A", "/World/B"])
        match, in_c, out_c = check_topology_match(str(p1), str(p2))
        assert match is True
        assert in_c == 2
        assert out_c == 2

    def test_mismatched_topology(self, tmp_path: Path) -> None:
        p1 = tmp_path / "in.usda"
        p2 = tmp_path / "out.usda"
        _make_simple_stage(p1, ["/World/A", "/World/B"])
        _make_simple_stage(p2, ["/World/A"])
        match, in_c, out_c = check_topology_match(str(p1), str(p2))
        assert match is False
        assert in_c == 2
        assert out_c == 1

    def test_empty_stages(self, tmp_path: Path) -> None:
        p1 = tmp_path / "in.usda"
        p2 = tmp_path / "out.usda"
        _make_simple_stage(p1, [])
        _make_simple_stage(p2, [])
        match, in_c, out_c = check_topology_match(str(p1), str(p2))
        assert match is True
        assert in_c == 0
        assert out_c == 0


# ---------------------------------------------------------------------------
# check_hierarchy_match
# ---------------------------------------------------------------------------


class TestCheckHierarchyMatch:
    def test_matching_hierarchy(self, tmp_path: Path) -> None:
        p1 = tmp_path / "in.usda"
        p2 = tmp_path / "out.usda"
        _make_simple_stage(p1, ["/World/A", "/World/B"])
        _make_simple_stage(p2, ["/World/A", "/World/B"])
        match, only_in, only_out = check_hierarchy_match(str(p1), str(p2))
        assert match is True
        assert only_in == []
        assert only_out == []

    def test_extra_prims_in_output_under_looks_filtered(self, tmp_path: Path) -> None:
        """Prims under /World/Looks/ in output are filtered out, but
        /World/Looks itself (the Scope) is not filtered and counts as extra."""
        p1 = tmp_path / "in.usda"
        p2 = tmp_path / "out.usda"
        _make_simple_stage(p1, ["/World/A"])
        stage2 = _make_simple_stage(p2, ["/World/A"])
        UsdShade.Material.Define(stage2, "/World/Looks/Mat1")
        stage2.GetRootLayer().Save()

        match, only_in, only_out = check_hierarchy_match(str(p1), str(p2))
        # /World/Looks scope prim itself is NOT filtered (doesn't start with /World/Looks/)
        # so it shows up as extra in output
        assert "/World/Looks" in only_out
        # The material child IS filtered
        assert "/World/Looks/Mat1" not in only_out

    def test_missing_prim_in_output(self, tmp_path: Path) -> None:
        p1 = tmp_path / "in.usda"
        p2 = tmp_path / "out.usda"
        _make_simple_stage(p1, ["/World/A", "/World/B"])
        _make_simple_stage(p2, ["/World/A"])
        match, only_in, only_out = check_hierarchy_match(str(p1), str(p2))
        assert match is False
        assert "/World/B" in only_in

    def test_extra_non_looks_prim_in_output(self, tmp_path: Path) -> None:
        p1 = tmp_path / "in.usda"
        p2 = tmp_path / "out.usda"
        _make_simple_stage(p1, ["/World/A"])
        _make_simple_stage(p2, ["/World/A", "/World/Extra"])
        match, only_in, only_out = check_hierarchy_match(str(p1), str(p2))
        assert match is False
        assert "/World/Extra" in only_out


# ---------------------------------------------------------------------------
# check_instances
# ---------------------------------------------------------------------------


class TestCheckInstances:
    def test_no_instances(self, tmp_path: Path) -> None:
        p1 = tmp_path / "in.usda"
        p2 = tmp_path / "out.usda"
        _make_simple_stage(p1)
        _make_simple_stage(p2)
        in_i, out_i, kept, deinst, paths = check_instances(str(p1), str(p2))
        assert in_i == 0
        assert out_i == 0
        assert kept == 0
        assert deinst == 0
        assert paths == []

    def test_instances_preserved(self, tmp_path: Path) -> None:
        # Need a prototype (reference) for instances to work
        proto_path = tmp_path / "proto.usda"
        proto_stage = Usd.Stage.CreateNew(str(proto_path))
        root = UsdGeom.Xform.Define(proto_stage, "/Proto")
        proto_stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(proto_stage, "/Proto/Mesh")
        proto_stage.GetRootLayer().Save()

        for fname in ["in.usda", "out.usda"]:
            p = tmp_path / fname
            stage = Usd.Stage.CreateNew(str(p))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            inst = UsdGeom.Xform.Define(stage, "/World/Inst1")
            inst.GetPrim().GetReferences().AddReference(str(proto_path))
            inst.GetPrim().SetInstanceable(True)
            stage.GetRootLayer().Save()

        in_i, out_i, kept, deinst, paths = check_instances(
            str(tmp_path / "in.usda"), str(tmp_path / "out.usda")
        )
        assert in_i == 1
        assert out_i == 1
        assert kept == 1
        assert deinst == 0

    def test_instances_deinstanced(self, tmp_path: Path) -> None:
        proto_path = tmp_path / "proto.usda"
        proto_stage = Usd.Stage.CreateNew(str(proto_path))
        root = UsdGeom.Xform.Define(proto_stage, "/Proto")
        proto_stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(proto_stage, "/Proto/Mesh")
        proto_stage.GetRootLayer().Save()

        # Input has instance
        p_in = tmp_path / "in.usda"
        stage_in = Usd.Stage.CreateNew(str(p_in))
        world = UsdGeom.Xform.Define(stage_in, "/World")
        stage_in.SetDefaultPrim(world.GetPrim())
        inst = UsdGeom.Xform.Define(stage_in, "/World/Inst1")
        inst.GetPrim().GetReferences().AddReference(str(proto_path))
        inst.GetPrim().SetInstanceable(True)
        stage_in.GetRootLayer().Save()

        # Output does NOT have instance
        p_out = tmp_path / "out.usda"
        stage_out = Usd.Stage.CreateNew(str(p_out))
        world2 = UsdGeom.Xform.Define(stage_out, "/World")
        stage_out.SetDefaultPrim(world2.GetPrim())
        inst2 = UsdGeom.Xform.Define(stage_out, "/World/Inst1")
        inst2.GetPrim().GetReferences().AddReference(str(proto_path))
        # NOT set instanceable -> not an instance
        stage_out.GetRootLayer().Save()

        in_i, out_i, kept, deinst, paths = check_instances(str(p_in), str(p_out))
        assert in_i == 1
        assert out_i == 0
        assert kept == 0
        assert deinst == 1
        assert "/World/Inst1" in paths


# ---------------------------------------------------------------------------
# validate_asset
# ---------------------------------------------------------------------------


class TestValidateAsset:
    def test_no_pipeline_state(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        r = validate_asset(wdir)
        assert r.status == "no_state"
        assert not r.ok
        assert "No pipeline state file" in r.errors[0]

    def test_incomplete_pipeline(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["build_dataset_usd"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        r = validate_asset(wdir)
        assert r.status == "incomplete"
        assert not r.ok

    def test_failed_steps(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["predict"], "failed_steps": ["apply"]}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        # Need predictions file since predict completed
        pred_dir = wdir / "predictions"
        pred_dir.mkdir()
        (pred_dir / "predictions.jsonl").write_text(
            json.dumps({"id": "1", "material": "Steel"}) + "\n"
        )
        r = validate_asset(wdir)
        assert r.status == "failed"
        assert any("Failed steps" in e for e in r.errors)

    def test_predict_only_completed(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["predict"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        pred_dir = wdir / "predictions"
        pred_dir.mkdir()
        (pred_dir / "predictions.jsonl").write_text(
            json.dumps({"id": "1", "material": "Steel"}) + "\n"
        )
        r = validate_asset(wdir)
        assert r.status == "completed"
        assert r.ok
        assert r.predictions_count == 1
        assert r.has_predictions is True

    def test_no_predictions_file(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["predict"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        r = validate_asset(wdir)
        assert not r.ok
        assert any("No predictions file" in e for e in r.errors)

    def test_apply_completed_with_output(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["predict", "apply"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))

        # Predictions
        pred_dir = wdir / "predictions"
        pred_dir.mkdir()
        (pred_dir / "predictions.jsonl").write_text(
            json.dumps({"id": "mesh1", "material": "Steel"}) + "\n"
        )

        # Output USD with material binding
        out_dir = wdir / "output"
        out_dir.mkdir()
        out_path = out_dir / "output.usd"
        stage = _make_simple_stage(out_path, ["/World/MeshA"])
        _bind_material(stage, "/World/MeshA", "/World/Looks/Steel")
        stage.GetRootLayer().Save()

        r = validate_asset(wdir)
        assert r.status == "completed"
        assert r.ok
        assert r.bindings_in_layer > 0
        assert r.material_defs > 0
        assert r.bindings_our == 1

    def test_apply_no_output_usd(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["predict", "apply"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        pred_dir = wdir / "predictions"
        pred_dir.mkdir()
        (pred_dir / "predictions.jsonl").write_text(
            json.dumps({"id": "1", "material": "X"}) + "\n"
        )
        r = validate_asset(wdir)
        assert not r.ok
        assert any("output USD not found" in e for e in r.errors)

    def test_apply_zero_bindings(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["predict", "apply"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        pred_dir = wdir / "predictions"
        pred_dir.mkdir()
        (pred_dir / "predictions.jsonl").write_text(
            json.dumps({"id": "1", "material": "X"}) + "\n"
        )
        out_dir = wdir / "output"
        out_dir.mkdir()
        out_path = out_dir / "output.usd"
        _make_simple_stage(out_path, ["/World/MeshA"])  # no bindings
        r = validate_asset(wdir)
        assert not r.ok
        assert any("Zero material bindings" in e for e in r.errors)

    def test_simulate_mode_warning(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        (wdir / ".simulate").touch()
        state = {"completed_steps": ["predict"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        pred_dir = wdir / "predictions"
        pred_dir.mkdir()
        (pred_dir / "predictions.jsonl").write_text(
            json.dumps({"id": "1", "material": "X"}) + "\n"
        )
        r = validate_asset(wdir)
        assert r.status == "completed"
        assert any("simulate mode" in w for w in r.warnings)

    def test_restored_predictions_preferred(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["predict"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))

        # Both raw and restored exist
        pred_dir = wdir / "predictions"
        pred_dir.mkdir()
        (pred_dir / "predictions.jsonl").write_text(json.dumps({"id": "1"}) + "\n")
        restored_dir = wdir / "restored"
        restored_dir.mkdir()
        (restored_dir / "restored_predictions.jsonl").write_text(
            json.dumps({"id": "r1"}) + "\n" + json.dumps({"id": "r2"}) + "\n"
        )

        r = validate_asset(wdir)
        assert r.predictions_count == 2  # restored used, not raw

    def test_empty_predictions(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".asset1"
        wdir.mkdir()
        state = {"completed_steps": ["predict"], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        pred_dir = wdir / "predictions"
        pred_dir.mkdir()
        (pred_dir / "predictions.jsonl").write_text("\n")
        r = validate_asset(wdir)
        assert r.predictions_count == 0
        assert any("0 entries" in w for w in r.warnings)

    def test_name_strips_leading_dot(self, tmp_path: Path) -> None:
        wdir = tmp_path / ".dotname"
        wdir.mkdir()
        state = {"completed_steps": [], "failed_steps": []}
        (wdir / ".pipeline_state.json").write_text(json.dumps(state))
        r = validate_asset(wdir)
        assert r.name == "dotname"


# ---------------------------------------------------------------------------
# format_asset_report
# ---------------------------------------------------------------------------


class TestFormatAssetReport:
    def test_pass_report_with_bindings(self) -> None:
        r = AssetReport(name="asset1", status="completed", bindings_in_layer=5)
        lines = format_asset_report(r)
        assert len(lines) >= 1
        assert "PASS" in lines[0]
        assert "asset1" in lines[0]

    def test_fail_report_shows_errors(self) -> None:
        r = AssetReport(name="bad", status="failed", errors=["something wrong"])
        lines = format_asset_report(r)
        assert any("FAIL" in line for line in lines)
        assert any("ERROR" in line for line in lines)

    def test_inherited_not_verbose(self) -> None:
        r = AssetReport(name="inst", status="inherited")
        r.warnings.append("Instance group member")
        lines = format_asset_report(r, verbose=False)
        assert lines == []

    def test_inherited_verbose(self) -> None:
        r = AssetReport(name="inst", status="inherited", predictions_count=10)
        r.warnings.append("Instance group member")
        lines = format_asset_report(r, verbose=True)
        assert len(lines) >= 1
        assert "IG" in lines[0]

    def test_no_bindings_shows_predictions(self) -> None:
        r = AssetReport(name="pred_only", status="completed", predictions_count=42)
        lines = format_asset_report(r)
        assert any("predictions" in line for line in lines)

    def test_topology_mismatch_warning(self) -> None:
        r = AssetReport(
            name="topo",
            status="failed",
            errors=["err"],
            topology_match=False,
        )
        lines = format_asset_report(r, verbose=True)
        assert any("topology mismatch" in line.lower() for line in lines)

    def test_hierarchy_mismatch_warning(self) -> None:
        r = AssetReport(
            name="hier",
            status="failed",
            errors=["err"],
            hierarchy_match=False,
        )
        lines = format_asset_report(r, verbose=True)
        assert any("hierarchy mismatch" in line.lower() for line in lines)

    def test_deinstanced_info(self) -> None:
        r = AssetReport(
            name="deinst",
            status="failed",
            errors=["err"],
            instances_deinstanced=3,
        )
        lines = format_asset_report(r)
        assert any("de-instanced" in line for line in lines)
