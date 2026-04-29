# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for remove_all_lights in USD prim utilities."""

import pytest

pxr = pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

from world_understanding.utils.usd.prim import remove_all_lights  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_stage_with_light_under_default_prim() -> Usd.Stage:
    """Light is a child of the default prim — the simple case."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdLux.DistantLight.Define(stage, "/Root/SunLight")
    UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    return stage


def _create_stage_with_light_outside_default_prim() -> Usd.Stage:
    """Light is a sibling of the default prim (e.g. /World/Light vs /Robot).

    This is the layout used by IsaacSim-exported assets such as the UR10.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Default prim is /Robot
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdGeom.Mesh.Define(stage, "/Robot/Mesh")

    # Light lives under /World — a sibling of the default prim
    UsdGeom.Xform.Define(stage, "/World")
    UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    return stage


def _create_stage_with_multiple_light_types() -> Usd.Stage:
    """Stage containing several light types scattered across the hierarchy."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    UsdLux.DistantLight.Define(stage, "/Root/Sun")
    UsdLux.DomeLight.Define(stage, "/Root/Env")
    UsdLux.SphereLight.Define(stage, "/Root/PointLight")
    UsdLux.RectLight.Define(stage, "/Env/AreaLight")  # outside default prim
    UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    return stage


def _create_stage_no_lights() -> Usd.Stage:
    """Stage with no lights at all."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    return stage


def _collect_light_paths(stage: Usd.Stage) -> list[str]:
    """Return paths of all active light prims in the stage."""
    paths = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasAPI(UsdLux.LightAPI) and prim.IsActive():
            paths.append(str(prim.GetPath()))
    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemoveAllLights:
    """Tests for remove_all_lights."""

    def test_removes_light_under_default_prim(self):
        stage = _create_stage_with_light_under_default_prim()
        assert _collect_light_paths(stage) == ["/Root/SunLight"]

        remove_all_lights(stage)

        assert _collect_light_paths(stage) == []
        # Mesh should still exist
        assert stage.GetPrimAtPath("/Root/Mesh").IsValid()

    def test_removes_light_outside_default_prim(self):
        """Regression test: lights outside the default prim must be removed."""
        stage = _create_stage_with_light_outside_default_prim()
        assert _collect_light_paths(stage) == ["/World/DistantLight"]

        remove_all_lights(stage)

        assert _collect_light_paths(stage) == []
        assert stage.GetPrimAtPath("/Robot/Mesh").IsValid()

    def test_removes_multiple_light_types(self):
        stage = _create_stage_with_multiple_light_types()
        assert len(_collect_light_paths(stage)) == 4

        remove_all_lights(stage)

        assert _collect_light_paths(stage) == []
        assert stage.GetPrimAtPath("/Root/Mesh").IsValid()

    def test_noop_when_no_lights(self):
        stage = _create_stage_no_lights()
        remove_all_lights(stage)
        assert stage.GetPrimAtPath("/Root/Mesh").IsValid()

    def test_survives_root_layer_export(self, tmp_path):
        """Light removal must persist through GetRootLayer().Export() (NVCF path)."""
        stage = _create_stage_with_light_outside_default_prim()
        remove_all_lights(stage)

        export_path = tmp_path / "exported.usdc"
        stage.GetRootLayer().Export(str(export_path))
        reloaded = Usd.Stage.Open(str(export_path))

        assert _collect_light_paths(reloaded) == []
        assert reloaded.GetPrimAtPath("/Robot/Mesh").IsValid()

    def test_survives_flatten_then_export(self, tmp_path):
        """Simulate the full NVCF pipeline: flatten -> remove lights -> export."""
        stage = _create_stage_with_light_outside_default_prim()

        # Flatten (simulates duplicate_stage / NVCF flatten)
        flat_layer = stage.Flatten()
        new_layer = Sdf.Layer.CreateAnonymous("flat.usda")
        new_layer.TransferContent(flat_layer)
        flat_stage = Usd.Stage.Open(new_layer)

        assert _collect_light_paths(flat_stage) == ["/World/DistantLight"]

        remove_all_lights(flat_stage)

        assert _collect_light_paths(flat_stage) == []

        # Export root layer and reload (NVCF path)
        export_path = tmp_path / "flattened.usdc"
        flat_stage.GetRootLayer().Export(str(export_path))
        reloaded = Usd.Stage.Open(str(export_path))

        assert _collect_light_paths(reloaded) == []

    def test_no_default_prim(self):
        """Stage without a default prim should still have lights removed."""
        stage = Usd.Stage.CreateInMemory()
        UsdLux.DistantLight.Define(stage, "/Lights/Sun")
        UsdGeom.Mesh.Define(stage, "/Scene/Mesh")
        # No default prim set

        remove_all_lights(stage)

        assert _collect_light_paths(stage) == []
        assert stage.GetPrimAtPath("/Scene/Mesh").IsValid()
