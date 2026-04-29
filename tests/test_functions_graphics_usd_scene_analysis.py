# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for usd_scene_analysis.detect_objects."""

from pxr import Usd, UsdGeom, Vt

from world_understanding.functions.graphics.usd_scene_analysis import detect_objects
from world_understanding.utils.usd.composition import collect_composition_arcs
from world_understanding.utils.usd.prim import collect_mesh_geometry_stats


class TestDetectObjects:
    """Integration tests for detect_objects."""

    def _make_stage_with_meshes(self):
        """Create an in-memory stage with Xforms and Meshes."""
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())

        # A few objects with meshes
        UsdGeom.Xform.Define(stage, "/World/Car")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Car/Body")
        mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        )
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))

        UsdGeom.Xform.Define(stage, "/World/Tree")
        mesh2 = UsdGeom.Mesh.Define(stage, "/World/Tree/Trunk")
        mesh2.GetPointsAttr().Set(Vt.Vec3fArray([(2, 0, 0), (3, 0, 0), (3, 2, 0)]))
        mesh2.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
        mesh2.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))

        return stage

    def test_returns_two_lists(self):
        """detect_objects returns (objects, instance_groups) tuple."""
        stage = self._make_stage_with_meshes()
        comp = collect_composition_arcs(stage)
        geom = collect_mesh_geometry_stats(stage)

        objects, instance_groups = detect_objects(stage, comp, geom)

        assert isinstance(objects, list)
        assert isinstance(instance_groups, list)

    def test_detects_objects_from_simple_scene(self):
        """Objects are detected from a simple hierarchy."""
        stage = self._make_stage_with_meshes()
        comp = collect_composition_arcs(stage)
        geom = collect_mesh_geometry_stats(stage)

        objects, _ = detect_objects(stage, comp, geom)

        # Should find at least some objects
        assert len(objects) >= 1

        # Each object has required keys
        for obj in objects:
            assert "path" in obj
            assert "name" in obj
            assert "source_classification" in obj

    def test_empty_stage_returns_empty(self):
        """Empty stage returns no objects."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

        comp = collect_composition_arcs(stage)
        geom = collect_mesh_geometry_stats(stage)

        objects, instance_groups = detect_objects(stage, comp, geom)
        assert isinstance(objects, list)
        assert isinstance(instance_groups, list)
