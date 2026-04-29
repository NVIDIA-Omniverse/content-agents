# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for collect_mesh_geometry_stats in prim.py."""

from pxr import Usd, UsdGeom, Vt

from world_understanding.utils.usd.prim import collect_mesh_geometry_stats


class TestCollectMeshGeometryStats:
    """Tests for collect_mesh_geometry_stats."""

    def test_empty_stage(self):
        """Empty stage returns zero counts."""
        stage = Usd.Stage.CreateInMemory()
        result = collect_mesh_geometry_stats(stage)
        assert result["total_prims"] == 0
        assert result["total_meshes"] == 0

    def test_stage_with_xform_only(self):
        """Stage with only Xform has prims but no meshes."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        result = collect_mesh_geometry_stats(stage)
        assert result["total_prims"] == 1
        assert result["total_meshes"] == 0

    def test_stage_with_mesh(self):
        """Stage with a mesh reports geometry stats."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
        mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        )
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))

        result = collect_mesh_geometry_stats(stage)
        assert result["total_meshes"] == 1
        assert result["total_vertices"] == 4
        assert result["total_faces"] == 1
        assert len(result["top_meshes_by_vertices"]) == 1
        assert result["top_meshes_by_vertices"][0]["path"] == "/Mesh"

    def test_skip_geometry(self):
        """skip_geometry=True skips vertex/face counting."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/Mesh")

        result = collect_mesh_geometry_stats(stage, skip_geometry=True)
        assert result["total_meshes"] == 1
        assert "total_vertices" not in result
        assert "total_faces" not in result

    def test_prim_type_counts(self):
        """Reports correct prim type distribution."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Mesh.Define(stage, "/World/Mesh1")
        UsdGeom.Mesh.Define(stage, "/World/Mesh2")
        UsdGeom.Camera.Define(stage, "/World/Cam")

        result = collect_mesh_geometry_stats(stage, skip_geometry=True)
        assert result["prim_type_counts"]["Mesh"] == 2
        assert result["prim_type_counts"]["Xform"] == 1
        assert result["prim_type_counts"]["Camera"] == 1
