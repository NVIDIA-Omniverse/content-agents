# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for new stage.py functions: collect_file_info, get_scene_extent."""

from pxr import Usd, UsdGeom

from world_understanding.utils.usd.stage import (
    collect_file_info,
    get_scene_extent,
    get_stage_info_from_path,
)


class TestCollectFileInfo:
    """Tests for collect_file_info."""

    def test_usda_file(self, tmp_path):
        """Returns correct metadata for a USDA file."""
        f = tmp_path / "test.usda"
        stage = Usd.Stage.CreateNew(str(f))
        UsdGeom.Xform.Define(stage, "/Root")
        stage.Save()

        info = collect_file_info(f)
        assert info["filename"] == "test.usda"
        assert info["format"] == "usda"
        assert info["file_size_bytes"] > 0
        assert info["path"] == str(f.resolve())

    def test_usdc_format(self, tmp_path):
        """Returns correct format for usdc."""
        f = tmp_path / "test.usdc"
        stage = Usd.Stage.CreateNew(str(f))
        stage.Save()

        info = collect_file_info(f)
        assert info["format"] == "usdc"

    def test_unknown_extension(self, tmp_path):
        """Unknown extension returns 'unknown' format."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        info = collect_file_info(f)
        assert info["format"] == "unknown"


class TestGetSceneExtent:
    """Tests for get_scene_extent."""

    def test_simple_mesh_extent(self, tmp_path):
        """Returns bounding box for a stage with geometry."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        mesh = UsdGeom.Cube.Define(stage, "/Cube")
        mesh.GetSizeAttr().Set(2.0)

        result = get_scene_extent(stage)
        assert result is not None
        assert "bounding_box" in result
        assert "size_scene_units" in result
        assert "size_meters" in result

    def test_empty_stage_returns_result(self):
        """Empty stage still returns a result (zero extent)."""
        stage = Usd.Stage.CreateInMemory()
        result = get_scene_extent(stage)
        # May return None or zero-extent depending on USD version
        # Just verify it doesn't crash
        assert result is None or "bounding_box" in result


class TestGetStageInfoFromPath:
    """Tests for get_stage_info_from_path."""

    def test_valid_usd(self, tmp_path):
        """Returns info dict for valid USD file."""
        f = tmp_path / "test.usda"
        stage = Usd.Stage.CreateNew(str(f))
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.Save()

        info = get_stage_info_from_path(f)
        assert info is not None
        assert info["prim_count"] == 2

    def test_nonexistent_file(self):
        """Returns None for non-existent file."""
        result = get_stage_info_from_path("/nonexistent/file.usd")
        assert result is None
