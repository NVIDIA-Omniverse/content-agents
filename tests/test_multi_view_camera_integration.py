# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for multi-view camera rendering.

Tests that the camera creation logic works end-to-end with actual USD stages.
"""

import pytest

try:
    from pxr import Usd, UsdGeom

    HAS_USD = True
except ImportError:
    HAS_USD = False

from world_understanding.functions.graphics.rendering import (
    CameraSpec,
    CameraViewType,
    RenderingConfig,
    prepare_render_prims,
)

pytestmark = pytest.mark.skipif(not HAS_USD, reason="USD not available")


@pytest.fixture
def simple_stage():
    """Create a simple USD stage with a cube mesh."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/Cube")
    return stage


class TestCameraCreationIntegration:
    """Test camera creation with different configurations."""

    def test_legacy_config_creates_cameras(self, simple_stage):
        """Legacy config creates cameras correctly."""
        config = RenderingConfig(
            camera_view_type=CameraViewType.CORNER,
            camera_ordering=["+x+y+z", "-x-y-z"],
            camera_prim_focus_margin=1.0,
        )

        stage, camera_paths, frames = prepare_render_prims(
            simple_stage, ["/Cube"], config, render_mode="prim_only"
        )

        # Should create 2 cameras
        assert len(camera_paths) == 2
        assert all("/Cameras/" in path for path in camera_paths)

        # Verify cameras exist in stage
        for camera_path in camera_paths:
            prim = stage.GetPrimAtPath(camera_path)
            assert prim.IsValid()
            assert prim.IsA(UsdGeom.Camera)

    def test_per_mode_cameras_creates_correct_count(self, simple_stage):
        """Per-mode camera config creates correct number of cameras."""
        config = RenderingConfig(
            camera_specs={
                "prim_only": [
                    CameraSpec(
                        direction="+x+y+z",
                        margin=1.0,
                        focal_length=100.0,
                        horizontal_aperture=1.0,
                        vertical_aperture=1.0,
                        near_clip_margin=0.1,
                        far_clip_margin=0.1,
                    ),
                    CameraSpec(
                        direction="-x-y-z",
                        margin=1.0,
                        focal_length=100.0,
                        horizontal_aperture=1.0,
                        vertical_aperture=1.0,
                        near_clip_margin=0.1,
                        far_clip_margin=0.1,
                    ),
                ],
                "prim_with_stage": [
                    CameraSpec(
                        direction="+x+y+z",
                        margin=6.0,
                        focal_length=100.0,
                        horizontal_aperture=1.0,
                        vertical_aperture=1.0,
                        near_clip_margin=0.1,
                        far_clip_margin=0.1,
                    ),
                ],
            }
        )

        # Test prim_only mode (2 cameras)
        stage1, camera_paths1, frames1 = prepare_render_prims(
            simple_stage, ["/Cube"], config, render_mode="prim_only"
        )
        assert len(camera_paths1) == 2

        # Test prim_with_stage mode (1 camera)
        stage2 = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage2, "/Cube")
        stage2, camera_paths2, frames2 = prepare_render_prims(
            stage2, ["/Cube"], config, render_mode="prim_with_stage"
        )
        assert len(camera_paths2) == 1

    def test_global_cameras_apply_to_all_modes(self, simple_stage):
        """Global (__all__) cameras apply to all modes."""
        config = RenderingConfig(
            camera_specs={
                "__all__": [
                    CameraSpec(
                        direction="+x+y+z",
                        margin=1.0,
                        focal_length=100.0,
                        horizontal_aperture=1.0,
                        vertical_aperture=1.0,
                        near_clip_margin=0.1,
                        far_clip_margin=0.1,
                    ),
                ]
            }
        )

        # Both modes should use the same global camera
        stage1, camera_paths1, frames1 = prepare_render_prims(
            simple_stage, ["/Cube"], config, render_mode="prim_only"
        )

        stage2 = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage2, "/Cube")
        stage2, camera_paths2, frames2 = prepare_render_prims(
            stage2, ["/Cube"], config, render_mode="prim_with_stage"
        )

        assert len(camera_paths1) == 1
        assert len(camera_paths2) == 1

    def test_render_mode_auto_inference(self, simple_stage):
        """When render_mode is None, it's inferred from should_render_prim_only."""
        config = RenderingConfig(
            camera_ordering=["+x+y+z"],
            should_render_prim_only=True,
            camera_prim_focus_margin=1.0,
        )

        # Don't pass render_mode - should be inferred as "prim_only"
        stage, camera_paths, frames = prepare_render_prims(
            simple_stage, ["/Cube"], config, render_mode=None
        )

        assert len(camera_paths) == 1

    def test_corner_and_side_cameras_mixed(self, simple_stage):
        """Mix of corner and side cameras works correctly."""
        config = RenderingConfig(
            camera_specs={
                "prim_only": [
                    CameraSpec(
                        direction="+x+y+z",  # Corner (auto-inferred)
                        margin=1.0,
                        focal_length=100.0,
                        horizontal_aperture=1.0,
                        vertical_aperture=1.0,
                        near_clip_margin=0.1,
                        far_clip_margin=0.1,
                    ),
                    CameraSpec(
                        direction="-z",  # Side (auto-inferred)
                        margin=1.0,
                        focal_length=100.0,
                        horizontal_aperture=1.0,
                        vertical_aperture=1.0,
                        near_clip_margin=0.1,
                        far_clip_margin=0.1,
                    ),
                ]
            }
        )

        stage, camera_paths, frames = prepare_render_prims(
            simple_stage, ["/Cube"], config, render_mode="prim_only"
        )

        assert len(camera_paths) == 2

        # Verify both camera types were created
        for camera_path in camera_paths:
            prim = stage.GetPrimAtPath(camera_path)
            assert prim.IsValid()
            assert prim.IsA(UsdGeom.Camera)
