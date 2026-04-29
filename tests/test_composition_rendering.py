# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for composition mode rendering.

Ensures the composition pipeline produces a highlight stage with isolated prims
and a plain stage that shows the full scene (all prims visible).

Regression test for: composition mode was rendering only the selected prim
instead of showing the full scene with the selected prim highlighted.
"""

import pytest

try:
    from pxr import Usd, UsdGeom

    HAS_USD = True
except ImportError:
    HAS_USD = False

from world_understanding.functions.graphics.rendering import (
    CameraFocusMode,
    RenderingConfig,
    prepare_prims_with_composition,
)

pytestmark = pytest.mark.skipif(not HAS_USD, reason="USD not available")


@pytest.fixture
def multi_mesh_stage():
    """Create a stage with multiple Mesh prims for composition testing.

    Uses UsdGeom.Mesh (not Cube/Sphere) because the rendering pipeline's
    visibility logic (disable_visibility_for_all_mesh_prims) only operates
    on Mesh-typed prims.
    """
    stage = Usd.Stage.CreateInMemory()
    for name, tx in [("MeshA", 0.0), ("MeshB", 2.0), ("MeshC", 4.0)]:
        xform = UsdGeom.Xform.Define(stage, f"/{name}")
        xform.AddTranslateOp().Set((tx, 0.0, 0.0))
        mesh = UsdGeom.Mesh.Define(stage, f"/{name}")
        # Minimal triangle so the mesh has a valid extent
        mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    return stage


def _get_visibility(stage: "Usd.Stage", prim_path: str, time) -> str:
    """Return the resolved visibility token for a prim at a given time."""
    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    vis_attr = mesh.GetVisibilityAttr()
    if not vis_attr or not vis_attr.HasValue():
        return UsdGeom.Tokens.inherited
    return vis_attr.Get(time=time)


class TestPrepareCompositionStages:
    """Test prepare_prims_with_composition stage setup."""

    def test_plain_stage_shows_all_prims(self, multi_mesh_stage):
        """Plain stage must have all prims visible (full scene) — regression test."""
        prim_paths = ["/MeshA", "/MeshB", "/MeshC"]
        config = RenderingConfig(
            camera_ordering=["+x"],
            camera_composition_margin=3.0,
            per_mode_focus_mode={"composition": CameraFocusMode.STAGE},
        )

        (_, _, _), (plain_stage, _, plain_frames) = prepare_prims_with_composition(
            multi_mesh_stage, prim_paths, config
        )

        # Every prim must be visible at every frame in the plain stage
        for frame in range(plain_frames):
            for path in prim_paths:
                vis = _get_visibility(plain_stage, path, Usd.TimeCode(frame))
                assert vis == UsdGeom.Tokens.inherited, (
                    f"Plain stage: {path} should be visible at frame {frame}, "
                    f"got '{vis}'. Composition must show the full scene."
                )

    def test_highlight_stage_isolates_prims(self, multi_mesh_stage):
        """Highlight stage must show only one prim per frame (isolated)."""
        prim_paths = ["/MeshA", "/MeshB", "/MeshC"]
        config = RenderingConfig(
            camera_ordering=["+x"],
            camera_composition_margin=3.0,
            per_mode_focus_mode={"composition": CameraFocusMode.STAGE},
        )

        (highlight_stage, _, highlight_frames), (_, _, _) = (
            prepare_prims_with_composition(multi_mesh_stage, prim_paths, config)
        )

        assert highlight_frames == len(prim_paths)

        for frame_idx, active_path in enumerate(prim_paths):
            for path in prim_paths:
                vis = _get_visibility(highlight_stage, path, Usd.TimeCode(frame_idx))
                if path == active_path:
                    assert vis == UsdGeom.Tokens.inherited, (
                        f"Highlight stage: {path} should be visible at its own "
                        f"frame {frame_idx}"
                    )
                else:
                    assert vis == UsdGeom.Tokens.invisible, (
                        f"Highlight stage: {path} should be hidden at frame "
                        f"{frame_idx} (active prim is {active_path})"
                    )

    def test_highlight_stage_uses_red_color(self, multi_mesh_stage):
        """Highlight stage must force red (1,0,0) for contour extraction."""
        prim_paths = ["/MeshA", "/MeshB"]
        config = RenderingConfig(
            camera_ordering=["+x"],
            # User sets non-red — should be overridden
            highlight_color=(0.5, 0.5, 0.0),
            per_mode_focus_mode={"composition": CameraFocusMode.STAGE},
        )

        (highlight_stage, _, _), _ = prepare_prims_with_composition(
            multi_mesh_stage, prim_paths, config
        )

        # At frame 0, MeshA should be highlighted in red
        prim = highlight_stage.GetPrimAtPath("/MeshA")
        mesh = UsdGeom.Mesh(prim)
        display_color = mesh.GetDisplayColorAttr().Get(time=Usd.TimeCode(0))
        assert display_color is not None and len(display_color) > 0
        r, g, b = display_color[0]
        assert (r, g, b) == pytest.approx((1.0, 0.0, 0.0), abs=0.01), (
            f"Highlight color should be red (1,0,0), got ({r},{g},{b})"
        )

    def test_both_stages_have_same_frame_count(self, multi_mesh_stage):
        """Both stages must produce the same number of frames."""
        prim_paths = ["/MeshA", "/MeshB", "/MeshC"]
        config = RenderingConfig(
            camera_ordering=["+x"],
            per_mode_focus_mode={"composition": CameraFocusMode.STAGE},
        )

        (_, _, h_frames), (_, _, p_frames) = prepare_prims_with_composition(
            multi_mesh_stage, prim_paths, config
        )

        assert h_frames == p_frames, (
            f"Frame count mismatch: highlight={h_frames}, plain={p_frames}"
        )

    def test_both_stages_have_same_camera_count(self, multi_mesh_stage):
        """Both stages must produce the same number of cameras."""
        prim_paths = ["/MeshA", "/MeshB"]
        config = RenderingConfig(
            camera_ordering=["+x", "+y", "+z"],
            per_mode_focus_mode={"composition": CameraFocusMode.STAGE},
        )

        (_, h_cams, _), (_, p_cams, _) = prepare_prims_with_composition(
            multi_mesh_stage, prim_paths, config
        )

        assert len(h_cams) == len(p_cams), (
            f"Camera count mismatch: highlight={len(h_cams)}, plain={len(p_cams)}"
        )
