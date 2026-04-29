# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Warp rendering backend.

Unit tests run without GPU/warp. Integration tests require warp + CUDA GPU.
"""

import math

import numpy as np
import pytest

from world_understanding.functions.graphics.render_warp import (
    _gf_matrix_to_transform_7f,
    _triangulate,
    _unpack_color_image,
    _unpack_depth_image,
)

# ---------------------------------------------------------------------------
# Unit tests (no GPU required)
# ---------------------------------------------------------------------------


class TestTriangulate:
    """Test _triangulate() for polygon → triangle conversion."""

    def test_single_triangle(self):
        """A single triangle (3 verts) passes through unchanged."""
        fvc = np.array([3])
        fvi = np.array([0, 1, 2])
        result = _triangulate(fvc, fvi)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_single_quad(self):
        """A quad (4 verts) produces 2 triangles via fan."""
        fvc = np.array([4])
        fvi = np.array([0, 1, 2, 3])
        result = _triangulate(fvc, fvi)
        # Fan from v0: (0,1,2) and (0,2,3)
        np.testing.assert_array_equal(result, [0, 1, 2, 0, 2, 3])

    def test_mixed_faces(self):
        """Mixed tri + quad produces correct triangle count."""
        fvc = np.array([3, 4])
        fvi = np.array([0, 1, 2, 3, 4, 5, 6])
        result = _triangulate(fvc, fvi)
        # tri: (0,1,2), quad fan: (3,4,5), (3,5,6)
        assert len(result) == 9  # 3 triangles * 3 verts

    def test_pentagon(self):
        """A pentagon (5 verts) produces 3 triangles."""
        fvc = np.array([5])
        fvi = np.array([0, 1, 2, 3, 4])
        result = _triangulate(fvc, fvi)
        # Fan from v0: (0,1,2), (0,2,3), (0,3,4)
        expected = [0, 1, 2, 0, 2, 3, 0, 3, 4]
        np.testing.assert_array_equal(result, expected)

    def test_empty_input(self):
        """Empty input produces empty output."""
        fvc = np.array([], dtype=np.int32)
        fvi = np.array([], dtype=np.int32)
        result = _triangulate(fvc, fvi)
        assert len(result) == 0

    def test_dtype_is_int32(self):
        """Result dtype should be int32."""
        fvc = np.array([3])
        fvi = np.array([0, 1, 2])
        result = _triangulate(fvc, fvi)
        assert result.dtype == np.int32


class TestGfMatrixToTransform7f:
    """Test _gf_matrix_to_transform_7f() transform conversion."""

    def test_identity_matrix(self):
        """Identity matrix → position (0,0,0) + identity quaternion (0,0,0,1)."""
        from pxr import Gf

        m = Gf.Matrix4d(1.0)
        result = _gf_matrix_to_transform_7f(m)
        assert len(result) == 7
        # Position should be origin
        assert abs(result[0]) < 1e-6
        assert abs(result[1]) < 1e-6
        assert abs(result[2]) < 1e-6
        # Quaternion should be identity (0, 0, 0, 1)
        assert abs(result[3]) < 1e-6  # qx
        assert abs(result[4]) < 1e-6  # qy
        assert abs(result[5]) < 1e-6  # qz
        assert abs(result[6] - 1.0) < 1e-6  # qw

    def test_translation_only(self):
        """Pure translation matrix extracts correct position."""
        from pxr import Gf

        m = Gf.Matrix4d(1.0)
        m.SetTranslateOnly(Gf.Vec3d(1.0, 2.0, 3.0))
        result = _gf_matrix_to_transform_7f(m)
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1] - 2.0) < 1e-6
        assert abs(result[2] - 3.0) < 1e-6

    def test_quaternion_is_normalized(self):
        """Result quaternion should be unit length."""
        from pxr import Gf

        # Create a rotation matrix (90 deg around Y)
        m = Gf.Matrix4d(1.0)
        m.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90))
        result = _gf_matrix_to_transform_7f(m)
        qx, qy, qz, qw = result[3], result[4], result[5], result[6]
        length = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        assert abs(length - 1.0) < 1e-5


class TestUnpackColorImage:
    """Test _unpack_color_image() uint32 → RGBA conversion."""

    def test_red_pixel(self):
        """Pure red pixel (R=255) unpacks correctly."""
        # Pack: R in bits 0-7, G in 8-15, B in 16-23, A in 24-31
        red_packed = np.array(
            [[[[255 | (0 << 8) | (0 << 16) | (255 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(red_packed, 0, 0)
        assert result.shape == (1, 1, 4)
        assert result[0, 0, 0] == 255  # R
        assert result[0, 0, 1] == 0  # G
        assert result[0, 0, 2] == 0  # B
        assert result[0, 0, 3] == 255  # A (forced to 255)

    def test_green_pixel(self):
        """Pure green pixel (G=255) unpacks correctly."""
        green_packed = np.array(
            [[[[0 | (255 << 8) | (0 << 16) | (255 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(green_packed, 0, 0)
        assert result[0, 0, 0] == 0  # R
        assert result[0, 0, 1] == 255  # G
        assert result[0, 0, 2] == 0  # B

    def test_blue_pixel(self):
        """Pure blue pixel (B=255) unpacks correctly."""
        blue_packed = np.array(
            [[[[0 | (0 << 8) | (255 << 16) | (255 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(blue_packed, 0, 0)
        assert result[0, 0, 0] == 0  # R
        assert result[0, 0, 1] == 0  # G
        assert result[0, 0, 2] == 255  # B

    def test_white_pixel(self):
        """White pixel (all 255) unpacks correctly."""
        white_packed = np.array(
            [[[[255 | (255 << 8) | (255 << 16) | (255 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(white_packed, 0, 0)
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 255
        assert result[0, 0, 2] == 255
        assert result[0, 0, 3] == 255

    def test_alpha_always_255(self):
        """Alpha channel is always forced to 255 regardless of packed value."""
        # Pack with alpha = 0 (should still output 255)
        packed = np.array(
            [[[[128 | (64 << 8) | (32 << 16) | (0 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(packed, 0, 0)
        assert result[0, 0, 3] == 255  # Alpha forced to 255

    def test_multi_camera(self):
        """Multiple cameras are indexed correctly."""
        # 1 world, 2 cameras, 1x1 pixels
        packed = np.array(
            [
                [
                    [[255]],  # cam 0: red channel only
                    [[255 << 8]],  # cam 1: green channel only
                ]
            ],
            dtype=np.uint32,
        )
        result0 = _unpack_color_image(packed, 0, 0)
        result1 = _unpack_color_image(packed, 0, 1)
        assert result0[0, 0, 0] == 255  # cam 0 = red
        assert result1[0, 0, 1] == 255  # cam 1 = green


class TestUnpackDepthImage:
    """Test _unpack_depth_image() extracts correct slice."""

    def test_extracts_correct_camera(self):
        """Depth extraction selects the right world/camera slice."""
        depth = np.zeros((1, 2, 4, 4), dtype=np.float32)
        depth[0, 0] = 1.0
        depth[0, 1] = 2.0

        result0 = _unpack_depth_image(depth, 0, 0)
        result1 = _unpack_depth_image(depth, 0, 1)
        assert result0.shape == (4, 4)
        np.testing.assert_allclose(result0, 1.0)
        np.testing.assert_allclose(result1, 2.0)

    def test_returns_copy(self):
        """Result should be a copy, not a view."""
        depth = np.ones((1, 1, 2, 2), dtype=np.float32)
        result = _unpack_depth_image(depth, 0, 0)
        result[0, 0] = 99.0
        assert depth[0, 0, 0, 0] == 1.0  # Original unchanged


class TestWarpBackendSensorSupport:
    """Test sensor capability methods without requiring warp."""

    def test_supported_sensor_modes_class_var(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        assert "depth" in WarpRenderingBackend.SUPPORTED_SENSOR_MODES
        assert "normal" in WarpRenderingBackend.SUPPORTED_SENSOR_MODES

    def test_supports_sensors_returns_true(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        assert backend.supports_sensors() is True

    def test_get_supported_sensor_modes(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        modes = backend.get_supported_sensor_modes()
        assert "depth" in modes
        assert "normal" in modes
        # Should return a copy
        modes.append("fake")
        assert "fake" not in backend.get_supported_sensor_modes()


class TestWarpBackendInit:
    """Test WarpRenderingBackend initialization."""

    def test_default_params(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        assert backend._device == "cuda:0"
        assert backend._color_boost == 3.0
        assert backend._enable_shadows is True
        assert backend._enable_backface_culling is True

    def test_custom_params(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend(
            device="cuda:1",
            color_boost=2.0,
            enable_shadows=False,
            enable_backface_culling=False,
        )
        assert backend._device == "cuda:1"
        assert backend._color_boost == 2.0
        assert backend._enable_shadows is False
        assert backend._enable_backface_culling is False


# ---------------------------------------------------------------------------
# Integration tests (require CUDA GPU + warp + Newton warp_raytrace)
# ---------------------------------------------------------------------------

_has_warp = False
try:
    import warp as wp

    wp.init()
    # Check we have a CUDA device
    if wp.is_cuda_available():
        from newton._src.sensors.warp_raytrace import RenderContext  # noqa: F401

        _has_warp = True
except (ImportError, RuntimeError):
    pass

requires_warp = pytest.mark.skipif(not _has_warp, reason="warp + CUDA not available")


@pytest.fixture
def simple_usd_stage_with_mesh():
    """Create a simple USD stage with a triangulated mesh and camera."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Create a simple quad mesh (2 triangles)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Quad")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(-1, -1, 0),
            Gf.Vec3f(1, -1, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(-1, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    mesh.GetDisplayColorAttr().Set([(0.8, 0.2, 0.2)])

    # Camera looking at the quad
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(50.0)
    camera.GetVerticalApertureAttr().Set(24.0)
    camera.GetHorizontalApertureAttr().Set(36.0)

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

    return stage


@requires_warp
class TestWarpIntegrationSingleCamera:
    """Integration test: single camera rendering with Warp."""

    def test_render_single_camera(self, simple_usd_stage_with_mesh):
        from world_understanding.functions.graphics.render_warp import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage_with_mesh,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0",
        )

        assert result["total_cameras"] == 1
        assert result["successful_cameras"] == 1
        assert result["failed_cameras"] == 0
        assert len(result["results"]) == 1
        assert result["results"][0]["frame_count"] == 1
        assert len(result["results"][0]["images"]) == 1

        # Check image dimensions
        img = result["results"][0]["images"][0]
        assert img.size == (64, 64)

    def test_render_with_depth_sensor(self, simple_usd_stage_with_mesh):
        from world_understanding.functions.graphics.render_warp import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage_with_mesh,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0",
            sensors=["depth"],
        )

        assert result["successful_cameras"] == 1
        sensors = result["results"][0]["sensors"]
        assert "depth" in sensors
        assert 0 in sensors["depth"]
        depth_arr = sensors["depth"][0]
        assert depth_arr.shape == (64, 64)

    def test_render_no_meshes_returns_error(self):
        """Rendering a stage with no meshes should report failure."""
        from pxr import Gf, Usd, UsdGeom

        from world_understanding.functions.graphics.render_warp import (
            render_all_cameras,
        )

        stage = Usd.Stage.CreateInMemory()
        camera = UsdGeom.Camera.Define(stage, "/Camera")
        camera.GetFocalLengthAttr().Set(50.0)
        camera.GetVerticalApertureAttr().Set(24.0)
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

        result = render_all_cameras(
            stage=stage,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0",
        )

        assert result["successful_cameras"] == 0
        assert result["failed_cameras"] == 1
        assert "error" in result["results"][0]


@requires_warp
class TestWarpIntegrationMultiFrame:
    """Integration test: multi-frame rendering with visibility."""

    def test_render_multiple_frames(self, simple_usd_stage_with_mesh):
        from world_understanding.functions.graphics.render_warp import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage_with_mesh,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0:2",
        )

        assert result["successful_cameras"] == 1
        assert result["results"][0]["frame_count"] == 3  # frames 0, 1, 2


@requires_warp
class TestWarpIntegrationBackend:
    """Integration test: WarpRenderingBackend class."""

    def test_backend_render(self, simple_usd_stage_with_mesh):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        result = backend.render(
            stage=simple_usd_stage_with_mesh,
            cameras=["/Camera"],
            image_width=64,
            image_height=64,
            frames="0",
        )

        assert result["total_cameras"] == 1
        assert result["successful_cameras"] == 1
        assert len(result["results"][0]["images"]) == 1

    def test_backend_cull_style_none(self, simple_usd_stage_with_mesh):
        """cull_style='none' should disable backface culling."""
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        # Should not raise
        result = backend.render(
            stage=simple_usd_stage_with_mesh,
            cameras=["/Camera"],
            image_width=64,
            cull_style="none",
            frames="0",
        )
        assert result["successful_cameras"] == 1
