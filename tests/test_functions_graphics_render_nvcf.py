# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NVCF render response parsing, including V2-to-V1 conversion."""

import base64
import io

import numpy as np
from PIL import Image

from world_understanding.functions.graphics.render_nvcf import (
    RenderingStatus,
    _convert_v2_sensor,
    _convert_v2_to_v1,
    _is_v2_response,
)


class TestIsV2Response:
    """Tests for V2 response detection."""

    def test_detects_v2_response(self):
        result = {
            "total_cameras": 1,
            "total_frames": 1,
            "rendered_data": {"Camera": {}},
        }
        assert _is_v2_response(result) is True

    def test_rejects_v1_response(self):
        result = {
            "images": {"0": {}},
            "status": "success",
        }
        assert _is_v2_response(result) is False

    def test_rejects_empty_dict(self):
        assert _is_v2_response({}) is False

    def test_rejects_partial_v2(self):
        # Has rendered_data but not total_cameras
        assert _is_v2_response({"rendered_data": {}}) is False


class TestConvertV2Sensor:
    """Tests for V2 sensor data conversion."""

    def test_converts_uint8_rgb(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0, 0] = [255, 0, 0, 255]
        sensor_obj = {
            "type": "array",
            "data": base64.b64encode(arr.tobytes()).decode(),
            "shape": [4, 4, 4],
            "dtype": "uint8",
        }
        result = _convert_v2_sensor(sensor_obj)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4, 4)
        assert result[0, 0, 0] == 255

    def test_returns_string_when_no_shape(self):
        sensor_obj = {"data": "abc123"}
        result = _convert_v2_sensor(sensor_obj)
        assert result == "abc123"

    def test_returns_empty_string_when_no_data(self):
        sensor_obj = {"shape": [4, 4]}
        result = _convert_v2_sensor(sensor_obj)
        assert result == ""


class TestConvertV2ToV1:
    """Tests for V2→V1 full response conversion."""

    def _make_v2_response(
        self, width: int = 4, height: int = 4, n_cameras: int = 1
    ) -> dict:
        """Create a minimal V2 response with an RGB image."""
        rendered_data = {}
        for i in range(n_cameras):
            arr = np.full((height, width, 4), 128, dtype=np.uint8)
            cam_name = f"Camera{i}"
            rendered_data[cam_name] = {
                "0": {
                    "rgb": {
                        "type": "array",
                        "data": base64.b64encode(arr.tobytes()).decode(),
                        "shape": [height, width, 4],
                        "dtype": "uint8",
                    }
                }
            }
        return {
            "total_cameras": n_cameras,
            "total_frames": 1,
            "rendered_data": rendered_data,
        }

    def test_v1_has_status_success(self):
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        assert v1["status"] == RenderingStatus.success

    def test_v1_has_images_key(self):
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        assert "images" in v1
        assert "0" in v1["images"]

    def test_v1_frame_camera_nesting(self):
        """V1 format nests frame→camera (opposite of V2 camera→frame)."""
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        frame_data = v1["images"]["0"]
        assert "Camera0" in frame_data

    def test_v1_rgb_converted_to_base64_png(self):
        """V2 raw array data should become a base64 PNG in V1 'images' key."""
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        camera_data = v1["images"]["0"]["Camera0"]
        assert "images" in camera_data  # rgb → images
        # Should be valid base64 PNG
        png_bytes = base64.b64decode(camera_data["images"])
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (4, 4)

    def test_multi_camera_response(self):
        v2 = self._make_v2_response(n_cameras=3)
        v1 = _convert_v2_to_v1(v2)
        frame_data = v1["images"]["0"]
        assert len(frame_data) == 3
        for i in range(3):
            assert f"Camera{i}" in frame_data

    def test_sensor_name_mapping(self):
        """V2 sensor names should be mapped to V1 equivalents."""
        arr = np.zeros((4, 4), dtype=np.float32)
        v2 = {
            "total_cameras": 1,
            "total_frames": 1,
            "rendered_data": {
                "Camera": {
                    "0": {
                        "rgb": {
                            "type": "array",
                            "data": base64.b64encode(
                                np.zeros((4, 4, 4), dtype=np.uint8).tobytes()
                            ).decode(),
                            "shape": [4, 4, 4],
                            "dtype": "uint8",
                        },
                        "distance_to_image_plane": {
                            "type": "array",
                            "data": base64.b64encode(arr.tobytes()).decode(),
                            "shape": [4, 4],
                            "dtype": "float32",
                        },
                        "instance_segmentation": {
                            "type": "array",
                            "data": base64.b64encode(
                                np.zeros((4, 4), dtype=np.uint32).tobytes()
                            ).decode(),
                            "shape": [4, 4],
                            "dtype": "uint32",
                        },
                    }
                }
            },
        }
        v1 = _convert_v2_to_v1(v2)
        camera_data = v1["images"]["0"]["Camera"]
        assert "images" in camera_data  # rgb → images
        assert "linear_depth" in camera_data  # distance_to_image_plane → linear_depth
        assert (
            "instance_id_segmentation" in camera_data
        )  # instance_segmentation → instance_id_segmentation

    def test_empty_rendered_data(self):
        v2 = {"total_cameras": 0, "total_frames": 0, "rendered_data": {}}
        v1 = _convert_v2_to_v1(v2)
        assert v1["status"] == RenderingStatus.success
        assert v1["images"] == {}
