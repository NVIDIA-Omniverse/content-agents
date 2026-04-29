# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for texture blending functions."""

import numpy as np
import pytest
from PIL import Image

from texture_agent.functions.texture_blending import blend_texture_onto_constant


class TestBlendTextureOntoConstant:
    """Tests for blend_texture_onto_constant()."""

    def test_full_opacity_replaces_base(self) -> None:
        """At opacity=1.0 with no alpha, texture fully replaces base."""
        base_color = (1.0, 0.0, 0.0)  # red
        texture = Image.new("RGB", (64, 64), (0, 0, 255))  # blue

        result = blend_texture_onto_constant(
            base_color, texture, output_size=(64, 64), opacity=1.0
        )

        arr = np.array(result)
        assert arr.shape == (64, 64, 3)
        # Should be all blue
        np.testing.assert_array_equal(arr[:, :, 0], 0)
        np.testing.assert_array_equal(arr[:, :, 1], 0)
        np.testing.assert_array_equal(arr[:, :, 2], 255)

    def test_zero_opacity_preserves_base(self) -> None:
        """At opacity=0.0, the result should be the base color."""
        base_color = (1.0, 0.0, 0.0)  # red
        texture = Image.new("RGB", (64, 64), (0, 0, 255))  # blue

        result = blend_texture_onto_constant(
            base_color, texture, output_size=(64, 64), opacity=0.0
        )

        arr = np.array(result)
        # Should be all red (base color)
        np.testing.assert_array_equal(arr[:, :, 0], 255)
        np.testing.assert_array_equal(arr[:, :, 1], 0)
        np.testing.assert_array_equal(arr[:, :, 2], 0)

    def test_rgba_texture_uses_alpha(self) -> None:
        """RGBA texture's alpha channel is used as the blend mask."""
        base_color = (1.0, 1.0, 1.0)  # white

        # Create RGBA texture: green with half the pixels transparent
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[:, :, 1] = 255  # green
        arr[:32, :, 3] = 255  # top half opaque
        arr[32:, :, 3] = 0  # bottom half transparent
        texture = Image.fromarray(arr, "RGBA")

        result = blend_texture_onto_constant(
            base_color, texture, output_size=(64, 64), opacity=1.0
        )

        result_arr = np.array(result)
        # Top half should be green
        assert result_arr[0, 0, 1] == 255
        assert result_arr[0, 0, 0] == 0
        # Bottom half should be white (base)
        assert result_arr[63, 0, 0] == 255
        assert result_arr[63, 0, 1] == 255

    def test_explicit_mask(self) -> None:
        """Explicit mask overrides texture alpha."""
        base_color = (0.0, 0.0, 0.0)  # black
        texture = Image.new("RGB", (64, 64), (255, 255, 255))  # white

        # Mask: left half white (texture), right half black (base)
        mask = Image.new("L", (64, 64), 0)
        mask_arr = np.array(mask)
        mask_arr[:, :32] = 255
        mask = Image.fromarray(mask_arr)

        result = blend_texture_onto_constant(
            base_color, texture, mask=mask, output_size=(64, 64)
        )

        result_arr = np.array(result)
        # Left half should be white
        assert result_arr[0, 0, 0] == 255
        # Right half should be black
        assert result_arr[0, 63, 0] == 0

    def test_output_size_is_respected(self) -> None:
        """Output image matches requested size."""
        base_color = (0.5, 0.5, 0.5)
        texture = Image.new("RGB", (100, 100), (128, 128, 128))

        result = blend_texture_onto_constant(
            base_color, texture, output_size=(256, 256)
        )

        assert result.size == (256, 256)

    def test_partial_opacity(self) -> None:
        """Partial opacity produces blended result."""
        base_color = (0.0, 0.0, 0.0)  # black
        texture = Image.new("RGB", (64, 64), (255, 255, 255))  # white

        result = blend_texture_onto_constant(
            base_color, texture, output_size=(64, 64), opacity=0.5
        )

        result_arr = np.array(result)
        # Should be roughly mid-gray (around 127-128)
        mean_val = result_arr.mean()
        assert 100 < mean_val < 160

    def test_base_color_mapping(self) -> None:
        """Base color (0-1 float) is correctly mapped to (0-255 int)."""
        base_color = (0.5, 0.25, 0.75)
        texture = Image.new("RGBA", (64, 64), (0, 0, 0, 0))  # fully transparent

        result = blend_texture_onto_constant(base_color, texture, output_size=(64, 64))

        result_arr = np.array(result)
        assert result_arr[0, 0, 0] == 127  # 0.5 * 255
        assert result_arr[0, 0, 1] == 63  # 0.25 * 255
        assert result_arr[0, 0, 2] == 191  # 0.75 * 255
