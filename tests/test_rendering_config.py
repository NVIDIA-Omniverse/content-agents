# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for unified rendering configuration parser."""

import pytest

from world_understanding.agentic.config import RendererConfig


def parse_rendering_config(config: dict) -> RendererConfig:
    """Helper function to parse rendering config (for backward compatibility with tests).

    Args:
        config: Renderer configuration dict

    Returns:
        RendererConfig with parsed rendering modes
    """
    renderer = RendererConfig(**config)
    # Add parsed rendering_modes_config as an attribute for tests
    if "rendering_modes" in config:
        renderer.rendering_modes_config = renderer.get_rendering_modes_config(
            config["rendering_modes"]
        )
    else:
        renderer.rendering_modes_config = {}
    return renderer


class TestRenderingConfigParser:
    """Test suite for rendering configuration parser."""

    def test_old_format_basic(self):
        """Test parsing old format (material_agent) with basic config."""
        config = {
            "backend": "remote",
            "image_width": 512,
            "image_height": 512,
            "camera_directions": ["+x+y+z", "-x-y-z"],
            "camera_prim_with_stage_margin": 3.0,
            "rendering_modes": ["prim_with_stage", "prim_only"],
        }

        result = parse_rendering_config(config)

        assert isinstance(result, RendererConfig)
        assert result.backend == "remote"
        assert result.image_width == 512
        assert len(result.rendering_modes_config) == 2
        assert "prim_with_stage" in result.rendering_modes_config
        assert "prim_only" in result.rendering_modes_config

        # Check prim_with_stage mode
        prim_with_stage = result.rendering_modes_config["prim_with_stage"]
        assert prim_with_stage.margin == 3.0
        assert prim_with_stage.cameras == ["+x+y+z", "-x-y-z"]
        assert prim_with_stage.camera_focus_mode == "prim"

    def test_old_format_single_camera(self):
        """Test old format with single camera direction (not a list)."""
        config = {
            "camera_directions": "+x+y+z",  # Single string, not list
            "rendering_modes": ["prim_only"],
        }

        result = parse_rendering_config(config)

        assert len(result.rendering_modes_config) == 1
        prim_only = result.rendering_modes_config["prim_only"]
        # prim_only mode adds -x-y-z when only the default camera +x+y+z is specified
        # to get better coverage from opposite viewing angles
        assert prim_only.cameras == ["+x+y+z", "-x-y-z"]

    def test_old_format_composition_mode(self):
        """Test old format with composition mode gets correct margin."""
        config = {
            "camera_directions": ["+x", "+y", "+z"],
            "camera_composition_margin": 12.0,
            "rendering_modes": ["composition"],
        }

        result = parse_rendering_config(config)

        composition = result.rendering_modes_config["composition"]
        assert composition.margin == 12.0
        assert composition.cameras == ["+x", "+y", "+z"]

    def test_new_format_basic(self):
        """Test parsing new format (simready_agent) with per-mode config."""
        config = {
            "backend": "remote",
            "image_width": 512,
            "rendering_modes": {
                "prim_with_stage": {
                    "margin": 6.0,
                    "cameras": ["+x+y+z", "-x+y+z", "-x-y+z", "+x-y+z"],
                    "camera_focus_mode": "prim",
                },
                "prim_only": {
                    "margin": 1.2,
                    "cameras": ["+x+y+z", "-x-y-z"],
                    "camera_focus_mode": "prim",
                },
            },
        }

        result = parse_rendering_config(config)

        assert isinstance(result, RendererConfig)
        assert len(result.rendering_modes_config) == 2

        # Check prim_with_stage
        prim_with_stage = result.rendering_modes_config["prim_with_stage"]
        assert prim_with_stage.margin == 6.0
        assert len(prim_with_stage.cameras) == 4
        assert prim_with_stage.camera_focus_mode == "prim"

        # Check prim_only
        prim_only = result.rendering_modes_config["prim_only"]
        assert prim_only.margin == 1.2
        assert len(prim_only.cameras) == 2

    def test_new_format_composition_mode(self):
        """Test new format with composition mode (stage focus)."""
        config = {
            "rendering_modes": {
                "composition": {
                    "margin": 12.0,
                    "cameras": ["+x", "+y", "+z"],
                    "camera_focus_mode": "stage",
                    "skip_occluded_images": False,
                }
            }
        }

        result = parse_rendering_config(config)

        composition = result.rendering_modes_config["composition"]
        assert composition.margin == 12.0
        assert composition.cameras == ["+x", "+y", "+z"]
        assert composition.camera_focus_mode == "stage"
        assert composition.skip_occluded_images is False

    def test_new_format_occlusion_settings(self):
        """Test new format with occlusion detection settings."""
        config = {
            "rendering_modes": {
                "prim_with_stage": {
                    "margin": 6.0,
                    "cameras": ["+x+y+z"],
                    "skip_occluded_images": True,
                    "occlusion_pixel_threshold": 50,
                }
            }
        }

        result = parse_rendering_config(config)

        mode = result.rendering_modes_config["prim_with_stage"]
        assert mode.skip_occluded_images is True
        assert mode.occlusion_pixel_threshold == 50

    def test_defaults(self):
        """Test that defaults are applied correctly."""
        config = {
            "rendering_modes": ["prim_only"]  # Minimal old format
        }

        result = parse_rendering_config(config)

        # Check global defaults
        assert result.backend == "remote"
        assert result.image_width == 512
        assert result.image_height == 512
        assert result.should_highlight_prim is False  # Default is False

        # Check mode defaults
        mode = result.rendering_modes_config["prim_only"]
        assert mode.margin == 1.2  # Default for prim_only
        # prim_only mode adds -x-y-z when only the default camera is specified
        assert mode.cameras == ["+x+y+z", "-x-y-z"]
        assert mode.camera_focus_mode == "prim"

    def test_invalid_format(self):
        """Test that invalid rendering_modes format raises error."""
        config = {
            "rendering_modes": "invalid_string"  # Neither list nor dict
        }

        with pytest.raises(ValueError, match="Invalid rendering_modes format"):
            parse_rendering_config(config)

    def test_empty_config(self):
        """Test parsing empty config uses all defaults."""
        config = {}

        result = parse_rendering_config(config)

        assert isinstance(result, RendererConfig)
        assert result.backend == "remote"
        assert len(result.rendering_modes_config) == 0  # No modes specified

    def test_backward_compatibility(self):
        """Test that old format produces same structure as new format."""
        # Old format config
        old_config = {
            "camera_directions": ["+x+y+z", "-x-y-z"],
            "camera_prim_with_stage_margin": 3.0,
            "rendering_modes": ["prim_with_stage"],
        }

        # Equivalent new format config
        new_config = {
            "rendering_modes": {
                "prim_with_stage": {
                    "margin": 3.0,
                    "cameras": ["+x+y+z", "-x-y-z"],
                    "camera_focus_mode": "prim",
                }
            }
        }

        old_result = parse_rendering_config(old_config)
        new_result = parse_rendering_config(new_config)

        # Should produce equivalent results
        old_mode = old_result.rendering_modes_config["prim_with_stage"]
        new_mode = new_result.rendering_modes_config["prim_with_stage"]

        assert old_mode.margin == new_mode.margin
        assert old_mode.cameras == new_mode.cameras
        assert old_mode.camera_focus_mode == new_mode.camera_focus_mode

    def test_kuka_config(self):
        """Test with actual Kuka robot arm config (old format)."""
        config = {
            "backend": "remote",
            "image_width": 512,
            "image_height": 512,
            "cull_style": "back",
            "should_highlight_prim": True,
            "should_assign_random_colors": True,
            "highlight_color": [0.7, 0.0, 0.0],
            "other_color_range": [0.1, 0.2],
            "camera_view_type": "corner",
            "camera_directions": ["+x+y+z", "-x-y-z"],
            "camera_prim_with_stage_margin": 3.0,
            "rendering_modes": ["prim_with_stage", "prim_only"],
        }

        result = parse_rendering_config(config)

        assert result.backend == "remote"
        assert result.image_width == 512
        assert len(result.rendering_modes_config) == 2
        assert result.rendering_modes_config["prim_with_stage"].margin == 3.0
        assert result.rendering_modes_config["prim_only"].margin == 1.2  # Default

    def test_sedan_config(self):
        """Test with actual SimReady sedan config (new format)."""
        config = {
            "backend": "remote",
            "image_width": 512,
            "image_height": 512,
            "rendering_modes": {
                "prim_only": {
                    "margin": 1.2,
                    "cameras": ["+x+y+z", "-x-y-z"],
                    "camera_focus_mode": "prim",
                    "skip_occluded_images": False,
                },
                "prim_with_stage": {
                    "margin": 6.0,
                    "cameras": [
                        "+x+y+z",
                        "-x+y+z",
                        "-x-y+z",
                        "+x-y+z",
                        "+x+y-z",
                        "-x+y-z",
                        "-x-y-z",
                        "+x-y-z",
                    ],
                    "camera_focus_mode": "prim",
                    "skip_occluded_images": True,
                },
                "composition": {
                    "margin": 12.0,
                    "cameras": ["+x", "+y", "+z"],
                    "camera_focus_mode": "stage",
                    "skip_occluded_images": False,
                },
            },
        }

        result = parse_rendering_config(config)

        assert len(result.rendering_modes_config) == 3
        assert "composition" in result.rendering_modes_config
        assert result.rendering_modes_config["composition"].camera_focus_mode == "stage"
        assert len(result.rendering_modes_config["prim_with_stage"].cameras) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
