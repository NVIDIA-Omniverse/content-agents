# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multi-view camera configuration parsing."""

from world_understanding.agentic.usd_tasks.renderer import (
    parse_camera_configuration,
)
from world_understanding.functions.graphics.rendering import CameraSpec, CameraViewType


class TestLevel1BackwardCompatibility:
    """Test Level 1: Legacy backward compatible format."""

    def test_camera_directions_legacy(self):
        """Old camera_directions format still works."""
        config = {
            "camera_view_type": "corner",
            "camera_directions": ["+x+y+z", "-x-y-z"],
            "camera_margin": 1.5,
            "camera_focal_length": 100.0,
        }
        result = parse_camera_configuration(config)

        assert "__all__" in result
        assert len(result["__all__"]) == 2
        assert all(cam.margin == 1.5 for cam in result["__all__"])
        assert all(cam.focal_length == 100.0 for cam in result["__all__"])
        assert result["__all__"][0].direction == "+x+y+z"
        assert result["__all__"][1].direction == "-x-y-z"

    def test_camera_corners_alias(self):
        """camera_corners is alias for camera_directions."""
        config1 = {"camera_directions": ["+x+y+z"], "camera_margin": 1.0}
        config2 = {"camera_corners": ["+x+y+z"], "camera_margin": 1.0}

        result1 = parse_camera_configuration(config1)
        result2 = parse_camera_configuration(config2)

        assert len(result1["__all__"]) == len(result2["__all__"])
        assert result1["__all__"][0].direction == result2["__all__"][0].direction

    def test_infer_view_type_from_direction(self):
        """View type inferred correctly from direction string."""
        corner_config = {"camera_directions": ["+x+y+z"]}
        side_config = {"camera_directions": ["+x"]}

        corner_result = parse_camera_configuration(corner_config)
        side_result = parse_camera_configuration(side_config)

        assert corner_result["__all__"][0].view_type == CameraViewType.CORNER
        assert side_result["__all__"][0].view_type == CameraViewType.SIDE

    def test_default_directions_corner(self):
        """Default corner directions used when none specified."""
        config = {"camera_view_type": "corner"}
        result = parse_camera_configuration(config)

        # Should get 8 default corner directions
        assert len(result["__all__"]) == 8
        expected_directions = [
            "+x+y+z",
            "-x+y+z",
            "-x-y+z",
            "+x-y+z",
            "+x+y-z",
            "-x+y-z",
            "-x-y-z",
            "+x-y-z",
        ]
        actual_directions = [cam.direction for cam in result["__all__"]]
        assert actual_directions == expected_directions

    def test_default_directions_side(self):
        """Default side directions used when none specified."""
        config = {"camera_view_type": "side"}
        result = parse_camera_configuration(config)

        # Should get 6 default side directions
        assert len(result["__all__"]) == 6
        expected_directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
        actual_directions = [cam.direction for cam in result["__all__"]]
        assert actual_directions == expected_directions


class TestLevel2EnhancedSimple:
    """Test Level 2: Enhanced simple camera configuration."""

    def test_camera_defaults(self):
        """Global defaults apply to all cameras."""
        config = {
            "cameras": {
                "defaults": {"margin": 1.5, "focal_length": 85.0},
                "views": [
                    {"direction": "+x+y+z"},
                    {"direction": "-x-y-z"},
                ],
            }
        }
        result = parse_camera_configuration(config)

        assert all(cam.margin == 1.5 for cam in result["__all__"])
        assert all(cam.focal_length == 85.0 for cam in result["__all__"])

    def test_per_view_overrides(self):
        """Per-view settings override defaults."""
        config = {
            "cameras": {
                "defaults": {"margin": 1.0, "focal_length": 100.0},
                "views": [
                    {"direction": "+x+y+z"},  # Uses defaults
                    {"direction": "-x-y-z", "margin": 2.0},  # Override margin
                ],
            }
        }
        result = parse_camera_configuration(config)

        assert result["__all__"][0].margin == 1.0
        assert result["__all__"][0].focal_length == 100.0
        assert result["__all__"][1].margin == 2.0
        assert result["__all__"][1].focal_length == 100.0  # Still uses default

    def test_mix_corner_and_side_views(self):
        """Can mix corner and side camera views."""
        config = {
            "cameras": {
                "views": [
                    {"direction": "+x+y+z"},  # Auto-inferred as corner
                    {"direction": "-z"},  # Auto-inferred as side
                ]
            }
        }
        result = parse_camera_configuration(config)

        assert result["__all__"][0].view_type == CameraViewType.CORNER
        assert result["__all__"][1].view_type == CameraViewType.SIDE


class TestLevel3PerMode:
    """Test Level 3: Per-mode camera configuration."""

    def test_per_mode_cameras(self):
        """Each mode gets its own camera list."""
        config = {
            "rendering_modes": {
                "prim_only": {"cameras": [{"direction": "+x+y+z", "margin": 1.1}]},
                "prim_with_stage": {
                    "cameras": [
                        {"direction": "+x+y+z", "margin": 2.0},
                        {"direction": "-x-y-z", "margin": 2.0},
                    ]
                },
            }
        }
        result = parse_camera_configuration(config)

        assert len(result["prim_only"]) == 1
        assert len(result["prim_with_stage"]) == 2
        assert result["prim_only"][0].margin == 1.1
        assert result["prim_with_stage"][0].margin == 2.0

    def test_mode_level_settings_compact_syntax(self):
        """Mode-level settings apply to all cameras (compact syntax #2)."""
        config = {
            "rendering_modes": {
                "prim_only": {
                    "margin": 1.5,
                    "focal_length": 85.0,
                    "cameras": ["+x+y+z", "-x-y-z"],
                }
            }
        }
        result = parse_camera_configuration(config)

        assert len(result["prim_only"]) == 2
        assert all(cam.margin == 1.5 for cam in result["prim_only"])
        assert all(cam.focal_length == 85.0 for cam in result["prim_only"])

    def test_shorthand_list_syntax(self):
        """Support shorthand list syntax for cameras (compact syntax #3)."""
        config = {
            "camera_defaults": {"margin": 1.5},
            "rendering_modes": {"prim_only": ["+x+y+z", "-x-y-z"]},  # Shorthand
        }
        result = parse_camera_configuration(config)

        assert len(result["prim_only"]) == 2
        assert all(cam.margin == 1.5 for cam in result["prim_only"])

    def test_use_cameras_from_reference(self):
        """Can reference another mode's cameras."""
        config = {
            "rendering_modes": {
                "prim_only": {"cameras": [{"direction": "+x+y+z"}]},
                "linear_depth": {"use_cameras_from": "prim_only"},
            }
        }
        result = parse_camera_configuration(config)

        assert result["linear_depth"] == result["prim_only"]
        assert len(result["linear_depth"]) == 1


class TestCompactSyntaxVariants:
    """Test various compact syntax options."""

    def test_inline_dict_syntax(self):
        """Compact syntax #1: Inline dictionaries."""
        config = {
            "rendering_modes": {
                "prim_only": {
                    "cameras": [
                        {"direction": "+x+y+z", "margin": 1.1},
                        {"direction": "-x-y-z", "margin": 1.1},
                    ]
                }
            }
        }
        result = parse_camera_configuration(config)

        assert len(result["prim_only"]) == 2
        assert all(cam.margin == 1.1 for cam in result["prim_only"])

    def test_mode_level_settings(self):
        """Compact syntax #2: Mode-level settings (most compact)."""
        config = {
            "rendering_modes": {
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
                }
            }
        }
        result = parse_camera_configuration(config)

        assert len(result["prim_with_stage"]) == 8
        assert all(cam.margin == 6.0 for cam in result["prim_with_stage"])

    def test_mixed_mode_and_per_camera_settings(self):
        """Mode-level settings can be overridden per-camera."""
        config = {
            "rendering_modes": {
                "prim_only": {
                    "margin": 1.0,  # Mode-level default
                    "cameras": [
                        "+x+y+z",  # Uses mode-level margin 1.0
                        {"direction": "-x-y-z", "margin": 2.0},  # Override
                    ],
                }
            }
        }
        result = parse_camera_configuration(config)

        assert result["prim_only"][0].margin == 1.0
        assert result["prim_only"][1].margin == 2.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_rendering_modes(self):
        """Empty rendering_modes falls back to legacy."""
        config = {
            "rendering_modes": {},
            "camera_directions": ["+x+y+z"],
        }
        result = parse_camera_configuration(config)

        assert "__all__" in result
        assert len(result["__all__"]) == 1

    def test_camera_defaults_merged(self):
        """camera_defaults field is properly merged."""
        config = {
            "camera_defaults": {"margin": 2.0, "focal_length": 120.0},
            "rendering_modes": {"prim_only": ["+x+y+z"]},
        }
        result = parse_camera_configuration(config)

        assert result["prim_only"][0].margin == 2.0
        assert result["prim_only"][0].focal_length == 120.0

    def test_global_defaults_parameter(self):
        """Global defaults parameter is properly applied."""
        config = {"rendering_modes": {"prim_only": ["+x+y+z"]}}
        global_defaults = {"margin": 5.0}

        result = parse_camera_configuration(config, global_defaults)

        assert result["prim_only"][0].margin == 5.0

    def test_precedence_order(self):
        """Test merge precedence: global < mode < camera."""
        global_defaults = {"margin": 1.0, "focal_length": 100.0}
        config = {
            "camera_defaults": {"margin": 2.0},  # Overrides global
            "rendering_modes": {
                "prim_only": {
                    "margin": 3.0,  # Overrides camera_defaults
                    "cameras": [
                        "+x+y+z",  # Uses mode margin 3.0
                        {"direction": "-x-y-z", "margin": 4.0},  # Overrides mode
                    ],
                }
            },
        }

        result = parse_camera_configuration(config, global_defaults)

        assert result["prim_only"][0].margin == 3.0  # Mode-level
        assert result["prim_only"][1].margin == 4.0  # Camera-level
        # focal_length not overridden anywhere, uses global
        assert result["prim_only"][0].focal_length == 100.0


class TestCameraSpecMethods:
    """Test CameraSpec class methods."""

    def test_auto_infer_corner_view_type(self):
        """CameraSpec auto-infers CORNER for multi-axis directions."""
        cam = CameraSpec(direction="+x+y+z")
        assert cam.view_type == CameraViewType.CORNER

    def test_auto_infer_side_view_type(self):
        """CameraSpec auto-infers SIDE for single-axis directions."""
        cam = CameraSpec(direction="+x")
        assert cam.view_type == CameraViewType.SIDE

    def test_merge_with_defaults_fills_none(self):
        """merge_with_defaults fills None values."""
        cam = CameraSpec(direction="+x+y+z", margin=None, focal_length=None)

        merged = cam.merge_with_defaults(default_margin=1.5, default_focal_length=120.0)

        assert merged.margin == 1.5
        assert merged.focal_length == 120.0

    def test_merge_with_defaults_keeps_specified(self):
        """merge_with_defaults keeps specified values."""
        cam = CameraSpec(direction="+x+y+z", margin=2.0, focal_length=None)

        merged = cam.merge_with_defaults(default_margin=1.5, default_focal_length=120.0)

        assert merged.margin == 2.0  # Kept
        assert merged.focal_length == 120.0  # Filled

    def test_to_camera_ordering_format(self):
        """to_camera_ordering_format returns direction string."""
        cam = CameraSpec(direction="+x+y+z", margin=1.0)
        assert cam.to_camera_ordering_format() == "+x+y+z"


class TestRealWorldScenarios:
    """Test real-world configuration scenarios."""

    def test_simready_agent_config(self):
        """Test SimReady agent-style configuration."""
        config = {
            "camera_defaults": {
                "focal_length": 100.0,
                "horizontal_aperture": 1.0,
                "vertical_aperture": 1.0,
            },
            "rendering_modes": {
                "prim_only": {"margin": 1.2, "cameras": ["+x+y+z", "-x-y-z"]},
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
                },
            },
        }

        result = parse_camera_configuration(config)

        # prim_only: 2 cameras with margin 1.2
        assert len(result["prim_only"]) == 2
        assert all(cam.margin == 1.2 for cam in result["prim_only"])

        # prim_with_stage: 8 cameras with margin 6.0
        assert len(result["prim_with_stage"]) == 8
        assert all(cam.margin == 6.0 for cam in result["prim_with_stage"])

        # All cameras have default focal length
        all_cameras = result["prim_only"] + result["prim_with_stage"]
        assert all(cam.focal_length == 100.0 for cam in all_cameras)

    def test_material_agent_config(self):
        """Test Material agent-style configuration."""
        config = {
            "camera_defaults": {"focal_length": 100.0},
            "rendering_modes": {
                "prim_with_stage": {"margin": 2.5, "cameras": ["+x+y+z", "-x-y-z"]},
                "prim_only": {
                    "margin": 1.05,
                    "cameras": ["+x+y+z", "-x+y+z", "-x-y+z", "+x-y+z"],
                },
                "linear_depth": ["+x+y+z"],  # Shorthand
                "instance_id_segmentation": {"use_cameras_from": "linear_depth"},
            },
        }

        result = parse_camera_configuration(config)

        assert len(result["prim_with_stage"]) == 2
        assert len(result["prim_only"]) == 4
        assert len(result["linear_depth"]) == 1
        assert len(result["instance_id_segmentation"]) == 1

        # Verify reference works
        assert result["instance_id_segmentation"] == result["linear_depth"]

    def test_backward_compatible_existing_config(self):
        """Existing configs work unchanged."""
        config = {
            "camera_view_type": "corner",
            "camera_directions": ["+x+y+z", "-x-y-z"],
            "camera_margin": 1.0,
            "camera_focal_length": 100.0,
            "camera_horizontal_aperture": 1.0,
            "camera_vertical_aperture": 1.0,
        }

        result = parse_camera_configuration(config)

        assert "__all__" in result
        assert len(result["__all__"]) == 2

        # Verify all fields preserved
        for cam in result["__all__"]:
            assert cam.margin == 1.0
            assert cam.focal_length == 100.0
            assert cam.horizontal_aperture == 1.0
            assert cam.vertical_aperture == 1.0
            assert cam.view_type == CameraViewType.CORNER
