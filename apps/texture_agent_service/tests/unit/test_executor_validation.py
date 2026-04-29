# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for executor terminal-state validation."""

from ...service.workers.executor import _get_step_validation_error


class TestGetStepValidationError:
    """Tests for executor validation of false-positive success states."""

    def test_discover_materials_fails_when_downstream_steps_need_materials(self):
        error = _get_step_validation_error(
            "discover_materials",
            {"materials_found": 0},
            [
                "prepare_uvs",
                "discover_materials",
                "generate_prompts",
                "generate_textures",
                "apply_textures",
            ],
        )

        assert error is not None
        assert "No discoverable materials" in error

    def test_discover_materials_allows_zero_materials_for_discovery_only_runs(self):
        error = _get_step_validation_error(
            "discover_materials",
            {"materials_found": 0},
            ["discover_materials"],
        )

        assert error is None

    def test_apply_textures_fails_when_no_output_usd_is_written(self):
        error = _get_step_validation_error(
            "apply_textures",
            {"output_usd_count": 0},
            ["discover_materials", "generate_textures", "apply_textures"],
        )

        assert error is not None
        assert "no output USD files" in error

    def test_apply_textures_succeeds_when_output_usd_exists(self):
        error = _get_step_validation_error(
            "apply_textures",
            {"output_usd_count": 1},
            ["discover_materials", "generate_textures", "apply_textures"],
        )

        assert error is None
