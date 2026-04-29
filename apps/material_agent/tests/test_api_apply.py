# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Apply API."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from material_agent.api.apply import ApplyInput, ApplyOutput, run_apply
from material_agent.api.types import AssignmentStats, DownloadStats


class TestApplyInput:
    """Tests for ApplyInput validation."""

    def test_apply_input_valid(self, tmp_path):
        """Test creating valid ApplyInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = ApplyInput(config=config_file, render_enabled=True)

        assert params.config == config_file
        assert params.render_enabled is True
        assert params.layer_only is False

    def test_apply_input_missing_config(self, tmp_path):
        """Test ApplyInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ApplyInput(config=config_file)

    def test_apply_input_with_overrides(self, tmp_path):
        """Test ApplyInput with all overrides."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        input_usd = tmp_path / "input.usd"
        predictions = tmp_path / "predictions.jsonl"
        output_usd = tmp_path / "output.usd"

        params = ApplyInput(
            config=config_file,
            input_usd_override=input_usd,
            predictions_override=predictions,
            output_usd_override=output_usd,
            layer_only=True,
            render_enabled=False,
            verbose=True,
        )

        assert params.input_usd_override == input_usd
        assert params.predictions_override == predictions
        assert params.output_usd_override == output_usd
        assert params.layer_only is True


class TestRunApply:
    """Tests for run_apply function."""

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_apply_success(self, mock_arun_pipeline, tmp_path):
        """Test successful apply execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock pipeline result
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=True,
            step_results={
                "apply": {
                    "output_usd_path": str(tmp_path / "output.usd"),
                    "unique_materials": ["steel", "rubber", "plastic"],
                    "matched_materials": {
                        "steel": [{"source_path": "/path/to/steel.mdl"}]
                    },
                    "resolved_materials": {"steel": "/local/steel.mdl"},
                    "materials_applied": {"steel": {"prims": ["prim1", "prim2"]}},
                    "assignment_stats": {
                        "materials_created": 3,
                        "materials_applied": 3,
                        "total_prims": 10,
                        "failed": 0,
                    },
                    "download_stats": {
                        "found_local": 2,
                        "downloaded": 1,
                        "failed": 0,
                        "skipped": 0,
                    },
                    "rendered_image_paths": [str(tmp_path / "render.png")],
                    "rendering_skipped": False,
                    "layer_only": False,
                }
            },
        )

        # Execute
        params = ApplyInput(config=config_file)
        result = run_apply(params)

        # Verify
        assert result.success is True
        assert result.output_usd_path == tmp_path / "output.usd"
        assert len(result.unique_materials) == 3
        assert result.assignment_stats.materials_created == 3
        assert result.download_stats.found_local == 2
        assert len(result.rendered_image_paths) == 1

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_apply_pipeline_failure(self, mock_arun_pipeline, tmp_path):
        """Test apply when pipeline fails."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock pipeline failure
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=False,
            error="Apply failed",
        )

        # Execute
        params = ApplyInput(config=config_file)
        result = run_apply(params)

        # Verify
        assert result.success is False
        assert result.error == "Apply failed"

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_apply_exception(self, mock_arun_pipeline, tmp_path):
        """Test apply when exception occurs."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock exception
        mock_arun_pipeline.side_effect = RuntimeError("Unexpected error")

        # Execute
        params = ApplyInput(config=config_file)
        result = run_apply(params)

        # Verify
        assert result.success is False
        assert "Unexpected error" in result.error


class TestApplyOutput:
    """Tests for ApplyOutput dataclass."""

    def test_apply_output_success(self, tmp_path):
        """Test creating successful ApplyOutput."""
        assignment_stats = AssignmentStats(materials_created=5, total_prims=20)
        download_stats = DownloadStats(found_local=3, downloaded=2)

        output = ApplyOutput(
            success=True,
            output_usd_path=tmp_path / "output.usd",
            unique_materials=["steel", "rubber"],
            assignment_stats=assignment_stats,
            download_stats=download_stats,
        )

        assert output.success is True
        assert output.output_usd_path == tmp_path / "output.usd"
        assert len(output.unique_materials) == 2
        assert output.assignment_stats.materials_created == 5

    def test_apply_output_error(self):
        """Test creating error ApplyOutput."""
        output = ApplyOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.output_usd_path is None
