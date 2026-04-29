# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Predict API."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from material_agent.api.predict import PredictInput, PredictOutput, run_predict


class TestPredictInput:
    """Tests for PredictInput validation."""

    def test_predict_input_valid(self, tmp_path):
        """Test creating valid PredictInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = PredictInput(config=config_file, resume=True, verbose=True)

        assert params.config == config_file
        assert params.resume is True
        assert params.verbose is True

    def test_predict_input_missing_config(self, tmp_path):
        """Test PredictInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            PredictInput(config=config_file)


class TestRunPredict:
    """Tests for run_predict function."""

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_predict_success(self, mock_arun_pipeline, tmp_path):
        """Test successful predict execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock pipeline result
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=True,
            step_results={
                "predict": {
                    "predictions_path": str(tmp_path / "predictions.jsonl"),
                    "report_path": str(tmp_path / "report.html"),
                    "num_predictions": 25,
                }
            },
        )

        # Execute
        params = PredictInput(config=config_file)
        result = run_predict(params)

        # Verify
        assert result.success is True
        assert result.predictions_path == tmp_path / "predictions.jsonl"
        assert result.report_path == tmp_path / "report.html"
        assert result.num_predictions == 25

        # Verify pipeline was called with only=predict
        call_args = mock_arun_pipeline.call_args[0][0]
        assert call_args.only_steps == ["predict"]

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_predict_pipeline_failure(self, mock_arun_pipeline, tmp_path):
        """Test predict when pipeline fails."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock pipeline failure
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=False,
            error="Pipeline failed",
        )

        # Execute
        params = PredictInput(config=config_file)
        result = run_predict(params)

        # Verify
        assert result.success is False
        assert result.error == "Pipeline failed"

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_predict_exception(self, mock_arun_pipeline, tmp_path):
        """Test predict when exception occurs."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock exception
        mock_arun_pipeline.side_effect = RuntimeError("Unexpected error")

        # Execute
        params = PredictInput(config=config_file)
        result = run_predict(params)

        # Verify
        assert result.success is False
        assert "Unexpected error" in result.error


class TestPredictOutput:
    """Tests for PredictOutput dataclass."""

    def test_predict_output_success(self, tmp_path):
        """Test creating successful PredictOutput."""
        output = PredictOutput(
            success=True,
            predictions_path=tmp_path / "predictions.jsonl",
            report_path=tmp_path / "report.html",
            num_predictions=50,
        )

        assert output.success is True
        assert output.predictions_path == tmp_path / "predictions.jsonl"
        assert output.num_predictions == 50

    def test_predict_output_error(self):
        """Test creating error PredictOutput."""
        output = PredictOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.predictions_path is None
