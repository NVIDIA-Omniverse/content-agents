# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Evaluate API."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.evaluate import (
    EvaluateInput,
    EvaluateOutput,
    run_evaluate,
)


class TestEvaluateInput:
    """Tests for EvaluateInput validation."""

    def test_evaluate_input_valid(self, tmp_path):
        """Test creating valid EvaluateInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = EvaluateInput(config=config_file, verbose=True)

        assert params.config == config_file
        assert params.verbose is True
        assert params.predictions_override is None

    def test_evaluate_input_missing_config(self, tmp_path):
        """Test EvaluateInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            EvaluateInput(config=config_file)

    def test_evaluate_input_missing_predictions(self, tmp_path):
        """Test EvaluateInput raises error for missing predictions file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        predictions_file = tmp_path / "missing.jsonl"

        with pytest.raises(FileNotFoundError, match="Predictions file not found"):
            EvaluateInput(
                config=config_file,
                predictions_override=predictions_file,
            )

    def test_evaluate_input_with_predictions(self, tmp_path):
        """Test EvaluateInput with predictions override."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.write_text("{}")

        params = EvaluateInput(
            config=config_file,
            predictions_override=predictions_file,
        )

        assert params.predictions_override == predictions_file


class TestRunEvaluate:
    """Tests for run_evaluate function."""

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_success(self, mock_create_workflow, tmp_path):
        """Test successful evaluation execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "evaluation_complete": True,
                "metrics": {
                    "functional_correctness_score": 4.2,
                    "success_rate": 85.0,
                    "total_cases": 50,
                },
                "evaluation_path": str(tmp_path / "evaluation.jsonl"),
                "html_report_path": str(tmp_path / "report.html"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = EvaluateInput(config=config_file)
        result = run_evaluate(params)

        # Verify
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.functional_correctness_score == 4.2
        assert result.evaluation_path == tmp_path / "evaluation.jsonl"
        assert result.html_report_path == tmp_path / "report.html"

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_with_predictions_override(
        self, mock_create_workflow, tmp_path
    ):
        """Test evaluation with predictions override."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.write_text("{}")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "evaluation_complete": True,
                "metrics": {"functional_correctness_score": 3.5},
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = EvaluateInput(
            config=config_file,
            predictions_override=predictions_file,
        )
        result = run_evaluate(params)

        # Verify context passed to workflow
        call_args = mock_workflow.arun.call_args[1]["initial_context"]
        assert call_args["predictions_path"] == str(predictions_file)
        assert result.success is True

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_not_complete(self, mock_create_workflow, tmp_path):
        """Test evaluation when workflow doesn't complete."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that doesn't complete
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value={"evaluation_complete": False})
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = EvaluateInput(config=config_file)
        result = run_evaluate(params)

        # Verify
        assert result.success is False
        assert "did not complete" in result.error.lower()

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_exception(self, mock_create_workflow, tmp_path):
        """Test evaluation when workflow raises exception."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that raises exception
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=ValueError("Evaluation failed"))
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = EvaluateInput(config=config_file)
        result = run_evaluate(params)

        # Verify
        assert result.success is False
        assert "Evaluation failed" in result.error


class TestEvaluateOutput:
    """Tests for EvaluateOutput dataclass."""

    def test_evaluate_output_success(self, tmp_path):
        """Test creating successful EvaluateOutput."""
        from material_agent.api.types import MetricsResult

        metrics = MetricsResult(functional_correctness_score=4.0)

        output = EvaluateOutput(
            success=True,
            metrics=metrics,
            evaluation_path=tmp_path / "eval.jsonl",
            html_report_path=tmp_path / "report.html",
        )

        assert output.success is True
        assert output.metrics == metrics
        assert output.evaluation_path == tmp_path / "eval.jsonl"

    def test_evaluate_output_error(self):
        """Test creating error EvaluateOutput."""
        output = EvaluateOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.metrics is None
