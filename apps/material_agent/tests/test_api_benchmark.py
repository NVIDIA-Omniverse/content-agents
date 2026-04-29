# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Benchmark API."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from material_agent.api.benchmark import (
    BenchmarkInput,
    BenchmarkOutput,
    run_benchmark,
)
from material_agent.api.defaults import DEFAULT_VLM_BACKEND, DEFAULT_VLM_MAX_WORKERS


class TestBenchmarkInput:
    """Tests for BenchmarkInput validation."""

    def test_benchmark_input_valid(self, tmp_path):
        """Test creating valid BenchmarkInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = BenchmarkInput(
            config=config_file,
            verbose=True,
        )

        assert params.config == config_file
        assert params.verbose is True
        assert params.resume is False

    def test_benchmark_input_missing_config(self, tmp_path):
        """Test BenchmarkInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            BenchmarkInput(config=config_file)

    def test_benchmark_input_with_overrides(self, tmp_path):
        """Test BenchmarkInput with path overrides."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        dataset_file = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "output"

        params = BenchmarkInput(
            config=config_file,
            dataset_override=dataset_file,
            output_dir_override=output_dir,
            resume=True,
            stream_predictions=False,
        )

        assert params.dataset_override == dataset_file
        assert params.output_dir_override == output_dir
        assert params.resume is True
        assert params.stream_predictions is False

    def test_benchmark_input_with_dict_config(self):
        """Test BenchmarkInput with dictionary config."""
        config_dict = {
            "model": {"service": "azure", "name": "gpt-4"},
            "dataset_path": "/path/to/dataset.jsonl",
        }

        params = BenchmarkInput(config=config_dict, verbose=True)

        assert params.config == config_dict
        assert params.verbose is True

    def test_benchmark_input_empty_dict(self):
        """Test BenchmarkInput raises error for empty dict config."""
        with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
            BenchmarkInput(config={})

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_with_dict_config(self, mock_create_workflow, tmp_path):
        """Test running benchmark with dictionary config."""
        # Setup - in-memory config
        config_dict = {
            "model": {
                "service": "azure",
                "name": "gpt-4",
                "deployment": "test-deployment",
            },
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "output_dir": str(tmp_path / "output"),
        }

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {
                    "functional_correctness_score": 4.0,
                    "success_rate": 80.0,
                    "total_cases": 50,
                },
                "evaluation_path": str(tmp_path / "evaluation.jsonl"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BenchmarkInput(config=config_dict, verbose=True)
        result = run_benchmark(params)

        # Verify
        assert result.success is True
        assert result.metrics.functional_correctness_score == 4.0

        # Verify config_dict was passed to workflow with defaults applied
        call_args = mock_workflow.arun.call_args[0][0]
        assert "config_dict" in call_args
        assert "config_path" not in call_args

        # Verify defaults were applied
        passed_config = call_args["config_dict"]
        # Original user values preserved
        assert passed_config["model"] == config_dict["model"]
        assert passed_config["dataset_path"] == config_dict["dataset_path"]
        # Defaults added
        assert "vlm" in passed_config
        assert passed_config["vlm"]["backend"] == DEFAULT_VLM_BACKEND
        assert "llm" in passed_config
        assert "judge" in passed_config
        assert passed_config["max_workers"] == DEFAULT_VLM_MAX_WORKERS


class TestRunBenchmark:
    """Tests for run_benchmark function."""

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_success(self, mock_create_workflow, tmp_path):
        """Test successful benchmark execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {
                    "functional_correctness_score": 4.5,
                    "success_rate": 90.0,
                    "exact_match_rate": 75.0,
                    "total_cases": 100,
                    "valid_cases": 95,
                    "successful_cases": 90,
                    "exact_matches": 75,
                    "failure_count": 5,
                    "score_distribution": {"5": 50, "4": 40},
                },
                "evaluation_path": str(tmp_path / "evaluation.jsonl"),
                "predictions_path": str(tmp_path / "predictions.jsonl"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BenchmarkInput(config=config_file)
        result = run_benchmark(params)

        # Verify
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.functional_correctness_score == 4.5
        assert result.metrics.success_rate == 90.0
        assert result.evaluation_path == tmp_path / "evaluation.jsonl"
        assert result.predictions_path == tmp_path / "predictions.jsonl"

        # Verify workflow was called correctly
        mock_create_workflow.assert_called_once()
        mock_workflow.arun.assert_called_once()

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_with_overrides(self, mock_create_workflow, tmp_path):
        """Test benchmark execution with overrides."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        dataset_file = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "output"

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {
                    "functional_correctness_score": 3.5,
                    "success_rate": 70.0,
                    "total_cases": 50,
                },
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BenchmarkInput(
            config=config_file,
            dataset_override=dataset_file,
            output_dir_override=output_dir,
            resume=True,
        )
        result = run_benchmark(params)

        # Verify context passed to workflow
        call_args = mock_workflow.arun.call_args[0][0]
        assert call_args["config_path"] == str(config_file)
        assert call_args["dataset_override"] == str(dataset_file)
        assert call_args["output_dir_override"] == str(output_dir)
        assert call_args["resume"] is True
        assert result.success is True

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_no_metrics(self, mock_create_workflow, tmp_path):
        """Test benchmark execution when workflow returns no metrics."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that returns no metrics
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value={})
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BenchmarkInput(config=config_file)
        result = run_benchmark(params)

        # Verify
        assert result.success is False
        assert result.error is not None
        assert "no metrics" in result.error.lower()

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_workflow_exception(self, mock_create_workflow, tmp_path):
        """Test benchmark execution when workflow raises exception."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that raises exception
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError("Workflow failed"))
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BenchmarkInput(config=config_file)
        result = run_benchmark(params)

        # Verify
        assert result.success is False
        assert "Workflow failed" in result.error


class TestBenchmarkOutput:
    """Tests for BenchmarkOutput dataclass."""

    def test_benchmark_output_success(self, tmp_path):
        """Test creating successful BenchmarkOutput."""
        from material_agent.api.types import MetricsResult

        metrics = MetricsResult(
            functional_correctness_score=4.5,
            success_rate=90.0,
        )

        output = BenchmarkOutput(
            success=True,
            metrics=metrics,
            evaluation_path=tmp_path / "eval.jsonl",
            predictions_path=tmp_path / "pred.jsonl",
        )

        assert output.success is True
        assert output.metrics == metrics
        assert output.evaluation_path == tmp_path / "eval.jsonl"
        assert output.error is None

    def test_benchmark_output_error(self):
        """Test creating error BenchmarkOutput."""
        output = BenchmarkOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.metrics is None
