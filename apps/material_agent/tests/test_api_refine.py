# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Refine API."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.refine import (
    IterationResult,
    RefineInput,
    RefineOutput,
    run_refine,
)


class TestRefineInput:
    """Tests for RefineInput validation."""

    def test_refine_input_valid(self, tmp_path):
        """Test creating valid RefineInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = RefineInput(
            config=config_file,
            max_iterations_override=5,
            verbose=True,
        )

        assert params.config == config_file
        assert params.max_iterations_override == 5
        assert params.verbose is True

    def test_refine_input_missing_config(self, tmp_path):
        """Test RefineInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            RefineInput(config=config_file)


class TestIterationResult:
    """Tests for IterationResult dataclass."""

    def test_iteration_result(self):
        """Test creating IterationResult."""
        result = IterationResult(
            iteration=1,
            judge_score=4.5,
            continue_iteration=True,
            materials_applied_count=10,
            prims_with_materials=25,
        )

        assert result.iteration == 1
        assert result.judge_score == 4.5
        assert result.continue_iteration is True
        assert result.materials_applied_count == 10


class TestRunRefine:
    """Tests for run_refine function."""

    @patch(
        "material_agent.workflows.factory.create_iterative_apply_workflow_from_config"
    )
    def test_run_refine_success(self, mock_create_workflow, tmp_path):
        """Test successful refine execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "iteration_count": 3,
                "iteration_results": [
                    {
                        "iteration": 1,
                        "judge_score": 3.5,
                        "continue_iteration": True,
                        "materials_applied_count": 8,
                        "prims_with_materials": 20,
                    },
                    {
                        "iteration": 2,
                        "judge_score": 4.0,
                        "continue_iteration": True,
                        "materials_applied_count": 10,
                        "prims_with_materials": 25,
                    },
                    {
                        "iteration": 3,
                        "judge_score": 4.5,
                        "continue_iteration": False,
                        "materials_applied_count": 12,
                        "prims_with_materials": 28,
                    },
                ],
                "final_iteration": {
                    "judge_score": 4.5,
                    "materials_applied_count": 12,
                },
                "termination_reason": "approved",
                "all_iteration_outputs": [
                    str(tmp_path / "iter1.usd"),
                    str(tmp_path / "iter2.usd"),
                    str(tmp_path / "iter3.usd"),
                ],
                "final_output_path": str(tmp_path / "final.usd"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = RefineInput(config=config_file)
        result = run_refine(params)

        # Verify
        assert result.success is True
        assert result.iteration_count == 3
        assert result.final_judge_score == 4.5
        assert result.termination_reason == "approved"
        assert len(result.iteration_results) == 3
        assert result.final_output_path == tmp_path / "final.usd"

        # Verify iteration results structure
        assert result.iteration_results[0].iteration == 1
        assert result.iteration_results[0].judge_score == 3.5
        assert result.iteration_results[2].continue_iteration is False

    @patch(
        "material_agent.workflows.factory.create_iterative_apply_workflow_from_config"
    )
    def test_run_refine_with_max_iterations_override(
        self, mock_create_workflow, tmp_path
    ):
        """Test refine with max iterations override."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "iteration_count": 1,
                "iteration_results": [],
                "final_iteration": {"judge_score": 3.0},
                "termination_reason": "max_iterations",
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = RefineInput(config=config_file, max_iterations_override=2)
        result = run_refine(params)

        # Verify context passed to workflow
        call_args = mock_workflow.arun.call_args[1]["initial_context"]
        assert call_args["max_iterations_override"] == 2
        assert result.iteration_count == 1

    @patch(
        "material_agent.workflows.factory.create_iterative_apply_workflow_from_config"
    )
    def test_run_refine_no_iterations(self, mock_create_workflow, tmp_path):
        """Test refine when no iterations completed."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that returns 0 iterations
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value={"iteration_count": 0})
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = RefineInput(config=config_file)
        result = run_refine(params)

        # Verify
        assert result.success is False
        assert "did not complete" in result.error.lower()

    @patch(
        "material_agent.workflows.factory.create_iterative_apply_workflow_from_config"
    )
    def test_run_refine_exception(self, mock_create_workflow, tmp_path):
        """Test refine when exception occurs."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that raises exception
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError("Refinement failed"))
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = RefineInput(config=config_file)
        result = run_refine(params)

        # Verify
        assert result.success is False
        assert "Refinement failed" in result.error


class TestRefineOutput:
    """Tests for RefineOutput dataclass."""

    def test_refine_output_success(self, tmp_path):
        """Test creating successful RefineOutput."""
        iteration_results = [
            IterationResult(iteration=1, judge_score=3.0, continue_iteration=True),
            IterationResult(iteration=2, judge_score=4.5, continue_iteration=False),
        ]

        output = RefineOutput(
            success=True,
            iteration_count=2,
            final_output_path=tmp_path / "final.usd",
            final_judge_score=4.5,
            termination_reason="approved",
            iteration_results=iteration_results,
            all_iteration_outputs=[tmp_path / "iter1.usd", tmp_path / "iter2.usd"],
        )

        assert output.success is True
        assert output.iteration_count == 2
        assert output.final_judge_score == 4.5
        assert len(output.iteration_results) == 2

    def test_refine_output_error(self):
        """Test creating error RefineOutput."""
        output = RefineOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.iteration_count == 0
