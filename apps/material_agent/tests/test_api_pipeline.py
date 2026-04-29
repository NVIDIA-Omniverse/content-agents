# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Pipeline API."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.pipeline import (
    PipelineInput,
    PipelineOutput,
    _dry_run_pipeline,
    arun_pipeline,
    run_pipeline,
)


class TestPipelineInput:
    """Tests for PipelineInput validation."""

    def test_pipeline_input_valid(self, tmp_path):
        """Test creating valid PipelineInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = PipelineInput(
            config=config_file,
            skip_steps=["build_dataset_usd"],
            only_steps=[],
            resume=True,
            dry_run=False,
            clean=True,
            verbose=True,
        )

        assert params.config == config_file
        assert params.skip_steps == ["build_dataset_usd"]
        assert params.resume is True
        assert params.clean is True

    def test_pipeline_input_missing_config(self, tmp_path):
        """Test PipelineInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            PipelineInput(config=config_file)

    def test_pipeline_input_defaults(self, tmp_path):
        """Test PipelineInput with default values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = PipelineInput(config=config_file)

        assert params.skip_steps == []
        assert params.only_steps == []
        assert params.resume is False
        assert params.dry_run is False
        assert params.clean is False

    def test_pipeline_input_empty_dict(self):
        """Test PipelineInput rejects empty config dictionaries."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PipelineInput(config={})


class TestRunPipeline:
    """Tests for run_pipeline function."""

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_success(self, mock_create_workflow, tmp_path):
        """Test successful pipeline execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "pipeline_results": {
                    "predict": {"predictions_path": "/path/to/predictions.jsonl"},
                    "apply": {"output_usd_path": "/path/to/output.usd"},
                }
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(config=config_file)
        result = run_pipeline(params)

        # Verify
        assert result.success is True
        assert "predict" in result.step_results
        assert "apply" in result.step_results
        assert result.completed_steps == ["predict", "apply"]

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_with_skip_steps(self, mock_create_workflow, tmp_path):
        """Test pipeline with skip steps."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={"pipeline_results": {"predict": {}, "apply": {}}}
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(
            config=config_file,
            skip_steps=["build_dataset_usd", "build_dataset_pdf_vectorstore"],
        )
        result = run_pipeline(params)

        # Verify context passed to workflow
        call_args = mock_workflow.arun.call_args[0][0]
        assert call_args["skip_steps"] == [
            "build_dataset_usd",
            "build_dataset_pdf_vectorstore",
        ]
        assert result.success is True

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_with_only_steps(self, mock_create_workflow, tmp_path):
        """Test pipeline with only specific steps."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={"pipeline_results": {"predict": {}}}
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(
            config=config_file,
            only_steps=["predict"],
        )
        result = run_pipeline(params)

        # Verify context
        call_args = mock_workflow.arun.call_args[0][0]
        assert call_args["only_steps"] == ["predict"]
        assert result.success is True

    @patch("material_agent.api.pipeline._dry_run_pipeline")
    def test_run_pipeline_dry_run(self, mock_dry_run, tmp_path):
        """Test pipeline dry run."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "project:\n  name: test\nsteps:\n  predict:\n    enabled: true"
        )

        # Mock dry run
        mock_dry_run.return_value = PipelineOutput(
            success=True,
            completed_steps=["predict", "apply"],
            skipped_steps=["build_dataset_usd"],
        )

        # Execute
        params = PipelineInput(config=config_file, dry_run=True)
        result = run_pipeline(params)

        # Verify
        assert result.success is True
        assert result.completed_steps == ["predict", "apply"]
        assert result.skipped_steps == ["build_dataset_usd"]

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_no_results(self, mock_create_workflow, tmp_path):
        """Test pipeline when workflow returns None."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that returns None
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value=None)
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(config=config_file)
        result = run_pipeline(params)

        # Verify
        assert result.success is False
        assert "did not complete" in result.error.lower()

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_exception(self, mock_create_workflow, tmp_path):
        """Test pipeline when exception occurs."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that raises exception
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError("Pipeline failed"))
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(config=config_file)
        result = run_pipeline(params)

        # Verify
        assert result.success is False
        assert "Pipeline failed" in result.error

    @pytest.mark.asyncio
    async def test_arun_pipeline_with_dict_config_and_default_listener(
        self, monkeypatch
    ):
        """Test async pipeline uses default listener and dict config."""
        listener = Mock()
        workflow = Mock()
        workflow.arun = AsyncMock(return_value={"pipeline_results": {"predict": {}}})

        monkeypatch.setattr(
            "world_understanding.agentic.events.create_default_listener",
            lambda verbose=False: listener,
        )
        monkeypatch.setattr(
            "material_agent.workflows.create_unified_pipeline_workflow",
            lambda: workflow,
        )

        params = PipelineInput(
            config={"project": {"name": "demo"}},
            skip_steps=["build_dataset_usd"],
            only_steps=["predict"],
            resume=True,
            clean=True,
            verbose=True,
            session_id="session-1",
        )
        result = await arun_pipeline(params)

        assert result.success is True
        call_args = workflow.arun.call_args[0][0]
        assert call_args["config_dict"] == {"project": {"name": "demo"}}
        assert call_args["event_listener"] is listener
        assert call_args["session_id"] == "session-1"
        listener.info.assert_any_call("Using in-memory config dictionary")
        listener.info.assert_any_call("Resume mode enabled")
        listener.info.assert_any_call(
            "Clean mode enabled (will delete working dir and output files)"
        )
        listener.info.assert_any_call("Using provided session ID: session-1")

    @pytest.mark.asyncio
    async def test_arun_pipeline_simulate_mode_from_file_and_partial_failure(
        self, monkeypatch, tmp_path
    ):
        """Test simulate-mode patching and workflow partial failure handling."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("project:\n  name: demo\n", encoding="utf-8")

        listener = Mock()
        workflow = Mock()
        workflow.arun = AsyncMock(
            return_value={
                "error": "predict failed",
                "failed_task": "predict",
                "pipeline_results": {"build_dataset_usd": {"num_prims": 3}},
            }
        )

        monkeypatch.setattr(
            "material_agent.api.simulate_config.patch_config_for_simulate",
            lambda config: {"patched": True, **config},
        )
        monkeypatch.setattr(
            "material_agent.workflows.create_unified_pipeline_workflow",
            lambda: workflow,
        )

        params = PipelineInput(
            config=config_file, simulate=True, event_listener=listener
        )
        result = await arun_pipeline(params)

        assert result.success is False
        assert result.error == "predict failed"
        assert result.completed_steps == ["build_dataset_usd"]
        assert result.step_results == {"build_dataset_usd": {"num_prims": 3}}
        call_args = workflow.arun.call_args[0][0]
        assert call_args["config_dict"]["patched"] is True
        assert call_args["config_path"] == str(config_file)
        listener.info.assert_any_call("Simulate mode: all backends patched to 'mock'")
        listener.event.assert_any_call(
            "workflow.failed",
            {
                "workflow_type": "pipeline",
                "error": "predict failed",
                "failed_task": "predict",
            },
        )

    def test_dry_run_pipeline_filters_steps_for_unified_config(self):
        """Test actual dry-run helper for unified config rules."""
        params = PipelineInput(
            config={
                "project": {"name": "demo"},
                "steps": {
                    "build_dataset_usd": {"enabled": True},
                    "predict": {"temperature": 0.0},
                    "apply": {"enabled": False},
                },
            },
            skip_steps=["build_dataset_usd"],
            only_steps=["predict"],
            dry_run=True,
        )

        result = _dry_run_pipeline(params)

        assert result.success is True
        assert result.completed_steps == ["predict"]
        assert result.skipped_steps == ["build_dataset_usd"]

    def test_dry_run_pipeline_returns_error_for_invalid_yaml(self, tmp_path):
        """Test dry-run helper wraps YAML parsing errors."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("[", encoding="utf-8")

        result = _dry_run_pipeline(PipelineInput(config=config_file))

        assert result.success is False
        assert result.error


class TestPipelineOutput:
    """Tests for PipelineOutput dataclass."""

    def test_pipeline_output_success(self):
        """Test creating successful PipelineOutput."""
        output = PipelineOutput(
            success=True,
            step_results={
                "predict": {"predictions_path": "/path/to/pred.jsonl"},
                "apply": {"output_usd_path": "/path/to/output.usd"},
            },
            completed_steps=["predict", "apply"],
            skipped_steps=["build_dataset_usd"],
        )

        assert output.success is True
        assert len(output.step_results) == 2
        assert output.completed_steps == ["predict", "apply"]
        assert output.skipped_steps == ["build_dataset_usd"]

    def test_pipeline_output_error(self):
        """Test creating error PipelineOutput."""
        output = PipelineOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.step_results == {}
