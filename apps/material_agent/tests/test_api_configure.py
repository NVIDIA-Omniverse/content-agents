# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Configure API."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.configure import (
    ConfigureInput,
    ConfigureOutput,
    run_configure,
)


class TestConfigureInput:
    """Tests for ConfigureInput validation."""

    def test_configure_input_valid(self, tmp_path):
        """Test creating valid ConfigureInput."""
        output_config = tmp_path / "new_config.yaml"

        params = ConfigureInput(
            output_config_path=output_config,
            force=False,
            verbose=True,
        )

        assert params.output_config_path == output_config
        assert params.force is False
        assert params.verbose is True

    def test_configure_input_file_exists_without_force(self, tmp_path):
        """Test ConfigureInput raises error when file exists and force is False."""
        output_config = tmp_path / "existing.yaml"
        output_config.write_text("# existing")

        with pytest.raises(FileExistsError, match="Configuration file already exists"):
            ConfigureInput(output_config_path=output_config, force=False)

    def test_configure_input_file_exists_with_force(self, tmp_path):
        """Test ConfigureInput allows overwrite when force is True."""
        output_config = tmp_path / "existing.yaml"
        output_config.write_text("# existing")

        params = ConfigureInput(output_config_path=output_config, force=True)

        assert params.output_config_path == output_config
        assert params.force is True


class TestRunConfigure:
    """Tests for run_configure function."""

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_success(self, mock_create_workflow, tmp_path):
        """Test successful configuration creation."""
        # Setup
        output_config = tmp_path / "new_config.yaml"

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "config_created": True,
                "config_path": str(output_config),
                "pipeline_name": "test_pipeline",
                "input_usd_path": "/path/to/input.usd",
                "materials_library_path": "/path/to/materials",
                "output_usd_path": "/path/to/output.usd",
                "dataset_dir": "/path/to/dataset",
                "predictions_dir": "/path/to/predictions",
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = ConfigureInput(output_config_path=output_config)
        result = run_configure(params)

        # Verify
        assert result.success is True
        assert result.config_path == output_config
        assert result.pipeline_name == "test_pipeline"
        assert result.input_usd_path == "/path/to/input.usd"
        assert result.materials_library_path == "/path/to/materials"

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_not_created(self, mock_create_workflow, tmp_path):
        """Test configure when workflow doesn't create config."""
        # Setup
        output_config = tmp_path / "new_config.yaml"

        # Mock workflow that doesn't complete
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value={"config_created": False})
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = ConfigureInput(output_config_path=output_config)
        result = run_configure(params)

        # Verify
        assert result.success is False
        assert "did not complete" in result.error.lower()

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_file_exists_error(self, mock_create_workflow, tmp_path):
        """Test configure when file exists and force is False."""
        # Setup - create existing file
        output_config = tmp_path / "existing.yaml"
        output_config.write_text("# existing")

        # This should raise during input validation, not during run
        with pytest.raises(FileExistsError, match="Configuration file already exists"):
            ConfigureInput(output_config_path=output_config, force=False)

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_exception(self, mock_create_workflow, tmp_path):
        """Test configure when exception occurs."""
        # Setup
        output_config = tmp_path / "new_config.yaml"

        # Mock workflow that raises exception
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError("Configuration failed"))
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = ConfigureInput(output_config_path=output_config)
        result = run_configure(params)

        # Verify
        assert result.success is False
        assert "Configuration failed" in result.error


class TestConfigureOutput:
    """Tests for ConfigureOutput dataclass."""

    def test_configure_output_success(self, tmp_path):
        """Test creating successful ConfigureOutput."""
        output = ConfigureOutput(
            success=True,
            config_path=tmp_path / "config.yaml",
            pipeline_name="test_pipeline",
            input_usd_path="/path/to/input.usd",
            materials_library_path="/path/to/materials",
            output_usd_path="/path/to/output.usd",
            dataset_dir="/path/to/dataset",
            predictions_dir="/path/to/predictions",
        )

        assert output.success is True
        assert output.config_path == tmp_path / "config.yaml"
        assert output.pipeline_name == "test_pipeline"

    def test_configure_output_error(self):
        """Test creating error ConfigureOutput."""
        output = ConfigureOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.config_path is None
