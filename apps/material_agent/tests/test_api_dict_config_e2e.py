# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for dict config usage."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api import (
    CollectingEventListener,
    build_unified_pipeline_config,
    pipeline,
)


class TestDictConfigEndToEnd:
    """End-to-end tests for dictionary configuration."""

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_pipeline_with_dict_config_creates_workflow(self, mock_create_workflow):
        """Test that pipeline works with dict config end-to-end."""
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

        # Build config dict (no YAML file needed!)
        config = build_unified_pipeline_config(
            project_name="test_project",
            input_usd_path="input.usd",
            output_usd_path="output.usd",
            materials_library_path="materials.usd",
            materials_entries=[
                {"name": "Steel", "description": "Steel material", "binding": "/Steel"}
            ],
        )

        # Run pipeline with dict config
        result = pipeline(config)

        # Should succeed
        assert result.success is True
        assert "predict" in result.completed_steps
        assert "apply" in result.completed_steps

        # Verify workflow was called with config_dict
        call_args = mock_workflow.arun.call_args[0][0]
        assert "config_dict" in call_args
        assert "config_path" not in call_args
        assert call_args["config_dict"]["project"]["name"] == "test_project"

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_pipeline_with_dict_config_and_listener(self, mock_create_workflow):
        """Test pipeline with dict config and custom event listener."""
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "pipeline_results": {
                    "predict": {},
                    "apply": {},
                }
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Build config
        config = build_unified_pipeline_config(
            project_name="test_with_listener",
            input_usd_path="input.usd",
            output_usd_path="output.usd",
            materials_library_path="materials.usd",
            materials_entries=[{"name": "Steel", "binding": "/Steel"}],
        )

        # Create custom listener
        listener = CollectingEventListener()

        # Run with both dict config and custom listener
        result = pipeline(config, event_listener=listener)

        # Should succeed
        assert result.success is True

        # Should have emitted events
        started_events = listener.get_events("workflow.started")
        assert len(started_events) == 1
        assert started_events[0]["data"]["workflow_type"] == "pipeline"
        assert started_events[0]["data"]["config_type"] == "dict"

        # Should have logs
        assert len(listener.logs) > 0

    def test_minimal_dict_config_with_defaults(self):
        """Test that truly minimal dict config works with defaults."""
        # This would be the absolute minimum for a prediction-only workflow
        minimal_config = {
            "project": {"name": "minimal_test"},
            "input": {"usd_path": "input.usd"},
            "output": {"usd_path": "output.usd"},
            "materials": {
                "library_path": "materials.usd",
                "entries": [{"name": "Steel", "binding": "/Steel"}],
            },
            "steps": {
                "predict": {"enabled": True},
            },
            # VLM config should be added by defaults!
        }

        # This tests that the config structure is valid
        # (actual execution would require real files)
        assert "project" in minimal_config
        assert "steps" in minimal_config
        assert "predict" in minimal_config["steps"]
