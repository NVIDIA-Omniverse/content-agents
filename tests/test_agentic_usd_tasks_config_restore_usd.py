# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RestoreUSDConfigTask YAML loading with Python-specific tags."""

import textwrap
from pathlib import Path

import pytest

from world_understanding.agentic.usd_tasks.config_restore_usd import (
    RestoreUSDConfigTask,
)


class TestRestoreUSDConfigTaskYAMLFallback:
    """Tests for the safe_load -> FullLoader fallback in RestoreUSDConfigTask."""

    def _write_config(self, tmp_path: Path, content: str) -> Path:
        config_path = tmp_path / "restore_config.yaml"
        config_path.write_text(textwrap.dedent(content))
        return config_path

    def test_loads_standard_yaml(self, tmp_path):
        """Standard YAML (no Python tags) loads via safe_load."""
        config_path = self._write_config(
            tmp_path,
            """\
            original_usd_path: /tmp/original.usd
            predictions_path: /tmp/predictions.jsonl
            output_predictions_path: /tmp/output.jsonl
            optimization_metadata:
              optimized: true
            """,
        )
        task = RestoreUSDConfigTask()
        context = {"config_path": str(config_path)}
        result = task.run(context, None)

        assert result["original_usd_path"] == "/tmp/original.usd"
        assert result["predictions_path"] == "/tmp/predictions.jsonl"
        assert result["optimization_metadata"] == {"optimized": True}

    def test_loads_yaml_with_python_name_tag(self, tmp_path):
        """YAML with !!python/name tag falls back to FullLoader."""
        config_path = tmp_path / "restore_config.yaml"
        config_path.write_text(
            "original_usd_path: /tmp/original.usd\n"
            "predictions_path: /tmp/predictions.jsonl\n"
            "output_predictions_path: /tmp/output.jsonl\n"
            "optimization_metadata:\n"
            "  none_val: !!python/none ''\n"
        )
        task = RestoreUSDConfigTask()
        context = {"config_path": str(config_path)}
        result = task.run(context, None)

        assert result["original_usd_path"] == "/tmp/original.usd"
        assert result["optimization_metadata"]["none_val"] is None

    def test_missing_config_path_raises(self):
        """Missing config_path raises ValueError."""
        task = RestoreUSDConfigTask()
        with pytest.raises(ValueError, match="config_path is required"):
            task.run({}, None)

    def test_empty_config_raises(self, tmp_path):
        """Empty config file raises ValueError."""
        config_path = self._write_config(tmp_path, "")
        task = RestoreUSDConfigTask()
        with pytest.raises(ValueError, match="Empty configuration"):
            task.run({"config_path": str(config_path)}, None)
