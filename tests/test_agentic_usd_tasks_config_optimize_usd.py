# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OptimizeUSDConfigTask backend validation logic."""

import textwrap
from pathlib import Path

import pytest

from world_understanding.agentic.usd_tasks.config_optimize_usd import (
    OptimizeUSDConfigTask,
)


def _write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "optimize_config.yaml"
    config_path.write_text(textwrap.dedent(content))
    return config_path


def _minimal_config(extra: str = "") -> str:
    return textwrap.dedent(
        f"""\
        input_usd_path: /tmp/input.usd
        output_usd_path: /tmp/output.usd
        optimization_config:
          backend: remote
        {extra}
        """
    )


class TestOptimizeUSDConfigTaskBackendValidation:
    """Tests for backend validation in OptimizeUSDConfigTask (lines 132-152)."""

    def test_invalid_backend_raises_value_error(self, tmp_path):
        """An unrecognised backend value must raise ValueError."""
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: cloud
            """,
        )
        task = OptimizeUSDConfigTask()
        with pytest.raises(ValueError, match="Invalid backend 'cloud'"):
            task.run({"config_path": str(config_path)}, None)

    def test_nvcf_backend_sets_poll_seconds_default(self, tmp_path):
        """backend='remote' without poll_seconds should default to 300."""
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: remote
            """,
        )
        task = OptimizeUSDConfigTask()
        result = task.run({"config_path": str(config_path)}, None)
        assert result["optimization_config"]["poll_seconds"] == 300

    def test_nvcf_backend_respects_explicit_poll_seconds(self, tmp_path):
        """backend='remote' with an explicit poll_seconds should keep the user value."""
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: remote
              poll_seconds: 60
            """,
        )
        task = OptimizeUSDConfigTask()
        result = task.run({"config_path": str(config_path)}, None)
        assert result["optimization_config"]["poll_seconds"] == 60

    def test_local_backend_skips_poll_seconds_default(self, tmp_path):
        """backend='local' must NOT inject poll_seconds into optimization_config."""
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: local
              scene_optimizer_settings:
                enable_deinstance: false
            """,
        )
        task = OptimizeUSDConfigTask()
        result = task.run({"config_path": str(config_path)}, None)
        assert "poll_seconds" not in result["optimization_config"]

    def test_local_backend_with_deinstance_no_warning(self, tmp_path, caplog):
        """backend='local' + enable_deinstance=True must not emit a warning (now supported)."""
        import logging

        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: local
              scene_optimizer_settings:
                enable_deinstance: true
            """,
        )
        task = OptimizeUSDConfigTask()
        with caplog.at_level(logging.WARNING):
            task.run({"config_path": str(config_path)}, None)

        deinstance_warnings = [
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING and "deinstance" in r.message
        ]
        assert deinstance_warnings == []
