# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest
from pytest import LogCaptureFixture
from typer.testing import CliRunner

from texture_agent.cli import app
from texture_agent.config import unified_config

runner = CliRunner()


def test_run_help_documents_resume_options() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--resume" in result.output
    assert "--session-id" in result.output


@pytest.mark.parametrize("option", ["--only", "--skip"])
def test_run_rejects_empty_step_filter_from_cli(
    option: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        unified_config,
        "load_config",
        lambda path, session_id=None: {},
    )
    monkeypatch.setattr(unified_config, "config_to_context", lambda config: {})

    result = runner.invoke(
        app,
        ["run", str(config_path), option, "", "--dry-run"],
    )

    assert result.exit_code == 1
    assert "empty step name" in caplog.text
