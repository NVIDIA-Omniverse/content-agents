# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest
from pytest import LogCaptureFixture
from typer.main import get_command
from typer.testing import CliRunner

from texture_agent.cli import app
from texture_agent.config import unified_config

runner = CliRunner()


def test_run_help_documents_resume_options() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0

    run_command = get_command(app).commands["run"]
    options_by_name = {param.name: param for param in run_command.params}

    resume_option = options_by_name["resume"]
    session_id_option = options_by_name["session_id"]

    assert "--resume" in resume_option.opts
    assert resume_option.help == "Reuse existing artifacts from the working directory"
    assert "--session-id" in session_id_option.opts
    assert session_id_option.help == "Reuse or override the config session ID"


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
