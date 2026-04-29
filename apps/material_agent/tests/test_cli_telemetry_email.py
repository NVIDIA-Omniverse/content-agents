# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI telemetry tagging via MA_USER_EMAIL."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer

import material_agent.cli as cli


class _SpanContextManager:
    """Simple context manager returning a mocked span."""

    def __init__(self, span: Mock):
        self._span = span

    def __enter__(self) -> Mock:
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _prepare_run_mocks(monkeypatch, tmp_path: Path, run_result):
    """Patch run() dependencies and return config path + run_pipeline mock."""
    config = tmp_path / "config.yaml"
    config.write_text("project:\n  name: test\n", encoding="utf-8")

    logger = Mock()
    listener = Mock()

    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: logger)
    monkeypatch.setattr(cli, "get_listener", lambda *args, **kwargs: listener)
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)

    import material_agent.api as api

    monkeypatch.setattr(api, "CLIEventListener", lambda **kwargs: object())

    class _PipelineInput:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    run_pipeline_mock = Mock(return_value=run_result)
    monkeypatch.setattr(api, "PipelineInput", _PipelineInput)
    monkeypatch.setattr(api, "run_pipeline", run_pipeline_mock)

    return config, run_pipeline_mock


def test_run_without_email_env_uses_default_behavior(monkeypatch, tmp_path):
    monkeypatch.delenv("MA_USER_EMAIL", raising=False)
    config, run_pipeline_mock = _prepare_run_mocks(
        monkeypatch,
        tmp_path,
        SimpleNamespace(success=True, step_results={}),
    )

    get_tracer_mock = Mock()
    monkeypatch.setattr(cli, "get_tracer", get_tracer_mock)

    cli.run(config=config)

    run_pipeline_mock.assert_called_once()
    get_tracer_mock.assert_not_called()


def test_run_with_email_env_sets_langfuse_attributes(monkeypatch, tmp_path):
    monkeypatch.setenv("MA_USER_EMAIL", "user@example.com")
    config, run_pipeline_mock = _prepare_run_mocks(
        monkeypatch,
        tmp_path,
        SimpleNamespace(success=True, step_results={}),
    )

    span = Mock()
    tracer = Mock()
    tracer.start_as_current_span.return_value = _SpanContextManager(span)
    monkeypatch.setattr(cli, "get_tracer", Mock(return_value=tracer))

    cli.run(config=config, session_id="session-123")

    run_pipeline_mock.assert_called_once()
    span.set_attribute.assert_any_call("maa.pipeline.user_email", "user@example.com")
    span.set_attribute.assert_any_call("langfuse.user.id", "user@example.com")
    span.set_attribute.assert_any_call("maa.pipeline.session_id", "session-123")
    span.set_attribute.assert_any_call("langfuse.session.id", "session-123")
    span.set_attribute.assert_any_call("maa.pipeline.status", "completed")


def test_run_with_email_env_generates_session_id_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MA_USER_EMAIL", "user@example.com")
    config, _ = _prepare_run_mocks(
        monkeypatch,
        tmp_path,
        SimpleNamespace(success=True, step_results={}),
    )

    span = Mock()
    tracer = Mock()
    tracer.start_as_current_span.return_value = _SpanContextManager(span)
    monkeypatch.setattr(cli, "get_tracer", Mock(return_value=tracer))
    monkeypatch.setattr(
        cli.uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )

    cli.run(config=config)

    expected_session_id = "12345678-1234-5678-1234-567812345678"
    span.set_attribute.assert_any_call("maa.pipeline.session_id", expected_session_id)
    span.set_attribute.assert_any_call("langfuse.session.id", expected_session_id)


def test_run_with_whitespace_email_env_uses_default_behavior(monkeypatch, tmp_path):
    monkeypatch.setenv("MA_USER_EMAIL", "   ")
    config, run_pipeline_mock = _prepare_run_mocks(
        monkeypatch,
        tmp_path,
        SimpleNamespace(success=True, step_results={}),
    )

    get_tracer_mock = Mock()
    monkeypatch.setattr(cli, "get_tracer", get_tracer_mock)

    cli.run(config=config)

    run_pipeline_mock.assert_called_once()
    get_tracer_mock.assert_not_called()


def test_run_sets_failed_status_when_pipeline_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("MA_USER_EMAIL", "user@example.com")
    config, run_pipeline_mock = _prepare_run_mocks(
        monkeypatch,
        tmp_path,
        SimpleNamespace(success=False, error="boom", step_results={}),
    )

    span = Mock()
    tracer = Mock()
    tracer.start_as_current_span.return_value = _SpanContextManager(span)
    monkeypatch.setattr(cli, "get_tracer", Mock(return_value=tracer))

    with pytest.raises(typer.Exit) as exc:
        cli.run(config=config)

    assert exc.value.exit_code == 1
    run_pipeline_mock.assert_called_once()
    span.set_attribute.assert_any_call("maa.pipeline.status", "failed")
