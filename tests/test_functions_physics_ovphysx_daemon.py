# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ovphysx daemon venv discovery and install hints."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from world_understanding.functions.physics import ovphysx_daemon as daemon_mod


def test_ovphysx_venv_python_path_uses_scripts_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    assert daemon_mod._ovphysx_venv_python_path(tmp_path) == (
        tmp_path / "Scripts" / "python.exe"
    )


def test_ovphysx_venv_python_path_uses_bin_on_posix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daemon_mod.os, "name", "posix")

    assert daemon_mod._ovphysx_venv_python_path(tmp_path) == (
        tmp_path / "bin" / "python"
    )


def test_resolve_python_accepts_windows_venv_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_path = tmp_path / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    assert daemon._resolve_python() == python_path


def test_missing_daemon_venv_hint_targets_platform_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    with pytest.raises(daemon_mod.OvPhysXDaemonUnavailableError) as exc_info:
        daemon._resolve_python()

    message = str(exc_info.value)
    assert "uv pip install --python" in message
    assert str(tmp_path / "Scripts" / "python.exe") in message
    assert "--extra-index-url https://pypi.nvidia.com" in message


def test_read_stdout_line_uses_threaded_pipe_reader_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PipeLikeStdout:
        def fileno(self) -> int:  # pragma: no cover - must not be used on Windows
            raise AssertionError("Windows pipe reader should not call fileno()")

        def readline(self) -> str:
            return '{"status": "ready"}\n'

    class _FakeProcess:
        stdout = _PipeLikeStdout()

    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = _FakeProcess()  # type: ignore[assignment]
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    assert daemon._read_stdout_line(1.0, "startup") == '{"status": "ready"}\n'


def test_threaded_stdout_reader_timeout_kills_process_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SlowStdout:
        def readline(self) -> str:
            time.sleep(0.2)
            return ""

    class _FakeProcess:
        stdout = _SlowStdout()
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            return 0

    process = _FakeProcess()
    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = process  # type: ignore[assignment]
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="startup timed out"):
        daemon._read_stdout_line(0.01, "startup")

    assert process.killed is True


def test_threaded_stdout_reader_wraps_readline_errors_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingStdout:
        def readline(self) -> str:
            raise RuntimeError("pipe broke")

    class _FakeProcess:
        stdout = _FailingStdout()

    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = _FakeProcess()  # type: ignore[assignment]
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="stdout read failed"):
        daemon._read_stdout_line(1.0, "evaluate")
