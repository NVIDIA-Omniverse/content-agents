# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the OVRTX container entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker-entrypoint.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="OVRTX entrypoint tests exercise POSIX shell, /tmp, symlinks, and signals",
)


def _entrypoint_env(tmp_path: Path, display: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    return {
        "HOME": str(tmp_path),
        "OVRTX_XVFB_DISPLAY": display,
        "OVRTX_XVFB_STARTUP_CHECKS": "30",
        "OVRTX_XVFB_STARTUP_DELAY": "0.1",
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHON_BIN": sys.executable,
    }


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _unused_display_id() -> str:
    base = 40000 + (os.getpid() % 10000)
    for display in range(base, base + 100):
        lock_file = Path(f"/tmp/.X{display}-lock")
        socket_file = Path(f"/tmp/.X11-unix/X{display}")
        if not lock_file.exists() and not socket_file.exists():
            return str(display)
    raise RuntimeError("could not find an unused X display id for entrypoint test")


def test_rejects_non_numeric_display(tmp_path: Path) -> None:
    result = subprocess.run(
        ["sh", str(ENTRYPOINT), "true"],
        env=_entrypoint_env(tmp_path, "abc"),
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 1
    assert "OVRTX_XVFB_DISPLAY must be a numeric display id" in result.stderr


def test_rejects_zero_startup_checks(tmp_path: Path) -> None:
    env = _entrypoint_env(tmp_path, _unused_display_id())
    env["OVRTX_XVFB_STARTUP_CHECKS"] = "0"

    result = subprocess.run(
        ["sh", str(ENTRYPOINT), "true"],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 1
    assert "OVRTX_XVFB_STARTUP_CHECKS must be greater than zero" in result.stderr


def test_refuses_symlinked_display_artifact(tmp_path: Path) -> None:
    display = _unused_display_id()
    lock_file = Path(f"/tmp/.X{display}-lock")
    socket_file = Path(f"/tmp/.X11-unix/X{display}")
    target = tmp_path / "lock-target"
    target.write_text("owned by test", encoding="utf-8")
    socket_target = tmp_path / "socket-target"
    socket_target.write_text("owned by test", encoding="utf-8")

    try:
        socket_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.symlink_to(target)
        socket_file.symlink_to(socket_target)
        result = subprocess.run(
            ["sh", str(ENTRYPOINT), "true"],
            env=_entrypoint_env(tmp_path, display),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    finally:
        if lock_file.exists() or lock_file.is_symlink():
            lock_file.unlink()
        if socket_file.exists() or socket_file.is_symlink():
            socket_file.unlink()

    assert result.returncode == 1
    assert (
        f"Refusing to remove symlinked Xvfb display artifact: {lock_file}"
        in result.stderr
    )
    assert (
        f"Refusing to remove symlinked Xvfb display artifact: {socket_file}"
        in result.stderr
    )


def test_xvfb_startup_failure_exits_before_service(tmp_path: Path) -> None:
    display = _unused_display_id()
    env = _entrypoint_env(tmp_path, display)
    marker = tmp_path / "service-started"
    env["SERVICE_MARKER"] = str(marker)

    _write_executable(
        tmp_path / "bin" / "Xvfb",
        "#!/bin/sh\nexit 42\n",
    )

    result = subprocess.run(
        ["sh", str(ENTRYPOINT), "sh", "-c", 'touch "$SERVICE_MARKER"'],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 1
    assert f"Xvfb failed to start on display :{display} (exit 42)" in result.stderr
    assert not marker.exists()


def test_waits_for_xvfb_socket_before_service(tmp_path: Path) -> None:
    display = _unused_display_id()
    env = _entrypoint_env(tmp_path, display)
    marker = tmp_path / "service-started"
    pid_file = tmp_path / "xvfb.pid"
    env["SERVICE_MARKER"] = str(marker)
    env["XVFB_PID_FILE"] = str(pid_file)

    _write_executable(
        tmp_path / "bin" / "Xvfb",
        """#!/bin/sh
exec >/dev/null 2>&1
display_num="${1#:}"
socket_file="/tmp/.X11-unix/X${display_num}"
echo "$$" > "$XVFB_PID_FILE"
sleep 0.2
mkdir -p /tmp/.X11-unix
exec "$PYTHON_BIN" - "$socket_file" <<'PY'
import socket
import sys
import time

sock = socket.socket(socket.AF_UNIX)
sock.bind(sys.argv[1])
sock.listen(1)
try:
    time.sleep(30)
finally:
    sock.close()
PY
""",
    )

    try:
        result = subprocess.run(
            ["sh", str(ENTRYPOINT), "sh", "-c", 'touch "$SERVICE_MARKER"'],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    finally:
        socket_file = Path(f"/tmp/.X11-unix/X{display}")
        if pid_file.exists():
            subprocess.run(
                ["kill", pid_file.read_text(encoding="utf-8").strip()], check=False
            )
        if socket_file.exists():
            socket_file.unlink()

    assert result.returncode == 0
    assert marker.exists()


def test_exits_if_xvfb_exits_after_service_start(tmp_path: Path) -> None:
    display = _unused_display_id()
    env = _entrypoint_env(tmp_path, display)
    env["OVRTX_SUPERVISOR_POLL_DELAY"] = "0.05"
    marker = tmp_path / "service-started"
    terminated_marker = tmp_path / "service-terminated"
    env["SERVICE_MARKER"] = str(marker)
    env["SERVICE_TERMINATED_MARKER"] = str(terminated_marker)

    _write_executable(
        tmp_path / "bin" / "Xvfb",
        """#!/bin/sh
exec >/dev/null 2>&1
display_num="${1#:}"
socket_file="/tmp/.X11-unix/X${display_num}"
mkdir -p /tmp/.X11-unix
exec "$PYTHON_BIN" - "$socket_file" <<'PY'
import os
import socket
import sys
import time

socket_path = sys.argv[1]
sock = socket.socket(socket.AF_UNIX)
sock.bind(socket_path)
sock.listen(1)
try:
    time.sleep(0.2)
finally:
    sock.close()
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
PY
""",
    )

    result = subprocess.run(
        [
            "sh",
            str(ENTRYPOINT),
            "sh",
            "-c",
            (
                "trap 'touch \"$SERVICE_TERMINATED_MARKER\"; exit 0' TERM; "
                'touch "$SERVICE_MARKER"; '
                "while :; do sleep 1; done"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 1
    assert marker.exists()
    assert terminated_marker.exists()
    assert f"Xvfb exited after startup on display :{display}" in result.stderr


def test_xvfb_regular_socket_path_file_does_not_start_service(
    tmp_path: Path,
) -> None:
    display = _unused_display_id()
    env = _entrypoint_env(tmp_path, display)
    env["OVRTX_XVFB_STARTUP_CHECKS"] = "2"
    env["OVRTX_XVFB_STARTUP_DELAY"] = "0.05"
    marker = tmp_path / "service-started"
    pid_file = tmp_path / "xvfb.pid"
    env["SERVICE_MARKER"] = str(marker)
    env["XVFB_PID_FILE"] = str(pid_file)

    _write_executable(
        tmp_path / "bin" / "Xvfb",
        """#!/bin/sh
display_num="${1#:}"
socket_file="/tmp/.X11-unix/X${display_num}"
echo "$$" > "$XVFB_PID_FILE"
mkdir -p /tmp/.X11-unix
touch "$socket_file"
exec sleep 30
""",
    )

    try:
        result = subprocess.run(
            ["sh", str(ENTRYPOINT), "sh", "-c", 'touch "$SERVICE_MARKER"'],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    finally:
        socket_file = Path(f"/tmp/.X11-unix/X{display}")
        if pid_file.exists():
            subprocess.run(
                ["kill", pid_file.read_text(encoding="utf-8").strip()], check=False
            )
        if socket_file.exists():
            socket_file.unlink()

    assert result.returncode == 1
    assert "Xvfb did not create display socket" in result.stderr
    assert not marker.exists()
