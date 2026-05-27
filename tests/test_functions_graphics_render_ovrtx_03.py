# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest


class _NoopFileLock:
    def __init__(self, path: str, timeout: float) -> None:
        self.path = path
        self.timeout = timeout

    def __enter__(self) -> "_NoopFileLock":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_runtime_requirements_file_is_used_for_dependency_pins() -> None:
    from world_understanding.functions.graphics import render_ovrtx

    requirements_file = render_ovrtx._OVRTX_RUNTIME_REQUIREMENTS_FILE
    assert requirements_file.exists()
    assert render_ovrtx._ovrtx_runtime_requirements_args() == [
        "-r",
        str(requirements_file),
    ]
    requirements = requirements_file.read_text(encoding="utf-8")
    assert "numpy==2.2.6" in requirements
    assert "pillow==12.2.0" in requirements


def test_bundled_python_runtime_libraries_are_removed_from_ovrtx_package(
    tmp_path: Path,
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
    plugins_dir = site_dir / "ovrtx" / "bin" / "plugins"
    plugins_dir.mkdir(parents=True)

    bundled_python_libraries = [
        plugins_dir / "libpython3.12.so",
        plugins_dir / "libpython3.12.so.1.0",
        site_dir / "ovrtx" / "lib" / "libpython3.12.so",
    ]
    for library in bundled_python_libraries:
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(b"python-runtime")
    keep_library = plugins_dir / "libovrtx.so"
    keep_library.write_bytes(b"ovrtx")

    removed = render_ovrtx._remove_ovrtx_bundled_python_libraries(venv_dir)

    assert removed == bundled_python_libraries
    assert all(not library.exists() for library in bundled_python_libraries)
    assert keep_library.exists()


def test_new_ovrtx_venv_removes_bundled_python_before_import_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    plugins_dir = (
        render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        / "ovrtx"
        / "bin"
        / "plugins"
    )
    bundled_python_library = plugins_dir / "libpython3.12.so.1.0"

    def fake_run_checked(cmd: list[str], label: str) -> None:
        if label == "uv venv creation":
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")
        if label == "uv pip install ovrtx":
            plugins_dir.mkdir(parents=True, exist_ok=True)
            bundled_python_library.write_bytes(b"python-runtime")

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        assert python_path_arg == python_path
        assert venv_dir_arg == venv_dir
        assert not bundled_python_library.exists()
        return render_ovrtx._OVRTX_VERSION

    monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert not bundled_python_library.exists()


def test_ovrtx_index_args_are_configurable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.delenv("WU_OVRTX_INDEX_URL", raising=False)
    assert render_ovrtx._ovrtx_index_args() == [
        "--index-url",
        render_ovrtx._OVRTX_INDEX_URL,
    ]

    monkeypatch.setenv("WU_OVRTX_INDEX_URL", "https://mirror.example/simple")
    assert render_ovrtx._ovrtx_index_args() == [
        "--index-url",
        "https://mirror.example/simple",
    ]

    monkeypatch.setenv("WU_OVRTX_INDEX_URL", "")
    with caplog.at_level(logging.WARNING):
        assert render_ovrtx._ovrtx_index_args() == []
    assert "pip/uv global index configuration" in caplog.text


def test_get_ovrtx_python_uses_cross_process_file_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    lock_calls: list[tuple[str, float]] = []

    class FakeFileLock:
        def __init__(self, path: str, timeout: float) -> None:
            lock_calls.append((path, timeout))

        def __enter__(self) -> "FakeFileLock":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(render_ovrtx, "FileLock", FakeFileLock)
    monkeypatch.setattr(
        render_ovrtx,
        "_probe_ovrtx_version",
        lambda python_path_arg, venv_dir_arg: render_ovrtx._OVRTX_VERSION,
    )

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert lock_calls == [
        (
            str(render_ovrtx._ovrtx_provision_lock_path(venv_dir)),
            render_ovrtx._OVRTX_PROVISION_LOCK_TIMEOUT_S,
        )
    ]


def test_provision_lock_path_honors_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("WU_OVRTX_LOCK_DIR", str(lock_dir))

    venv_dir = tmp_path / "cache" / "ovrtx_venv"
    lock_path = render_ovrtx._ovrtx_provision_lock_path(venv_dir)

    assert lock_path.parent == lock_dir
    assert lock_path.name.startswith("ovrtx_venv-")
    assert lock_path.name.endswith(".lock")


def test_auto_provision_disabled_preserves_existing_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    rmtree_calls: list[tuple[Path, bool]] = []

    monkeypatch.setattr(
        render_ovrtx,
        "_probe_ovrtx_version",
        lambda python_path_arg, venv_dir_arg: "0.2.0.280040",
    )
    monkeypatch.setattr(
        render_ovrtx.shutil,
        "rmtree",
        lambda path, ignore_errors=False: rmtree_calls.append(
            (Path(path), ignore_errors)
        ),
    )

    with pytest.raises(RuntimeError, match="AUTO_PROVISION is disabled"):
        render_ovrtx._get_ovrtx_python(venv_dir=venv_dir)
    assert rmtree_calls == []
    assert python_path.exists()


def test_provision_only_cli_calls_provisioner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(
        render_ovrtx,
        "_get_ovrtx_python",
        lambda venv_dir=None: "/tmp/ovrtx/bin/python",
    )

    assert render_ovrtx._main(["--provision-only"]) == 0
    assert capsys.readouterr().out == "OvRTX Python ready: /tmp/ovrtx/bin/python\n"


def test_daemon_start_clears_pythonpath_in_daemon_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setenv("PYTHONPATH", "/app/pythonpath")
    captured_env: dict[str, str] = {}

    class FakeProcess:
        pid = 12345
        stdin = None
        stdout = None
        stderr: list[str] = []

        def poll(self) -> None:
            return None

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        captured_env.update(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
    monkeypatch.setattr(render_ovrtx.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        render_ovrtx._OvRTXDaemon,
        "_read_stdout_line",
        lambda self, timeout_s, phase: json.dumps({"status": "ready"}),
    )

    daemon = render_ovrtx._OvRTXDaemon(
        ovrtx_python=str(tmp_path / "python"),
        daemon_script_path=str(tmp_path / "daemon.py"),
    )
    daemon.ensure_running()

    assert "PYTHONPATH" not in captured_env
    assert os.environ["PYTHONPATH"] == "/app/pythonpath"


def test_existing_wrong_version_venv_is_recreated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")

    versions = iter(["0.2.0.280040", render_ovrtx._OVRTX_VERSION])
    probe_calls: list[tuple[Path, Path]] = []

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        probe_calls.append((python_path_arg, venv_dir_arg))
        return next(versions)

    rmtree_calls: list[tuple[Path, bool]] = []

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        rmtree_calls.append((Path(path), ignore_errors))
        python_path.unlink(missing_ok=True)

    run_checked_calls: list[tuple[list[str], str]] = []

    def fake_run_checked(cmd: list[str], label: str) -> None:
        run_checked_calls.append((cmd, label))
        if label == "uv venv creation":
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")

    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
    monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert probe_calls == [(python_path, venv_dir), (python_path, venv_dir)]
    assert rmtree_calls == [(venv_dir, True)]
    assert [label for _, label in run_checked_calls] == [
        "uv venv creation",
        "uv pip install ovrtx",
        "uv pip install ovrtx runtime deps",
    ]


def test_managed_marker_without_version_does_not_skip_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n",
        encoding="utf-8",
    )

    versions = iter(["0.2.0.280040", render_ovrtx._OVRTX_VERSION])
    probe_calls: list[tuple[Path, Path]] = []

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        probe_calls.append((python_path_arg, venv_dir_arg))
        return next(versions)

    rmtree_calls: list[tuple[Path, bool]] = []

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        rmtree_calls.append((Path(path), ignore_errors))
        python_path.unlink(missing_ok=True)

    def fake_run_checked(cmd: list[str], label: str) -> None:
        if label == "uv venv creation":
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")

    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
    monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert probe_calls == [(python_path, venv_dir), (python_path, venv_dir)]
    assert rmtree_calls == [(venv_dir, True)]
    marker = (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).read_text(encoding="utf-8")
    assert f"ovrtx_version={render_ovrtx._OVRTX_VERSION}" in marker


def test_matching_legacy_managed_runtime_survives_marker_backfill_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n",
        encoding="utf-8",
    )

    probe_calls: list[tuple[Path, Path]] = []

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        probe_calls.append((python_path_arg, venv_dir_arg))
        return render_ovrtx._OVRTX_VERSION

    def fake_write_marker(venv_dir_arg: Path) -> None:
        assert venv_dir_arg == venv_dir
        raise PermissionError("read-only runtime")

    rmtree_calls: list[tuple[Path, bool]] = []

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        rmtree_calls.append((Path(path), ignore_errors))

    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
    monkeypatch.setattr(render_ovrtx, "_write_ovrtx_managed_marker", fake_write_marker)
    monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert probe_calls == [(python_path, venv_dir)]
    assert rmtree_calls == []


def test_managed_marker_with_current_version_uses_fast_path(tmp_path: Path) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n",
        encoding="utf-8",
    )

    assert render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)


def test_managed_marker_version_parser_is_line_based(tmp_path: Path) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    marker_path = tmp_path / render_ovrtx._OVRTX_MANAGED_MARKER
    marker_path.write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"note=ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n",
        encoding="utf-8",
    )

    assert (
        render_ovrtx._read_ovrtx_managed_marker_version(marker_path)
        == render_ovrtx._OVRTX_VERSION
    )


def test_unreadable_managed_marker_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    marker_path = venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER
    marker_path.write_text(
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n",
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == marker_path:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert not render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)


def test_marker_version_mismatch_skips_bundled_library_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "ovrtx_version=0.0.0\n",
        encoding="utf-8",
    )

    def fail_if_scanned(unused_venv_dir: Path) -> list[Path]:
        raise AssertionError("bundled libraries should not be scanned")

    monkeypatch.setattr(
        render_ovrtx,
        "_ovrtx_bundled_python_libraries",
        fail_if_scanned,
    )

    assert not render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)
