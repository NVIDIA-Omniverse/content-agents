# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_probe_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ovrtx_inprocess_openusd_probe.py"
    )
    spec = importlib.util.spec_from_file_location("ovrtx_inprocess_probe", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_probe_repeats_both_import_orders(monkeypatch, tmp_path: Path) -> None:
    probe = _load_probe_module()
    calls: list[dict[str, Any]] = []

    def fake_run_snippet(
        python: Path,
        snippet: str,
        *,
        env: dict[str, str],
        timeout_s: float,
    ) -> dict[str, Any]:
        calls.append(
            {
                "python": python,
                "order": env["WU_OVRTX_INPROCESS_ORDER"],
                "run": env["WU_OVRTX_INPROCESS_RUN"],
                "timeout_s": timeout_s,
            }
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "launch_failed": False,
            "timeout_s": timeout_s,
            "stdout": "renderer_constructed=Renderer\n",
            "stderr_tail": "",
        }

    monkeypatch.setattr(probe, "_run_snippet", fake_run_snippet)
    monkeypatch.setattr(probe, "_gpu_summary", lambda: "gpu unavailable")
    monkeypatch.setattr(probe, "_git_rev", lambda repo_root, rev: "abc123")

    report, success = probe.run_probe(
        python=Path("python"),
        repo_root=tmp_path,
        runs=2,
        constructor_timeout_s=7.0,
        minimal_render=False,
        render_timeout_s=11.0,
    )

    assert success
    assert report["repo_head"] == "abc123"
    assert report["minimal_render_results"] == []
    assert report["phase_success"] == {
        "constructor": True,
        "minimal_render": None,
    }
    assert [(call["run"], call["order"]) for call in calls] == [
        ("1", "pxr_then_ovrtx"),
        ("1", "ovrtx_then_pxr"),
        ("2", "pxr_then_ovrtx"),
        ("2", "ovrtx_then_pxr"),
    ]
    assert {call["timeout_s"] for call in calls} == {7.0}


def test_run_probe_marks_timeout_as_failure(monkeypatch, tmp_path: Path) -> None:
    probe = _load_probe_module()

    def fake_run_snippet(
        python: Path,
        snippet: str,
        *,
        env: dict[str, str],
        timeout_s: float,
    ) -> dict[str, Any]:
        return {
            "returncode": None,
            "timed_out": True,
            "launch_failed": False,
            "timeout_s": timeout_s,
            "stdout": "pxr_import=ok\n",
            "stderr_tail": "",
        }

    monkeypatch.setattr(probe, "_run_snippet", fake_run_snippet)
    monkeypatch.setattr(probe, "_gpu_summary", lambda: "gpu unavailable")
    monkeypatch.setattr(probe, "_git_rev", lambda repo_root, rev: "abc123")

    report, success = probe.run_probe(
        python=Path("python"),
        repo_root=tmp_path,
        runs=1,
        constructor_timeout_s=7.0,
        minimal_render=False,
        render_timeout_s=11.0,
    )

    assert not success
    assert all(result["timed_out"] for result in report["constructor_results"])


def test_run_probe_marks_launch_failure_as_failure(monkeypatch, tmp_path: Path) -> None:
    probe = _load_probe_module()

    def fake_run_snippet(
        python: Path,
        snippet: str,
        *,
        env: dict[str, str],
        timeout_s: float,
    ) -> dict[str, Any]:
        return {
            "returncode": None,
            "timed_out": False,
            "launch_failed": True,
            "timeout_s": timeout_s,
            "stdout": "",
            "stderr_tail": "python not found",
        }

    monkeypatch.setattr(probe, "_run_snippet", fake_run_snippet)
    monkeypatch.setattr(probe, "_gpu_summary", lambda: "gpu unavailable")
    monkeypatch.setattr(probe, "_git_rev", lambda repo_root, rev: "abc123")

    report, success = probe.run_probe(
        python=Path("missing-python"),
        repo_root=tmp_path,
        runs=1,
        constructor_timeout_s=7.0,
        minimal_render=False,
        render_timeout_s=11.0,
    )

    assert not success
    assert all(result["launch_failed"] for result in report["constructor_results"])


def test_run_snippet_returns_structured_launch_failure(monkeypatch) -> None:
    probe = _load_probe_module()

    def fake_run(*args: Any, **kwargs: Any) -> None:
        raise OSError("cannot launch")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    result = probe._run_snippet(
        Path("missing-python"),
        "print('unused')",
        env={},
        timeout_s=3.0,
    )

    assert result == {
        "returncode": None,
        "timed_out": False,
        "launch_failed": True,
        "timeout_s": 3.0,
        "stdout": "",
        "stderr_tail": "cannot launch",
    }


def test_run_probe_can_include_minimal_render(monkeypatch, tmp_path: Path) -> None:
    probe = _load_probe_module()
    smoke_usd = tmp_path / "smoke.usda"
    smoke_usd.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_run_snippet(
        python: Path,
        snippet: str,
        *,
        env: dict[str, str],
        timeout_s: float,
    ) -> dict[str, Any]:
        if "WU_OVRTX_SMOKE_USD" in env:
            assert env["WU_OVRTX_SMOKE_USD"] == str(smoke_usd)
            assert env["WU_OVRTX_INPROCESS_ORDER"] in {
                "pxr_then_ovrtx",
                "ovrtx_then_pxr",
            }
            return {
                "returncode": 0,
                "timed_out": False,
                "launch_failed": False,
                "timeout_s": timeout_s,
                "stdout": (
                    "stage_open=True\n"
                    'minimal_render={"max_rgb": 255, "nonzero_rgb": 1}\n'
                ),
                "stderr_tail": "",
            }
        return {
            "returncode": 0,
            "timed_out": False,
            "launch_failed": False,
            "timeout_s": timeout_s,
            "stdout": "renderer_constructed=Renderer\n",
            "stderr_tail": "",
        }

    monkeypatch.setattr(probe, "_run_snippet", fake_run_snippet)
    monkeypatch.setattr(probe, "_gpu_summary", lambda: "gpu unavailable")
    monkeypatch.setattr(probe, "_git_rev", lambda repo_root, rev: "abc123")

    report, success = probe.run_probe(
        python=Path("python"),
        repo_root=tmp_path,
        runs=1,
        constructor_timeout_s=7.0,
        minimal_render=True,
        render_timeout_s=11.0,
        smoke_usd=smoke_usd,
        tmp_dir=tmp_path,
    )

    assert success
    assert report["phase_success"] == {
        "constructor": True,
        "minimal_render": True,
    }
    assert [result["order"] for result in report["minimal_render_results"]] == [
        "pxr_then_ovrtx",
        "ovrtx_then_pxr",
    ]
    assert all(
        result["case"] == "minimal_render" and result["smoke_usd"] == str(smoke_usd)
        for result in report["minimal_render_results"]
    )


def test_run_probe_rejects_minimal_render_when_stage_does_not_open(
    monkeypatch, tmp_path: Path
) -> None:
    probe = _load_probe_module()

    def fake_run_snippet(
        python: Path,
        snippet: str,
        *,
        env: dict[str, str],
        timeout_s: float,
    ) -> dict[str, Any]:
        if "WU_OVRTX_SMOKE_USD" in env:
            return {
                "returncode": 0,
                "timed_out": False,
                "launch_failed": False,
                "timeout_s": timeout_s,
                "stdout": (
                    "stage_open=False\n"
                    'minimal_render={"max_rgb": 255, "nonzero_rgb": 1}\n'
                ),
                "stderr_tail": "",
            }
        return {
            "returncode": 0,
            "timed_out": False,
            "launch_failed": False,
            "timeout_s": timeout_s,
            "stdout": "renderer_constructed=Renderer\n",
            "stderr_tail": "",
        }

    monkeypatch.setattr(probe, "_run_snippet", fake_run_snippet)
    monkeypatch.setattr(probe, "_gpu_summary", lambda: "gpu unavailable")
    monkeypatch.setattr(probe, "_git_rev", lambda repo_root, rev: "abc123")

    report, success = probe.run_probe(
        python=Path("python"),
        repo_root=tmp_path,
        runs=1,
        constructor_timeout_s=7.0,
        minimal_render=True,
        render_timeout_s=11.0,
    )

    assert not success
    assert report["phase_success"]["minimal_render"] is False
    assert all(
        result["validation_error"] == "smoke USD did not open"
        for result in report["minimal_render_results"]
    )


def test_run_probe_rejects_blank_minimal_render(monkeypatch, tmp_path: Path) -> None:
    probe = _load_probe_module()

    def fake_run_snippet(
        python: Path,
        snippet: str,
        *,
        env: dict[str, str],
        timeout_s: float,
    ) -> dict[str, Any]:
        if "WU_OVRTX_SMOKE_USD" in env:
            return {
                "returncode": 0,
                "timed_out": False,
                "launch_failed": False,
                "timeout_s": timeout_s,
                "stdout": (
                    'stage_open=True\nminimal_render={"max_rgb": 0, "nonzero_rgb": 0}\n'
                ),
                "stderr_tail": "",
            }
        return {
            "returncode": 0,
            "timed_out": False,
            "launch_failed": False,
            "timeout_s": timeout_s,
            "stdout": "renderer_constructed=Renderer\n",
            "stderr_tail": "",
        }

    monkeypatch.setattr(probe, "_run_snippet", fake_run_snippet)
    monkeypatch.setattr(probe, "_gpu_summary", lambda: "gpu unavailable")
    monkeypatch.setattr(probe, "_git_rev", lambda repo_root, rev: "abc123")

    report, success = probe.run_probe(
        python=Path("python"),
        repo_root=tmp_path,
        runs=1,
        constructor_timeout_s=7.0,
        minimal_render=True,
        render_timeout_s=11.0,
    )

    assert not success
    assert report["phase_success"]["minimal_render"] is False
    assert all(
        result["validation_error"] == "minimal render RGB output is blank"
        for result in report["minimal_render_results"]
    )
