# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the temporary physics_sane scaffold adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from world_understanding.functions.physics import physics_sane_adapter as adapter
from world_understanding.functions.physics.physics_sanity import (
    PhysicsSanityFinding,
    PhysicsSanityResult,
)


def test_adapter_maps_clean_inspection_to_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_inspect_usd_physics(
        usd_path: str | Path,
        *,
        expect_physics: bool | None = None,
        task_text: str | None = None,
        single_asset: bool | None = None,
        asset_validator_report: dict[str, Any] | None = None,
    ) -> PhysicsSanityResult:
        calls.append(
            {
                "usd_path": usd_path,
                "expect_physics": expect_physics,
                "task_text": task_text,
                "single_asset": single_asset,
                "asset_validator_report": asset_validator_report,
            }
        )
        return PhysicsSanityResult(
            usd_path=str(usd_path),
            opened=True,
            physics_expected=bool(expect_physics),
            summary={
                "physics_scene_count": 1,
                "rigid_body_count": 1,
                "collider_count": 1,
            },
            findings=[],
        )

    monkeypatch.setattr(adapter, "inspect_usd_physics", fake_inspect_usd_physics)

    result = adapter.run_physics_sane_adapter(
        "asset.usda",
        task_description="Check collision, mass, and rigid body setup",
        policy={"single_asset": True},
        asset_validator_report={"status": "success", "issues": []},
    )

    assert calls == [
        {
            "usd_path": "asset.usda",
            "expect_physics": True,
            "task_text": None,
            "single_asset": True,
            "asset_validator_report": {"status": "success", "issues": []},
        }
    ]
    assert result == {
        "template": "physics_sane",
        "status": "completed",
        "verdict": "pass",
        "passed": True,
        "issues": [],
        "metrics": {
            "physics_expected": True,
            "opened": True,
            "physics_scene_count": 1,
            "rigid_body_count": 1,
            "collider_count": 1,
        },
        "evidence": {"usd_path": "asset.usda"},
    }


def test_adapter_maps_fail_and_warn_findings_to_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_inspect_usd_physics(
        usd_path: str | Path,
        **_: Any,
    ) -> PhysicsSanityResult:
        return PhysicsSanityResult(
            usd_path=str(usd_path),
            opened=True,
            physics_expected=True,
            summary={"rigid_body_count": 1},
            findings=[
                PhysicsSanityFinding(
                    code="physics.no_colliders",
                    severity="fail",
                    message="Rigid body has no collider.",
                    prim_path="/World/Body",
                    details={"rigid_body": "/World/Body"},
                ),
                PhysicsSanityFinding(
                    code="physics.mass_scale_suspicious",
                    severity="warn",
                    message="Mass scale looks high.",
                ),
            ],
        )

    monkeypatch.setattr(adapter, "inspect_usd_physics", fake_inspect_usd_physics)

    result = adapter.run_physics_sane_adapter("asset.usda", expect_physics=True)

    assert result["verdict"] == "fail"
    assert result["passed"] is False
    assert result["issues"] == [
        {
            "code": "physics.no_colliders",
            "severity": "fail",
            "message": "Rigid body has no collider.",
            "subject": "/World/Body",
            "prim_path": "/World/Body",
            "details": {"rigid_body": "/World/Body"},
        },
        {
            "code": "physics.mass_scale_suspicious",
            "severity": "warn",
            "message": "Mass scale looks high.",
            "subject": "asset.usda",
            "prim_path": None,
            "details": {},
        },
    ]


def test_adapter_maps_warning_only_inspection_to_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_inspect_usd_physics(
        usd_path: str | Path,
        **_: Any,
    ) -> PhysicsSanityResult:
        return PhysicsSanityResult(
            usd_path=str(usd_path),
            opened=True,
            physics_expected=True,
            findings=[
                PhysicsSanityFinding(
                    code="physics.asset_validator_unavailable",
                    severity="warn",
                    message="Validator unavailable.",
                )
            ],
        )

    monkeypatch.setattr(adapter, "inspect_usd_physics", fake_inspect_usd_physics)

    result = adapter.run_physics_sane_adapter("asset.usda", expect_physics=True)

    assert result["verdict"] == "warn"
    assert result["passed"] is True


def test_adapter_policy_expectation_takes_precedence_over_task_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_expectations: list[bool | None] = []

    def fake_inspect_usd_physics(
        usd_path: str | Path,
        *,
        expect_physics: bool | None = None,
        **_: Any,
    ) -> PhysicsSanityResult:
        captured_expectations.append(expect_physics)
        return PhysicsSanityResult(
            usd_path=str(usd_path),
            opened=True,
            physics_expected=bool(expect_physics),
        )

    monkeypatch.setattr(adapter, "inspect_usd_physics", fake_inspect_usd_physics)

    adapter.run_physics_sane_adapter(
        "asset.usda",
        task_description="Visual only, no physics checks",
        policy={"expect_physics": True},
    )
    adapter.run_physics_sane_adapter(
        "asset.usda",
        task_description="Check collision behavior",
        expect_physics=False,
        policy={"expect_physics": True},
    )

    assert captured_expectations == [True, False]


def test_adapter_explicit_single_asset_takes_precedence_over_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_single_asset: list[bool | None] = []

    def fake_inspect_usd_physics(
        usd_path: str | Path,
        *,
        single_asset: bool | None = None,
        **_: Any,
    ) -> PhysicsSanityResult:
        captured_single_asset.append(single_asset)
        return PhysicsSanityResult(
            usd_path=str(usd_path),
            opened=True,
            physics_expected=False,
        )

    monkeypatch.setattr(adapter, "inspect_usd_physics", fake_inspect_usd_physics)

    adapter.run_physics_sane_adapter(
        "asset.usda",
        single_asset=False,
        policy={"single_asset": True},
    )

    assert captured_single_asset == [False]


def test_adapter_accepts_none_policy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[bool | None, bool | None]] = []

    def fake_inspect_usd_physics(
        usd_path: str | Path,
        *,
        expect_physics: bool | None = None,
        single_asset: bool | None = None,
        **_: Any,
    ) -> PhysicsSanityResult:
        captured.append((expect_physics, single_asset))
        return PhysicsSanityResult(
            usd_path=str(usd_path),
            opened=True,
            physics_expected=bool(expect_physics),
        )

    monkeypatch.setattr(adapter, "inspect_usd_physics", fake_inspect_usd_physics)

    adapter.run_physics_sane_adapter(
        "asset.usda",
        policy={"expect_physics": None, "single_asset": None},
    )

    assert captured == [(False, None)]


def test_adapter_maps_unopened_inspection_to_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_inspect_usd_physics(
        usd_path: str | Path,
        **_: Any,
    ) -> PhysicsSanityResult:
        return PhysicsSanityResult(
            usd_path=str(usd_path),
            opened=False,
            physics_expected=True,
            findings=[
                PhysicsSanityFinding(
                    code="physics.usd_open_failed",
                    severity="fail",
                    message="Failed to open USD stage.",
                )
            ],
        )

    monkeypatch.setattr(adapter, "inspect_usd_physics", fake_inspect_usd_physics)

    result = adapter.run_physics_sane_adapter("asset.usda", expect_physics=True)

    assert result["verdict"] == "fail"
    assert result["passed"] is False
    assert result["issues"][0]["code"] == "physics.usd_open_failed"


def test_adapter_reports_structured_template_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_inspect_usd_physics(
        usd_path: str | Path,
        **_: Any,
    ) -> PhysicsSanityResult:
        raise RuntimeError(f"boom: {usd_path}")

    monkeypatch.setattr(adapter, "inspect_usd_physics", fake_inspect_usd_physics)

    result = adapter.run_physics_sane_adapter(
        "asset.usda",
        task_description="Check collision behavior",
    )

    assert result["template"] == "physics_sane"
    assert result["status"] == "error"
    assert result["verdict"] == "fail"
    assert result["passed"] is False
    assert result["metrics"] == {"physics_expected": True, "opened": False}
    assert result["issues"] == [
        {
            "code": "agent.template_error",
            "severity": "fail",
            "message": "physics_sane adapter failed: boom: asset.usda",
            "subject": "asset.usda",
            "prim_path": None,
            "details": {"error_type": "RuntimeError"},
        }
    ]


def test_adapter_rejects_non_bool_policy_values() -> None:
    with pytest.raises(TypeError, match="policy.expect_physics must be a bool"):
        adapter.run_physics_sane_adapter(
            "asset.usda",
            policy={"expect_physics": "yes"},
        )
    with pytest.raises(TypeError, match="policy.single_asset must be a bool"):
        adapter.run_physics_sane_adapter(
            "asset.usda",
            single_asset=False,
            policy={"single_asset": "yes"},
        )
