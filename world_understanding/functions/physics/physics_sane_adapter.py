# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary physics_sane scaffold adapter.

This module is intentionally contract-neutral for issue #78 Lane 3. It wraps
the deterministic physics sanity inspection utilities and returns plain
JSON-friendly dictionaries that the pre-#45 scaffold can consume later without
depending on final Validation Agent models.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from world_understanding.functions.physics.physics_sanity import (
    PhysicsSanityFinding,
    PhysicsSanityResult,
    infer_physics_expected,
    inspect_usd_physics,
)

PHYSICS_SANE_TEMPLATE_NAME = "physics_sane"
TEMPLATE_ERROR_CODE = "agent.template_error"

ProvisionalVerdict = Literal["pass", "warn", "fail"]


def run_physics_sane_adapter(
    usd_path: str | Path,
    *,
    task_description: str | None = None,
    expect_physics: bool | None = None,
    single_asset: bool | None = None,
    policy: Mapping[str, Any] | None = None,
    asset_validator_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the provisional ``physics_sane`` template adapter.

    Args:
        usd_path: USD/USDZ input to inspect.
        task_description: Optional prompt text used for physics expectation
            inference when no explicit policy is supplied.
        expect_physics: Explicit physics expectation. Takes precedence over
            ``policy["expect_physics"]`` and prompt inference.
        single_asset: Optional multi-mesh policy. Takes precedence over
            ``policy["single_asset"]``.
        policy: Optional draft scaffold policy mapping.
        asset_validator_report: Optional USD validation report to pass
            through to the underlying inspector.

    Returns:
        A provisional template-result dictionary with stable issue payloads and
        summary metrics. The shape is deliberately simple so a future contract
        layer can copy or translate it.
    """

    normalized_usd_path = str(Path(usd_path))
    resolved_expect_physics = _resolve_policy_bool(
        "expect_physics",
        explicit_value=expect_physics,
        policy=policy,
    )
    resolved_single_asset = _resolve_policy_bool(
        "single_asset",
        explicit_value=single_asset,
        policy=policy,
    )
    physics_expected = infer_physics_expected(
        task_description,
        resolved_expect_physics,
    )

    try:
        inspection = inspect_usd_physics(
            normalized_usd_path,
            expect_physics=physics_expected,
            task_text=None,
            single_asset=resolved_single_asset,
            asset_validator_report=(
                dict(asset_validator_report)
                if asset_validator_report is not None
                else None
            ),
        )
    except Exception as exc:
        return _template_error_result(normalized_usd_path, physics_expected, exc)

    issues = [
        _finding_to_issue(finding, inspection.usd_path)
        for finding in inspection.findings
    ]
    verdict = _inspection_verdict(inspection)

    return {
        "template": PHYSICS_SANE_TEMPLATE_NAME,
        "status": "completed",
        "verdict": verdict,
        "passed": verdict != "fail",
        "issues": issues,
        "metrics": {
            **inspection.summary,
            "physics_expected": inspection.physics_expected,
            "opened": inspection.opened,
        },
        "evidence": {
            "usd_path": inspection.usd_path,
        },
    }


def _resolve_policy_bool(
    key: str,
    *,
    explicit_value: bool | None,
    policy: Mapping[str, Any] | None,
) -> bool | None:
    if policy is None or key not in policy:
        return explicit_value

    value = policy[key]
    if value is not None and not isinstance(value, bool):
        raise TypeError(
            f"policy.{key} must be a bool or None, got {type(value).__name__}"
        )
    if explicit_value is not None:
        return explicit_value
    return value


def _finding_to_issue(
    finding: PhysicsSanityFinding,
    usd_path: str,
) -> dict[str, Any]:
    return {
        "code": finding.code,
        "severity": finding.severity,
        "message": finding.message,
        "subject": finding.prim_path or usd_path,
        "prim_path": finding.prim_path,
        "details": dict(finding.details),
    }


def _inspection_verdict(result: PhysicsSanityResult) -> ProvisionalVerdict:
    if not result.opened or any(
        finding.severity == "fail" for finding in result.findings
    ):
        return "fail"
    if any(finding.severity == "warn" for finding in result.findings):
        return "warn"
    return "pass"


def _template_error_result(
    usd_path: str,
    physics_expected: bool,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "template": PHYSICS_SANE_TEMPLATE_NAME,
        "status": "error",
        "verdict": "fail",
        "passed": False,
        "issues": [
            {
                "code": TEMPLATE_ERROR_CODE,
                "severity": "fail",
                "message": f"physics_sane adapter failed: {exc}",
                "subject": usd_path,
                "prim_path": None,
                "details": {"error_type": type(exc).__name__},
            }
        ],
        "metrics": {
            "physics_expected": physics_expected,
            "opened": False,
        },
        "evidence": {
            "usd_path": usd_path,
        },
    }
