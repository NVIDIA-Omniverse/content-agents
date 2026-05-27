# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility mappers from the issue #78 scaffold to stable V1 contracts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from world_understanding.validation.models import (
    IssueSeverity,
    TemplateStatus,
    ValidationFocusConfig,
    ValidationInputGroups,
    ValidationIssue,
    ValidationPlan,
    ValidationPlanStep,
    ValidationProject,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateResult,
    ValidationVerdict,
)
from world_understanding.validation.templates import ValidationContractError


def validation_request_from_scaffold_request(request: Any) -> ValidationRequest:
    """Map a ``DraftValidationRequest``-like object into ``ValidationRequest``.

    The scaffold has runner-only fields such as ``base_dir`` and ``dry_run``.
    They are intentionally not part of the stable request contract; callers
    that need to rerun from this mapped request should resolve inputs first.
    """

    return ValidationRequest(
        task_description=str(request.task_description),
        inputs=tuple(str(input_path) for input_path in request.inputs),
        project=ValidationProject(working_dir=str(request.working_dir)),
        focus=ValidationFocusConfig(prim_paths=tuple(request.focus_prim_paths)),
        requested_templates=tuple(str(name) for name in request.requested_templates),
        policy=_mapping_attr(request, "policy"),
        metadata=_mapping_attr(request, "metadata"),
    )


def validation_plan_from_scaffold_plan(plan: Any) -> ValidationPlan:
    """Map a ``DraftValidationPlan``-like object into ``ValidationPlan``."""

    inventory = plan.input_inventory
    inventory_data = inventory.to_dict()
    return ValidationPlan(
        steps=tuple(
            ValidationPlanStep(
                template_name=str(step.template_name),
                reason=str(step.reason),
                inputs_needed=tuple(str(item) for item in step.inputs_needed),
            )
            for step in plan.steps
        ),
        input_groups=ValidationInputGroups.from_inventory_dict(inventory_data),
        focus_prim_paths=tuple(str(path) for path in inventory.focus_prim_paths),
        reasoning_summary=str(plan.reasoning_summary),
        artifact_paths=_artifact_paths_from_inventory(inventory_data),
    )


def validation_issue_from_scaffold_issue(
    issue: Any,
    *,
    template_name: str | None = None,
) -> ValidationIssue:
    """Map a scaffold issue into a stable issue contract."""

    return ValidationIssue(
        code=str(issue.code),
        severity=_issue_severity(issue.severity),
        message=str(issue.message),
        template_name=template_name,
        subject=str(issue.subject) if issue.subject is not None else None,
        details=_mapping_attr(issue, "details"),
    )


def validation_template_result_from_scaffold_result(
    result: Any,
) -> ValidationTemplateResult:
    """Map a scaffold template result into a stable template result contract."""

    template_name = str(result.template_name)
    return ValidationTemplateResult(
        template_name=template_name,
        status=_template_status(result.status),
        issues=tuple(
            validation_issue_from_scaffold_issue(issue, template_name=template_name)
            for issue in result.issues
        ),
        metrics=_mapping_attr(result, "metrics"),
        evidence=_mapping_attr(result, "evidence"),
        metadata=_mapping_attr(result, "metadata"),
    )


def validation_result_from_scaffold_result(result: Any) -> ValidationResult:
    """Map a scaffold validation result into the stable V1 report contract."""

    metadata = _mapping_attr(result, "metadata")
    artifact_paths = metadata.pop("artifact_paths", None)
    if not isinstance(artifact_paths, Mapping):
        artifact_paths = {}
    return ValidationResult(
        verdict=_validation_verdict(result.verdict),
        request=validation_request_from_scaffold_request(result.request),
        plan=validation_plan_from_scaffold_plan(result.plan),
        template_results=tuple(
            validation_template_result_from_scaffold_result(template_result)
            for template_result in result.template_results
        ),
        issues=tuple(
            validation_issue_from_scaffold_issue(issue) for issue in result.issues
        ),
        metrics=_mapping_attr(result, "metrics"),
        evidence=_mapping_attr(result, "evidence"),
        artifact_paths={str(key): str(value) for key, value in artifact_paths.items()},
        metadata=metadata,
    )


def _artifact_paths_from_inventory(inventory: Mapping[str, Any]) -> dict[str, str]:
    working_dir = inventory.get("working_dir")
    if not working_dir:
        return {}
    return {
        "plan": str(Path(str(working_dir)) / "plan.json"),
        "validation_result": str(Path(str(working_dir)) / "validation_result.json"),
    }


def _mapping_attr(obj: Any, name: str) -> dict[str, Any]:
    value = getattr(obj, name, None)
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _issue_severity(value: Any) -> IssueSeverity:
    if value == "info":
        return "info"
    if value in {"warn", "warning"}:
        return "warn"
    if value == "fail":
        return "fail"
    raise ValidationContractError(f"Unknown scaffold issue severity: {value!r}")


def _template_status(value: Any) -> TemplateStatus:
    if value in {"passed", "warn", "failed", "needs_refinement", "skipped", "error"}:
        return cast(TemplateStatus, value)
    raise ValidationContractError(f"Unknown scaffold template status: {value!r}")


def _validation_verdict(value: Any) -> ValidationVerdict:
    if value in {"pass", "warn", "fail", "needs_refinement", "planned"}:
        return cast(ValidationVerdict, value)
    raise ValidationContractError(f"Unknown scaffold validation verdict: {value!r}")
