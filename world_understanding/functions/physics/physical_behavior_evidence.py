# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract-neutral physical_behavior evidence helpers.

These helpers resolve and lightly validate motion/refine artifacts consumed by
the Validation Agent ``physical_behavior`` template. This module intentionally
does not render, sample video, run simulation, call VLMs, or import the stable
Validation Agent contract layer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

BehaviorEvidenceKind = Literal[
    "time_sampled_usd",
    "animation_usd",
    "video",
    "sampled_frame",
    "simulation_json",
    "trajectory_metrics",
]
BehaviorEvidenceSeverity = Literal["warn", "fail"]
BehaviorPlaceholderVerdict = Literal["warn", "fail"]
BehaviorPlaceholderStatus = Literal["skipped", "unavailable"]

EvidenceInput = str | Path | Mapping[str, Any]
EvidenceInputSpec = EvidenceInput | Sequence[EvidenceInput] | None

SUPPORTED_EVIDENCE_KINDS: tuple[BehaviorEvidenceKind, ...] = (
    "time_sampled_usd",
    "animation_usd",
    "video",
    "sampled_frame",
    "simulation_json",
    "trajectory_metrics",
)

USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
JSON_EXTENSIONS = frozenset({".json"})
JSONL_EXTENSIONS = frozenset({".jsonl"})

BEHAVIOR_EVIDENCE_MISSING = "physics.behavior_evidence_missing"
BEHAVIOR_EVIDENCE_UNSUPPORTED = "physics.behavior_evidence_unsupported"
BEHAVIOR_EVIDENCE_MALFORMED = "physics.behavior_evidence_malformed"
BEHAVIOR_JUDGE_UNAVAILABLE = "physics.behavior_judge_unavailable"


@dataclass(frozen=True)
class PhysicalBehaviorEvidence:
    """One physical-behavior evidence artifact after lightweight resolution."""

    original: str
    path: str
    kind: BehaviorEvidenceKind | None
    exists: bool
    required: bool
    role: str | None = None
    description: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class PhysicalBehaviorIssue:
    """Simple issue shape for later Validation Agent model mapping."""

    code: str
    severity: BehaviorEvidenceSeverity
    message: str
    subject: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class PhysicalBehaviorEvidenceResolution:
    """Resolved evidence artifacts and any blocking or warning issues."""

    evidence: tuple[PhysicalBehaviorEvidence, ...] = ()
    issues: tuple[PhysicalBehaviorIssue, ...] = ()
    supported_kinds: tuple[BehaviorEvidenceKind, ...] = SUPPORTED_EVIDENCE_KINDS
    behavior_evidence_required: bool = False

    @property
    def passed(self) -> bool:
        """Whether no fail-severity evidence issues were found."""

        return not any(issue.severity == "fail" for issue in self.issues)

    @property
    def available_evidence(self) -> tuple[PhysicalBehaviorEvidence, ...]:
        """Evidence artifacts that exist and have a supported kind."""

        return tuple(
            item for item in self.evidence if item.exists and item.kind is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "issues": [issue.to_dict() for issue in self.issues],
            "supported_kinds": list(self.supported_kinds),
            "behavior_evidence_required": self.behavior_evidence_required,
            "passed": self.passed,
        }


def resolve_physical_behavior_evidence(
    evidence_inputs: EvidenceInputSpec = None,
    *,
    base_dir: str | Path | None = None,
    behavior_evidence_required: bool = False,
    default_required: bool | None = None,
) -> PhysicalBehaviorEvidenceResolution:
    """Resolve physical-behavior evidence specs without heavy dependencies.

    Supported input specs are either paths or mappings with these draft keys:
    ``path``, ``kind``, ``required``, ``role``, and ``description``.

    Missing or unsupported evidence returns structured issues instead of
    raising. Malformed developer specs, such as a mapping without ``path``, are
    also converted to issues so the template can skip/fail cleanly.
    """

    required_default = (
        behavior_evidence_required if default_required is None else default_required
    )
    raw_inputs = _normalize_evidence_inputs(evidence_inputs)
    if not raw_inputs:
        severity = _severity_for_required(behavior_evidence_required)
        return PhysicalBehaviorEvidenceResolution(
            issues=(
                PhysicalBehaviorIssue(
                    code=BEHAVIOR_EVIDENCE_MISSING,
                    severity=severity,
                    message="No physical behavior evidence was supplied.",
                    details={"behavior_evidence_required": behavior_evidence_required},
                ),
            ),
            behavior_evidence_required=behavior_evidence_required,
        )

    base_path = _resolve_base_dir(base_dir)
    evidence: list[PhysicalBehaviorEvidence] = []
    issues: list[PhysicalBehaviorIssue] = []

    for input_index, raw_input in enumerate(raw_inputs):
        try:
            spec = _coerce_evidence_spec(raw_input, required_default)
        except (TypeError, ValueError) as exc:
            required = _required_for_malformed_spec(raw_input, required_default)
            issues.append(
                PhysicalBehaviorIssue(
                    code=BEHAVIOR_EVIDENCE_MALFORMED,
                    severity=_severity_for_required(required),
                    message=str(exc),
                    subject=_subject_for_malformed_spec(raw_input),
                    details={
                        "input_index": input_index,
                        "required": required,
                    },
                )
            )
            continue

        path = _resolve_path(spec["path"], base_path)
        exists = path.exists()
        kind, kind_details, kind_issue = _resolve_evidence_kind(path, spec["kind"])
        item = PhysicalBehaviorEvidence(
            original=spec["original"],
            path=str(path),
            kind=kind,
            exists=exists,
            required=spec["required"],
            role=spec["role"],
            description=spec["description"],
            details=kind_details,
        )
        evidence.append(item)

        if not exists:
            issues.append(
                PhysicalBehaviorIssue(
                    code=BEHAVIOR_EVIDENCE_MISSING,
                    severity=_severity_for_required(spec["required"]),
                    message=f"Physical behavior evidence does not exist: {path}",
                    subject=str(path),
                    details={"kind": kind, "role": spec["role"]},
                )
            )
            continue

        if kind_issue is not None:
            issues.append(kind_issue)

    return PhysicalBehaviorEvidenceResolution(
        evidence=tuple(evidence),
        issues=tuple(issues),
        behavior_evidence_required=behavior_evidence_required,
    )


def make_physical_behavior_placeholder_result(
    resolution: PhysicalBehaviorEvidenceResolution,
    *,
    task_description: str | None = None,
    behavior_evidence_required: bool | None = None,
) -> dict[str, Any]:
    """Return a legacy resolver-only ``physical_behavior`` result.

    This compatibility helper is retained for callers that only need evidence
    resolution output. It does not judge behavior. If
    ``behavior_evidence_required`` is not supplied, the resolver-level and
    per-item requiredness are preserved.
    """

    available_evidence = resolution.available_evidence
    effective_required = _effective_behavior_evidence_required(
        resolution,
        behavior_evidence_required,
    )
    issues = _placeholder_issue_dicts(resolution, effective_required)
    status: BehaviorPlaceholderStatus

    if any(issue["severity"] == "fail" for issue in issues):
        verdict: BehaviorPlaceholderVerdict = "fail"
        status = "skipped"
    elif not available_evidence:
        verdict = _severity_for_required(effective_required)
        status = "skipped"
    else:
        severity = _severity_for_required(effective_required)
        verdict = severity
        status = "unavailable"
        issues.append(
            PhysicalBehaviorIssue(
                code=BEHAVIOR_JUDGE_UNAVAILABLE,
                severity=severity,
                message=(
                    "Physical behavior evidence is present, but the behavior "
                    "judge/refine summary is unavailable."
                ),
                details={
                    "behavior_evidence_required": effective_required,
                    "evidence_kinds": _available_evidence_kinds(available_evidence),
                },
            ).to_dict()
        )

    return {
        "template": "physical_behavior",
        "status": status,
        "verdict": verdict,
        "passed": verdict != "fail",
        "issues": issues,
        "metrics": {
            "behavior_evidence_required": effective_required,
            "evidence_count": len(resolution.evidence),
            "available_evidence_count": len(available_evidence),
            "evidence_kinds": _available_evidence_kinds(available_evidence),
        },
        "evidence": [item.to_dict() for item in resolution.evidence],
        "task_description": task_description,
    }


def _normalize_evidence_inputs(
    evidence_inputs: EvidenceInputSpec,
) -> tuple[EvidenceInput, ...]:
    """Normalize absent, scalar, and sequence evidence inputs to a tuple."""

    if evidence_inputs is None:
        return ()
    if isinstance(evidence_inputs, str | Path) or isinstance(evidence_inputs, Mapping):
        return (evidence_inputs,)
    return tuple(evidence_inputs)


def _coerce_evidence_spec(
    raw_input: EvidenceInput,
    default_required: bool,
) -> dict[str, Any]:
    """Coerce a path or mapping input into a validated internal evidence spec."""

    if isinstance(raw_input, str | Path):
        return {
            "original": str(raw_input),
            "path": raw_input,
            "kind": None,
            "required": default_required,
            "role": None,
            "description": None,
        }

    if not isinstance(raw_input, Mapping):
        raise TypeError(
            "Physical behavior evidence input must be a path or mapping, "
            f"got {type(raw_input).__name__}"
        )

    path = raw_input.get("path")
    if not isinstance(path, str | Path):
        raise ValueError("Physical behavior evidence mapping requires a path")

    kind = raw_input.get("kind")
    if kind is not None and kind not in SUPPORTED_EVIDENCE_KINDS:
        raise ValueError(
            "Unsupported physical behavior evidence kind "
            f"{kind!r}; supported kinds are {list(SUPPORTED_EVIDENCE_KINDS)}"
        )

    required = raw_input.get("required", default_required)
    if not isinstance(required, bool):
        raise TypeError(
            "Physical behavior evidence required flag must be a bool, "
            f"got {type(required).__name__}"
        )

    role = raw_input.get("role")
    if role is not None and not isinstance(role, str):
        raise TypeError("Physical behavior evidence role must be a string")

    description = raw_input.get("description")
    if description is not None and not isinstance(description, str):
        raise TypeError("Physical behavior evidence description must be a string")

    return {
        "original": str(path),
        "path": path,
        "kind": kind,
        "required": required,
        "role": role,
        "description": description,
    }


def _required_for_malformed_spec(raw_input: object, default_required: bool) -> bool:
    """Recover the intended required flag from a malformed mapping when possible."""

    if isinstance(raw_input, Mapping):
        required = raw_input.get("required", default_required)
        if isinstance(required, bool):
            return required
    return default_required


def _subject_for_malformed_spec(raw_input: object) -> str | None:
    """Return the malformed mapping path as an issue subject when it is usable."""

    if not isinstance(raw_input, Mapping):
        return None
    path = raw_input.get("path")
    if isinstance(path, str | Path):
        return str(path)
    return None


def _resolve_base_dir(base_dir: str | Path | None) -> Path:
    """Resolve the base directory used for relative evidence paths."""

    if base_dir is None:
        return Path.cwd().resolve()
    path = Path(base_dir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    """Resolve an evidence path relative to the supplied base directory."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve(strict=False)


def _resolve_evidence_kind(
    path: Path,
    explicit_kind: str | None,
) -> tuple[BehaviorEvidenceKind | None, dict[str, Any], PhysicalBehaviorIssue | None]:
    """Resolve or infer the supported physical-behavior evidence kind."""

    suffix = path.suffix.lower()
    if explicit_kind is not None:
        kind = cast(BehaviorEvidenceKind, explicit_kind)
        return (
            kind,
            {"inferred": False},
            _json_issue_if_needed(
                path,
                kind,
            ),
        )

    if suffix in USD_EXTENSIONS:
        return "time_sampled_usd", {"inferred": True}, None
    if suffix in VIDEO_EXTENSIONS:
        return "video", {"inferred": True}, None
    if suffix in IMAGE_EXTENSIONS:
        return "sampled_frame", {"inferred": True}, None
    if suffix in JSON_EXTENSIONS:
        return _infer_json_evidence_kind(path)
    if suffix in JSONL_EXTENSIONS:
        return _infer_jsonl_evidence_kind(path)

    return (
        None,
        {"inferred": False, "extension": suffix or None},
        PhysicalBehaviorIssue(
            code=BEHAVIOR_EVIDENCE_UNSUPPORTED,
            severity="fail",
            message=(
                "Unsupported physical behavior evidence extension "
                f"{suffix or '<none>'}."
            ),
            subject=str(path),
            details={"supported_kinds": list(SUPPORTED_EVIDENCE_KINDS)},
        ),
    )


def _infer_json_evidence_kind(
    path: Path,
) -> tuple[BehaviorEvidenceKind | None, dict[str, Any], PhysicalBehaviorIssue | None]:
    """Infer the behavior evidence kind for JSON inputs from lightweight content."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "simulation_json", {"inferred": True}, None
    except (OSError, UnicodeError) as exc:
        return (
            "simulation_json",
            {"inferred": True},
            PhysicalBehaviorIssue(
                code=BEHAVIOR_EVIDENCE_MALFORMED,
                severity="fail",
                message=f"Physical behavior JSON evidence could not be read: {exc}",
                subject=str(path),
                details={"error_type": type(exc).__name__},
            ),
        )
    except json.JSONDecodeError as exc:
        return (
            "simulation_json",
            {"inferred": True},
            PhysicalBehaviorIssue(
                code=BEHAVIOR_EVIDENCE_MALFORMED,
                severity="fail",
                message=f"Physical behavior JSON evidence is malformed: {exc}",
                subject=str(path),
                details={"error_type": type(exc).__name__},
            ),
        )

    if isinstance(data, Mapping):
        keys = set(data.keys())
        if keys & {"trajectory", "trajectory_metrics", "trajectoryMetrics"}:
            return (
                "trajectory_metrics",
                {"inferred": True, "json_keys": sorted(keys)},
                None,
            )
        if "metrics" in keys and keys & {"positions", "velocities", "settled"}:
            return (
                "trajectory_metrics",
                {"inferred": True, "json_keys": sorted(keys)},
                None,
            )
        return "simulation_json", {"inferred": True, "json_keys": sorted(keys)}, None

    return (
        "simulation_json",
        {"inferred": True, "json_top_level": type(data).__name__},
        None,
    )


def _infer_jsonl_evidence_kind(
    path: Path,
) -> tuple[BehaviorEvidenceKind | None, dict[str, Any], PhysicalBehaviorIssue | None]:
    """Infer line-delimited trajectory metrics from lightweight JSONL checks."""

    if not path.exists():
        return "trajectory_metrics", {"inferred": True}, None
    issue, line_count = _jsonl_issue_and_line_count(path)
    return (
        "trajectory_metrics",
        {"inferred": True, "jsonl_line_count": line_count},
        issue,
    )


def _json_issue_if_needed(
    path: Path,
    explicit_kind: BehaviorEvidenceKind,
) -> PhysicalBehaviorIssue | None:
    """Validate explicitly typed JSON evidence enough to report malformed input."""

    if explicit_kind not in {"simulation_json", "trajectory_metrics"}:
        return None
    if not path.exists():
        return None
    if path.suffix.lower() in JSONL_EXTENSIONS:
        return _jsonl_issue_if_needed(path)
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        return PhysicalBehaviorIssue(
            code=BEHAVIOR_EVIDENCE_MALFORMED,
            severity="fail",
            message=f"Physical behavior JSON evidence could not be read: {exc}",
            subject=str(path),
            details={"error_type": type(exc).__name__},
        )
    except json.JSONDecodeError as exc:
        return PhysicalBehaviorIssue(
            code=BEHAVIOR_EVIDENCE_MALFORMED,
            severity="fail",
            message=f"Physical behavior JSON evidence is malformed: {exc}",
            subject=str(path),
            details={"error_type": type(exc).__name__},
        )
    return None


def _jsonl_issue_if_needed(path: Path) -> PhysicalBehaviorIssue | None:
    """Validate line-delimited JSON enough to report malformed metrics input."""

    issue, _ = _jsonl_issue_and_line_count(path)
    return issue


def _jsonl_issue_and_line_count(
    path: Path,
) -> tuple[PhysicalBehaviorIssue | None, int]:
    """Validate line-delimited JSON and count non-empty records in one pass."""

    line_count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                line_count += 1
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    return (
                        PhysicalBehaviorIssue(
                            code=BEHAVIOR_EVIDENCE_MALFORMED,
                            severity="fail",
                            message=(
                                "Physical behavior JSONL evidence is malformed "
                                f"on line {line_number}: {exc}"
                            ),
                            subject=str(path),
                            details={
                                "error_type": type(exc).__name__,
                                "line_number": line_number,
                            },
                        ),
                        line_count,
                    )
    except (OSError, UnicodeError) as exc:
        return (
            PhysicalBehaviorIssue(
                code=BEHAVIOR_EVIDENCE_MALFORMED,
                severity="fail",
                message=f"Physical behavior JSONL evidence could not be read: {exc}",
                subject=str(path),
                details={"error_type": type(exc).__name__},
            ),
            line_count,
        )
    return None, line_count


def _severity_for_required(required: bool) -> BehaviorEvidenceSeverity:
    """Map evidence requiredness to the provisional warn/fail severity."""

    return "fail" if required else "warn"


def _effective_behavior_evidence_required(
    resolution: PhysicalBehaviorEvidenceResolution,
    behavior_evidence_required: bool | None,
) -> bool:
    """Combine explicit, resolver-level, and per-item requiredness."""

    return (
        bool(behavior_evidence_required)
        or resolution.behavior_evidence_required
        or any(item.required for item in resolution.evidence)
    )


def _placeholder_issue_dicts(
    resolution: PhysicalBehaviorEvidenceResolution,
    effective_required: bool,
) -> list[dict[str, Any]]:
    """Convert resolution issues and upgrade missing-evidence severity if needed."""

    issue_dicts: list[dict[str, Any]] = []
    for issue in resolution.issues:
        issue_dict = issue.to_dict()
        if (
            effective_required
            and issue_dict["code"] == BEHAVIOR_EVIDENCE_MISSING
            and issue_dict["severity"] == "warn"
        ):
            issue_dict["severity"] = "fail"
            issue_dict["details"] = {
                **issue_dict["details"],
                "behavior_evidence_required": True,
            }
        issue_dicts.append(issue_dict)
    return issue_dicts


def _available_evidence_kinds(
    evidence: Sequence[PhysicalBehaviorEvidence],
) -> list[BehaviorEvidenceKind]:
    """Return stable unique evidence kinds for available evidence artifacts."""

    return sorted({item.kind for item in evidence if item.kind is not None})
