# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provisional render_valid adapter for the pre-#45 validation scaffold."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeGuard

from PIL import Image as PILImage

from world_understanding.functions.graphics.render_validation import (
    FrameId,
    RenderValidationIssue,
    detect_duplicate_frames,
    validate_image_artifact,
    validate_render_response,
)

RENDER_VALID_TEMPLATE = "render_valid"
RENDER_EVIDENCE_MISSING = "render.evidence_missing"
RENDER_FRAME_ID_MISMATCH = "render.frame_id_mismatch"
RENDER_RESPONSE_IMAGE_KEYS = ("images", "image_files", "output_paths", "outputs")

RenderValidStatus = Literal["pass", "fail", "skipped"]
PathEvidence = str | Path
PathEvidenceSpec = PathEvidence | Sequence[PathEvidence]
ResponseImageEvidence = str | Path | PILImage.Image


def run_render_valid_adapter(
    *,
    image_paths: PathEvidenceSpec = (),
    animation_frame_paths: PathEvidenceSpec | None = None,
    frame_ids: Sequence[FrameId] | None = None,
    render_response: Any | None = None,
    expected_cameras: Sequence[str] | None = None,
    expected_frames: Sequence[FrameId] | None = None,
    render_output_dir: str | Path | None = None,
    backend: str | None = None,
    detect_error_material_artifacts: bool = False,
) -> dict[str, Any]:
    """Run provisional render_valid checks and return a scaffold-friendly dict.

    This adapter intentionally depends only on the contract-neutral render
    validation utilities. It does not import Lane 1 scaffold models or the final
    #45 validation contracts. Relative image filenames found inside render
    responses are validated only when ``render_output_dir`` is supplied.
    """
    normalized_images = _normalize_paths(image_paths)
    normalized_frames = _normalize_optional_paths(animation_frame_paths)
    has_render_response = render_response is not None
    has_animation_frames = bool(normalized_frames)
    if not normalized_images and not has_animation_frames and not has_render_response:
        return _skipped_result(backend)

    image_results = [
        validate_image_artifact(
            path,
            backend=backend,
            detect_error_material_artifacts=detect_error_material_artifacts,
        )
        for path in normalized_images
    ]
    frame_image_results = [
        validate_image_artifact(
            path,
            backend=backend,
            detect_error_material_artifacts=detect_error_material_artifacts,
        )
        for path in (normalized_frames or ())
    ]

    duplicate_frame_result = None
    adapter_issues: list[dict[str, Any]] = []
    if (
        has_animation_frames
        and frame_ids is not None
        and len(frame_ids) != len(normalized_frames or ())
    ):
        adapter_issues.append(
            _adapter_issue(
                code=RENDER_FRAME_ID_MISMATCH,
                message=(
                    "frame_ids must have the same length as animation_frame_paths."
                ),
                check="duplicate_frames",
                details={
                    "frame_id_count": len(frame_ids),
                    "animation_frame_count": len(normalized_frames or ()),
                },
            )
        )
    elif has_animation_frames and len(normalized_frames or ()) >= 2:
        duplicate_frame_result = detect_duplicate_frames(
            normalized_frames or (),
            frame_ids=frame_ids,
            backend=backend,
        )

    render_response_metadata = None
    render_response_result = None
    response_images: tuple[ResponseImageEvidence, ...] = ()
    if has_render_response:
        render_response_result = validate_render_response(
            render_response,
            expected_cameras=expected_cameras,
            expected_frames=expected_frames,
            backend=backend,
        )
        render_response_metadata = render_response_result.metadata
        response_images = _extract_render_response_images(
            render_response,
            render_output_dir=render_output_dir,
        )

    response_image_results = [
        validate_image_artifact(
            image,
            backend=backend,
            detect_error_material_artifacts=detect_error_material_artifacts,
        )
        for image in response_images
    ]

    issues = adapter_issues + _collect_issues(
        image_results=[result.to_dict() for result in image_results],
        frame_image_results=[result.to_dict() for result in frame_image_results],
        response_image_results=[result.to_dict() for result in response_image_results],
        duplicate_frame_result=(
            duplicate_frame_result.to_dict()
            if duplicate_frame_result is not None
            else None
        ),
        render_response_result=(
            render_response_result.to_dict()
            if render_response_result is not None
            else None
        ),
    )
    status: RenderValidStatus = "fail" if issues else "pass"

    return {
        "template": RENDER_VALID_TEMPLATE,
        "status": status,
        "issues": issues,
        "metrics": {
            "image_count": len(normalized_images),
            "animation_frame_count": (
                len(normalized_frames) if normalized_frames is not None else 0
            ),
            "render_response_image_count": len(response_images),
            "issue_count": len(issues),
            "render_response_present": has_render_response,
        },
        "evidence": {
            "image_paths": [str(path) for path in normalized_images],
            "animation_frame_paths": (
                [str(path) for path in normalized_frames]
                if normalized_frames is not None
                else []
            ),
            "render_response_images": [
                _response_image_label(image) for image in response_images
            ],
        },
        "results": {
            "images": [result.to_dict() for result in image_results],
            "animation_frames": [result.to_dict() for result in frame_image_results],
            "render_response_images": [
                result.to_dict() for result in response_image_results
            ],
            "duplicate_frames": (
                duplicate_frame_result.to_dict()
                if duplicate_frame_result is not None
                else None
            ),
            "render_response": (
                render_response_result.to_dict()
                if render_response_result is not None
                else None
            ),
        },
        "metadata": {
            "backend": backend,
            "render_response": (
                render_response_metadata.to_dict()
                if render_response_metadata is not None
                else None
            ),
        },
    }


def _skipped_result(backend: str | None) -> dict[str, Any]:
    issue = RenderValidationIssue(
        code=RENDER_EVIDENCE_MISSING,
        message="No render evidence was supplied for render_valid checks.",
        severity="warning",
    )
    issue_payload = issue.to_dict()
    issue_payload["template"] = RENDER_VALID_TEMPLATE
    issue_payload["check"] = "evidence"
    return {
        "template": RENDER_VALID_TEMPLATE,
        "status": "skipped",
        "issues": [issue_payload],
        "metrics": {
            "image_count": 0,
            "animation_frame_count": 0,
            "render_response_image_count": 0,
            "issue_count": 1,
            "render_response_present": False,
        },
        "evidence": {
            "image_paths": [],
            "animation_frame_paths": [],
            "render_response_images": [],
        },
        "results": {
            "images": [],
            "animation_frames": [],
            "render_response_images": [],
            "duplicate_frames": None,
            "render_response": None,
        },
        "metadata": {
            "backend": backend,
            "render_response": None,
        },
    }


def _normalize_paths(paths: PathEvidenceSpec) -> tuple[Path, ...]:
    if isinstance(paths, str | Path):
        return (Path(paths),)
    return tuple(Path(path) for path in paths)


def _normalize_optional_paths(
    paths: PathEvidenceSpec | None,
) -> tuple[Path, ...] | None:
    if paths is None:
        return None
    return _normalize_paths(paths)


def _adapter_issue(
    *,
    code: str,
    message: str,
    check: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    issue: dict[str, Any] = RenderValidationIssue(
        code=code,
        message=message,
        details=details,
    ).to_dict()
    issue["template"] = RENDER_VALID_TEMPLATE
    issue["check"] = check
    return issue


def _extract_render_response_images(
    response: Any,
    *,
    render_output_dir: str | Path | None,
) -> tuple[ResponseImageEvidence, ...]:
    images: list[ResponseImageEvidence] = []
    output_dir = Path(render_output_dir) if render_output_dir is not None else None
    for entry in _render_response_entries(response):
        if not isinstance(entry, Mapping):
            continue
        for raw_image in _entry_response_image_entries(entry):
            image = _coerce_response_image(raw_image, output_dir)
            if image is not None:
                images.append(image)
    return tuple(images)


def _entry_response_image_entries(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    for key in RENDER_RESPONSE_IMAGE_KEYS:
        image_entries = entry.get(key)
        if image_entries is None:
            continue
        if _is_non_string_sequence(image_entries):
            raw_images = tuple(image_entries)
            if raw_images:
                return raw_images
            continue
        return (image_entries,)
    return ()


def _render_response_entries(response: Any) -> tuple[Any, ...]:
    if isinstance(response, Mapping):
        results = response.get("results")
        if results is None:
            return (response,)
        if _is_non_string_sequence(results):
            return tuple(results)
        return ()
    if _is_non_string_sequence(response):
        return tuple(response)
    return ()


def _coerce_response_image(
    value: object,
    output_dir: Path | None,
) -> ResponseImageEvidence | None:
    if isinstance(value, PILImage.Image):
        return value
    if isinstance(value, str | Path):
        if not str(value):
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        if output_dir is None:
            return None
        return output_dir / path
    return None


def _response_image_label(image: ResponseImageEvidence) -> str | None:
    if isinstance(image, str | Path):
        return str(image)
    return None


def _is_non_string_sequence(value: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _collect_issues(
    *,
    image_results: Sequence[Mapping[str, Any]],
    frame_image_results: Sequence[Mapping[str, Any]],
    response_image_results: Sequence[Mapping[str, Any]],
    duplicate_frame_result: Mapping[str, Any] | None,
    render_response_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for index, result in enumerate(image_results):
        issues.extend(
            _tag_issues(
                result.get("issues", ()),
                check="image",
                evidence_index=index,
            )
        )

    for index, result in enumerate(frame_image_results):
        issues.extend(
            _tag_issues(
                result.get("issues", ()),
                check="animation_frame",
                evidence_index=index,
            )
        )

    for index, result in enumerate(response_image_results):
        issues.extend(
            _tag_issues(
                result.get("issues", ()),
                check="render_response_image",
                evidence_index=index,
            )
        )

    if duplicate_frame_result is not None:
        issues.extend(
            _tag_issues(
                duplicate_frame_result.get("issues", ()),
                check="duplicate_frames",
            )
        )

    if render_response_result is not None:
        issues.extend(
            _tag_issues(
                render_response_result.get("issues", ()),
                check="render_response",
            )
        )

    return issues


def _tag_issues(
    raw_issues: object,
    *,
    check: str,
    evidence_index: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_issues, list | tuple):
        return []

    tagged: list[dict[str, Any]] = []
    for raw_issue in raw_issues:
        if not isinstance(raw_issue, Mapping):
            continue
        issue = dict(raw_issue)
        issue["template"] = RENDER_VALID_TEMPLATE
        issue["check"] = check
        if evidence_index is not None:
            issue["evidence_index"] = evidence_index
        tagged.append(issue)
    return tagged
