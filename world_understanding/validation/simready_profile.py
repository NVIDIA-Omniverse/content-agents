# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SimReady Validation Agent profile materialization helpers.

The SimReady profile is intentionally a fixture matrix, not a single
``ValidationRequest``. This module converts each ready fixture into an
executable V1 request config that can be run by ``validation-agent run``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import yaml
from pydantic import ValidationError

from world_understanding.validation.models import (
    ValidationPlannerConfig,
    ValidationProject,
    ValidationRenderConfig,
    ValidationRequest,
)

DEFAULT_FIXTURE_CONFIG_NAME: Final = "validation.yaml"
DEFERRED_FIXTURE_STATUSES: Final = frozenset(
    {
        "deferred_until_behavior_evidence_exists",
    }
)


class SimReadyProfileError(ValueError):
    """Raised when a SimReady profile cannot produce V1 request configs."""


@dataclass(frozen=True)
class SimReadyFixtureRequest:
    """Executable Validation Agent request derived from one profile fixture."""

    fixture_id: str
    request: ValidationRequest
    output_dir: Path
    status: str
    config_path: Path | None = None


def build_simready_fixture_requests(
    profile_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    fixture_ids: Sequence[str] = (),
    include_deferred: bool = False,
) -> tuple[SimReadyFixtureRequest, ...]:
    """Build executable V1 requests from a SimReady fixture profile.

    By default deferred behavior-evidence fixtures are skipped. Pass
    ``include_deferred=True`` to materialize them for explicit dependency-lane
    testing.
    """

    path = _resolve_profile_path(profile_path)
    profile = _load_profile(path)
    output_root = _profile_output_root(profile, output_dir)
    fixtures = _selected_fixtures(
        profile,
        fixture_ids=fixture_ids,
        include_deferred=include_deferred,
    )
    return tuple(
        _build_fixture_request(
            profile_path=path,
            profile=profile,
            fixture=fixture,
            output_dir=output_root / _fixture_id(fixture),
        )
        for fixture in fixtures
    )


def write_simready_fixture_configs(
    profile_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    fixture_ids: Sequence[str] = (),
    include_deferred: bool = False,
    config_filename: str = DEFAULT_FIXTURE_CONFIG_NAME,
) -> tuple[SimReadyFixtureRequest, ...]:
    """Write per-fixture ``ValidationRequest`` configs runnable by the V1 CLI."""

    path = _resolve_profile_path(profile_path)
    profile = _load_profile(path)
    output_root = _profile_output_root(profile, output_dir)
    config_name = _safe_config_filename(config_filename)
    fixture_requests = build_simready_fixture_requests(
        path,
        output_dir=output_root,
        fixture_ids=fixture_ids,
        include_deferred=include_deferred,
    )

    written: list[SimReadyFixtureRequest] = []
    for fixture_request in fixture_requests:
        fixture_dir = output_root / fixture_request.fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        config_path = fixture_dir / config_name
        request = _make_config_relative_request(
            fixture_request.request,
            profile_path=path,
            config_dir=config_path.parent,
        )
        config_path.write_text(
            yaml.safe_dump(
                request.model_dump(mode="json", exclude_none=True),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        written.append(
            SimReadyFixtureRequest(
                fixture_id=fixture_request.fixture_id,
                request=request,
                output_dir=fixture_dir,
                status=fixture_request.status,
                config_path=config_path,
            )
        )
    return tuple(written)


def _resolve_profile_path(profile_path: str | Path) -> Path:
    path = Path(profile_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise SimReadyProfileError(f"SimReady profile not found: {path}")
    return path


def _safe_config_filename(config_filename: str) -> str:
    path = Path(config_filename)
    if path.is_absolute() or ".." in path.parts:
        raise SimReadyProfileError(
            "SimReady fixture config filename must be a relative path without '..'"
        )
    if path.name != config_filename:
        raise SimReadyProfileError(
            "SimReady fixture config filename must not contain path separators"
        )
    if not config_filename.strip():
        raise SimReadyProfileError("SimReady fixture config filename is required")
    return config_filename


def _load_profile(path: Path) -> Mapping[str, Any]:
    try:
        raw_profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SimReadyProfileError(
            f"Unable to read SimReady profile {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise SimReadyProfileError(
            f"Invalid SimReady profile YAML {path}: {exc}"
        ) from exc
    if not isinstance(raw_profile, Mapping):
        raise SimReadyProfileError(
            "SimReady profile must be a mapping: "
            f"{path} (got {type(raw_profile).__name__})"
        )
    return raw_profile


def _profile_output_root(
    profile: Mapping[str, Any],
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser()

    project = _mapping_value(profile, "project")
    working_dir = project.get("working_dir")
    if not isinstance(working_dir, str) or not working_dir.strip():
        raise SimReadyProfileError("SimReady profile project.working_dir is required")
    return Path(working_dir).expanduser()


def _selected_fixtures(
    profile: Mapping[str, Any],
    *,
    fixture_ids: Sequence[str],
    include_deferred: bool,
) -> tuple[Mapping[str, Any], ...]:
    fixtures = _mapping_sequence(profile, "fixtures")
    fixtures_by_id = {_fixture_id(fixture): fixture for fixture in fixtures}
    if fixture_ids:
        missing_ids = sorted(set(fixture_ids) - set(fixtures_by_id))
        if missing_ids:
            raise SimReadyProfileError(
                "Unknown SimReady fixture id(s): " + ", ".join(missing_ids)
            )
        selected = tuple(fixtures_by_id[fixture_id] for fixture_id in fixture_ids)
    else:
        selected = fixtures

    if include_deferred or fixture_ids:
        return selected
    return tuple(
        fixture
        for fixture in selected
        if _fixture_status(fixture) not in DEFERRED_FIXTURE_STATUSES
    )


def _build_fixture_request(
    *,
    profile_path: Path,
    profile: Mapping[str, Any],
    fixture: Mapping[str, Any],
    output_dir: Path,
) -> SimReadyFixtureRequest:
    fixture_id = _fixture_id(fixture)
    status = _fixture_status(fixture)
    try:
        request = ValidationRequest(
            task_description=_required_string(fixture, "task_description"),
            inputs=_runtime_inputs_from_fixture(fixture),
            project=_project_from_profile(profile, fixture_id, output_dir),
            planner=_planner_from_profile(profile),
            render=ValidationRenderConfig.model_validate(
                dict(_optional_mapping_value(profile, "render"))
            ),
            requested_templates=_string_sequence(
                fixture.get("expected_templates"),
                "expected_templates",
            ),
            policy=_policy_from_fixture(profile, fixture),
            metadata=_metadata_from_fixture(profile_path, profile, fixture),
        )
    except ValidationError as exc:
        raise SimReadyProfileError(
            f"Fixture {fixture_id!r} is not a valid ValidationRequest: {exc}"
        ) from exc
    return SimReadyFixtureRequest(
        fixture_id=fixture_id,
        request=request,
        output_dir=output_dir,
        status=status,
    )


def _project_from_profile(
    profile: Mapping[str, Any],
    fixture_id: str,
    output_dir: Path,
) -> ValidationProject:
    project = _mapping_value(profile, "project")
    project_name = project.get("name", "simready_validation")
    if not isinstance(project_name, str) or not project_name.strip():
        raise SimReadyProfileError("SimReady profile project.name must be a string")
    return ValidationProject(
        name=f"{project_name}:{fixture_id}",
        working_dir=str(output_dir),
    )


def _planner_from_profile(profile: Mapping[str, Any]) -> ValidationPlannerConfig:
    raw_planner = dict(_optional_mapping_value(profile, "planner"))
    allowed_templates = raw_planner.pop("allowed_templates", ())
    metadata = dict(_optional_mapping_value(raw_planner, "metadata"))
    if allowed_templates:
        metadata["allowed_templates"] = list(
            _string_sequence(allowed_templates, "planner.allowed_templates")
        )
    raw_planner["metadata"] = metadata
    return ValidationPlannerConfig.model_validate(raw_planner)


def _policy_from_fixture(
    profile: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    profile_block = _optional_mapping_value(profile, "profile")
    if profile_policy := _optional_mapping_value(profile_block, "policy"):
        policy.update(dict(profile_policy))
    if gate_policy := _optional_mapping_value(profile_block, "gate_policy"):
        policy["gate_policy"] = dict(gate_policy)
    input_groups = _optional_mapping_value(fixture, "expected_input_groups")
    if input_groups:
        policy["expected_input_groups"] = dict(input_groups)
        reference_images = _string_sequence(
            input_groups.get("reference_images"),
            "expected_input_groups.reference_images",
            required=False,
        )
        if reference_images:
            policy["reference_image_paths"] = list(reference_images)
    if issue_namespaces := _string_sequence(
        fixture.get("expected_issue_namespaces"),
        "expected_issue_namespaces",
        required=False,
    ):
        policy["expected_issue_namespaces"] = list(issue_namespaces)
    if source_configs := _optional_mapping_value(fixture, "source_configs"):
        policy["source_configs"] = dict(source_configs)
    fixture_policy = dict(_optional_mapping_value(fixture, "policy"))
    if (
        "gate_policy" in fixture_policy
        and isinstance(fixture_policy["gate_policy"], Mapping)
        and isinstance(policy.get("gate_policy"), Mapping)
    ):
        fixture_policy["gate_policy"] = {
            **dict(cast(Mapping[str, Any], policy["gate_policy"])),
            **dict(fixture_policy["gate_policy"]),
        }
    policy.update(fixture_policy)
    return policy


def _metadata_from_fixture(
    profile_path: Path,
    profile: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    project = _mapping_value(profile, "project")
    profile_block = _optional_mapping_value(profile, "profile")
    fixture_id = _fixture_id(fixture)
    metadata: dict[str, Any] = {
        "profile_name": profile_block.get("name", project.get("name")),
        "profile_source": str(profile_path),
        "fixture_id": fixture_id,
        "fixture_status": _fixture_status(fixture),
    }
    if purpose := fixture.get("purpose"):
        metadata["fixture_purpose"] = purpose
    if description := project.get("description"):
        metadata["project_description"] = description
    if artifact_layout := _optional_mapping_value(profile_block, "artifact_layout"):
        metadata["artifact_layout"] = dict(artifact_layout)
    return metadata


def _runtime_inputs_from_fixture(fixture: Mapping[str, Any]) -> tuple[str, ...]:
    inputs = _string_sequence(fixture.get("inputs"), "inputs")
    input_groups = _optional_mapping_value(fixture, "expected_input_groups")
    reference_images = set(
        _string_sequence(
            input_groups.get("reference_images"),
            "expected_input_groups.reference_images",
            required=False,
        )
    )
    if not reference_images:
        return inputs
    return tuple(
        input_path for input_path in inputs if input_path not in reference_images
    )


def _make_config_relative_request(
    request: ValidationRequest,
    *,
    profile_path: Path,
    config_dir: Path,
) -> ValidationRequest:
    project = request.project.model_copy(update={"working_dir": "."})
    return request.model_copy(
        update={
            "inputs": tuple(
                _config_relative_path(input_path, profile_path, config_dir)
                for input_path in request.inputs
            ),
            "project": project,
            "policy": _config_relative_policy(
                request.policy,
                profile_path=profile_path,
                config_dir=config_dir,
            ),
        }
    )


def _config_relative_policy(
    policy: Mapping[str, Any],
    *,
    profile_path: Path,
    config_dir: Path,
) -> dict[str, Any]:
    rewritten = dict(policy)
    for key in (
        "animation_frame_paths",
        "animation_usd_paths",
        "behavior_video_paths",
        "current_image_paths",
        "reference_image_paths",
        "render_image_paths",
        "sampled_video_frame_paths",
        "simulation_json_paths",
        "time_sampled_usd_paths",
        "trajectory_metrics_paths",
        "video_paths",
    ):
        if key in rewritten:
            rewritten[key] = list(
                _config_relative_path_sequence(
                    rewritten[key],
                    profile_path,
                    config_dir,
                )
            )

    for key in ("physical_behavior_refine_summary_path", "refine_summary_path"):
        if key in rewritten:
            rewritten[key] = _config_relative_path_value(
                rewritten[key],
                profile_path,
                config_dir,
            )

    for key in (
        "physical_behavior_refine_output_dir",
        "physics_refine_output_dir",
        "refine_output_dir",
    ):
        value = rewritten.get(key)
        if isinstance(value, str):
            rewritten[key] = _config_relative_path(value, profile_path, config_dir)

    for key in ("physical_behavior_evidence", "behavior_evidence"):
        if key in rewritten:
            rewritten[key] = _config_relative_evidence_value(
                rewritten[key],
                profile_path,
                config_dir,
            )

    focused_images = rewritten.get("focused_image_paths")
    if isinstance(focused_images, Mapping):
        rewritten["focused_image_paths"] = {
            str(prim_path): list(
                _config_relative_path_sequence(paths, profile_path, config_dir)
            )
            for prim_path, paths in focused_images.items()
        }

    input_groups = rewritten.get("expected_input_groups")
    if isinstance(input_groups, Mapping):
        rewritten["expected_input_groups"] = {
            str(group_name): list(
                _config_relative_path_sequence(paths, profile_path, config_dir)
            )
            for group_name, paths in input_groups.items()
        }
    return rewritten


def _config_relative_evidence_value(
    value: Any,
    profile_path: Path,
    config_dir: Path,
) -> Any:
    if isinstance(value, str):
        return _config_relative_path(value, profile_path, config_dir)
    if isinstance(value, Mapping):
        item = dict(value)
        # Evidence specs reserve "path" for the artifact path; other keys
        # are metadata and should pass through unchanged.
        path_value = item.get("path")
        if isinstance(path_value, str):
            item["path"] = _config_relative_path(path_value, profile_path, config_dir)
        return item
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _config_relative_evidence_value(item, profile_path, config_dir)
            for item in value
        ]
    return value


def _config_relative_path_value(
    value: Any,
    profile_path: Path,
    config_dir: Path,
) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _config_relative_path(value, profile_path, config_dir)
    return list(_config_relative_path_sequence(value, profile_path, config_dir))


def _config_relative_path_sequence(
    value: Any,
    profile_path: Path,
    config_dir: Path,
) -> tuple[str, ...]:
    return tuple(
        _config_relative_path(path, profile_path, config_dir)
        for path in _string_sequence(value, "policy paths", required=False)
    )


def _config_relative_path(path_value: str, profile_path: Path, config_dir: Path) -> str:
    source_path = _resolve_profile_path_value(path_value, profile_path)
    resolved_config_dir = config_dir.expanduser().resolve(strict=False)
    try:
        return os.path.relpath(source_path, resolved_config_dir)
    except ValueError:
        return str(source_path)


def _resolve_profile_path_value(path_value: str, profile_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)

    cwd_candidate = (Path.cwd() / path).resolve(strict=False)
    if cwd_candidate.exists():
        return cwd_candidate

    profile_candidate = (profile_path.parent / path).resolve(strict=False)
    if profile_candidate.exists():
        return profile_candidate

    return cwd_candidate


def _fixture_id(fixture: Mapping[str, Any]) -> str:
    return _required_string(fixture, "id")


def _fixture_status(fixture: Mapping[str, Any]) -> str:
    return _required_string(fixture, "status")


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SimReadyProfileError(f"SimReady profile field {key!r} is required")
    return value


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise SimReadyProfileError(f"SimReady profile field {key!r} must be a mapping")
    return value


def _optional_mapping_value(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SimReadyProfileError(f"SimReady profile field {key!r} must be a mapping")
    return value


def _mapping_sequence(
    mapping: Mapping[str, Any],
    key: str,
) -> tuple[Mapping[str, Any], ...]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise SimReadyProfileError(f"SimReady profile field {key!r} must be a list")
    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise SimReadyProfileError(
                f"SimReady profile field {key!r}[{index}] must be a mapping"
            )
        items.append(item)
    return tuple(items)


def _string_sequence(
    value: Any,
    label: str,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    if value is None:
        if required:
            raise SimReadyProfileError(
                f"SimReady profile field {label!r} must be a list of strings"
            )
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
        raise SimReadyProfileError(
            f"SimReady profile field {label!r} must be a list of strings"
        )
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise SimReadyProfileError(
                f"SimReady profile field {label!r}[{index}] must be a string"
            )
        strings.append(item)
    return tuple(strings)
