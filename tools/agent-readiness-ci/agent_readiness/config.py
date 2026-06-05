from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an agent-readiness config is invalid."""


@dataclass
class Job:
    id: str
    priority: str
    title: str
    success_criteria: list[str]


@dataclass
class Thresholds:
    p0_live_pass_rate_min: float = 0.8
    api_design_score_min: float = 0.7
    fail_on_p0_fail: bool = True
    fail_on_unverified_blockers: bool = True


@dataclass
class ReadinessConfig:
    product_id: str
    name: str
    repo_path: Path
    jobs: list[Job]
    raw: dict[str, Any] = field(repr=False)
    thresholds: Thresholds = field(default_factory=Thresholds)


def _require_mapping(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError("config must be a YAML mapping")
    return data


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"required field '{key}' is missing or empty")
    return value.strip()


def _load_jobs(data: dict[str, Any]) -> list[Job]:
    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ConfigError("required field 'jobs' must contain at least one job")
    jobs: list[Job] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_jobs, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"jobs[{index}] must be a mapping")
        job_id = _require_str(item, "id")
        if job_id in seen_ids:
            raise ConfigError(f"duplicate job id '{job_id}'")
        seen_ids.add(job_id)
        priority = str(item.get("priority", "P0")).strip()
        if priority not in {"P0", "P1", "P2"}:
            raise ConfigError(f"job '{job_id}' has invalid priority '{priority}'")
        title = _require_str(item, "title")
        criteria = item.get("success_criteria")
        if not isinstance(criteria, list) or not all(isinstance(c, str) and c.strip() for c in criteria):
            raise ConfigError(f"job '{job_id}' must define non-empty success_criteria")
        jobs.append(Job(id=job_id, priority=priority, title=title, success_criteria=[c.strip() for c in criteria]))
    return jobs


def _load_thresholds(data: dict[str, Any]) -> Thresholds:
    raw = data.get("thresholds") or {}
    if not isinstance(raw, dict):
        raise ConfigError("thresholds must be a mapping when provided")
    return Thresholds(
        p0_live_pass_rate_min=float(raw.get("p0_live_pass_rate_min", 0.8)),
        api_design_score_min=float(raw.get("api_design_score_min", 0.7)),
        fail_on_p0_fail=bool(raw.get("fail_on_p0_fail", True)),
        fail_on_unverified_blockers=bool(raw.get("fail_on_unverified_blockers", True)),
    )


def load_config(path: Path) -> ReadinessConfig:
    try:
        data = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    product_id = _require_str(data, "product_id")
    name = _require_str(data, "name")
    repo_value = data.get("repo_path", ".")
    if not isinstance(repo_value, str) or not repo_value.strip():
        raise ConfigError("repo_path must be a path string when provided")
    repo_path = Path(repo_value).expanduser()
    if not repo_path.is_absolute():
        repo_path = (path.parent / repo_path).resolve()
    else:
        repo_path = repo_path.resolve()
    if not repo_path.exists():
        raise ConfigError(f"repo_path does not exist: {repo_path}")

    return ReadinessConfig(
        product_id=product_id,
        name=name,
        repo_path=repo_path,
        jobs=_load_jobs(data),
        raw=data,
        thresholds=_load_thresholds(data),
    )
