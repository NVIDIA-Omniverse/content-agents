# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for nvbug-6125716 / OMPE-91693.

The agent-service `docker-compose.yml` files used to re-list VLM provider
API keys under `environment:` with `${VAR:-}` substitution. Compose
substitution evaluates against the project-directory `.env` (the compose
file's parent dir), not the `env_file: ../../.env` directive, so the
substitution resolved to empty strings and clobbered the env_file values
the user had configured. Pipelines silently lost their API keys and
fell back to "no key" failure modes.

This test pins both halves of the fix:

1. None of the agent-service `docker-compose.yml` files re-list a VLM
   provider API key under `environment:` with a `${...}` substitution.
2. Each agent-service `docker-compose.yml` has an `env_file:` entry
   pointing at repo-root `.env`, which is the canonical key source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

SERVICE_COMPOSE_FILES = [
    REPO_ROOT / "apps" / "material_agent_service" / "docker-compose.yml",
    REPO_ROOT / "apps" / "physics_agent_service" / "docker-compose.yml",
    REPO_ROOT / "apps" / "texture_agent_service" / "docker-compose.yml",
    REPO_ROOT / "apps" / "joint_agent_service" / "docker-compose.yml",
]

# Keys that flow through env_file and must not be re-substituted on the
# environment side. NGC_API_KEY is rendering/login only and is allowed to
# substitute (its absence is harmless when RENDER_ENDPOINT is local).
VLM_KEYS = {
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
}

ENV_VLM_SUBST_RE = re.compile(
    r"-\s*(?P<key>[A-Z_]+)\s*=\s*\$\{(?P<ref>[A-Z_]+)(?::-[^}]*)?\}"
)


@pytest.mark.parametrize(
    "compose_file",
    SERVICE_COMPOSE_FILES,
    ids=lambda p: p.parent.name,
)
def test_no_vlm_key_substitution_under_environment(compose_file: Path) -> None:
    """Regression: VLM keys must flow through env_file, not `${VAR:-}` lines."""
    if not compose_file.exists():
        pytest.skip(f"{compose_file} not present")

    text = compose_file.read_text()

    # Strip comment-only lines so the pattern in the explanatory header
    # comment ("do NOT re-list ... with `${VAR:-}`") is not matched.
    non_comment_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    non_comment = "\n".join(non_comment_lines)

    offenders: list[str] = []
    for match in ENV_VLM_SUBST_RE.finditer(non_comment):
        key = match.group("key")
        ref = match.group("ref")
        if key in VLM_KEYS or ref in VLM_KEYS:
            offenders.append(match.group(0))

    assert not offenders, (
        f"{compose_file.relative_to(REPO_ROOT)} re-substitutes VLM keys under "
        f"`environment:` — this clobbers env_file values per OMPE-91693 / "
        f"nvbug-6125716. Offending lines: {offenders}"
    )


@pytest.mark.parametrize(
    "compose_file",
    SERVICE_COMPOSE_FILES,
    ids=lambda p: p.parent.name,
)
def test_repo_root_env_file_referenced(compose_file: Path) -> None:
    """Each agent-service compose file must source repo-root .env as canonical key store."""
    if not compose_file.exists():
        pytest.skip(f"{compose_file} not present")

    parsed = yaml.safe_load(compose_file.read_text())
    services = parsed.get("services", {})
    assert services, f"{compose_file} declares no services"

    found_root_env_file = False
    for _name, svc in services.items():
        env_file = svc.get("env_file")
        if env_file is None:
            continue
        entries = env_file if isinstance(env_file, list) else [env_file]
        for entry in entries:
            path = entry["path"] if isinstance(entry, dict) else entry
            # Compose evaluates env_file paths relative to the compose file dir;
            # repo-root .env is two levels up from apps/<svc>_service/.
            resolved = (compose_file.parent / path).resolve()
            if resolved == (REPO_ROOT / ".env").resolve():
                found_root_env_file = True
                break

    assert found_root_env_file, (
        f"{compose_file.relative_to(REPO_ROOT)} must reference repo-root .env "
        f"via env_file: (per OMPE-91693 / nvbug-6125716 fix)"
    )
