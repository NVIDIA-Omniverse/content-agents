# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for nvbug-6121967 (OMPE-91522) and nvbug-6122154 (OMPE-91541).

Two related leaks recurred during 0.3.x: the public material/physics agent
service `docker-compose.yml` referenced an internal-only env var
`INFERENCE_NVIDIA_API_KEY`, and the public texture-agent Dockerfile retained
an `--extra-index-url https://gitlab-master.nvidia.com/...` that's unreachable
externally. Scan every file the staging copy publishes for known internal
markers so a future edit cannot silently re-introduce them in any of those
files (not just the two specific ones the original bugs surfaced in).

The candidate file set is built by globbing the public-mirror file types
(`README_PUBLIC.md`, `*.env_example_public`, `Dockerfile`, `docker-compose.yml`,
`docker-compose.multi-gpu.yml`, etc.) under the repo root and excluding
paths that `scripts/internal/copy_to_staging.sh` strips before publishing.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Glob patterns for files that ship in the public mirror. The
# `internal/` convention plus the explicit exclude list below strips
# anything internal-only; everything else is fair game for the public
# scan. See docs/internal/staging_guideline.md.
PUBLIC_FILE_GLOBS: list[str] = [
    # Top-level README / changelog / .env example.
    "README_PUBLIC.md",
    "CHANGELOG_PUBLIC.md",
    ".env_example_public",
    # Per-app variants (each agent has its own README_PUBLIC.md and the
    # services have their own .env_example_public).
    "apps/*/README_PUBLIC.md",
    "apps/*/.env_example_public",
    # Per-app `README.md`. The staging script replaces `README.md` with
    # the sibling `README_PUBLIC.md` when one exists, so a bare
    # `README.md` only ships when its directory has no `_PUBLIC` sibling
    # — `_filter_replaced_readmes` strips the redundant ones.
    "apps/*/README.md",
    "apps/*/NIGHTLY_BENCHMARK.md",
    # Per-app developer / API docs that ship publicly. Anything under
    # `apps/*/docs/internal/` is filtered by EXCLUDE_PATH_FRAGMENTS.
    "apps/*/docs/*.md",
    "apps/*/docs/**/*.md",
    # Public Dockerfiles + compose files. The internal `Dockerfile.ci` and
    # `docker-compose.internal.yml` variants are excluded below.
    "apps/*/Dockerfile",
    "apps/*/docker-compose.yml",
    "apps/*/docker-compose.multi-gpu.yml",
]

# Path fragments that the staging script strips before publishing — the
# `internal/` convention plus the joint-agent app and the
# `world_understanding_internal` package. See `scripts/internal/copy_to_staging.sh`.
EXCLUDE_PATH_FRAGMENTS: tuple[str, ...] = (
    "/internal/",
    "/.internal/",
    "/joint_agent/",
    "/joint_agent_service/",
    "/joint_agent_benchmark/",
    "/world_understanding_internal/",
    # Filename variants that are internal-only.
    "Dockerfile.ci",
    "docker-compose.internal.yml",
)


def _is_excluded(path: Path) -> bool:
    posix = "/" + path.relative_to(REPO_ROOT).as_posix()
    name = path.name
    return any(
        fragment in posix or fragment == name for fragment in EXCLUDE_PATH_FRAGMENTS
    )


def _filter_replaced_readmes(files: set[Path]) -> set[Path]:
    """Drop `<dir>/README.md` when `<dir>/README_PUBLIC.md` exists in the
    candidate set — the staging script replaces the former with the latter
    at copy time, so the internal version never ships."""
    public_dirs = {p.parent for p in files if p.name == "README_PUBLIC.md"}
    return {p for p in files if not (p.name == "README.md" and p.parent in public_dirs)}


def _gather_public_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in PUBLIC_FILE_GLOBS:
        for match in REPO_ROOT.glob(pattern):
            if match.is_file() and not _is_excluded(match):
                files.add(match)
    return sorted(_filter_replaced_readmes(files))


# Strings that must never appear in a published file. `nvidia_inference`
# is intentionally NOT in this list. The factory is registered only by
# `packages/world_understanding_internal/`, which is excluded from
# staging, so the identifier is a no-op in the public mirror — but
# scanning for it would create a lot of noise (it appears in switch
# statements, env-var lookups, and example-config dispatch code that
# legitimately ships to support an opt-in internal install). The
# more specific markers below (`perflab_azure_openai`, `llmgateway`,
# `INFERENCE_NVIDIA_API_KEY`, `NSTORAGE_API_KEY`) catch the
# user-visible leaks where they actually appear.
FORBIDDEN_STRINGS: dict[str, str] = {
    "gitlab-master.nvidia.com": "nvbug-6122154 — internal package index URL",
    "INFERENCE_NVIDIA_API_KEY": "nvbug-6121967 — internal-only inference API var",
    "nvbugspro.nvidia.com": "internal NVBugs URLs leak issue tracker context",
    "jirasw.nvidia.com": "internal Jira URLs leak ticket context",
    "urm.nvidia.com": "internal NVIDIA artifact registry",
    "perflab_azure_openai": "internal-only Azure OpenAI gateway backend identifier",
    "NSTORAGE_API_KEY": "internal-only NStorage credential",
    "llmgateway": (
        "internal-only LLM Gateway backend identifier (e.g. llmgateway_azure_openai)"
    ),
}

PUBLIC_FILES = _gather_public_files()


def test_public_file_set_is_non_empty() -> None:
    # Defensive: the globs must match something. If a future change moves
    # all public files out of the matched paths, fail loudly so the scope
    # gap is obvious instead of silently passing zero parametrizations.
    assert PUBLIC_FILES, (
        "No public-mirror files matched. PUBLIC_FILE_GLOBS may need an "
        "update — see scripts/internal/copy_to_staging.sh for what ships."
    )


@pytest.mark.parametrize(
    "public_file",
    PUBLIC_FILES,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_public_artifact_has_no_internal_strings(public_file: Path) -> None:
    content = public_file.read_text(encoding="utf-8")
    leaks = [
        f"{needle!r} ({why})"
        for needle, why in FORBIDDEN_STRINGS.items()
        if needle in content
    ]
    assert not leaks, (
        f"{public_file.relative_to(REPO_ROOT)} leaks internal-only strings "
        f"into the public mirror: {leaks}. Either remove the reference, or "
        "if the file legitimately needs it, exclude the file from staging "
        "via scripts/internal/copy_to_staging.sh and add the path fragment "
        "to EXCLUDE_PATH_FRAGMENTS in this test."
    )
