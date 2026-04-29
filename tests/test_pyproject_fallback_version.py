# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for nvbug-6122035 / OMPE-91525.

When users `unzip` a public source archive (no `.git`), `uv-dynamic-versioning`
aborts with `RuntimeError: This does not appear to be a Git project` unless
each `pyproject.toml` declares a `fallback-version`. Test pins both:

1. Every `pyproject.toml` in the repo declares `[tool.uv-dynamic-versioning].fallback-version`.
2. The fallback equals `VERSION.md` — so a chained `uv pip install -e apps/<svc>`
   still satisfies `world-understanding>=X.Y.Z` floors. The Codex round-2 review
   on !394 caught a `0.0.0` regression that would have broken those chained
   installs silently.
"""

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PYPROJECT_PATHS = [
    REPO_ROOT / "pyproject.toml",
    *sorted((REPO_ROOT / "apps").glob("*/pyproject.toml")),
]


def _expected_version() -> str:
    return (REPO_ROOT / "VERSION.md").read_text(encoding="utf-8").strip()


@pytest.mark.parametrize(
    "pyproject_path", PYPROJECT_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_fallback_version_declared_and_matches_version_md(pyproject_path: Path) -> None:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    udv = data.get("tool", {}).get("uv-dynamic-versioning")
    if udv is None:
        pytest.skip(
            f"{pyproject_path.relative_to(REPO_ROOT)} does not use uv-dynamic-versioning"
        )

    fallback = udv.get("fallback-version")
    assert fallback, (
        f"{pyproject_path.relative_to(REPO_ROOT)} is missing "
        "[tool.uv-dynamic-versioning].fallback-version. ZIP-download installs "
        "(no .git) will fail without it. See nvbug-6122035."
    )
    expected = _expected_version()
    assert fallback == expected, (
        f"{pyproject_path.relative_to(REPO_ROOT)} fallback-version={fallback!r} "
        f"does not match VERSION.md={expected!r}. Bump these in "
        "lockstep — see CLAUDE.md 'Version Bumping' step."
    )
