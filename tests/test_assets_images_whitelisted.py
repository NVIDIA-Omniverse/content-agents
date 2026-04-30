# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for OMPE-89656.

The repo's `.gitignore` blocks top-level `assets/` content (large binaries
under `assets/<other>/` should never be committed), but README teasers and
public-shipping media under `assets/images/` MUST be trackable so they
ship in the public mirror.

A regression in the gitignore rule order would silently drop the SimReady
teaser GIFs from the public release; the README would render with broken
image links. This test pins the whitelist by:

1. Asserting `git check-ignore` returns exit 1 (= NOT ignored) for each
   shipped teaser GIF under `assets/images/`.
2. Asserting at least one teaser GIF actually exists on disk under
   `assets/images/` so the test cannot pass vacuously.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_IMAGES_DIR = REPO_ROOT / "assets" / "images"


def _git_check_ignore(path: Path) -> bool:
    """Return True if `path` is ignored by git, False otherwise."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git binary not on PATH")
    result = subprocess.run(
        [git, "-C", str(REPO_ROOT), "check-ignore", "--quiet", str(path)],
        check=False,
    )
    # Exit 0 = ignored, 1 = not ignored, others = error.
    if result.returncode not in (0, 1):
        pytest.fail(f"git check-ignore failed with exit {result.returncode} for {path}")
    return result.returncode == 0


def test_teaser_gifs_are_not_gitignored() -> None:
    """Each shipped teaser GIF under assets/images/ must be trackable."""
    if not ASSETS_IMAGES_DIR.exists():
        pytest.skip(f"{ASSETS_IMAGES_DIR} not present")

    teasers = sorted(ASSETS_IMAGES_DIR.rglob("*.gif"))
    assert teasers, (
        f"No *.gif files found under {ASSETS_IMAGES_DIR.relative_to(REPO_ROOT)}; "
        f"the test cannot validate the whitelist vacuously"
    )

    ignored = [p.relative_to(REPO_ROOT) for p in teasers if _git_check_ignore(p)]
    assert not ignored, (
        f"The following teaser files under assets/images/ are gitignored — "
        f"they would silently drop from the public release per OMPE-89656: "
        f"{ignored}"
    )


def test_assets_images_directory_descent_allowed() -> None:
    """Sanity: the `assets/images/` directory itself must not be ignored."""
    if not ASSETS_IMAGES_DIR.exists():
        pytest.skip(f"{ASSETS_IMAGES_DIR} not present")
    assert not _git_check_ignore(ASSETS_IMAGES_DIR), (
        "assets/images/ directory is gitignored — descent into the public-"
        "shipping media tree is blocked, breaking OMPE-89656"
    )
