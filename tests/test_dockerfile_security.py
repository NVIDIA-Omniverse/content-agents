# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that Dockerfiles contain required security packages and CVE documentation."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Dockerfiles for each service
PHYSICS_AGENT_DOCKERFILES = [
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile.ci",
]

JOINT_AGENT_DOCKERFILES = [
    REPO_ROOT / "apps" / "joint_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "joint_agent_service" / "Dockerfile.ci",
]

MATERIAL_AGENT_DOCKERFILES = [
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile.ci",
]

ALL_DOCKERFILES = (
    PHYSICS_AGENT_DOCKERFILES + JOINT_AGENT_DOCKERFILES + MATERIAL_AGENT_DOCKERFILES
)

# gnupg packages required by CVE-2025-68973 fix
GNUPG_PACKAGES = [
    "gnupg",
    "gpg",
    "gpg-agent",
    "gpgconf",
    "gpgsm",
    "gpg-wks-client",
    "dirmngr",
    "libgpgme11t64",
]

# Matches apt-get install blocks (handles line continuations with \)
_APT_INSTALL_RE = re.compile(
    r"(?:apt-get|apt)\s+install.*?(?=(?:^[A-Z]|\Z))", re.DOTALL | re.MULTILINE
)


def _extract_apt_install_sections(content: str) -> str:
    """Extract all apt-get install command sections from Dockerfile content."""
    return " ".join(_APT_INSTALL_RE.findall(content))


@pytest.mark.parametrize(
    "dockerfile",
    PHYSICS_AGENT_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_physics_agent_has_gnupg_packages(dockerfile: Path) -> None:
    """CVE-2025-68973: physics-agent-service Dockerfiles must install gnupg packages."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")
    content = dockerfile.read_text()
    apt_sections = _extract_apt_install_sections(content)
    for pkg in GNUPG_PACKAGES:
        assert re.search(rf"\b{re.escape(pkg)}\b", apt_sections), (
            f"{dockerfile.name} is missing gnupg package '{pkg}' "
            f"in apt install commands (required for CVE-2025-68973)"
        )


@pytest.mark.parametrize(
    "dockerfile",
    ALL_DOCKERFILES,
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_dockerfile_has_cve_2026_4519_comment(dockerfile: Path) -> None:
    """CVE-2026-4519: all Dockerfiles must document the webbrowser.open() CVE status."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")
    content = dockerfile.read_text()
    comment_lines = [
        line for line in content.splitlines() if line.strip().startswith("#")
    ]
    assert any("CVE-2026-4519" in line for line in comment_lines), (
        f"{dockerfile.name} is missing CVE-2026-4519 documentation comment"
    )
