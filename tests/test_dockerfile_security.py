# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that Dockerfiles contain required security packages and CVE documentation."""

import re
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
OVRTX_RUNTIME_REQUIREMENTS = (
    REPO_ROOT
    / "world_understanding"
    / "functions"
    / "graphics"
    / "ovrtx_runtime_requirements.txt"
)
OVRTX_PROVISION_COMMAND = (
    "python -m world_understanding.functions.graphics.render_ovrtx --provision-only"
)
OVRTX_TEMP_UV_CACHE = "/tmp/wu-ovrtx-uv-cache"
ROOT_UV_CACHE = "/root/.cache/uv"

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

DISCOVERED_DOCKERFILES = sorted((REPO_ROOT / "apps").glob("*/Dockerfile*"))


def _read_discovered_dockerfiles() -> tuple[dict[Path, str], dict[Path, OSError]]:
    """Read discovered Dockerfiles without failing pytest collection."""
    contents: dict[Path, str] = {}
    read_errors: dict[Path, OSError] = {}
    for path in DISCOVERED_DOCKERFILES:
        if not path.is_file():
            continue
        try:
            contents[path] = path.read_text()
        except OSError as exc:
            read_errors[path] = exc
    return contents, read_errors


DOCKERFILE_CONTENT, DOCKERFILE_READ_ERRORS = _read_discovered_dockerfiles()
CUDA_UBUNTU_2404_DOCKERFILES = [
    path
    for path, content in DOCKERFILE_CONTENT.items()
    if "nvcr.io/nvidia/cuda:" in content and "ubuntu24.04" in content
]

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

_APT_COMMAND_PREFIX = (
    r"\b(?:apt-get|apt)(?:\s+--?[A-Za-z0-9][A-Za-z0-9-]*(?:=\S+)?)*\s+"
)

# Matches apt-get install blocks (handles line continuations with \)
_APT_INSTALL_RE = re.compile(
    rf"{_APT_COMMAND_PREFIX}install\b.*?(?=(?:^[A-Z]|\Z))",
    re.DOTALL | re.MULTILINE,
)
_DOCKERFILE_INSTRUCTION_RE = (
    r"ADD|ARG|CMD|COPY|ENTRYPOINT|ENV|EXPOSE|FROM|HEALTHCHECK|LABEL|MAINTAINER|"
    r"ONBUILD|RUN|SHELL|STOPSIGNAL|USER|VOLUME|WORKDIR"
)
_RUN_RE = re.compile(
    rf"^RUN\s+(.*?)(?=^(?:{_DOCKERFILE_INSTRUCTION_RE})\b|\Z)",
    re.MULTILINE | re.DOTALL,
)
_APT_UPGRADE_RE = re.compile(rf"{_APT_COMMAND_PREFIX}(?:dist-)?upgrade\b")
_APT_CLEAN_RE = re.compile(rf"{_APT_COMMAND_PREFIX}clean\b")
_PILLOW_REQ_PIN_RE = re.compile(
    r"^\s*pillow\s*==\s*([A-Za-z0-9][A-Za-z0-9.!+_-]*)\s*(?:#.*)?$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_apt_install_sections(content: str) -> str:
    """Extract all apt-get install command sections from Dockerfile content."""
    return " ".join(_APT_INSTALL_RE.findall(content))


def _strip_comment_lines(content: str) -> str:
    """Remove comment-only lines before command regex checks."""
    return "\n".join(
        line for line in content.splitlines() if not line.lstrip().startswith("#")
    )


def _extract_cuda_ubuntu24_sections(content: str) -> str:
    """Extract stages that inherit from CUDA Ubuntu 24.04 runtime images."""
    sections: list[str] = []
    current: list[str] = []
    in_cuda_ubuntu24_stage = False

    for line in content.splitlines():
        if line.startswith("FROM "):
            if in_cuda_ubuntu24_stage:
                sections.append("\n".join(current))
            in_cuda_ubuntu24_stage = (
                "nvcr.io/nvidia/cuda:" in line and "ubuntu24.04" in line
            )
            current = [line] if in_cuda_ubuntu24_stage else []
        elif in_cuda_ubuntu24_stage:
            current.append(line)

    if in_cuda_ubuntu24_stage:
        sections.append("\n".join(current))
    return "\n".join(sections)


def _extract_run_sections(content: str) -> list[str]:
    """Extract Dockerfile RUN instructions so comments cannot satisfy checks."""
    return _RUN_RE.findall(_strip_comment_lines(content))


def _assert_refreshes_cuda_base_os_packages(dockerfile: Path, content: str) -> None:
    """Assert CUDA Ubuntu Dockerfiles refresh and clean inherited OS packages."""
    cuda_sections = _extract_cuda_ubuntu24_sections(content)
    run_sections = _extract_run_sections(cuda_sections)
    upgrade_runs = [
        section for section in run_sections if _APT_UPGRADE_RE.search(section)
    ]
    assert upgrade_runs, (
        f"{dockerfile.name} uses an NGC CUDA Ubuntu 24.04 base image but does "
        "not run apt-get upgrade before installing packages in that stage. CUDA "
        "runtime tags can lag Ubuntu security pockets, leaving stale "
        "OpenSSL/PAM/GnuTLS packages that fail image scans."
    )
    for run_section in upgrade_runs:
        upgrade_match = _APT_UPGRADE_RE.search(run_section)
        clean_match = _APT_CLEAN_RE.search(run_section)
        archive_index = run_section.find("/var/cache/apt/archives")
        assert upgrade_match is not None
        assert (
            clean_match is not None and clean_match.start() > upgrade_match.start()
        ), (
            f"{dockerfile.name} runs apt-get upgrade but does not clean apt caches "
            "later in the same RUN layer."
        )
        assert archive_index > upgrade_match.start(), (
            f"{dockerfile.name} runs apt-get upgrade but does not remove downloaded "
            "apt package archives later in the same RUN layer."
        )


def _assert_secure_pillow_pin(path: Path, content: str) -> None:
    """Assert a requirements file explicitly pins a non-vulnerable Pillow."""
    versions = [
        Version(match.group(1)) for match in _PILLOW_REQ_PIN_RE.finditer(content)
    ]
    assert versions, (
        f"{path.name} does not explicitly pin Pillow. Image scans require "
        "Pillow >= 12.2.0 for the isolated OVRTX runtime."
    )
    for version in versions:
        assert version >= Version("12.2.0"), (
            f"{path.name} pins Pillow {version}; image scans require Pillow "
            ">= 12.2.0 for the isolated OVRTX runtime."
        )


def _assert_ovrtx_provision_cleans_uv_cache(dockerfile: Path, content: str) -> None:
    """Assert OVRTX provisioning cannot leave scanner-visible uv archives."""
    provision_runs = [
        section
        for section in _extract_run_sections(content)
        if OVRTX_PROVISION_COMMAND in section
    ]
    assert provision_runs, f"{dockerfile.name} does not run OVRTX provisioning"

    for run_section in provision_runs:
        provision_index = run_section.index(OVRTX_PROVISION_COMMAND)
        uv_cache_index = run_section.find(f"UV_CACHE_DIR={OVRTX_TEMP_UV_CACHE}")
        assert 0 <= uv_cache_index < provision_index, (
            f"{dockerfile.name} provisions OVRTX without a temporary UV_CACHE_DIR; "
            "uv can otherwise persist scanner-visible OVRTX wheel archives."
        )
        for cache_path in (OVRTX_TEMP_UV_CACHE, ROOT_UV_CACHE):
            cleanup_index = run_section.find(cache_path, provision_index)
            assert cleanup_index > provision_index, (
                f"{dockerfile.name} provisions OVRTX but does not remove "
                f"{cache_path} later in the same RUN layer."
            )


def test_apt_command_patterns_accept_flags_before_subcommands() -> None:
    """Apt guardrails should accept common flags before the subcommand."""
    assert _APT_UPGRADE_RE.search("apt-get -y upgrade")
    assert _APT_UPGRADE_RE.search("apt-get --yes upgrade")
    assert _APT_CLEAN_RE.search("apt-get --quiet clean")

    install_section = _extract_apt_install_sections(
        "RUN apt-get --no-install-recommends install \\\n    gnupg \\\n    gpg\n"
    )
    assert "gnupg" in install_section
    assert "gpg" in install_section


def test_secure_pillow_pin_accepts_inline_comments() -> None:
    """Requirements comments should not hide a secure Pillow pin."""
    _assert_secure_pillow_pin(
        Path("requirements.txt"),
        "# Historical vulnerable pin: pillow==12.1.1\n"
        "pillow==12.2.0  # image scan floor\n",
    )


@pytest.mark.parametrize(
    "dockerfile",
    DISCOVERED_DOCKERFILES,
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_discovered_dockerfiles_are_readable(dockerfile: Path) -> None:
    """Unreadable Dockerfiles should fail as tests, not during collection."""
    error = DOCKERFILE_READ_ERRORS.get(dockerfile)
    if error is not None:
        pytest.fail(f"Could not read discovered Dockerfile {dockerfile}: {error}")


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


def test_joint_ci_pythonpath_precedes_ovrtx_provisioning() -> None:
    """The runtime-stage OVRTX provision command imports copied site-packages."""
    dockerfile = REPO_ROOT / "apps" / "joint_agent_service" / "Dockerfile.ci"
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    content = dockerfile.read_text()
    pythonpath_idx = content.index('ENV PYTHONPATH="/app:/usr/local/lib/python3.12')
    provision_idx = content.index(
        "python -m world_understanding.functions.graphics.render_ovrtx --provision-only"
    )

    assert pythonpath_idx < provision_idx


@pytest.mark.parametrize(
    "dockerfile",
    JOINT_AGENT_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_joint_agent_ovrtx_uses_shared_provisioner(dockerfile: Path) -> None:
    """Joint images must use the shared OVRTX provisioner and its sanitizer."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    content = _strip_comment_lines(dockerfile.read_text())
    assert OVRTX_PROVISION_COMMAND in content
    assert "ovrtx==" not in content


@pytest.mark.parametrize(
    "dockerfile",
    JOINT_AGENT_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_joint_agent_ovrtx_provisioning_cleans_uv_cache(
    dockerfile: Path,
) -> None:
    """Joint image scans must not see OVRTX libpython copies in uv caches."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    _assert_ovrtx_provision_cleans_uv_cache(dockerfile, dockerfile.read_text())


@pytest.mark.parametrize(
    "dockerfile",
    JOINT_AGENT_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_joint_agent_dockerfiles_are_covered_by_scan_guardrails(
    dockerfile: Path,
) -> None:
    """Joint service images are the image-scan regression surface."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")
    assert dockerfile in CUDA_UBUNTU_2404_DOCKERFILES
    content = dockerfile.read_text()
    _assert_refreshes_cuda_base_os_packages(dockerfile, content)


@pytest.mark.parametrize(
    "dockerfile",
    CUDA_UBUNTU_2404_DOCKERFILES,
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_cuda_ubuntu24_dockerfiles_refresh_base_os_packages(
    dockerfile: Path,
) -> None:
    """CUDA Ubuntu 24.04 runtime images must refresh stale base OS packages."""
    content = dockerfile.read_text()
    _assert_refreshes_cuda_base_os_packages(dockerfile, content)


def test_ovrtx_runtime_requirements_pillow_pin_is_not_vulnerable() -> None:
    """The shared OVRTX runtime dependency pin must not use vulnerable Pillow."""
    if not OVRTX_RUNTIME_REQUIREMENTS.exists():
        pytest.skip(
            f"{OVRTX_RUNTIME_REQUIREMENTS} not present in this checkout "
            "(e.g. public mirror)"
        )

    content = OVRTX_RUNTIME_REQUIREMENTS.read_text()
    _assert_secure_pillow_pin(OVRTX_RUNTIME_REQUIREMENTS, content)
