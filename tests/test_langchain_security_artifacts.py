# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for recurring LangChain security scanner findings."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent

LANGCHAIN_CORE_NOTICE_RE = re.compile(
    r"(?:\|\s*langchain-core\s*\|\s*|<td>langchain-core</td><td>)"
    r"(?P<version>[0-9][^|\s<]*)",
    re.IGNORECASE,
)

# Include every checked-in notice artifact that feeds scanner evidence. The root
# notice is intentionally part of the guard even when a PR only regenerates a
# service notice, because stale root evidence can keep recurring in source
# security scans.
ROOT_NOTICE_FILE = REPO_ROOT / "THIRD_PARTY_NOTICE.md"

NOTICE_FILES = [
    ROOT_NOTICE_FILE,
    REPO_ROOT / "apps" / "material_agent_service" / "3RD_PARTY.md",
    REPO_ROOT / "apps" / "material_agent_service" / "3rd_party_licenses.html",
]


def _is_vulnerable_langchain_core(version: str) -> bool:
    parsed = Version(version)
    return parsed < Version("0.3.85") or (Version("1.0.0") <= parsed < Version("1.3.3"))


def _locked_langchain_core_versions() -> list[str]:
    lock_data = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return [
        package["version"]
        for package in lock_data.get("package", [])
        if package.get("name") == "langchain-core"
    ]


def _dependency_name(requirement: str) -> str | None:
    requirement = requirement.split(";", 1)[0].strip()
    requirement = re.sub(r"\[.*?\]", "", requirement)
    match = re.match(r"([A-Za-z0-9._-]+)", requirement)
    if not match:
        return None
    return canonicalize_name(match.group(1))


def _pyproject_dependency_names(pyproject_file: Path) -> list[str]:
    pyproject_data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
    project_data = pyproject_data.get("project", {})
    dependencies = list(project_data.get("dependencies", []))
    for optional_dependencies in project_data.get("optional-dependencies", {}).values():
        dependencies.extend(optional_dependencies)
    return [name for dep in dependencies if (name := _dependency_name(dep))]


def _repo_pyproject_files() -> list[Path]:
    excluded_dirs = {
        ".codex",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
    }
    pyproject_files: list[Path] = []
    for pyproject_file in REPO_ROOT.rglob("pyproject.toml"):
        relative_parts = pyproject_file.relative_to(REPO_ROOT).parts
        if excluded_dirs.intersection(relative_parts):
            continue
        pyproject_files.append(pyproject_file)
    return sorted(pyproject_files)


def _pyprojects_declaring_langchain_core() -> list[Path]:
    declaring_files: list[Path] = []
    for pyproject_file in _repo_pyproject_files():
        dependency_names = _pyproject_dependency_names(pyproject_file)
        if "langchain-core" in dependency_names:
            declaring_files.append(pyproject_file)
    return declaring_files


def _repo_pyproject_files_by_name() -> dict[str, Path]:
    pyproject_files: dict[str, Path] = {}
    for pyproject_file in _repo_pyproject_files():
        pyproject_data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
        project_name = pyproject_data.get("project", {}).get("name")
        if isinstance(project_name, str):
            pyproject_files[canonicalize_name(project_name)] = pyproject_file
    return pyproject_files


def _transitive_pyproject_dependency_names(pyproject_file: Path) -> set[str]:
    local_pyprojects = _repo_pyproject_files_by_name()
    dependency_names: set[str] = set()
    visited_pyprojects: set[Path] = set()

    def visit(current_file: Path) -> None:
        current_file = current_file.resolve()
        if current_file in visited_pyprojects:
            return
        visited_pyprojects.add(current_file)

        for dependency_name in _pyproject_dependency_names(current_file):
            dependency_names.add(dependency_name)
            child_pyproject = local_pyprojects.get(dependency_name)
            if child_pyproject is not None:
                visit(child_pyproject)

    visit(pyproject_file)
    return dependency_names


def _format_relative_paths(paths: list[Path]) -> list[str]:
    return [path.relative_to(REPO_ROOT).as_posix() for path in paths]


def _notice_pyproject_file(notice_file: Path) -> Path | None:
    if notice_file == ROOT_NOTICE_FILE:
        return REPO_ROOT / "pyproject.toml"

    relative_parts = notice_file.relative_to(REPO_ROOT).parts
    if len(relative_parts) >= 3 and relative_parts[0] == "apps":
        app_pyproject = REPO_ROOT / "apps" / relative_parts[1] / "pyproject.toml"
        if app_pyproject.exists():
            return app_pyproject
    return None


def test_pyproject_discovery_covers_material_service() -> None:
    pyproject_paths = _format_relative_paths(_repo_pyproject_files())
    assert "apps/material_agent_service/pyproject.toml" in pyproject_paths


def test_notice_pyproject_file_maps_app_notice_to_app_pyproject() -> None:
    assert _notice_pyproject_file(
        REPO_ROOT / "apps" / "material_agent_service" / "3RD_PARTY.md"
    ) == (REPO_ROOT / "apps" / "material_agent_service" / "pyproject.toml")


def test_material_service_dependency_closure_includes_langchain_core() -> None:
    dependency_names = _transitive_pyproject_dependency_names(
        REPO_ROOT / "apps" / "material_agent_service" / "pyproject.toml"
    )

    assert "world-understanding" in dependency_names
    assert "langchain-core" in dependency_names


def test_dependency_name_normalizes_extras_markers_and_underscores() -> None:
    assert (
        _dependency_name('langchain_core[foo]>=1.4.0; python_version >= "3.12"')
        == "langchain-core"
    )
    assert _dependency_name("langchain.core>=1.4.0") == "langchain-core"


def _pyproject_declares_langchain_core() -> bool:
    return bool(_pyprojects_declaring_langchain_core())


def _pyprojects_declaring_langchain_core_message() -> str:
    return ", ".join(_format_relative_paths(_pyprojects_declaring_langchain_core()))


def test_locked_langchain_core_is_not_in_vulnerable_range() -> None:
    versions = _locked_langchain_core_versions()
    if not versions:
        declaring_files = _pyprojects_declaring_langchain_core_message()
        assert not _pyproject_declares_langchain_core(), (
            "langchain-core is declared in pyproject.toml but uv.lock does not "
            f"contain a resolved langchain-core package: {declaring_files}"
        )
        return

    vulnerable_versions = [
        version for version in versions if _is_vulnerable_langchain_core(version)
    ]
    assert not vulnerable_versions, (
        "uv.lock contains langchain-core versions in the vulnerable "
        f"CVE-2026-44843 range: {vulnerable_versions}"
    )


@pytest.mark.parametrize(
    "notice_file",
    NOTICE_FILES,
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_notice_artifacts_do_not_list_vulnerable_langchain_core(
    notice_file: Path,
) -> None:
    if not notice_file.exists():
        pytest.fail(f"{notice_file.relative_to(REPO_ROOT)} is missing")
    content = notice_file.read_text(encoding="utf-8")
    versions = [
        match.group("version") for match in LANGCHAIN_CORE_NOTICE_RE.finditer(content)
    ]
    if not versions:
        if notice_file == ROOT_NOTICE_FILE:
            assert not _locked_langchain_core_versions(), (
                f"{notice_file.relative_to(REPO_ROOT)} does not list "
                "langchain-core, but uv.lock still contains it"
            )
        else:
            pyproject_file = _notice_pyproject_file(notice_file)
            dependency_names = (
                _transitive_pyproject_dependency_names(pyproject_file)
                if pyproject_file is not None
                else set()
            )
            pyproject_name = (
                pyproject_file.relative_to(REPO_ROOT).as_posix()
                if pyproject_file is not None
                else "no app pyproject.toml"
            )
            assert "langchain-core" not in dependency_names, (
                f"{notice_file.relative_to(REPO_ROOT)} does not list "
                "langchain-core, but the dependency closure for "
                f"{pyproject_name} includes it"
            )
        return

    vulnerable_versions = [
        version for version in versions if _is_vulnerable_langchain_core(version)
    ]
    assert not vulnerable_versions, (
        f"{notice_file.relative_to(REPO_ROOT)} contains stale langchain-core "
        f"versions in the vulnerable CVE-2026-44843 range: {vulnerable_versions}"
    )


@pytest.mark.parametrize(
    "notice_file",
    [
        REPO_ROOT / "apps" / "material_agent_service" / "3RD_PARTY.md",
        REPO_ROOT / "apps" / "material_agent_service" / "3rd_party_licenses.html",
    ],
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_material_service_notices_do_not_resurrect_vision_endpoint(
    notice_file: Path,
) -> None:
    if not notice_file.exists():
        pytest.fail(f"{notice_file.relative_to(REPO_ROOT)} is missing")
    assert "vision-endpoint" not in notice_file.read_text(encoding="utf-8"), (
        f"{notice_file.relative_to(REPO_ROOT)} lists the removed vision-endpoint "
        "package, which previously reintroduced stale LangChain 0.3.x evidence "
        "into scanner inputs."
    )
