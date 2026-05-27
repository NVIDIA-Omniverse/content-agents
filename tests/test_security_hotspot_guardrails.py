# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small CI guardrails for security hotspot classes fixed in PR #231."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOTS = (REPO_ROOT / "apps", REPO_ROOT / "world_understanding")
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "docs",
    "node_modules",
    "tests",
}

DOCKERIGNORE_CREDENTIAL_PATTERNS = {
    ".aws/",
    ".azure/",
    ".config/gcloud/",
    ".env*",
    "!.env_example",
    "!.env_example_public",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "*.key",
    "*.p12",
    "*.pem",
    "*.pfx",
    "credentials",
    "credentials.*",
    "id_ed25519",
    "id_rsa",
}
STAGING_EXCLUDED_DOCKER_INPUTS = ("uv.lock", "packages/world_understanding_internal")
LLM_JSON_BOUNDARIES = {
    REPO_ROOT
    / "apps"
    / "joint_agent"
    / "joint_agent"
    / "tasks"
    / "identify_asset.py": ("asset_type", "asset_subtype"),
    REPO_ROOT
    / "apps"
    / "physics_agent"
    / "physics_agent"
    / "tasks"
    / "identify_asset.py": ("asset_type", "asset_subtype"),
    REPO_ROOT / "world_understanding" / "agentic" / "usd_tasks" / "identify_asset.py": (
        "asset_type",
        "asset_subtype",
    ),
    REPO_ROOT
    / "apps"
    / "material_agent"
    / "material_agent"
    / "scene"
    / "reconcile.py": ("remap",),
}
USD_NATIVE_RESOLVER_BOUNDARIES = (
    REPO_ROOT
    / "apps"
    / "material_agent"
    / "material_agent"
    / "tasks"
    / "apply_materials_to_usd.py",
    REPO_ROOT
    / "apps"
    / "material_agent"
    / "material_agent"
    / "tasks"
    / "iterative_completion.py",
)
LEGACY_MD5_HELPER = REPO_ROOT / "world_understanding" / "utils" / "compat_hash.py"


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _iter_source_files(suffix: str) -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
            current = Path(dirpath)
            if any(part in EXCLUDED_DIRS for part in current.parts):
                continue
            files.extend(current / name for name in filenames if name.endswith(suffix))
    return sorted(files)


def _iter_dockerfiles() -> list[Path]:
    return sorted(path for path in (REPO_ROOT / "apps").rglob("Dockerfile*"))


def _iter_public_dockerfiles() -> list[Path]:
    return sorted(path for path in (REPO_ROOT / "apps").rglob("Dockerfile"))


def _docker_instructions(content: str) -> list[str]:
    instructions: list[str] = []
    current: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        current.append(stripped)
        if not stripped.endswith("\\"):
            instructions.append(
                re.sub(r"\s+", " ", " ".join(part.rstrip("\\") for part in current))
            )
            current = []
    return instructions


def _md5_call_is_allowed(path: Path, node: ast.Call) -> bool:
    return path == LEGACY_MD5_HELPER and any(
        keyword.arg == "usedforsecurity"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in node.keywords
    )


def _find_md5_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    class Visitor(ast.NodeVisitor):
        hashlib_aliases = {"hashlib"}
        md5_aliases: set[str] = set()

        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            for alias in node.names:
                if alias.name == "hashlib":
                    self.hashlib_aliases.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
            if node.module == "hashlib":
                for alias in node.names:
                    if alias.name == "md5":
                        self.md5_aliases.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            direct = (
                isinstance(node.func, ast.Name) and node.func.id in self.md5_aliases
            )
            attribute = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "md5"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.hashlib_aliases
            )
            if (direct or attribute) and not _md5_call_is_allowed(path, node):
                violations.append(f"{_rel(path)}:{node.lineno}: avoid direct MD5")
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def test_dockerignore_filters_common_credential_files() -> None:
    patterns = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    missing = sorted(DOCKERIGNORE_CREDENTIAL_PATTERNS - patterns)
    assert not missing, ".dockerignore missing credential patterns: " + ", ".join(
        missing
    )


@pytest.mark.parametrize("dockerfile", _iter_dockerfiles(), ids=_rel)
def test_dockerfiles_avoid_recursive_context_copy_and_app_chown(
    dockerfile: Path,
) -> None:
    instructions = _docker_instructions(dockerfile.read_text(encoding="utf-8"))
    broad_copy = re.compile(
        r"^(?:COPY|ADD)(?: --\S+)* \. (?:/app|\.|/workspace)(?: |$)",
        re.IGNORECASE,
    )
    app_chown = re.compile(r"\bchown\s+-R\b.*(?:^| )/app(?: |$)", re.IGNORECASE)

    violations = [
        instruction
        for instruction in instructions
        if broad_copy.search(instruction) or app_chown.search(instruction)
    ]
    assert not violations, f"{_rel(dockerfile)}:\n" + "\n".join(violations)


@pytest.mark.parametrize("dockerfile", _iter_public_dockerfiles(), ids=_rel)
def test_public_dockerfiles_do_not_require_internal_or_staging_excluded_inputs(
    dockerfile: Path,
) -> None:
    content = dockerfile.read_text(encoding="utf-8")
    violations = [value for value in STAGING_EXCLUDED_DOCKER_INPUTS if value in content]
    assert not violations, f"{_rel(dockerfile)} references: {', '.join(violations)}"


@pytest.mark.parametrize("dockerfile", _iter_public_dockerfiles(), ids=_rel)
def test_non_root_editable_dockerfiles_do_not_write_bytecode_to_app(
    dockerfile: Path,
) -> None:
    content = dockerfile.read_text(encoding="utf-8")
    if (
        "uv pip install" not in content
        or " -e " not in content
        or "USER " not in content
    ):
        pytest.skip("Dockerfile does not combine editable install with non-root user")

    assert "PYTHONDONTWRITEBYTECODE=1" in content


@pytest.mark.parametrize(
    ("path", "expected_keys"),
    list(LLM_JSON_BOUNDARIES.items()),
    ids=[_rel(path) for path in LLM_JSON_BOUNDARIES],
)
def test_llm_json_boundaries_require_answer_keys(
    path: Path,
    expected_keys: tuple[str, ...],
) -> None:
    if not path.exists():
        assert "/joint_agent/" in path.as_posix(), f"{_rel(path)} is missing"
        pytest.skip("Joint Agent is excluded from this public release")

    content = path.read_text(encoding="utf-8")
    assert "extract_json_from_llm_response" in content
    assert "expected_keys" in content
    for key in expected_keys:
        assert repr(key) in content or f'"{key}"' in content


@pytest.mark.parametrize("path", USD_NATIVE_RESOLVER_BOUNDARIES, ids=_rel)
def test_generated_usd_boundaries_reject_native_resolver_paths(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    assert "world_understanding.utils.usd.asset_paths" in content
    assert "is_uri_asset_path" in content
    assert "resolve_relative_asset_path_under_base" in content
    assert "Keeping absolute/URL path" not in content
    assert "falling back to simple copy" not in content


def test_runtime_python_code_does_not_introduce_direct_md5_calls() -> None:
    violations: list[str] = []
    for path in _iter_source_files(".py"):
        violations.extend(_find_md5_calls(path))

    assert not violations, "\n".join(violations)
