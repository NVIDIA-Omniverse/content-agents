# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for nvbug-6122150 / OMPE-91540.

`*_agent_service` apps used to declare `[tool.hatch.build.targets.wheel].only-include`
which ships files but does not register a Python package — so `pip install -e
apps/<svc>_agent_service` produced an empty wheel with no `_editable_impl_*.pth`,
and `from client.client import ...` from any cwd outside the service directory
raised `ModuleNotFoundError`.

The fix switched to `packages = ["service", "client"]`. This test pins that
declaration so a future edit cannot revert to the broken `only-include` form.
"""

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

AGENT_SERVICE_PYPROJECTS = sorted(
    (REPO_ROOT / "apps").glob("*_agent_service/pyproject.toml")
) + sorted((REPO_ROOT / "apps").glob("*_simple_service/pyproject.toml"))

REQUIRED_PACKAGES = {"service", "client"}


@pytest.mark.parametrize(
    "pyproject_path",
    AGENT_SERVICE_PYPROJECTS,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_service_pyproject_declares_packages(pyproject_path: Path) -> None:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    wheel_target = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel")
    )
    if wheel_target is None:
        pytest.skip(
            f"{pyproject_path.relative_to(REPO_ROOT)} has no hatch wheel target"
        )

    packages = wheel_target.get("packages")
    assert packages is not None, (
        f"{pyproject_path.relative_to(REPO_ROOT)} must declare "
        "`[tool.hatch.build.targets.wheel].packages`. The previous "
        "`only-include` form ships files without registering them as a "
        "Python package, so editable installs produced an empty wheel and "
        "`from client.client import ...` failed. See nvbug-6122150."
    )
    missing = REQUIRED_PACKAGES - set(packages)
    assert not missing, (
        f"{pyproject_path.relative_to(REPO_ROOT)} packages={packages!r} is "
        f"missing {sorted(missing)}. Both `service` and `client` must be "
        "registered so the documented `from client.client import ...` import "
        "works for editable installs."
    )
