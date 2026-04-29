from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COPY_TO_STAGING = REPO_ROOT / "scripts" / "internal" / "copy_to_staging.sh"
pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None,
    reason="copy_to_staging.sh requires rsync",
)

PUBLIC_ASSET_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
}
PUBLIC_USD_ROOTS = (
    Path("apps/material_agent/data/examples"),
    Path("apps/material_agent/data/materials"),
    Path("apps/material_agent/data/regression"),
    Path("apps/material_agent/data/templates"),
    Path("apps/material_agent_service/examples"),
    Path("apps/material_agent_service/materials/default"),
)


def _run(
    *args: str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def _collect_public_asset_files(target_dir: Path) -> list[Path]:
    asset_files: list[Path] = []
    for root in PUBLIC_USD_ROOTS:
        root_path = target_dir / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in PUBLIC_ASSET_SUFFIXES:
                asset_files.append(path.relative_to(target_dir))
    return sorted(asset_files)


def test_copy_to_staging_keeps_public_assets_trackable(tmp_path: Path) -> None:
    target_dir = tmp_path / "staging"
    _run("git", "init", str(target_dir))

    _run("bash", str(COPY_TO_STAGING), str(target_dir), cwd=REPO_ROOT)

    asset_files = _collect_public_asset_files(target_dir)
    assert asset_files, "expected staged public assets to exist"

    for rel_path in asset_files:
        status = _run(
            "git",
            "-C",
            str(target_dir),
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            rel_path.as_posix(),
        )
        assert status.stdout.strip() == f"?? {rel_path.as_posix()}"

        _run(
            "git",
            "-C",
            str(target_dir),
            "add",
            "-A",
            "--",
            rel_path.as_posix(),
        )
        staged = _run(
            "git",
            "-C",
            str(target_dir),
            "diff",
            "--cached",
            "--name-only",
            "--",
            rel_path.as_posix(),
        )
        assert staged.stdout.strip() == rel_path.as_posix()
