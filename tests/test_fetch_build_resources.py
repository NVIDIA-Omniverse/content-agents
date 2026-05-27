# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import os
import subprocess
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_build_resources.sh"


def url_sha256(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def fetch_url_for_arch(arch: str) -> str:
    env = os.environ.copy()
    env["SO_CORE_ARCH"] = arch
    env["SO_CORE_PRINT_URL_ONLY"] = "1"
    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_fetch_build_resources_defaults_to_x86_package_for_x86_64() -> None:
    url = fetch_url_for_arch("x86_64")

    assert "manylinux_2_35_x86_64.release.zip" in url


def test_fetch_build_resources_defaults_to_aarch64_package_for_aarch64() -> None:
    url = fetch_url_for_arch("aarch64")

    assert "manylinux_2_35_aarch64.release.zip" in url


def test_fetch_build_resources_rejects_unknown_architecture() -> None:
    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "riscv64"
    env["SO_CORE_PRINT_URL_ONLY"] = "1"

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "unsupported Scene Optimizer Core architecture: riscv64" in result.stderr


def test_fetch_build_resources_allows_unknown_architecture_with_explicit_url() -> None:
    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "riscv64"
    env["SO_CORE_URL"] = "https://example.invalid/scene_optimizer_core_custom.zip"
    env["SO_CORE_PRINT_URL_ONLY"] = "1"

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == env["SO_CORE_URL"]


def test_fetch_build_resources_refetches_wrong_architecture_package(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "scene_optimizer_core"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        (existing / subdir).mkdir(parents=True)
    tf_binary = existing / "usdpy" / "pxr" / "Tf" / "_tf.so"
    tf_binary.parent.mkdir(parents=True)
    tf_binary.write_text("not an aarch64 ELF", encoding="utf-8")

    package_zip = tmp_path / "scene_optimizer_core.zip"
    with zipfile.ZipFile(package_zip, "w") as archive:
        for subdir in ("python", "lib", "extraLibs", "usdpy"):
            archive.writestr(f"{subdir}/.keep", "")
        archive.writestr("usdpy/pxr/Tf/_tf.so", "replacement package")

    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "aarch64"
    env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
    env["SO_CORE_URL"] = package_zip.as_uri()

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "does not match manylinux_2_35_aarch64; refetching" in result.stdout
    marker = (existing / ".so_core_platform").read_text(encoding="utf-8")
    assert "platform=manylinux_2_35_aarch64" in marker
    assert f"url_sha256={url_sha256(package_zip.as_uri())}" in marker
    assert package_zip.as_uri() not in marker


def test_fetch_build_resources_refetches_when_url_changes(tmp_path: Path) -> None:
    existing = tmp_path / "scene_optimizer_core"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        (existing / subdir).mkdir(parents=True)
    (existing / ".so_core_platform").write_text(
        "platform=manylinux_2_35_aarch64\n"
        f"url_sha256={url_sha256('https://example.invalid/old_scene_optimizer_core.zip')}\n",
        encoding="utf-8",
    )

    package_zip = tmp_path / "scene_optimizer_core.zip"
    with zipfile.ZipFile(package_zip, "w") as archive:
        for subdir in ("python", "lib", "extraLibs", "usdpy"):
            archive.writestr(f"{subdir}/.keep", "")

    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "aarch64"
    env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
    env["SO_CORE_URL"] = package_zip.as_uri()

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "does not match manylinux_2_35_aarch64; refetching" in result.stdout
    marker = (existing / ".so_core_platform").read_text(encoding="utf-8")
    assert "platform=manylinux_2_35_aarch64" in marker
    assert f"url_sha256={url_sha256(package_zip.as_uri())}" in marker
    assert package_zip.as_uri() not in marker


def test_fetch_build_resources_preserves_existing_package_when_refetch_fails(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "scene_optimizer_core"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        (existing / subdir).mkdir(parents=True)
    sentinel = existing / "usdpy" / "existing.txt"
    sentinel.write_text("existing package is still usable", encoding="utf-8")

    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "aarch64"
    env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
    env["SO_CORE_URL"] = (tmp_path / "missing_scene_optimizer_core.zip").as_uri()

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "does not match manylinux_2_35_aarch64; refetching" in result.stdout
    assert sentinel.read_text(encoding="utf-8") == "existing package is still usable"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        assert (existing / subdir).is_dir()
