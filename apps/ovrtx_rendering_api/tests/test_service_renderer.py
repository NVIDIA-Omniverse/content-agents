# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for apps/ovrtx_rendering_api/service/renderer.py.

Focus on the pure-Python helpers — the pxr and ovrtx imports in ``render()``
are not exercised here since they require a GPU-equipped environment.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

# ``service`` is on sys.path via ``pythonpath = ["apps/ovrtx_rendering_api"]``
# in the root pyproject.toml's [tool.pytest.ini_options].
from service.renderer import (
    _ZIP_MAX_FILES,
    _extract_zip_bundle,
    _is_usdz_payload,
)


def _make_bundle(tmp_path: Path, names: list[str]) -> Path:
    """Build a bundle.zip containing the given entries at the archive root."""
    src = tmp_path / "src"
    src.mkdir()
    for name in names:
        p = src / name
        p.parent.mkdir(parents=True, exist_ok=True)
        # A one-line USDA is enough — we never open the stage in these tests.
        p.write_text('#usda 1.0\ndef Xform "Root" {}\n')

    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(src / name, name)
    return zip_path


class TestExtractZipBundle:
    def test_picks_stage_usda_root_from_client_bundle(self, tmp_path: Path):
        """Matches the layout produced by render_nvcf._bundle_stage_with_local_assets."""
        zip_path = _make_bundle(
            tmp_path,
            ["stage.usda", "mdl_materials/wood/wood.mdl", "textures/albedo.png"],
        )
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).name == "stage.usda"
        assert Path(main_usd).exists()
        # Assets must be extracted alongside so relative paths resolve.
        assert (extracted / "bundle" / "mdl_materials" / "wood" / "wood.mdl").exists()
        assert (extracted / "bundle" / "textures" / "albedo.png").exists()

    def test_prefers_main_over_scene_and_stage(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["main.usda", "scene.usd", "stage.usdc"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).name == "main.usda"

    def test_prefers_scene_over_stage(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["scene.usd", "stage.usdc"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).name == "scene.usd"

    def test_falls_back_to_alphabetical(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["zebra.usda", "alpha.usda"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).name == "alpha.usda"

    def test_usdz_mode_prefers_first_usd_in_archive_order(self, tmp_path: Path):
        """USDZ packages use their first USD layer as the package root."""
        zip_path = _make_bundle(tmp_path, ["zebra.usda", "alpha.usda"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(
            str(zip_path),
            str(extracted),
            prefer_first_usd=True,
        )

        assert Path(main_usd).name == "zebra.usda"

    def test_discovers_nested_usd(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["assets/sub/main.usda"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).relative_to(extracted / "bundle") == Path(
            "assets/sub/main.usda"
        )

    def test_discovers_uppercase_usd_extensions(self, tmp_path: Path):
        """USD files with uppercase extensions must be found on Linux too."""
        zip_path = _make_bundle(tmp_path, ["MAIN.USDA", "Scene.USD"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        # Priority still picks "main" via stem.lower(), even with uppercase ext.
        assert Path(main_usd).name == "MAIN.USDA"

    def test_empty_bundle_raises(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["README.md", "notes/info.txt"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="No USD layer found"):
            _extract_zip_bundle(str(zip_path), str(extracted))

    def test_rejects_path_traversal_entries(self, tmp_path: Path):
        """A malicious bundle must not be able to write outside extract_dir."""
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            zf.writestr("../escape.usda", '#usda 1.0\ndef Xform "Bad" {}\n')

        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="unsafe entry path"):
            _extract_zip_bundle(str(zip_path), str(extracted))

        # And the escape target must not have been written.
        assert not (tmp_path / "escape.usda").exists()

    def test_rejects_zip_bomb_by_entry_count(self, tmp_path: Path):
        """Too many entries trips the ZIP-bomb guard before extraction."""
        zip_path = tmp_path / "bomb.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            # Writing _ZIP_MAX_FILES+1 real entries is slow; patch the
            # central directory by emitting tiny entries beyond the limit.
            for i in range(_ZIP_MAX_FILES):
                zf.writestr(f"pad_{i}.bin", b"")

        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="too many entries"):
            _extract_zip_bundle(str(zip_path), str(extracted))

    def test_rejects_zip_bomb_by_uncompressed_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Oversized uncompressed total trips the guard before extraction."""
        zip_path = tmp_path / "bomb.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')

        # Drive the threshold down to something we can exceed trivially.
        monkeypatch.setattr("service.renderer._ZIP_MAX_UNCOMPRESSED_BYTES", 8)

        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="uncompressed size too large"):
            _extract_zip_bundle(str(zip_path), str(extracted))

    def test_rejects_symlink_entries(self, tmp_path: Path):
        """Symlink entries in the ZIP must not be materialized on disk."""
        zip_path = tmp_path / "symlinks.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            # Symlink entry pointing at an absolute host path. Encode the
            # Unix symlink mode (0xA1FF = S_IFLNK|0o777) in external_attr,
            # mirroring what unzip / Info-ZIP writes.
            link = zipfile.ZipInfo("link.usda")
            link.create_system = 3  # Unix
            link.external_attr = (0xA1FF) << 16
            zf.writestr(link, "/etc/passwd")

        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="symlink entry"):
            _extract_zip_bundle(str(zip_path), str(extracted))

        assert not (extracted / "bundle" / "link.usda").exists()


class TestIsUsdzPayload:
    """Detection decides whether a ZIP is a .usdz package or a bundle to extract."""

    def _write_usdz_shape(self, zip_path: Path) -> None:
        """Write an archive matching the Pixar USDZ structural spec."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("root.usdc", b"PXR-USDC\x00\x00\x00\x00")
            zf.writestr("textures/albedo.png", b"\x89PNG\r\n\x1a\n")

    def _write_deflated_bundle(self, zip_path: Path) -> None:
        """Write a render_nvcf-style bundle (DEFLATED)."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            zf.writestr("textures/albedo.png", b"\x89PNG\r\n\x1a\n")

    def test_url_extension_wins_for_s3(self, tmp_path: Path):
        zip_path = tmp_path / "asset.usdz"
        self._write_usdz_shape(zip_path)

        assert _is_usdz_payload("s3://bucket/asset.usdz", str(zip_path)) is True
        assert _is_usdz_payload("https://host/path/asset.USDZ", str(zip_path)) is True

    def test_url_with_query_string(self, tmp_path: Path):
        zip_path = tmp_path / "asset.usdz"
        self._write_usdz_shape(zip_path)

        # Query parameters must not hide the .usdz suffix.
        assert (
            _is_usdz_payload(
                "https://host/asset.usdz?version=42&signed=yes", str(zip_path)
            )
            is True
        )

    def test_non_usdz_url_is_not_package(self, tmp_path: Path):
        """render_nvcf bundles upload as .zip; those must go through extraction."""
        zip_path = tmp_path / "bundle.zip"
        self._write_deflated_bundle(zip_path)

        assert _is_usdz_payload("https://bucket/bundle.zip", str(zip_path)) is False

    def test_url_wins_over_content(self, tmp_path: Path):
        """A .zip URL never takes the USDZ path even if contents look USDZ-shaped."""
        zip_path = tmp_path / "looks_like_usdz.zip"
        self._write_usdz_shape(zip_path)

        assert (
            _is_usdz_payload("https://host/looks_like_usdz.zip", str(zip_path)) is False
        )

    def test_data_uri_usdz_by_content(self, tmp_path: Path):
        """Data URIs have no path; fall back to structural signature."""
        zip_path = tmp_path / "payload.bin"
        self._write_usdz_shape(zip_path)

        assert _is_usdz_payload("data:application/zip;base64,", str(zip_path)) is True

    def test_data_uri_deflated_is_not_usdz(self, tmp_path: Path):
        """A DEFLATED client bundle delivered via data URI still extracts."""
        zip_path = tmp_path / "payload.bin"
        self._write_deflated_bundle(zip_path)

        assert _is_usdz_payload("data:application/zip;base64,", str(zip_path)) is False

    def test_data_uri_stored_but_non_usd_first(self, tmp_path: Path):
        """STORED alone is not enough — first entry must be a USD layer."""
        zip_path = tmp_path / "payload.bin"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("README.md", b"not usd")
            zf.writestr("root.usdc", b"PXR-USDC\x00\x00\x00\x00")

        assert _is_usdz_payload("data:application/zip;base64,", str(zip_path)) is False

    def test_data_uri_empty_zip(self, tmp_path: Path):
        zip_path = tmp_path / "payload.bin"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED):
            pass

        assert _is_usdz_payload("data:application/zip;base64,", str(zip_path)) is False
