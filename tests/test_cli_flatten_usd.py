# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the flatten-usd CLI command."""

from typer.testing import CliRunner

from world_understanding.cli import app

runner = CliRunner()


class TestFlattenUsdCommand:
    """Tests for wu flatten-usd CLI command."""

    def test_help(self):
        """Command --help works and shows description."""
        result = runner.invoke(app, ["flatten-usd", "--help"])
        assert result.exit_code == 0
        assert "Flatten a composed USD stage" in result.output

    def test_missing_source(self):
        """Missing source argument shows error."""
        result = runner.invoke(app, ["flatten-usd"])
        assert result.exit_code != 0

    def test_nonexistent_source(self):
        """Non-existent source file shows error."""
        result = runner.invoke(app, ["flatten-usd", "/nonexistent/file.usd"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_unsupported_extension(self, tmp_path):
        """Unsupported file extension shows error."""
        bad_file = tmp_path / "file.txt"
        bad_file.write_text("not usd")
        result = runner.invoke(app, ["flatten-usd", str(bad_file)])
        assert result.exit_code != 0
        assert "Unsupported" in result.output

    def test_flatten_simple_usda(self, tmp_path):
        """Flatten a simple USDA file."""
        from pxr import Usd, UsdGeom

        source = tmp_path / "test.usda"
        stage = Usd.Stage.CreateNew(str(source))
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.Save()

        dest = tmp_path / "test_flat.usda"
        result = runner.invoke(app, ["flatten-usd", str(source), str(dest), "--force"])
        assert result.exit_code == 0
        assert "successful" in result.output.lower()
        assert dest.exists()

    def test_flatten_default_destination(self, tmp_path):
        """Flatten with default destination creates <stem>_flat.<ext>."""
        from pxr import Usd, UsdGeom

        source = tmp_path / "scene.usda"
        stage = Usd.Stage.CreateNew(str(source))
        UsdGeom.Xform.Define(stage, "/Root")
        stage.Save()

        result = runner.invoke(app, ["flatten-usd", str(source)])
        assert result.exit_code == 0
        expected = tmp_path / "scene_flat.usda"
        assert expected.exists()

    def test_flatten_no_overwrite_without_force(self, tmp_path):
        """Existing destination without --force shows error."""
        from pxr import Usd

        source = tmp_path / "src.usda"
        dest = tmp_path / "dst.usda"
        stage = Usd.Stage.CreateNew(str(source))
        stage.Save()
        dest.write_text("existing")

        result = runner.invoke(app, ["flatten-usd", str(source), str(dest)])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_flatten_verbose(self, tmp_path):
        """Verbose mode shows additional stage info."""
        from pxr import Usd, UsdGeom

        source = tmp_path / "v.usda"
        stage = Usd.Stage.CreateNew(str(source))
        UsdGeom.Xform.Define(stage, "/World")
        stage.Save()

        dest = tmp_path / "v_flat.usda"
        result = runner.invoke(
            app, ["flatten-usd", str(source), str(dest), "--verbose"]
        )
        assert result.exit_code == 0
        assert "Total prims" in result.output
