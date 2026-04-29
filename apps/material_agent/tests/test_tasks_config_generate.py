# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GenerateConfigTask: manifest loading, _build_config, and _remap_material_paths."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from material_agent.tasks.config_generate import GenerateConfigTask


class TestBuildConfig:
    """Tests for GenerateConfigTask._build_config()."""

    def test_build_config_with_materials_manifest(self):
        """_build_config includes materials.path when manifest is provided."""
        task = GenerateConfigTask()
        config = task._build_config(
            pipeline_name="test",
            input_usd_path="input.usd",
            materials_library_path=None,
            materials_manifest="materials.yaml",
        )

        assert "materials" in config
        assert config["materials"]["path"] == "materials.yaml"
        # Should not have library_path or entries when using manifest
        assert "library_path" not in config["materials"]
        assert "entries" not in config["materials"]

    def test_build_config_with_library_path(self):
        """_build_config includes library_path and example entries when no manifest."""
        task = GenerateConfigTask()
        config = task._build_config(
            pipeline_name="test",
            input_usd_path="input.usd",
            materials_library_path="/path/to/library.usd",
            materials_manifest=None,
        )

        assert "materials" in config
        assert config["materials"]["library_path"] == "/path/to/library.usd"
        assert "entries" in config["materials"]
        assert len(config["materials"]["entries"]) > 0

    def test_build_config_without_materials(self):
        """_build_config omits materials section when neither manifest nor library."""
        task = GenerateConfigTask()
        config = task._build_config(
            pipeline_name="test",
            input_usd_path="input.usd",
            materials_library_path=None,
            materials_manifest=None,
        )

        assert "materials" not in config

    def test_build_config_with_reference_images(self):
        """_build_config includes reference_images in input section."""
        task = GenerateConfigTask()
        config = task._build_config(
            pipeline_name="test",
            input_usd_path="input.usd",
            materials_library_path=None,
            materials_manifest=None,
            reference_images=["ref1.jpg", "ref2.jpg"],
        )

        assert config["input"]["reference_images"] == ["ref1.jpg", "ref2.jpg"]

    def test_build_config_without_reference_images(self):
        """_build_config omits reference_images when empty."""
        task = GenerateConfigTask()
        config = task._build_config(
            pipeline_name="test",
            input_usd_path="input.usd",
            materials_library_path=None,
            materials_manifest=None,
        )

        assert "reference_images" not in config["input"]

    def test_build_config_always_produces_apply_mode(self):
        """_build_config always produces predict + apply steps (no refine mode)."""
        task = GenerateConfigTask()
        config = task._build_config(
            pipeline_name="test",
            input_usd_path="input.usd",
            materials_library_path=None,
            materials_manifest=None,
        )

        assert "predict" in config["steps"]
        assert "apply" in config["steps"]
        assert "refine" not in config["steps"]


class TestManifestValidation:
    """Tests for manifest loading/validation in GenerateConfigTask.run()."""

    def test_manifest_file_not_found_raises(self, tmp_path):
        """run() raises ValueError when manifest file doesn't exist."""
        task = GenerateConfigTask()
        context = {
            "output_config_path": str(tmp_path / "config.yaml"),
            "force": True,
            "materials_manifest": str(tmp_path / "nonexistent.yaml"),
        }

        with patch("material_agent.tasks.config_generate.typer") as mock_typer:
            mock_typer.prompt = MagicMock(
                side_effect=["test_pipeline", "input.usd", "output.usd"]
            )
            with pytest.raises(ValueError, match="Materials manifest file not found"):
                task.run(context)

    def _prompt_side_effects(self, *extra_prompts):
        """Build side_effect list for typer.prompt: pipeline, input, ref images, output."""
        # Prompts in order: pipeline_name, input_usd, ref_image (empty to stop), output_usd
        return ["test_pipeline", "input.usd", *extra_prompts, "", "output.usd"]

    def test_malformed_manifest_list_yaml(self, tmp_path):
        """run() handles YAML that parses as a list (not dict) without AttributeError."""
        manifest_path = tmp_path / "bad_manifest.yaml"
        manifest_path.write_text("- item1\n- item2\n")

        task = GenerateConfigTask()
        context = {
            "output_config_path": str(tmp_path / "config.yaml"),
            "force": True,
            "materials_manifest": str(manifest_path),
        }

        # The task should handle this gracefully (entries=[], no AttributeError)
        with patch("material_agent.tasks.config_generate.typer") as mock_typer:
            mock_typer.prompt = MagicMock(side_effect=self._prompt_side_effects())
            result = task.run(context)
            assert result["config_created"] is True

    def test_empty_manifest_yaml(self, tmp_path):
        """run() handles empty YAML manifest gracefully."""
        manifest_path = tmp_path / "empty_manifest.yaml"
        manifest_path.write_text("")

        task = GenerateConfigTask()
        context = {
            "output_config_path": str(tmp_path / "config.yaml"),
            "force": True,
            "materials_manifest": str(manifest_path),
        }

        with patch("material_agent.tasks.config_generate.typer") as mock_typer:
            mock_typer.prompt = MagicMock(side_effect=self._prompt_side_effects())
            result = task.run(context)
            assert result["config_created"] is True

    def test_valid_manifest_resolves_library_path(self, tmp_path):
        """run() resolves relative library_path from manifest."""
        manifest_path = tmp_path / "manifest.yaml"
        manifest_data = {
            "library_path": "libs/materials.usd",
            "entries": [
                {
                    "name": "Steel",
                    "description": "Shiny steel",
                    "binding": "/Looks/Steel",
                }
            ],
        }
        manifest_path.write_text(yaml.dump(manifest_data))

        task = GenerateConfigTask()
        context = {
            "output_config_path": str(tmp_path / "config.yaml"),
            "force": True,
            "materials_manifest": str(manifest_path),
        }

        with patch("material_agent.tasks.config_generate.typer") as mock_typer:
            mock_typer.prompt = MagicMock(side_effect=self._prompt_side_effects())
            result = task.run(context)
            expected = str(tmp_path / "libs" / "materials.usd")
            assert result["materials_library_path"] == expected

    def test_valid_manifest_with_absolute_library_path(self, tmp_path):
        """run() uses absolute library_path as-is from manifest."""
        manifest_path = tmp_path / "manifest.yaml"
        manifest_data = {
            "library_path": "/absolute/path/materials.usd",
            "entries": [{"name": "Steel", "description": "Steel", "binding": "/Steel"}],
        }
        manifest_path.write_text(yaml.dump(manifest_data))

        task = GenerateConfigTask()
        context = {
            "output_config_path": str(tmp_path / "config.yaml"),
            "force": True,
            "materials_manifest": str(manifest_path),
        }

        with patch("material_agent.tasks.config_generate.typer") as mock_typer:
            mock_typer.prompt = MagicMock(side_effect=self._prompt_side_effects())
            result = task.run(context)
            assert result["materials_library_path"] == "/absolute/path/materials.usd"


class TestLogRetrievalSummary:
    """Tests for MaterialRetrievalTask._log_retrieval_summary()."""

    def test_logs_summary_for_matched_materials(self):
        """Summary logs material names and match counts."""
        from material_agent.tasks.material_retrieval import MaterialRetrievalTask

        task = MaterialRetrievalTask()
        listener = MagicMock()

        matched = {
            "Steel": [{"source_path": "/path/steel.mdl", "s3_path": None}],
            "Rubber": [],
        }
        task._log_retrieval_summary(matched, listener)

        # Should have called info multiple times (header + per-material)
        assert listener.info.call_count >= 3

    def test_logs_empty_materials(self):
        """Summary handles empty materials dict."""
        from material_agent.tasks.material_retrieval import MaterialRetrievalTask

        task = MaterialRetrievalTask()
        listener = MagicMock()

        task._log_retrieval_summary({}, listener)

        # Should log "No materials were retrieved"
        calls = [str(c) for c in listener.info.call_args_list]
        assert any("No materials" in c for c in calls)
