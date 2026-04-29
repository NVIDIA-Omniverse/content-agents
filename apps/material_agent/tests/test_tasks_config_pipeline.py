# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material_agent.tasks.config_pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from material_agent.tasks.config_pipeline import PipelineConfigTask


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


class TestPipelineConfigTask:
    def test_run_requires_config_path(self) -> None:
        with pytest.raises(ValueError, match="config_path"):
            PipelineConfigTask().run({})

    def test_run_rejects_missing_and_empty_files(self, tmp_path: Path) -> None:
        task = PipelineConfigTask()

        with pytest.raises(FileNotFoundError):
            task.run({"config_path": str(tmp_path / "missing.yaml")})

        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")
        with pytest.raises(ValueError, match="empty"):
            task.run({"config_path": str(empty_config)})

    def test_run_loads_metadata_and_injects_materials(self, tmp_path: Path) -> None:
        task = PipelineConfigTask()
        predict_config = _write_yaml(tmp_path / "predict.yaml", {"prompt": "hello"})
        config_path = _write_yaml(
            tmp_path / "pipeline.yaml",
            {
                "pipeline": {
                    "name": "demo",
                    "description": "desc",
                    "working_dir": "work",
                    "keep_temp_files": False,
                },
                "materials": {
                    "library_path": "materials.usd",
                    "entries": [
                        {
                            "name": "Steel",
                            "description": "Brushed steel",
                            "binding": "/World/Looks/Steel",
                        },
                        {"name": "Plastic", "binding": "/World/Looks/Plastic"},
                    ],
                },
                "build_dataset_prepare_dataset": {"prompts": {}},
                "predict": {"config": str(predict_config.name)},
                "benchmark": {"ignored": True},
                "validate_predictions": {},
                "harmonize_predictions": {},
                "apply": {},
                "refine": {},
            },
        )

        context = task.run(
            {
                "config_path": str(config_path),
                "skip_steps": ["apply"],
            }
        )

        assert context["pipeline_name"] == "demo"
        assert context["pipeline_description"] == "desc"
        assert context["working_dir"] == (tmp_path / "work").resolve()
        assert context["keep_temp_files"] is False
        assert context["steps_to_run"] == [
            "build_dataset_prepare_dataset",
            "predict",
            "validate_predictions",
            "harmonize_predictions",
            "refine",
        ]

        materials_data = context["materials_data"]
        assert materials_data["library_path"] == str(
            (tmp_path / "materials.usd").resolve()
        )

        prepare_config = context["step_configs"]["build_dataset_prepare_dataset"]
        assert prepare_config["materials_list"] == ["Steel", "Plastic"]
        assert "Steel: Brushed steel" in prepare_config["_materials_formatted"]

        predict_step = context["step_configs"]["predict"]
        assert predict_step["_external_config_path"] == predict_config

        validate_step = context["step_configs"]["validate_predictions"]
        assert validate_step["material_names"] == ["Steel", "Plastic"]

        harmonize_step = context["step_configs"]["harmonize_predictions"]
        assert harmonize_step["material_names"] == ["Steel", "Plastic"]

        refine_step = context["step_configs"]["refine"]
        assert (
            refine_step["apply"]["materials_mapping"]["Steel"] == "/World/Looks/Steel"
        )
        assert (
            refine_step["apply"]["materials_mapping"]["material_library_path"]
            == materials_data["library_path"]
        )

    def test_run_errors_when_no_steps_survive_filters(self, tmp_path: Path) -> None:
        task = PipelineConfigTask()
        config_path = _write_yaml(tmp_path / "pipeline.yaml", {"predict": {}})

        with pytest.raises(ValueError, match="No valid steps"):
            task.run(
                {
                    "config_path": str(config_path),
                    "only_steps": ["apply"],
                }
            )

    def test_parse_materials_validates_shape_and_resolves_paths(
        self, tmp_path: Path
    ) -> None:
        task = PipelineConfigTask()
        config_path = tmp_path / "pipeline.yaml"

        assert task._parse_materials({}, config_path) is None

        parsed = task._parse_materials(
            {
                "materials": {
                    "library_path": "materials.usd",
                    "entries": [
                        {
                            "name": "Steel",
                            "description": "Shiny",
                            "binding": "/Looks/Steel",
                        }
                    ],
                }
            },
            config_path,
        )
        assert parsed == {
            "library_path": str((tmp_path / "materials.usd").resolve()),
            "entries": [
                {
                    "name": "Steel",
                    "description": "Shiny",
                    "binding": "/Looks/Steel",
                }
            ],
        }

        with pytest.raises(ValueError, match="must be a dictionary"):
            task._parse_materials({"materials": "bad"}, config_path)

        with pytest.raises(ValueError, match="must be a list"):
            task._parse_materials({"materials": {"entries": "bad"}}, config_path)

        with pytest.raises(ValueError, match="must be a dictionary"):
            task._parse_materials({"materials": {"entries": ["bad"]}}, config_path)

        with pytest.raises(ValueError, match="missing 'name'"):
            task._parse_materials(
                {"materials": {"entries": [{"binding": "/Looks/Steel"}]}}, config_path
            )

    def test_process_steps_honors_predict_benchmark_skip_and_only(
        self, tmp_path: Path
    ) -> None:
        task = PipelineConfigTask()
        listener = Mock()
        config_path = tmp_path / "pipeline.yaml"

        steps_to_run, step_configs = task._process_steps(
            {
                "predict": {"enabled": True},
                "benchmark": {"enabled": True},
                "apply": {"enabled": True},
            },
            config_path,
            tmp_path,
            {"skip_steps": ["apply"], "only_steps": ["predict", "apply"]},
            None,
            listener,
        )

        assert steps_to_run == ["predict"]
        assert step_configs["predict"]["_inline_config"] is True
        listener.warning.assert_called_once()
        listener.info.assert_any_call("Skipping step: apply (--skip)")

    def test_resolve_step_config_handles_external_and_inline_configs(
        self, tmp_path: Path
    ) -> None:
        task = PipelineConfigTask()
        listener = Mock()
        config_path = tmp_path / "pipeline.yaml"

        external = _write_yaml(tmp_path / "predict.yaml", {"value": 1})
        resolved = task._resolve_step_config(
            "predict",
            {"config": external.name},
            config_path,
            tmp_path,
            listener,
        )
        assert resolved["value"] == 1
        assert resolved["_external_config_path"] == external

        inline = task._resolve_step_config(
            "apply",
            {"enabled": True},
            config_path,
            tmp_path,
            listener,
        )
        assert inline["_inline_config"] is True
        assert inline["_pipeline_config_path"] == config_path

        with pytest.raises(FileNotFoundError, match="not found"):
            task._resolve_step_config(
                "predict",
                {"config": "missing.yaml"},
                config_path,
                tmp_path,
                listener,
            )

        empty_external = _write_yaml(tmp_path / "empty.yaml", {})
        with pytest.raises(ValueError, match="is empty"):
            task._resolve_step_config(
                "predict",
                {"config": empty_external.name},
                config_path,
                tmp_path,
                listener,
            )

    def test_inject_materials_into_steps(self) -> None:
        task = PipelineConfigTask()
        listener = Mock()
        materials_data = {
            "library_path": "/tmp/materials.usd",
            "entries": [
                {
                    "name": "Steel",
                    "description": "Brushed",
                    "binding": "/Looks/Steel",
                },
                {"name": "Plastic", "binding": "/Looks/Plastic"},
            ],
        }

        prepare = task._inject_materials_into_step(
            "build_dataset_prepare_dataset",
            {"prompts": {}},
            materials_data,
            listener,
        )
        assert prepare["materials_list"] == ["Steel", "Plastic"]
        assert prepare["_materials_formatted"] == "Steel: Brushed\nPlastic"

        validate = task._inject_materials_into_step(
            "validate_predictions", {}, materials_data, listener
        )
        assert validate["material_names"] == ["Steel", "Plastic"]

        harmonize = task._inject_materials_into_step(
            "harmonize_predictions", {}, materials_data, listener
        )
        assert harmonize["material_names"] == ["Steel", "Plastic"]

        apply = task._inject_materials_into_step("apply", {}, materials_data, listener)
        assert (
            apply["materials_mapping"]["material_library_path"] == "/tmp/materials.usd"
        )
        assert apply["materials_mapping"]["Steel"] == "/Looks/Steel"

        refine = task._inject_materials_into_step(
            "refine", {}, materials_data, listener
        )
        assert refine["apply"]["materials_mapping"]["Plastic"] == "/Looks/Plastic"

        untouched_apply = task._inject_materials_into_step(
            "apply",
            {"materials_mapping": {"existing": "keep"}},
            materials_data,
            listener,
        )
        assert untouched_apply["materials_mapping"] == {"existing": "keep"}

    def test_format_materials_for_prompt(self) -> None:
        assert (
            PipelineConfigTask()._format_materials_for_prompt(
                [
                    {"name": "Steel", "description": "Brushed"},
                    {"name": "Plastic"},
                ]
            )
            == "Steel: Brushed\nPlastic"
        )
