# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from physics_agent.tasks.apply_physics import ApplyPhysicsTask
from physics_agent.tasks.config_apply_physics import ApplyPhysicsConfigTask
from physics_agent.tasks.config_identify_asset import IdentifyAssetConfigTask
from physics_agent.tasks.config_predict import PredictConfigTask
from physics_agent.tasks.config_prepare_dataset import PrepareDatasetConfigTask
from physics_agent.tasks.config_usd_dataset import USDDatasetConfigTask
from physics_agent.tasks.dataset_loading import DatasetLoadingTask
from physics_agent.tasks.predictions import SavePredictionsTask


class MemoryObjectStore:
    def __init__(self, data: dict[str, object] | None = None):
        self._data = dict(data or {})

    def exists(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._data.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._data[key] = value


def write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_predict_config_task_loads_dataset_and_system_prompt(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "dataset.jsonl"
    dataset_path.write_text(json.dumps({"id": "prim-1"}) + "\n", encoding="utf-8")
    (dataset_dir / "dataset.json").write_text(
        json.dumps({"system_prompt": "classify carefully"}),
        encoding="utf-8",
    )

    config_path = tmp_path / "predict.yaml"
    write_yaml(
        config_path,
        {
            "dataset": "dataset/dataset.jsonl",
            "output_dir": "predictions",
            "vlm": {"model": "test-model"},
            "llm": {"model": "parser-model"},
            "report": {
                "image_max_size": 256,
                "image_format": "jpeg",
                "image_quality": 72,
            },
            "output_key": "asset_class",
            "allow_empty_predictions": True,
        },
    )

    result = PredictConfigTask().run({"config_path": str(config_path), "resume": True})

    assert result["dataset"] == [{"id": "prim-1"}]
    assert result["dataset_path"] == str(dataset_path.resolve())
    assert result["output_dir"] == str((tmp_path / "predictions").resolve())
    assert result["image_base_dir"] == str(dataset_dir.resolve())
    assert result["system_prompt"] == "classify carefully"
    assert result["output_key"] == "asset_class"
    assert result["allow_empty_predictions"] is True
    assert result["report_image_max_size"] == 256
    assert result["report_image_format"] == "jpeg"
    assert result["report_image_quality"] == 72
    assert result["resume"] is True


def test_predict_config_task_validates_allow_empty_predictions(tmp_path: Path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps({"id": "prim-1"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="allow_empty_predictions"):
        PredictConfigTask().run(
            {
                "config_dict": {
                    "dataset": str(dataset_path),
                    "allow_empty_predictions": "yes",
                }
            }
        )


def test_prepare_and_usd_dataset_config_tasks_resolve_paths(tmp_path: Path):
    usd_dir = tmp_path / "usd_dataset"
    usd_dir.mkdir()
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"ref")

    prepare_config_path = tmp_path / "prepare.yaml"
    write_yaml(
        prepare_config_path,
        {
            "usd_dir": "usd_dataset",
            "dataset": "prepared",
            "models": ["001", "002"],
            "reference_images": ["reference.png"],
            "prompts": {"system": "hello"},
            "include_prim_path_context": False,
            "include_geometric_context": True,
        },
    )

    prepare_result = PrepareDatasetConfigTask().run(
        {"config_path": str(prepare_config_path)}
    )
    assert prepare_result["usd_dir"] == str(usd_dir.resolve())
    assert prepare_result["dataset_path"] == str((tmp_path / "prepared").resolve())
    assert prepare_result["models"] == ["001", "002"]
    assert prepare_result["reference_images"] == [str(reference.resolve())]
    assert prepare_result["prompts"]["system"] == "hello"
    assert prepare_result["include_prim_path_context"] is False
    assert prepare_result["include_geometric_context"] is True

    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    usd_dataset_result = USDDatasetConfigTask().run(
        {
            "config_dict": {
                "usd_path": str(usd_path),
                "renderer": {"backend": "remote"},
                "prim_filters": {"types": ["UsdGeom.Mesh"]},
            }
        }
    )
    assert usd_dataset_result["usd_path"] == str(usd_path)
    assert Path(usd_dataset_result["output_dir"]).parts[-2:] == ("dataset", "usd")
    assert usd_dataset_result["renderer"]["backend"] == "remote"
    assert usd_dataset_result["prim_filters"]["types"] == ["UsdGeom.Mesh"]


def test_identify_asset_config_task_applies_defaults_and_relative_paths(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    config_path = tmp_path / "identify.yaml"
    write_yaml(
        config_path,
        {
            "usd_path": "asset.usd",
            "renderer": {"backend": "ovrtx", "image_width": 256},
            "prompts": {"system": "identify it"},
        },
    )

    result = IdentifyAssetConfigTask().run({"config_path": str(config_path)})

    assert result["usd_path"] == str(usd_path.resolve())
    assert result["output_dir"] == str((tmp_path / "identification").resolve())
    assert result["render_config"]["backend"] == "ovrtx"
    assert result["render_config"]["image_width"] == 256
    assert result["render_config"]["image_height"] == 512
    assert result["identify_system_prompt"] == "identify it"


def test_apply_physics_config_task_validates_mass_scale_policy(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    config_path = tmp_path / "apply.yaml"
    write_yaml(
        config_path,
        {
            "usd_path": "asset.usd",
            "predictions_path": "predictions.jsonl",
            "output_usd_path": "out.usda",
            "mass_scale_policy": "skip_mass",
            "allow_empty_predictions": True,
        },
    )

    result = ApplyPhysicsConfigTask().run({"config_path": str(config_path)})

    assert result["mass_scale_policy"] == "skip_mass"
    assert result["allow_empty_predictions"] is True
    assert result["usd_path"] == str(usd_path.resolve())

    default_result = ApplyPhysicsConfigTask().run(
        {
            "config_dict": {
                "usd_path": str(usd_path),
                "predictions_path": str(predictions_path),
                "output_usd_path": str(tmp_path / "default_out.usda"),
            }
        }
    )
    assert default_result["mass_scale_policy"] == "skip_mass"
    assert default_result["allow_empty_predictions"] is False

    with pytest.raises(ValueError, match="mass_scale_policy"):
        ApplyPhysicsConfigTask().run(
            {
                "config_dict": {
                    "usd_path": str(usd_path),
                    "predictions_path": str(predictions_path),
                    "output_usd_path": str(tmp_path / "out.usda"),
                    "mass_scale_policy": "bad",
                }
            }
        )

    with pytest.raises(ValueError, match="allow_empty_predictions"):
        ApplyPhysicsConfigTask().run(
            {
                "config_dict": {
                    "usd_path": str(usd_path),
                    "predictions_path": str(predictions_path),
                    "output_usd_path": str(tmp_path / "out.usda"),
                    "allow_empty_predictions": "yes",
                }
            }
        )


def test_apply_physics_task_forwards_authoring_policies(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_apply_physics(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "out.usda")

    monkeypatch.setattr(
        "physics_agent.tasks.apply_physics.apply_physics",
        fake_apply_physics,
    )

    result = ApplyPhysicsTask().run(
        {
            "usd_path": str(tmp_path / "in.usda"),
            "predictions_path": str(tmp_path / "predictions.jsonl"),
            "output_usd_path": str(tmp_path / "out.usda"),
            "collision_approx": "none",
            "output_key": "analysis",
            "mass_scale_policy": "fail",
            "allow_empty_predictions": True,
        }
    )

    assert result["output_usd_path"] == str(tmp_path / "out.usda")
    assert captured["collision_approx"] == "none"
    assert captured["output_key"] == "analysis"
    assert captured["mass_scale_policy"] == "fail"
    assert captured["allow_empty_predictions"] is True


def test_dataset_loading_task_validates_entries_and_updates_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("", encoding="utf-8")
    valid_image = tmp_path / "valid.png"
    valid_image.write_bytes(b"image")

    task = DatasetLoadingTask()

    class StubListener:
        def __init__(self) -> None:
            self.warnings: list[str] = []

        def warning(self, message: str) -> None:
            self.warnings.append(message)

        def debug(self, message: str) -> None:
            pass

    listener = StubListener()
    task._listener = listener

    valid_entry = {"id": "a", "media": {"images": [{"path": "valid.png"}]}}
    invalid_entry = {"id": "b", "images": ["missing.png"]}
    assert task._validate_entry(valid_entry, dataset_path, listener) is True
    assert task._validate_entry(invalid_entry, dataset_path, listener) is False

    filtered = task._validate_dataset([valid_entry, invalid_entry], dataset_path)
    assert filtered == [valid_entry]
    assert listener.warnings

    context: dict[str, object] = {}
    task._update_context(context, filtered, dataset_path, {"entries": 1})
    assert context["dataset"] == filtered
    assert context["dataset_path"] == str(dataset_path)
    assert context["image_base_dir"] == str(tmp_path)

    def fake_base_run(self, context, object_store=None):
        context["base_run_called"] = True
        return context

    monkeypatch.setattr(
        "world_understanding.agentic.dataset.BaseDatasetLoadingTask.run", fake_base_run
    )
    result = task.run({"dataset_path": str(dataset_path)})
    assert result["base_run_called"] is True


def test_save_predictions_task_supports_existing_file_and_object_store(tmp_path: Path):
    task = SavePredictionsTask()
    predictions_path = tmp_path / "existing.jsonl"
    predictions_path.write_text("", encoding="utf-8")

    already_saved = task.run(
        {"predictions_path": str(predictions_path), "predictions_count": 3}
    )
    assert already_saved["predictions_saved"] is True

    object_store = MemoryObjectStore(
        {
            "predictions": [
                {"id": "prim-1", "vlm_response": {"label": "metal"}},
                {"id": "prim-2", "vlm_response": {"label": "plastic"}},
            ]
        }
    )
    result = task.run(
        {"output_dir": str(tmp_path), "output_key": "asset_class"}, object_store
    )

    output_path = tmp_path / "predictions.jsonl"
    saved_lines = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert output_path.exists()
    assert result["predictions_saved"] is True
    assert saved_lines[0]["asset_class"] == {"label": "metal"}

    no_predictions = task.run({"output_dir": str(tmp_path)})
    assert no_predictions["predictions_saved"] is False


def test_save_predictions_task_adds_mass_scale_quality_warnings(tmp_path: Path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "metadata": {"world_bbox_meters": {"size": [8.0, 0.8, 0.8]}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    object_store = MemoryObjectStore(
        {
            "predictions": [
                {
                    "id": "prim-1",
                    "vlm_response": {
                        "physical_properties": {
                            "density": 2700,
                            "estimated_mass_kg": 25000,
                        }
                    },
                }
            ],
        }
    )

    SavePredictionsTask().run(
        {
            "output_dir": str(tmp_path),
            "output_key": "classification",
            "dataset_path": str(dataset_path),
        },
        object_store,
    )

    saved = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8"))

    assert saved["quality_warnings"][0]["code"] == "mass_scale_suspicious"


def test_save_predictions_task_unwraps_nested_output_key_payload(tmp_path: Path):
    object_store = MemoryObjectStore(
        {
            "predictions": [
                {
                    "id": "prim-1",
                    "vlm_response": {
                        "classification": {
                            "component_type": "optical",
                            "physical_properties": {"density": 2500},
                        },
                        "original_response": "raw VLM output",
                    },
                }
            ],
        }
    )

    SavePredictionsTask().run(
        {"output_dir": str(tmp_path), "output_key": "classification"}, object_store
    )

    saved = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8"))

    assert saved["classification"] == {
        "component_type": "optical",
        "physical_properties": {"density": 2500},
        "original_response": "raw VLM output",
    }


def test_save_predictions_task_preserves_existing_quality_warnings(tmp_path: Path):
    object_store = MemoryObjectStore(
        {
            "predictions": [
                {
                    "id": "prim-1",
                    "vlm_response": {"physical_properties": {}},
                    "quality_warnings": [
                        {
                            "code": "custom_quality_check",
                            "severity": "warning",
                            "message": "custom warning",
                        }
                    ],
                }
            ],
        }
    )

    SavePredictionsTask().run(
        {"output_dir": str(tmp_path), "output_key": "classification"}, object_store
    )

    saved = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8"))

    assert saved["quality_warnings"] == [
        {
            "code": "custom_quality_check",
            "severity": "warning",
            "message": "custom warning",
        }
    ]
