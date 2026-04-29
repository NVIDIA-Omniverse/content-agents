# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import yaml

from physics_agent.tasks.identify_asset import IdentifyAssetTask
from physics_agent.tasks.inference import VLMInferenceTask
from physics_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask
from physics_agent.workflows import (
    create_identify_asset_workflow_from_config,
    create_optimize_usd_workflow_from_config,
    create_prediction_workflow_from_config,
    create_prepare_dataset_workflow_from_config,
    create_restore_usd_workflow_from_config,
    create_unified_pipeline_workflow,
    create_usd_data_preparation_workflow_from_config,
)


class RecordingListener:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, name: str, payload: dict[str, object]) -> None:
        self.events.append((name, payload))

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass


class MemoryObjectStore:
    def __init__(self, data: dict[str, object] | None = None):
        self._data = dict(data or {})

    def exists(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._data.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._data[key] = value


class FakeVLM:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_workflow_factories_build_expected_task_sequences():
    assert [task.name for task in create_prediction_workflow_from_config().tasks] == [
        "PredictConfig",
        "ModelProvisioning",
        "DatasetLoading",
        "VLMInference",
        "GeneratePredictionReport",
        "SavePredictions",
    ]
    assert [
        task.name for task in create_prepare_dataset_workflow_from_config().tasks
    ] == [
        "PrepareDatasetConfig",
        "PrepareDataset",
    ]
    assert [
        task.name for task in create_identify_asset_workflow_from_config().tasks
    ] == [
        "IdentifyAssetConfig",
        "RenderScenePreview",
        "ModelProvisioning",
        "IdentifyAsset",
    ]
    assert [
        task.__class__.__name__
        for task in create_optimize_usd_workflow_from_config().tasks
    ] == ["OptimizeUSDConfigTask", "OptimizeUSDTask"]
    assert [
        task.__class__.__name__
        for task in create_restore_usd_workflow_from_config().tasks
    ] == ["RestoreUSDConfigTask", "RestoreUSDTask"]
    assert (
        create_usd_data_preparation_workflow_from_config().name
        == "USD → Dataset Preparation"
    )
    assert [task.name for task in create_unified_pipeline_workflow().tasks] == [
        "UnifiedConfigLoading",
        "UnifiedPipelineExecutor",
    ]


def test_identify_asset_task_parses_and_saves_results(tmp_path: Path):
    output_dir = tmp_path / "identify"
    vlm = FakeVLM(
        """```json
        {"asset_type":"vehicle","asset_subtype":"forklift","confidence":"high"}
        ```"""
    )
    images = [str(tmp_path / f"img-{i}.png") for i in range(7)]

    result = IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": images,
            "identify_system_prompt": "identify it",
            "output_dir": str(output_dir),
        }
    )

    saved = json.loads((output_dir / "identification.json").read_text(encoding="utf-8"))
    assert result["identification"]["asset_type"] == "vehicle"
    assert saved["asset_subtype"] == "forklift"
    assert len(vlm.calls[0]["images"]) == 6

    no_images = IdentifyAssetTask().run({"vlm": vlm, "output_dir": str(output_dir)})
    assert no_images["identification"]["asset_type"] == "unknown"


def test_vlm_inference_task_streams_predictions_and_supports_resume(
    tmp_path: Path, monkeypatch
):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_entries = [
        {"id": "prim-1", "images": ["a.png"]},
        {"id": "prim-2", "image_path": "b.png"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(entry) for entry in dataset_entries) + "\n",
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "prim-1", "classification": {"label": "old"}}) + "\n",
        encoding="utf-8",
    )
    listener = RecordingListener()

    captured: dict[str, object] = {}

    def fake_batch_classify_assets(**kwargs):
        captured["processed_ids"] = kwargs["processed_ids"]
        captured["max_workers"] = kwargs["max_workers"]
        kwargs["on_progress"]("prim-2", "ok")
        kwargs["on_prediction"](
            "prim-2", {"classification": {"label": "metal"}, "confidence": "high"}
        )
        kwargs["on_result"](
            {"id": "prim-2", "status": "success", "vlm_response": {"label": "metal"}},
            dataset_entries[1],
        )
        return [
            {"id": "prim-2", "status": "success", "vlm_response": {"label": "metal"}},
            {"id": "prim-3", "status": "error", "error": "failed"},
        ]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    object_store = MemoryObjectStore({"dataset": dataset_entries})
    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "resume": True,
            "output_key": "classification",
            "vlm_config": {"max_retries": 5},
            "max_workers": 8,
            "event_listener": listener,
        },
        object_store,
    )

    saved_lines = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]
    assert result["predictions_count"] == 2
    assert result["failed_count"] == 1
    assert result["inference_complete"] is True
    assert captured["processed_ids"] == {"prim-1"}
    assert captured["max_workers"] == 8
    assert saved_lines[-1]["classification"] == {"label": "metal"}
    assert any(event[0] == "prediction.completed" for event in listener.events)


def test_unified_pipeline_executor_autowires_steps_and_collects_outputs(
    tmp_path: Path, monkeypatch
):
    executor = UnifiedPipelineExecutorTask()
    listener = RecordingListener()
    recorded_configs: dict[str, dict[str, object]] = {}
    optimize_output = tmp_path / "optimized.usdc"
    original_usd = tmp_path / "original.usd"
    original_usd.write_text("#usda 1.0\n", encoding="utf-8")

    step_results = {
        "optimize_usd": {
            "optimized_usd_path": str(optimize_output),
            "optimization_metadata": {"mode": "split"},
            "optimization_success": True,
            "original_usd_path": str(original_usd),
        },
        "build_dataset_usd": {"output_dir": str(tmp_path / "dataset-usd")},
        "identify_asset": {
            "identification": {"asset_type": "vehicle", "asset_subtype": "forklift"},
            "identification_path": str(tmp_path / "identification.json"),
        },
        "build_dataset_prepare_dataset": {
            "dataset_path": str(tmp_path / "dataset"),
            "dataset_jsonl_path": str(tmp_path / "dataset" / "dataset.jsonl"),
        },
        "predict": {
            "predictions_path": str(tmp_path / "predictions.jsonl"),
            "predictions_count": 1,
            "output_key": "classification",
        },
        "restore_usd": {
            "restored_predictions_path": str(tmp_path / "restored_predictions.jsonl"),
            "restore_success": True,
            "predictions_count": 1,
            "restore_stats": {"restored": 1},
        },
        "apply_physics": {
            "output_usd_path": str(tmp_path / "physics" / "original_physics.usda")
        },
    }

    class FakeWorkflow:
        def __init__(self, step_name: str) -> None:
            self.step_name = step_name

        def run(self, context: dict[str, object]) -> dict[str, object]:
            config_path = Path(context["config_path"])
            recorded_configs[self.step_name] = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
            return step_results[self.step_name]

    import physics_agent.workflows as workflows_module

    monkeypatch.setattr(
        workflows_module,
        "create_optimize_usd_workflow_from_config",
        lambda: FakeWorkflow("optimize_usd"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_usd_data_preparation_workflow_from_config",
        lambda: FakeWorkflow("build_dataset_usd"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_identify_asset_workflow_from_config",
        lambda: FakeWorkflow("identify_asset"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_prepare_dataset_workflow_from_config",
        lambda: FakeWorkflow("build_dataset_prepare_dataset"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_prediction_workflow_from_config",
        lambda: FakeWorkflow("predict"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_restore_usd_workflow_from_config",
        lambda: FakeWorkflow("restore_usd"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_apply_physics_workflow_from_config",
        lambda: FakeWorkflow("apply_physics"),
    )

    context = {
        "steps_to_run": [
            "optimize_usd",
            "build_dataset_usd",
            "identify_asset",
            "build_dataset_prepare_dataset",
            "predict",
            "restore_usd",
            "apply_physics",
        ],
        "step_configs": {
            "optimize_usd": {"renderer": {"backend": "remote", "_internal": "omit"}},
            "build_dataset_usd": {"usd_path": str(original_usd)},
            "identify_asset": {"usd_path": str(original_usd)},
            "build_dataset_prepare_dataset": {
                "prompts": {"system": "Classify the part."}
            },
            "predict": {"report": {"image_format": "jpeg", "image_quality": 72}},
            "restore_usd": {},
            "apply_physics": {"usd_path": str(original_usd)},
        },
        "working_dir": str(tmp_path / "run"),
        "session_id": "session-1",
        "project_name": "physics-demo",
        "event_listener": listener,
    }

    result = executor.run(context)

    assert result["pipeline_state"] == "completed"
    assert result["pipeline_results"]["predict"]["predictions_count"] == 1
    assert recorded_configs["build_dataset_usd"]["usd_path"] == str(optimize_output)
    assert recorded_configs["identify_asset"]["usd_path"] == str(optimize_output)
    assert recorded_configs["build_dataset_prepare_dataset"]["prompts"][
        "system"
    ].startswith("This is a vehicle (forklift). ")
    assert recorded_configs["predict"]["report"]["image_format"] == "jpeg"
    assert recorded_configs["restore_usd"]["original_usd_path"] == str(original_usd)
    assert recorded_configs["restore_usd"]["predictions_path"] == str(
        tmp_path / "predictions.jsonl"
    )
    assert recorded_configs["restore_usd"]["output_predictions_path"].endswith(
        "restored_predictions.jsonl"
    )
    assert recorded_configs["apply_physics"]["usd_path"] == str(optimize_output)
    assert recorded_configs["apply_physics"]["predictions_path"] == str(
        tmp_path / "predictions.jsonl"
    )
    assert any(event[0] == "pipeline.completed" for event in listener.events)
