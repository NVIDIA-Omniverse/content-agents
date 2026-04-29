# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from physics_agent.tasks.prepare_dataset import PrepareDatasetTask
from physics_agent.tasks.reporting import GeneratePredictionReportTask


class MemoryObjectStore:
    def __init__(self, data: dict[str, object] | None = None):
        self._data = dict(data or {})

    def exists(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._data.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._data[key] = value


def test_prepare_dataset_task_builds_v02_dataset_with_context_and_failures(
    tmp_path: Path,
):
    usd_dir = tmp_path / "usd"
    model_dir = usd_dir / "."
    model_dir.mkdir(parents=True)

    renders_dir = model_dir / "renders"
    renders_dir.mkdir()
    (renders_dir / "composition.png").write_bytes(b"composition")
    (renders_dir / "prim_only.png").write_bytes(b"prim")

    (model_dir / "dataset.json").write_text(
        json.dumps({"statistics": {"total_prims": 1}}),
        encoding="utf-8",
    )
    prim_record = {
        "prim_path": "/World/Body",
        "world_bbox_meters": {"size": [1.0, 2.0, 3.0]},
        "relative_metrics": {"relative_size": [0.1, 0.2, 0.3]},
        "renders": [
            {
                "path": "renders/composition.png",
                "view": "+x+y+z",
                "camera": "cam-a",
                "render_mode": "composition",
            },
            {
                "path": "renders/prim_only.png",
                "view": "-x-y-z",
                "camera": "cam-b",
                "render_mode": "prim_only",
            },
        ],
    }
    (model_dir / "prims.jsonl").write_text(
        json.dumps(prim_record) + "\n",
        encoding="utf-8",
    )

    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    structure_assignments = tmp_path / "assignments.json"
    structure_assignments.write_text(
        json.dumps(
            {
                "assignments": {
                    "/World/Body": {"component_name": "chassis"},
                }
            }
        ),
        encoding="utf-8",
    )

    dataset_path = tmp_path / "prepared"
    result = PrepareDatasetTask().run(
        {
            "usd_dir": str(usd_dir),
            "dataset_path": str(dataset_path),
            "models": [".", "missing-model"],
            "reference_images": [str(reference)],
            "config": {
                "include_prim_path_context": True,
                "include_geometric_context": True,
                "structure_assignments_path": str(structure_assignments),
                "render_mode_filter": ["composition", "prim_only"],
                "prompts": {
                    "system": "system prompt",
                    "user": "Describe the component.",
                    "vlm_image_prompts": [
                        {"composition": "Composition view."},
                        {"prim_only": "Prim-only view."},
                    ],
                },
            },
        }
    )

    assert result["failed_models"] == ["missing-model"]
    assert result["dataset_jsonl_path"].endswith("dataset.jsonl")
    assert result["dataset_config_path"].endswith("dataset.json")
    assert len(result["dataset_entries"]) == 1

    entry = result["dataset_entries"][0]
    assert entry["id"] == "/World/Body"
    assert "The prim path of this 3D asset is: /World/Body" in entry["user_prompt"]
    assert "Bounding box volume" in entry["user_prompt"]
    assert "chassis" in entry["user_prompt"]

    media_images = entry["media"]["images"]
    assert media_images[0]["type"] == "reference"
    assert (
        media_images[0]["metadata"]["vlm_prompt"]
        == "This is a reference image of the asset."
    )
    assert media_images[1]["metadata"]["render_mode"] == "composition"
    assert "Camera Position" in media_images[1]["metadata"]["vlm_prompt"]

    dataset_config = json.loads((dataset_path / "dataset.json").read_text())
    assert dataset_config["metadata"]["num_entries"] == 1
    assert dataset_config["inference"]["prompts"][0]["system_prompt"] == "system prompt"


def test_generate_prediction_report_task_creates_html_report(
    tmp_path: Path, monkeypatch
):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "classification": {
                    "component_type": "panel",
                    "component_name": "hood",
                    "material": "metal",
                    "confidence": "high",
                    "physical_properties": {
                        "density": 7.8,
                        "static_friction": 0.3,
                        "dynamic_friction": 0.2,
                        "restitution": 0.1,
                    },
                    "original_response": "raw output",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "user_prompt": "Describe this asset.",
                "media": {
                    "images": [{"path": "render.png", "metadata": {"view": "+x"}}]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "physics_agent.tasks.reporting.format_images_html",
        lambda *args, **kwargs: "<div class='images'>stub</div>",
    )
    monkeypatch.setattr(
        "physics_agent.tasks.reporting.format_system_prompt_section",
        lambda prompt: f"<section>{prompt}</section>",
    )
    monkeypatch.setattr(
        "physics_agent.tasks.reporting.validate_image_options",
        lambda image_format, image_quality, image_max_size: (
            image_format or "png",
            image_quality or 85,
            image_max_size or 128,
        ),
    )

    object_store = MemoryObjectStore(
        {
            "dataset": [
                {
                    "id": "prim-1",
                    "text": "fallback prompt",
                    "images": ["render.png"],
                    "image_metadata": [{"view": "+x"}],
                }
            ]
        }
    )
    result = GeneratePredictionReportTask().run(
        {
            "predictions_path": str(predictions_path),
            "dataset_path": str(dataset_path),
            "predictions_count": 1,
            "failed_count": 0,
            "token_stats": {
                "total_tokens": 42,
                "total_input_tokens": 30,
                "total_output_tokens": 12,
                "invocation_count": 1,
                "by_model": {
                    "demo-model": {
                        "count": 1,
                        "input_tokens": 30,
                        "output_tokens": 12,
                        "total_tokens": 42,
                    }
                },
            },
            "actual_system_prompt_used": "system prompt",
            "report_image_max_size": 96,
            "report_image_format": "jpeg",
            "report_image_quality": 77,
        },
        object_store,
    )

    report_path = Path(result["report_path"])
    html = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "Physics Agent - Prediction Report" in html
    assert "hood" in html
    assert "metal" in html
    assert "demo-model" in html
    assert "<div class='images'>stub</div>" in html
    assert "<section>system prompt</section>" in html
