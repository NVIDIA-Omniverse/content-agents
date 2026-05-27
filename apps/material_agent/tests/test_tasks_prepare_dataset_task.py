# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for material_agent.tasks.prepare_dataset."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from material_agent.tasks import prepare_dataset as prepare_dataset_module
from material_agent.tasks.prepare_dataset import (
    PrepareDatasetTask,
    extract_material_name_from_mdl_path,
    match_display_color_to_material,
)


def _write_png(path: Path, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)
    return path


def _write_model_inputs(base_dir: Path, model_name: str) -> Path:
    model_dir = base_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dataset.json").write_text(
        json.dumps({"statistics": {"total_prims": 1}}), encoding="utf-8"
    )
    (model_dir / "usd_model.json").write_text("{}", encoding="utf-8")
    return model_dir


def test_extract_material_name_from_mdl_path_parses_nv_materials() -> None:
    mdl_path = "../../materials/3D_Library_Material/nv007_tin_plating/tin_plating.mdl"
    assert extract_material_name_from_mdl_path(mdl_path) == "Tin Plating"
    assert extract_material_name_from_mdl_path("") is None


def test_match_display_color_to_material_uses_rounded_rgb() -> None:
    mapping = [{"color": [0.1234, 0.5678, 0.9999], "material": "Anodized Aluminum"}]
    assert (
        match_display_color_to_material([0.12339, 0.56781, 0.99991], mapping)
        == "Anodized Aluminum"
    )
    assert match_display_color_to_material([0.0, 0.0, 0.0], mapping) is None


def test_default_prompts_include_unknown_visual_evidence_contract() -> None:
    assert (
        '"material": "__UNKNOWN__"'
        in prepare_dataset_module._VLM_SYSTEM_PROMPT_TEMPLATE
    )
    assert "no visible geometry" in prepare_dataset_module._VLM_SYSTEM_PROMPT_TEMPLATE
    assert (
        "Do NOT infer the material from the prim path"
        in prepare_dataset_module._VLM_SYSTEM_PROMPT_TEMPLATE
    )
    assert (
        "blank, uniformly colored" in prepare_dataset_module._VLM_USER_PROMPT_TEMPLATE
    )
    assert (
        '"__UNKNOWN__" for that part while preserving its prim-path entry'
        in prepare_dataset_module._VLM_MULTI_PRIM_USER_PROMPT_TEMPLATE
    )


def test_prepare_dataset_task_builds_v02_dataset_entries(tmp_path: Path) -> None:
    usd_dir = tmp_path / "usd_inputs"
    dataset_dir = tmp_path / "prepared_dataset"
    model_dir = _write_model_inputs(usd_dir, "MODEL_A")
    _write_png(model_dir / "render_b.png", "red")
    _write_png(model_dir / "render_a.png", "blue")
    reference_image = _write_png(tmp_path / "reference.png", "white")

    prim_data = {
        "prim_path": "/Root/PartA",
        "display_color": [0.1, 0.2, 0.3],
        "world_bbox_meters": {"size": [1.0, 2.0, 3.0]},
        "relative_metrics": {
            "relative_size": [0.2, 0.4, 0.6],
            "relative_center": [0.5, -0.5, 1.25],
        },
        "metadata": {
            "custom_data": {"annotation": "Main bracket"},
            "hoops_metadata": {
                "PTC_COMMON_NAME": "Bracket",
                "PTC_WM_NUMBER": "ABC-123",
            },
            "references": ["child_a", "child_b"],
        },
        "material_bindings": {
            "mdl_path": "../../materials/3D_Library_Material/nv007_tin_plating/tin_plating.mdl"
        },
        "renders": [
            {
                "path": "render_b.png",
                "view": "rear_left",
                "camera": "cam-b",
                "render_mode": "shaded",
            },
            {
                "path": "render_a.png",
                "view": "front_right",
                "camera": "cam-a",
                "render_mode": "shaded",
            },
        ],
    }
    (model_dir / "prims.jsonl").write_text(
        json.dumps(prim_data) + "\n", encoding="utf-8"
    )

    listener = MagicMock()
    task = PrepareDatasetTask()
    context = {
        "usd_dir": usd_dir,
        "dataset_path": dataset_dir,
        "models": ["MODEL_A"],
        "config": {
            "materials_list": ["Steel", "Plastic"],
            "include_ground_truth": True,
            "include_prim_path_context": True,
            "include_display_color_context": True,
            "include_geometric_context": True,
            "display_color_to_material": [
                {"color": [0.1, 0.2, 0.3], "material": "Color Match"}
            ],
            "reference_images": [str(reference_image)],
            "reference_image_max_size": 64,
            "render_mode_filter": ["shaded"],
            "prompts": {
                "vlm_system": "Materials:\n{materials_list}",
                "vlm_user": "Context:\n{context}",
                "vlm_image_prompts": {
                    "reference_images": ["Reference product photo"],
                    "shaded": "Rendered highlighted part",
                },
            },
        },
    }

    with patch(
        "material_agent.tasks.prepare_dataset.get_listener", return_value=listener
    ):
        result = task.run(context)

    assert len(result["dataset_entries"]) == 1
    entry = result["dataset_entries"][0]
    assert entry["id"] == "/Root/PartA"
    assert entry["ground_truth"] == {"material": "Color Match"}
    assert (
        "prim path of the 3D USD stage for this part is /Root/PartA"
        in entry["user_prompt"]
    )
    assert "Bounding box dimensions (meters)" in entry["user_prompt"]
    assert "Part annotation: Main bracket" in entry["user_prompt"]
    assert "Reference images precede rendered images" in entry["user_prompt"]

    images = entry["media"]["images"]
    assert len(images) == 3
    assert images[0]["type"] == "reference"
    assert images[0]["metadata"]["vlm_prompt"] == "Reference product photo"
    assert images[1]["path"].endswith("render_a.png")
    assert images[2]["path"].endswith("render_b.png")
    assert (
        "Camera Position: Looking from front_right towards the center"
        in images[1]["metadata"]["vlm_prompt"]
    )

    dataset_jsonl = dataset_dir / "dataset.jsonl"
    dataset_config = dataset_dir / "dataset.json"
    assert result["dataset_path"] == dataset_dir
    assert result["dataset_jsonl_path"] == dataset_jsonl
    assert result["num_entries"] == 1
    assert dataset_jsonl.exists()
    assert dataset_config.exists()
    config_data = json.loads(dataset_config.read_text(encoding="utf-8"))
    assert config_data["schema_version"] == "0.2"
    assert config_data["metadata"]["num_entries"] == 1
    assert config_data["inference"]["prompts"][0]["system_prompt"] == (
        "Materials:\nSteel, Plastic"
    )


def test_prepare_dataset_task_raises_when_any_model_fails(tmp_path: Path) -> None:
    usd_dir = tmp_path / "usd_inputs"
    dataset_dir = tmp_path / "prepared_dataset"
    model_dir = _write_model_inputs(usd_dir, "MODEL_OK")
    _write_png(model_dir / "render.png", "green")
    prim_data = {
        "prim_path": "/Root/PartB",
        "material_bindings": {
            "mdl_path": "../../materials/3D_Library_Material/nv010_brushed_aluminum/test.mdl"
        },
        "renders": [
            {
                "path": "render.png",
                "view": "front",
                "camera": "cam",
                "render_mode": "shaded",
            }
        ],
    }
    (model_dir / "prims.jsonl").write_text(
        json.dumps(prim_data) + "\n", encoding="utf-8"
    )

    listener = MagicMock()
    task = PrepareDatasetTask()
    context = {
        "usd_dir": usd_dir,
        "dataset_path": dataset_dir,
        "models": ["MODEL_OK", "MODEL_MISSING"],
        "config": {"materials_list": "Steel, Aluminum"},
    }

    with patch(
        "material_agent.tasks.prepare_dataset.get_listener", return_value=listener
    ):
        try:
            task.run(context)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ValueError for missing model inputs")

    assert "Failed to prepare data for 1 model(s): MODEL_MISSING" in message
    assert (dataset_dir / "dataset.jsonl").exists()
