# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from physics_agent.config import (
    STEP_ORDER,
    STEP_OUTPUT_DIRS,
    ConfigValidator,
    ProjectPathResolver,
    UnifiedPipelineConfigTask,
    get_default_config,
    get_step_defaults,
)
from physics_agent.workflows import create_unified_pipeline_workflow


def test_schema_helpers_and_unified_workflow_exports():
    defaults = get_default_config()

    assert defaults["project"]["name"] == "physics_agent_project"
    assert defaults["input"]["reference_images"] == []
    assert STEP_ORDER[0] == "optimize_usd"
    assert STEP_OUTPUT_DIRS["predict"] == "predictions"
    assert get_step_defaults("unknown_step") == {"enabled": True}

    workflow = create_unified_pipeline_workflow()
    assert [task.name for task in workflow.tasks] == [
        "UnifiedConfigLoading",
        "UnifiedPipelineExecutor",
    ]


def test_project_path_resolver_resolves_paths_and_summarizes(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n")
    ref_image = tmp_path / "ref.png"
    ref_image.write_bytes(b"ref")
    config_path = tmp_path / "configs" / "pipeline.yaml"
    config_path.parent.mkdir()
    config_path.write_text("project:\n  name: demo\n")

    resolver = ProjectPathResolver(
        config={
            "project": {
                "name": "demo",
                "working_dir": "work",
                "session_id": "session-123",
            },
            "input": {
                "usd_path": "../asset.usd",
                "reference_images": ["../ref.png"],
            },
            "steps": {},
            "advanced": {},
        },
        config_file_path=config_path,
    )

    resolver.create_working_directories()
    resolver.validate_input_paths()
    summary = resolver.get_path_summary()

    assert resolver.input_usd == usd_path.resolve()
    assert resolver.reference_images == [ref_image.resolve()]
    assert (
        resolver.get_step_output_dir("predict") == resolver.working_dir / "predictions"
    )
    assert resolver.get_step_dataset_file("predict").name == "dataset.jsonl"
    assert resolver.get_step_predictions_file().name == "predictions.jsonl"
    assert resolver.get_usd_dataset_dir() == resolver.working_dir / "dataset" / "usd"
    assert resolver.get_dataset_dir() == resolver.working_dir / "dataset"
    assert summary["input"]["usd_path"] == str(usd_path.resolve())
    assert summary["step_outputs"]["predict"] == str(
        resolver.working_dir / "predictions"
    )


def test_config_validator_handles_required_fields_and_warns(
    monkeypatch: pytest.MonkeyPatch,
):
    validator = ConfigValidator()

    with pytest.raises(ValueError, match="Missing required section"):
        validator.validate({"project": {"name": "demo"}})

    with pytest.raises(ValueError, match="project.name"):
        validator.validate({"project": {}, "input": {"usd_path": "/tmp/a.usd"}})

    warnings: list[str] = []

    def record_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr("physics_agent.config.validator.logger.warning", record_warning)
    validator.validate(
        {
            "project": {"name": "demo"},
            "input": {"usd_path": "/tmp/a.usd"},
            "steps": {"unknown_step": {"enabled": True}},
        }
    )

    assert warnings
    assert "Unknown step 'unknown_step'" in warnings[0]

    with pytest.raises(ValueError, match="predict.output_key must be a string"):
        validator.validate_step_requirements(
            "predict",
            {"output_key": 123},
            {"project": {"name": "demo"}, "input": {"usd_path": "/tmp/a.usd"}},
        )


def test_unified_pipeline_config_task_builds_autowired_step_configs(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n")
    reference = tmp_path / "ref.png"
    reference.write_bytes(b"ref")

    config = {
        "project": {"name": "demo", "working_dir": "runs/demo"},
        "input": {
            "usd_path": str(usd_path),
            "reference_images": [str(reference)],
        },
        "steps": {
            "build_dataset_usd": {"enabled": True},
            "identify_asset": {"enabled": True},
            "build_dataset_prepare_dataset": {"enabled": True},
            "predict": {
                "enabled": True,
                "vlm": {"model": "demo-model"},
            },
            "restore_usd": {"enabled": True},
        },
    }

    context = {
        "config_dict": config,
        "session_id": "session-xyz",
        "only_steps": [
            "build_dataset_usd",
            "identify_asset",
            "build_dataset_prepare_dataset",
            "predict",
            "restore_usd",
        ],
    }

    result = UnifiedPipelineConfigTask().run(context)

    assert result["project_name"] == "demo"
    assert result["session_id"] == "session-xyz"
    assert result["config"]["project"]["session_id"] == "session-xyz"
    assert result["steps_to_run"] == [
        step_name for step_name in STEP_ORDER if step_name in context["only_steps"]
    ]

    build_dataset_usd = result["step_configs"]["build_dataset_usd"]
    assert build_dataset_usd["usd_path"] == str(usd_path)
    assert build_dataset_usd["output_dir"].endswith("dataset/usd")
    assert build_dataset_usd["renderer"]["rgb_rendering_modes"]
    assert "composition" in build_dataset_usd["renderer"]["rgb_rendering_modes"]

    identify_asset = result["step_configs"]["identify_asset"]
    assert identify_asset["usd_path"] == str(usd_path)
    assert identify_asset["output_dir"].endswith("identification")

    prepare_dataset = result["step_configs"]["build_dataset_prepare_dataset"]
    assert prepare_dataset["usd_dir"].endswith("dataset/usd")
    assert prepare_dataset["dataset"].endswith("dataset")
    assert prepare_dataset["models"] == ["."]
    assert prepare_dataset["reference_images"] == [str(reference)]

    predict = result["step_configs"]["predict"]
    assert predict["dataset"].endswith("dataset/dataset.jsonl")
    assert predict["output_dir"].endswith("predictions")
    assert predict["output_key"] == "classification"
    assert predict["vlm"]["model"] == "demo-model"

    restore = result["step_configs"]["restore_usd"]
    assert restore["original_usd_path"] == str(usd_path)
    assert restore["output_predictions_path"].endswith("restored_predictions.jsonl")


def test_unified_pipeline_config_task_rejects_empty_enabled_steps(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n")

    with pytest.raises(ValueError, match="No steps enabled"):
        UnifiedPipelineConfigTask().run(
            {
                "config_dict": {
                    "project": {"name": "demo"},
                    "input": {"usd_path": str(usd_path)},
                    "steps": {},
                }
            }
        )
