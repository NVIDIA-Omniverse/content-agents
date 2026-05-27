# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for UnifiedPipelineExecutorTask helper and runtime behavior."""

from __future__ import annotations

import asyncio
import enum
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from material_agent.tasks.unified_pipeline_executor import (
    UnifiedPipelineExecutorTask,
    _make_yaml_safe,
)


class _Mode(enum.Enum):
    FAST = "fast"


def test_make_yaml_safe_normalizes_nested_non_primitives() -> None:
    safe = _make_yaml_safe(
        {
            Path("root"): {
                "path": Path("/tmp/example"),
                "mode": _Mode.FAST,
                "items": (1, Path("child")),
                "flags": {"b", "a"},
                "object": object(),
            }
        }
    )

    assert safe["root"]["path"] == "/tmp/example"
    assert safe["root"]["mode"] == "fast"
    assert safe["root"]["items"] == [1, "child"]
    assert safe["root"]["flags"] == ["a", "b"]
    assert isinstance(safe["root"]["object"], str)


def test_create_temp_config_file_strips_private_renderer_keys(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()

    config_path = executor._create_temp_config_file(
        "render",
        {
            "renderer": {
                "backend": "remote",
                "_unified_config": object(),
                "_rendering_modes_config": object(),
            },
            "path": Path("/tmp/output.usd"),
            "mode": _Mode.FAST,
            "flags": {"b", "a"},
        },
        tmp_path,
    )

    with open(config_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    assert loaded["renderer"] == {"backend": "remote"}
    assert loaded["path"] == "/tmp/output.usd"
    assert loaded["mode"] == "fast"
    assert loaded["flags"] == ["a", "b"]


@pytest.mark.parametrize(
    ("step_name", "result", "expected"),
    [
        (
            "build_dataset_prepare_dataset",
            {
                "dataset_path": "dataset",
                "dataset_jsonl_path": "dataset.jsonl",
                "vlm_prompt_path": "prompt.txt",
                "num_entries": 7,
            },
            {
                "dataset_path": "dataset",
                "dataset_jsonl_path": "dataset.jsonl",
                "vlm_prompt_path": "prompt.txt",
                "num_entries": 7,
            },
        ),
        (
            "build_dataset_pdf_vectorstore",
            {"output_dir": "vec"},
            {"vectorstore_dir": "vec"},
        ),
        (
            "build_dataset_usd",
            {"output_dir": "usd_data", "num_prims": 4, "num_images": 9},
            {
                "output_dir": "usd_data",
                "usd_dataset_dir": "usd_data",
                "num_prims": 4,
                "num_images": 9,
            },
        ),
        (
            "cluster_prims",
            {
                "cluster_map_path": "clusters/map.jsonl",
                "dataset_representatives_path": "dataset/reps.jsonl",
                "cluster_prims_ran": True,
                "cluster_summary_path": "clusters/cluster_summary.json",
                "cluster_report_path": "clusters/cluster_report.html",
                "cluster_total_prims": 117,
                "cluster_count": 88,
                "cluster_representative_count": 88,
                "cluster_reduction_percent": 24.786,
                "cluster_multi_member_count": 13,
                "cluster_singleton_count": 75,
                "cluster_max_size": 25,
                "cluster_capped_count": 0,
            },
            {
                "cluster_map_path": "clusters/map.jsonl",
                "dataset_representatives_path": "dataset/reps.jsonl",
                "cluster_prims_ran": True,
                "cluster_summary_path": "clusters/cluster_summary.json",
                "cluster_report_path": "clusters/cluster_report.html",
                "cluster_total_prims": 117,
                "cluster_count": 88,
                "cluster_representative_count": 88,
                "cluster_reduction_percent": 24.786,
                "cluster_multi_member_count": 13,
                "cluster_singleton_count": 75,
                "cluster_max_size": 25,
                "cluster_capped_count": 0,
            },
        ),
        (
            "cluster_prims",
            {
                "cluster_map_path": "clusters/map.jsonl",
                "dataset_representatives_path": "dataset/reps.jsonl",
            },
            {
                "cluster_map_path": "clusters/map.jsonl",
                "dataset_representatives_path": "dataset/reps.jsonl",
                "cluster_prims_ran": False,
                "cluster_summary_path": None,
                "cluster_report_path": None,
                "cluster_total_prims": 0,
                "cluster_count": 0,
                "cluster_representative_count": 0,
                "cluster_reduction_percent": 0.0,
                "cluster_multi_member_count": 0,
                "cluster_singleton_count": 0,
                "cluster_max_size": None,
                "cluster_capped_count": 0,
            },
        ),
        (
            "predict",
            {"predictions_path": "preds.jsonl", "predictions_count": 5},
            {"predictions_path": "preds.jsonl", "predictions_count": 5},
        ),
        (
            "expand_cluster_predictions",
            {"predictions_path": "expanded.jsonl"},
            {"predictions_path": "expanded.jsonl"},
        ),
        (
            "harmonize_predictions",
            {
                "predictions_path": "harmonized.jsonl",
                "harmonized_count": 3,
                "remap": {"a": "b"},
            },
            {
                "predictions_path": "harmonized.jsonl",
                "harmonized_count": 3,
                "remap": {"a": "b"},
            },
        ),
        (
            "evaluate",
            {
                "evaluation_path": "evaluation.json",
                "html_report_path": "report.html",
                "metrics": {"acc": 1.0},
            },
            {
                "evaluation_path": "evaluation.json",
                "html_report_path": "report.html",
                "metrics": {"acc": 1.0},
            },
        ),
        (
            "optimize_usd",
            {
                "optimized_usd_path": "optimized.usdc",
                "optimization_success": True,
                "original_usd_path": "original.usd",
                "original_prim_count": 10,
                "optimization_metadata": {"map": {}},
            },
            {
                "optimized_usd_path": "optimized.usdc",
                "optimization_success": True,
                "original_usd_path": "original.usd",
                "original_prim_count": 10,
                "optimization_metadata": {"map": {}},
            },
        ),
        (
            "apply",
            {"output_usd_path": "applied.usd", "materials_applied": 4},
            {"output_usd_path": "applied.usd", "materials_applied": 4},
        ),
        (
            "validate_input",
            {
                "validation_result": {"issues": []},
                "validation_summary": "ok",
                "validation_is_valid": True,
                "validation_fixed_usd_path": None,
                "validation_skipped": None,
                "validation_error": None,
                "validation_success": True,
            },
            {
                "validation_result": {"issues": []},
                "validation_summary": "ok",
                "validation_is_valid": True,
                "validation_fixed_usd_path": None,
                "validation_skipped": None,
                "validation_error": None,
                "validation_success": True,
            },
        ),
        (
            "refine",
            {"final_output_path": "refined.usd"},
            {
                "output_usd_path": "refined.usd",
                "final_output_path": "refined.usd",
            },
        ),
        (
            "render",
            {
                "rendered_image_paths": ["a.png"],
                "rendered_image_path": "a.png",
                "flattened_usd_path": "flat.usd",
            },
            {
                "rendered_image_paths": ["a.png"],
                "rendered_image_path": "a.png",
                "flattened_usd_path": "flat.usd",
            },
        ),
        (
            "validate_output",
            {
                "validation_result": {"ok": True},
                "validation_summary": "done",
                "validation_is_valid": True,
                "validation_regression": False,
                "validation_new_issues": [],
                "validation_skipped": None,
                "validation_error": None,
                "validation_success": True,
            },
            {
                "validation_result": {"ok": True},
                "validation_summary": "done",
                "validation_is_valid": True,
                "validation_regression": False,
                "validation_new_issues": [],
                "validation_skipped": None,
                "validation_error": None,
                "validation_success": True,
            },
        ),
        (
            "restore_usd",
            {
                "restored_usd_path": "restored.usd",
                "restored_predictions_path": "restored.jsonl",
                "restore_success": True,
                "predictions_count": 12,
            },
            {
                "restored_usd_path": "restored.usd",
                "restored_predictions_path": "restored.jsonl",
                "restore_success": True,
                "predictions_count": 12,
            },
        ),
    ],
)
def test_extract_step_outputs_maps_expected_keys(
    step_name: str,
    result: dict[str, object],
    expected: dict[str, object],
) -> None:
    executor = UnifiedPipelineExecutorTask()

    assert executor._extract_step_outputs(step_name, result) == expected


def test_run_cleans_outputs_and_updates_context(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    (working_dir / "stale.txt").write_text("old", encoding="utf-8")

    output_usd = tmp_path / "output" / "scene.usd"
    output_usd.parent.mkdir()
    output_usd.write_text("usd", encoding="utf-8")
    (output_usd.parent / "scene_flat.usd").write_text("flat", encoding="utf-8")
    renders_dir = output_usd.parent / "renders"
    renders_dir.mkdir()
    (renders_dir / "preview.png").write_text("x", encoding="utf-8")

    listener = MagicMock()
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["optimize_usd", "build_dataset_usd"],
        "step_configs": {
            "optimize_usd": {"enabled": True},
            "build_dataset_usd": {"enabled": True},
        },
        "clean": True,
        "session_id": "session-1",
        "project_name": "project-1",
        "path_resolver": SimpleNamespace(output_usd=output_usd),
    }
    pipeline_state = {
        "session_id": "session-1",
        "project_name": "project-1",
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }

    def step_outputs(step_name, step_config, ctx, object_store, state):
        if step_name == "optimize_usd":
            return {
                "optimized_usd_path": str(tmp_path / "optimized.usdc"),
                "original_usd_path": str(tmp_path / "input.usd"),
                "original_prim_count": 12,
            }
        return {
            "output_dir": str(tmp_path / "dataset"),
            "num_prims": 8,
            "num_images": 16,
        }

    executor._execute_step = MagicMock(side_effect=step_outputs)

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = executor.run(context)

    assert result["pipeline_state"] == "completed"
    assert result["original_prim_count"] == 12
    assert result["num_prims"] == 8
    assert result["num_images"] == 16
    assert not (working_dir / "stale.txt").exists()
    assert not output_usd.exists()
    assert not (output_usd.parent / "scene_flat.usd").exists()
    assert not renders_dir.exists()
    assert result["pipeline_results"]["optimize_usd"]["original_prim_count"] == 12
    assert result["pipeline_results"]["build_dataset_usd"]["num_prims"] == 8


def test_run_skips_failed_optimize_usd_and_continues(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    working_dir = tmp_path / "work"
    listener = MagicMock()
    event_listener = MagicMock()
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["optimize_usd", "build_dataset_usd"],
        "step_configs": {
            "optimize_usd": {"input_usd_path": str(tmp_path / "input.usd")},
            "build_dataset_usd": {"usd_path": str(tmp_path / "scene.usd")},
        },
        "session_id": "session-2",
        "project_name": "project-2",
        "event_listener": event_listener,
    }
    pipeline_state = {
        "session_id": "session-2",
        "project_name": "project-2",
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }

    def execute_step(step_name, step_config, ctx, object_store, state):
        if step_name == "optimize_usd":
            raise RuntimeError("scene optimizer crashed")
        return {"output_dir": "dataset", "num_prims": 5, "num_images": 9}

    executor._execute_step = MagicMock(side_effect=execute_step)

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = executor.run(context)

    state_file = working_dir / ".pipeline_state.json"
    saved = json.loads(state_file.read_text(encoding="utf-8"))

    assert result["pipeline_state"] == "completed"
    assert result["original_prim_count"] == 5
    assert result["pipeline_results"]["build_dataset_usd"]["num_images"] == 9
    assert saved["optimize_usd_skipped_original_input"] == str(tmp_path / "input.usd")
    assert saved["completed_steps"] == ["build_dataset_usd"]
    event_listener.event.assert_any_call(
        "step.skipped",
        {
            "step_name": "optimize_usd",
            "reason": "optimize_usd failed: scene optimizer crashed",
        },
    )


def test_run_records_step_error_for_failed_step(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    pipeline_state = {
        "session_id": "session-error",
        "project_name": "project-error",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    executor._execute_step = MagicMock(side_effect=RuntimeError("predict crashed"))
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["predict"],
        "step_configs": {"predict": {"enabled": True}},
        "session_id": "session-error",
        "project_name": "project-error",
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(RuntimeError, match="Pipeline failed at step 'predict'"),
    ):
        executor.run(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert saved["failed_steps"] == ["predict"]
    assert saved["step_errors"] == {"predict": "predict crashed"}
    assert saved["current_step"] is None


def test_run_clears_prior_step_error_on_success(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    pipeline_state = {
        "session_id": "session-retry",
        "project_name": "project-retry",
        "completed_steps": [],
        "failed_steps": ["predict"],
        "step_errors": {"predict": "old prediction failure"},
        "step_outputs": {},
        "current_step": None,
    }
    executor._execute_step = MagicMock(return_value={"predictions_path": "preds.jsonl"})
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["predict"],
        "step_configs": {"predict": {"enabled": True}},
        "session_id": "session-retry",
        "project_name": "project-retry",
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = executor.run(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert result["pipeline_state"] == "completed"
    assert saved["completed_steps"] == ["predict"]
    assert saved["failed_steps"] == []
    assert saved["step_errors"] == {}
    assert saved["step_outputs"]["predict"] == {"predictions_path": "preds.jsonl"}


def test_run_cancel_checker_stops_before_next_step(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    completed: list[str] = []
    pipeline_state = {
        "session_id": "cancel-sync",
        "project_name": "cancel-project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }

    def execute_step(step_name, *_args, **_kwargs):
        completed.append(step_name)
        return {"step": step_name}

    executor._execute_step = MagicMock(side_effect=execute_step)
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["build_dataset_usd", "predict"],
        "step_configs": {
            "build_dataset_usd": {"enabled": True},
            "predict": {"enabled": True},
        },
        "cancel_checker": lambda: bool(completed),
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        executor.run(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert completed == ["build_dataset_usd"]
    assert saved["completed_steps"] == ["build_dataset_usd"]
    assert saved["current_step"] is None
    listener.event.assert_any_call(
        "step.cancelled",
        {
            "step_name": "predict",
            "message": "Pipeline cancellation requested",
        },
    )


@pytest.mark.asyncio
async def test_arun_skips_restore_usd_without_optimize(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    pipeline_state = {
        "session_id": None,
        "project_name": None,
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }
    executor._aexecute_step = AsyncMock()
    context = {
        "working_dir": str(tmp_path / "work"),
        "steps_to_run": ["restore_usd"],
        "step_configs": {"restore_usd": {"enabled": True}},
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = await executor.arun(context)

    executor._aexecute_step.assert_not_awaited()
    assert result["pipeline_state"] == "completed"
    assert result["pipeline_results"] == {}


@pytest.mark.asyncio
async def test_arun_cancel_checker_stops_before_next_step(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    completed: list[str] = []
    pipeline_state = {
        "session_id": "cancel-async",
        "project_name": "cancel-project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }

    async def execute_step(step_name, *_args, **_kwargs):
        completed.append(step_name)
        return {"step": step_name}

    executor._aexecute_step = AsyncMock(side_effect=execute_step)
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["build_dataset_usd", "predict"],
        "step_configs": {
            "build_dataset_usd": {"enabled": True},
            "predict": {"enabled": True},
        },
        "cancel_checker": lambda: bool(completed),
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await executor.arun(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert completed == ["build_dataset_usd"]
    assert saved["completed_steps"] == ["build_dataset_usd"]
    assert saved["current_step"] is None
    listener.event.assert_any_call(
        "step.cancelled",
        {
            "step_name": "predict",
            "message": "Pipeline cancellation requested",
        },
    )


@pytest.mark.asyncio
async def test_arun_skips_failed_optimize_usd_and_continues(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    event_listener = MagicMock()
    working_dir = tmp_path / "work"
    pipeline_state = {
        "session_id": "session-3",
        "project_name": "project-3",
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }
    executor._aexecute_step = AsyncMock(
        side_effect=[
            RuntimeError("async optimizer crashed"),
            {"output_dir": "dataset", "num_prims": 4, "num_images": 6},
        ]
    )
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["optimize_usd", "build_dataset_usd"],
        "step_configs": {
            "optimize_usd": {"input_usd_path": str(tmp_path / "input.usd")},
            "build_dataset_usd": {"usd_path": str(tmp_path / "scene.usd")},
        },
        "session_id": "session-3",
        "project_name": "project-3",
        "event_listener": event_listener,
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = await executor.arun(context)

    state_file = working_dir / ".pipeline_state.json"
    saved = json.loads(state_file.read_text(encoding="utf-8"))

    assert result["pipeline_state"] == "completed"
    assert result["original_prim_count"] == 4
    assert result["pipeline_results"]["build_dataset_usd"]["num_images"] == 6
    assert saved["optimize_usd_skipped_original_input"] == str(tmp_path / "input.usd")
    assert saved["completed_steps"] == ["build_dataset_usd"]
    event_listener.event.assert_any_call(
        "step.skipped",
        {
            "step_name": "optimize_usd",
            "reason": "optimize_usd failed: async optimizer crashed",
        },
    )


@pytest.mark.asyncio
async def test_arun_records_step_error_for_failed_step(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    pipeline_state = {
        "session_id": "async-error",
        "project_name": "async-project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    executor._aexecute_step = AsyncMock(side_effect=RuntimeError("async predict broke"))
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["predict"],
        "step_configs": {"predict": {"enabled": True}},
        "session_id": "async-error",
        "project_name": "async-project",
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(RuntimeError, match="Pipeline failed at step 'predict'"),
    ):
        await executor.arun(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert saved["failed_steps"] == ["predict"]
    assert saved["step_errors"] == {"predict": "async predict broke"}
    assert saved["current_step"] is None
