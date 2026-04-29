# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution-path coverage for unified pipeline executor step methods."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import material_agent.tasks.unified_pipeline_executor as upe
import material_agent.workflows as workflows
from material_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask

_WORKFLOW_FACTORY_BY_STEP = {
    "validate_input": "create_validate_input_workflow_from_config",
    "optimize_usd": "create_optimize_usd_workflow_from_config",
    "build_dataset_usd": "create_usd_data_preparation_workflow_from_config",
    "build_dataset_pdf_vectorstore": "create_pdf_vectorstore_workflow_from_config",
    "build_dataset_prepare_dataset": "create_prepare_dataset_workflow_from_config",
    "cluster_prims": "create_cluster_prims_workflow_from_config",
    "predict": "create_prediction_workflow_from_config",
    "expand_cluster_predictions": "create_expand_cluster_predictions_workflow_from_config",
    "benchmark": "create_benchmark_workflow_from_config",
    "validate_predictions": "create_validate_predictions_workflow_from_config",
    "harmonize_predictions": "create_harmonize_predictions_workflow_from_config",
    "evaluate": "create_evaluation_workflow_from_config",
    "apply": "create_apply_workflow_from_config",
    "refine": "create_iterative_apply_workflow_from_config",
    "restore_usd": "create_restore_usd_workflow_from_config",
    "validate_output": "create_validate_output_workflow_from_config",
    "render": "create_render_workflow_from_config",
}


class _WorkflowCapture:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.last_context: dict[str, Any] = {}
        self.last_config: dict[str, Any] = {}

    def _capture(self, step_context: dict[str, Any]) -> None:
        self.last_context = dict(step_context)
        with open(step_context["config_path"], encoding="utf-8") as f:
            self.last_config = yaml.safe_load(f) or {}

    def run(self, step_context: dict[str, Any]) -> dict[str, Any]:
        self._capture(step_context)
        return dict(self.result)

    async def arun(self, step_context: dict[str, Any]) -> dict[str, Any]:
        self._capture(step_context)
        return dict(self.result)


def _patch_workflow_factory(
    monkeypatch: pytest.MonkeyPatch,
    step_name: str,
    workflow: _WorkflowCapture,
) -> None:
    fn_name = _WORKFLOW_FACTORY_BY_STEP[step_name]
    monkeypatch.setattr(workflows, fn_name, lambda: workflow)


def test_execute_step_evaluate_wires_paths_and_report_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture(
        {
            "evaluation_path": "eval/evaluation.json",
            "html_report_path": "eval/report.html",
            "metrics": {"f1": 0.91},
        }
    )
    _patch_workflow_factory(monkeypatch, "evaluate", workflow)

    pipeline_state = {
        "step_outputs": {
            "harmonize_predictions": {"predictions_path": "preds/harmonized.jsonl"},
            "build_dataset_prepare_dataset": {
                "dataset_jsonl_path": "dataset/dataset.jsonl",
                "vlm_prompt_path": "dataset/vlm_prompt.txt",
            },
        }
    }
    context = {
        "working_dir": str(tmp_path / "work"),
        "original_prim_count": 0,
        "num_prims": 12,
        "num_images": 24,
    }
    step_config = {
        "report": {"image_max_size": 640, "image_format": "jpeg", "image_quality": 75}
    }

    outputs = executor._execute_step(
        "evaluate",
        step_config,
        context,
        object_store=None,
        pipeline_state=pipeline_state,
    )

    assert outputs["evaluation_path"] == "eval/evaluation.json"
    assert workflow.last_config["predictions_path"] == "preds/harmonized.jsonl"
    assert workflow.last_config["dataset_path"] == "dataset/dataset.jsonl"
    assert workflow.last_config["system_prompt_file"] == "dataset/vlm_prompt.txt"
    assert str(workflow.last_config["output_dir"]).endswith("/evaluation")
    assert workflow.last_context["report_image_max_size"] == 640
    assert workflow.last_context["report_image_format"] == "jpeg"
    assert workflow.last_context["report_image_quality"] == 75
    assert workflow.last_context["original_prim_count"] == 0
    assert workflow.last_context["num_prims"] == 12
    assert workflow.last_context["num_images"] == 24


def test_execute_step_restore_usd_wires_fallbacks_and_metadata_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture(
        {
            "restored_predictions_path": "restored/restored_predictions.jsonl",
            "restore_success": True,
            "predictions_count": 5,
        }
    )
    _patch_workflow_factory(monkeypatch, "restore_usd", workflow)

    working_dir = tmp_path / "work"
    metadata_path = working_dir / "optimized" / "optimized_input.metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({"map": {"a": "/A"}}), encoding="utf-8")

    context = {
        "working_dir": str(working_dir),
        "path_resolver": SimpleNamespace(input_usd=Path("/tmp/original.usd")),
    }
    pipeline_state = {"step_outputs": {}}

    outputs = executor._execute_step(
        "restore_usd", {}, context, object_store=None, pipeline_state=pipeline_state
    )

    assert outputs["restore_success"] is True
    assert workflow.last_config["original_usd_path"] == "/tmp/original.usd"
    assert str(workflow.last_config["predictions_path"]).endswith(
        "/predictions/predictions.jsonl"
    )
    assert str(workflow.last_config["output_predictions_path"]).endswith(
        "/restored/restored_predictions.jsonl"
    )
    assert workflow.last_config["optimization_metadata"] == {"map": {"a": "/A"}}


def test_execute_step_validate_output_injects_baseline_from_validate_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture(
        {
            "validation_result": {"issues": []},
            "validation_summary": "ok",
            "validation_is_valid": True,
            "validation_regression": False,
            "validation_new_issues": [],
            "validation_success": True,
        }
    )
    _patch_workflow_factory(monkeypatch, "validate_output", workflow)

    class _Resolver:
        input_usd = None

        def resolve_path(self, value: str) -> Path:
            return Path("/resolved") / value

    context = {
        "working_dir": str(tmp_path / "work"),
        "path_resolver": _Resolver(),
        "config": {"input": {"usd_path": "scene/input.usd"}},
    }
    pipeline_state = {
        "step_outputs": {
            "apply": {"output_usd_path": "/tmp/applied.usd"},
            "validate_input": {"validation_result": {"issues": ["warn-a"]}},
        }
    }

    executor._execute_step(
        "validate_output", {}, context, object_store=None, pipeline_state=pipeline_state
    )

    assert workflow.last_config["input_usd_path"] == "/tmp/applied.usd"
    assert workflow.last_config["original_usd_path"] == "/resolved/scene/input.usd"
    assert workflow.last_config["baseline_validation"] == {"issues": ["warn-a"]}


def test_execute_step_optimize_stores_metadata_on_pipeline_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture(
        {
            "optimized_usd_path": "/tmp/optimized.usdc",
            "optimization_success": True,
            "original_usd_path": "/tmp/original.usd",
            "original_prim_count": 3,
            "optimization_metadata": {"index_remap": {"1": "2"}},
        }
    )
    _patch_workflow_factory(monkeypatch, "optimize_usd", workflow)

    pipeline_state = {
        "step_outputs": {
            "validate_input": {"validation_fixed_usd_path": "/tmp/fixed.usd"}
        }
    }
    outputs = executor._execute_step(
        "optimize_usd",
        {},
        {"working_dir": str(tmp_path / "work")},
        object_store=None,
        pipeline_state=pipeline_state,
    )

    assert outputs["optimized_usd_path"] == "/tmp/optimized.usdc"
    assert pipeline_state["optimization_metadata"] == {"index_remap": {"1": "2"}}
    assert workflow.last_config["input_usd_path"] == "/tmp/fixed.usd"


@pytest.mark.parametrize("step_name", ["unknown_step", "assign"])
def test_execute_step_raises_on_unknown_step(tmp_path: Path, step_name: str) -> None:
    executor = UnifiedPipelineExecutorTask()
    with pytest.raises(ValueError, match="Unknown step"):
        executor._execute_step(
            step_name,
            {},
            {"working_dir": str(tmp_path / "work")},
            object_store=None,
            pipeline_state={"step_outputs": {}},
        )


@pytest.mark.parametrize(
    "result_payload, error_snippet",
    [
        ({}, "workflow returned empty result"),
        (
            {"workflow_terminated": True, "failed_task": "predict", "error": "boom"},
            "failed at task 'predict'",
        ),
    ],
)
def test_execute_step_raises_on_workflow_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_payload: dict[str, Any],
    error_snippet: str,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture(result_payload)
    _patch_workflow_factory(monkeypatch, "predict", workflow)

    with pytest.raises(RuntimeError, match=error_snippet):
        executor._execute_step(
            "predict",
            {},
            {"working_dir": str(tmp_path / "work")},
            object_store=None,
            pipeline_state={"step_outputs": {}},
        )


@pytest.mark.asyncio
async def test_aexecute_step_refine_uses_restore_outputs_and_reference_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture({"final_output_path": "/tmp/final.usd"})
    _patch_workflow_factory(monkeypatch, "refine", workflow)

    context = {
        "working_dir": str(tmp_path / "work"),
        "pipeline_config": {"input": {"reference_images": ["a.png", "b.png"]}},
    }
    pipeline_state = {
        "step_outputs": {
            "optimize_usd": {"original_usd_path": "/tmp/original.usd"},
            "restore_usd": {"restored_predictions_path": "/tmp/restored.jsonl"},
        }
    }

    outputs = await executor._aexecute_step(
        "refine", {}, context, object_store=None, pipeline_state=pipeline_state
    )

    assert outputs["final_output_path"] == "/tmp/final.usd"
    assert workflow.last_config["input_usd_path"] == "/tmp/original.usd"
    assert workflow.last_config["predictions_path"] == "/tmp/restored.jsonl"
    assert workflow.last_config["judge"]["reference_images"] == ["a.png", "b.png"]


@pytest.mark.asyncio
async def test_aexecute_step_expand_cluster_predictions_handles_skip_and_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture({"predictions_path": "/tmp/expanded.jsonl"})
    _patch_workflow_factory(monkeypatch, "expand_cluster_predictions", workflow)

    context = {"working_dir": str(tmp_path / "work")}

    pipeline_state_skipped = {
        "step_outputs": {"cluster_prims": {"cluster_prims_ran": False}}
    }
    await executor._aexecute_step(
        "expand_cluster_predictions",
        {},
        context,
        object_store=None,
        pipeline_state=pipeline_state_skipped,
    )
    assert workflow.last_config["cluster_prims_ran"] is False

    pipeline_state_run = {
        "step_outputs": {
            "cluster_prims": {
                "cluster_prims_ran": True,
                "cluster_map_path": "/tmp/cluster_map.jsonl",
            },
            "predict": {"predictions_path": "/tmp/preds.jsonl"},
        }
    }
    await executor._aexecute_step(
        "expand_cluster_predictions",
        {},
        context,
        object_store=None,
        pipeline_state=pipeline_state_run,
    )
    assert workflow.last_config["predictions_path"] == "/tmp/preds.jsonl"
    assert workflow.last_config["cluster_map_path"] == "/tmp/cluster_map.jsonl"


@pytest.mark.asyncio
async def test_aexecute_step_validate_output_prefers_fixed_baseline_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture(
        {
            "validation_result": {"issues": []},
            "validation_summary": "ok",
            "validation_is_valid": True,
            "validation_regression": False,
            "validation_new_issues": [],
            "validation_success": True,
        }
    )
    _patch_workflow_factory(monkeypatch, "validate_output", workflow)

    pipeline_state = {
        "step_outputs": {
            "refine": {"final_output_path": "/tmp/refined.usd"},
            "validate_input": {"validation_fixed_usd_path": "/tmp/fixed_input.usd"},
        }
    }

    await executor._aexecute_step(
        "validate_output",
        {},
        {"working_dir": str(tmp_path / "work")},
        object_store=None,
        pipeline_state=pipeline_state,
    )

    assert workflow.last_config["input_usd_path"] == "/tmp/refined.usd"
    assert workflow.last_config["original_usd_path"] == "/tmp/fixed_input.usd"


@pytest.mark.asyncio
async def test_aexecute_step_evaluate_wires_fallback_paths_and_report_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workflow = _WorkflowCapture(
        {
            "evaluation_path": "/tmp/eval.json",
            "html_report_path": "/tmp/report.html",
            "metrics": {"acc": 0.8},
        }
    )
    _patch_workflow_factory(monkeypatch, "evaluate", workflow)

    working_dir = tmp_path / "work"
    result = await executor._aexecute_step(
        "evaluate",
        {"report": {"image_format": "png", "image_quality": 92}},
        {
            "working_dir": str(working_dir),
            "event_listener": object(),
            "original_prim_count": 1,
            "num_prims": 2,
            "num_images": 3,
        },
        object_store=None,
        pipeline_state={"step_outputs": {}},
    )

    assert result["html_report_path"] == "/tmp/report.html"
    assert str(workflow.last_config["predictions_path"]).endswith(
        "/predictions/predictions.jsonl"
    )
    assert str(workflow.last_config["dataset_path"]).endswith("/dataset/dataset.jsonl")
    assert str(workflow.last_config["output_dir"]).endswith("/evaluation")
    assert workflow.last_context["report_image_format"] == "png"
    assert workflow.last_context["report_image_quality"] == 92
    assert workflow.last_context["original_prim_count"] == 1
    assert workflow.last_context["num_prims"] == 2
    assert workflow.last_context["num_images"] == 3
    assert "event_listener" in workflow.last_context


@pytest.mark.asyncio
async def test_aexecute_step_render_harmonize_validate_and_restore_async_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    working_dir = tmp_path / "work"

    render_wf = _WorkflowCapture({"rendered_image_path": "/tmp/r.png"})
    _patch_workflow_factory(monkeypatch, "render", render_wf)
    await executor._aexecute_step(
        "render",
        {},
        {"working_dir": str(working_dir)},
        object_store=None,
        pipeline_state={
            "step_outputs": {"apply": {"output_usd_path": "/tmp/apply.usd"}}
        },
    )
    assert render_wf.last_config["input_usd_path"] == "/tmp/apply.usd"

    harmonize_wf = _WorkflowCapture({"predictions_path": "/tmp/h.jsonl"})
    _patch_workflow_factory(monkeypatch, "harmonize_predictions", harmonize_wf)
    await executor._aexecute_step(
        "harmonize_predictions",
        {},
        {"working_dir": str(working_dir)},
        object_store=None,
        pipeline_state={
            "step_outputs": {
                "benchmark": {"predictions_path": "/tmp/bench.jsonl"},
                "optimize_usd": {"optimized_usd_path": "/tmp/opt.usdc"},
            }
        },
    )
    assert harmonize_wf.last_config["predictions_path"] == "/tmp/bench.jsonl"
    assert harmonize_wf.last_config["optimized_usd_path"] == "/tmp/opt.usdc"

    validate_wf = _WorkflowCapture({"predictions_path": "/tmp/v.jsonl"})
    _patch_workflow_factory(monkeypatch, "validate_predictions", validate_wf)
    await executor._aexecute_step(
        "validate_predictions",
        {},
        {"working_dir": str(working_dir)},
        object_store=None,
        pipeline_state={
            "step_outputs": {
                "harmonize_predictions": {"predictions_path": "/tmp/harmonized.jsonl"}
            }
        },
    )
    assert validate_wf.last_config["predictions_path"] == "/tmp/harmonized.jsonl"

    restore_wf = _WorkflowCapture({"restored_predictions_path": "/tmp/restored.jsonl"})
    _patch_workflow_factory(monkeypatch, "restore_usd", restore_wf)
    await executor._aexecute_step(
        "restore_usd",
        {},
        {"working_dir": str(working_dir)},
        object_store=None,
        pipeline_state={
            "optimization_metadata": {"x": 1},
            "step_outputs": {
                "optimize_usd": {"original_usd_path": "/tmp/original.usd"},
                "predict": {"predictions_path": "/tmp/pred.jsonl"},
            },
        },
    )
    assert restore_wf.last_config["original_usd_path"] == "/tmp/original.usd"
    assert restore_wf.last_config["predictions_path"] == "/tmp/pred.jsonl"
    assert restore_wf.last_config["optimization_metadata"] == {"x": 1}


@pytest.mark.asyncio
async def test_arun_clean_resume_and_restore_skip_paths(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    executor._aexecute_step = AsyncMock()

    working_dir = tmp_path / "work"
    working_dir.mkdir(parents=True)
    (working_dir / "stale.txt").write_text("stale", encoding="utf-8")

    output_file = tmp_path / "out" / "scene.usd"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("usd", encoding="utf-8")
    (output_file.parent / "scene_flat.usd").write_text("flat", encoding="utf-8")
    renders_dir = output_file.parent / "renders"
    renders_dir.mkdir()
    (renders_dir / "frame.png").write_text("x", encoding="utf-8")

    listener = MagicMock()
    pipeline_state = {
        "completed_steps": ["already_done"],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }

    with (
        patch.object(upe, "_load_pipeline_state", return_value=pipeline_state),
        patch.object(upe, "get_listener", return_value=listener),
    ):
        result = await executor.arun(
            {
                "working_dir": str(working_dir),
                "steps_to_run": ["already_done", "restore_usd"],
                "step_configs": {"already_done": {}, "restore_usd": {}},
                "resume": True,
                "clean": True,
                "path_resolver": SimpleNamespace(output_usd=output_file),
            }
        )

    executor._aexecute_step.assert_not_awaited()
    assert result["pipeline_state"] == "completed"
    assert not output_file.exists()
    assert not (output_file.parent / "scene_flat.usd").exists()
    assert not renders_dir.exists()


@pytest.mark.asyncio
async def test_arun_raises_on_no_steps_and_dangerous_clean_path() -> None:
    executor = UnifiedPipelineExecutorTask()
    with pytest.raises(ValueError, match="No steps to run in pipeline"):
        await executor.arun({})

    with pytest.raises(ValueError, match="Refusing to delete"):
        await executor.arun(
            {
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {}},
                "clean": True,
                "working_dir": "/",
            }
        )
