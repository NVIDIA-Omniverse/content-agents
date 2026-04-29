"""Unit tests for apply_physics auto-wire resolution.

Covers the four pipeline topologies the executor is expected to handle:
  1. optimize_usd ran → use raw predict output for optimized/deinstanced USD.
  2. restore_usd ran but short-circuited (optimize_usd disabled) → fall
     back to raw predict output (safe because no optimization ran).
  3. restore_usd produced a remapped predictions file without optimize_usd → use it.
  4. predict-only (no optimize, no restore) → use raw predict output.

Also verifies predict.output_key propagates back out.
"""

from physics_agent.tasks.unified_pipeline_executor import (
    resolve_apply_physics_inputs,
)


def test_optimize_topology_uses_raw_predictions_for_optimized_usd():
    step_outputs = {
        "optimize_usd": {"optimized_usd_path": "/w/optimized.usda"},
        "predict": {"predictions_path": "/w/predictions.jsonl"},
        "restore_usd": {"restored_predictions_path": "/w/restored.jsonl"},
    }
    predictions_path, output_key = resolve_apply_physics_inputs(step_outputs)
    assert predictions_path == "/w/predictions.jsonl"
    assert output_key == "classification"


def test_falls_back_to_predict_when_restore_usd_shortcircuited_without_optimize():
    # optimize_usd disabled → restore_usd records an output entry but no path.
    # Raw predict output is safe because predictions reference original prims.
    step_outputs = {
        "predict": {"predictions_path": "/w/predictions.jsonl"},
        "restore_usd": {"restored_predictions_path": None},
    }
    predictions_path, output_key = resolve_apply_physics_inputs(step_outputs)
    assert predictions_path == "/w/predictions.jsonl"
    assert output_key == "classification"


def test_optimized_without_restore_still_uses_raw_predictions():
    step_outputs = {
        "optimize_usd": {"optimized_usd_path": "/w/optimized.usda"},
        "predict": {"predictions_path": "/w/predictions.jsonl"},
    }
    predictions_path, output_key = resolve_apply_physics_inputs(step_outputs)
    assert predictions_path == "/w/predictions.jsonl"
    assert output_key == "classification"


def test_optimized_with_restore_noop_still_uses_raw_predictions():
    step_outputs = {
        "optimize_usd": {"optimized_usd_path": "/w/optimized.usda"},
        "predict": {"predictions_path": "/w/predictions.jsonl"},
        "restore_usd": {"restored_predictions_path": None},
    }
    predictions_path, output_key = resolve_apply_physics_inputs(step_outputs)
    assert predictions_path == "/w/predictions.jsonl"
    assert output_key == "classification"


def test_non_optimized_topology_can_use_restored_predictions():
    step_outputs = {
        "predict": {"predictions_path": "/w/predictions.jsonl"},
        "restore_usd": {"restored_predictions_path": "/w/restored.jsonl"},
    }
    predictions_path, output_key = resolve_apply_physics_inputs(step_outputs)
    assert predictions_path == "/w/restored.jsonl"
    assert output_key == "classification"


def test_predict_only_topology():
    step_outputs = {"predict": {"predictions_path": "/w/predictions.jsonl"}}
    predictions_path, output_key = resolve_apply_physics_inputs(step_outputs)
    assert predictions_path == "/w/predictions.jsonl"
    assert output_key == "classification"


def test_no_predictions_available_returns_none():
    predictions_path, output_key = resolve_apply_physics_inputs({})
    assert predictions_path is None
    assert output_key == "classification"


def test_custom_output_key_propagates_from_predict():
    step_outputs = {
        "predict": {
            "predictions_path": "/w/predictions.jsonl",
            "output_key": "analysis",
        },
    }
    predictions_path, output_key = resolve_apply_physics_inputs(step_outputs)
    assert predictions_path == "/w/predictions.jsonl"
    assert output_key == "analysis"


def test_output_key_falls_back_to_classification_when_predict_didnt_record_one():
    step_outputs = {
        "predict": {"predictions_path": "/w/predictions.jsonl", "output_key": None},
    }
    _, output_key = resolve_apply_physics_inputs(step_outputs)
    assert output_key == "classification"
