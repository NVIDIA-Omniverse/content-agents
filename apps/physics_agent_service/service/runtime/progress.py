"""Shared progress model for pipeline steps.

Single source of truth for `/pipeline/{id}/status` + SSE overall-progress
math. The EventBus (`bus.py`) and the store-backed fallback in
`session/manager.py` both read from here so the two paths can't drift.

Percentages cover the default service pipeline (optimize_usd is disabled
by default in `build_default_pipeline_config`, but we keep a slot for it
so a user who enables it still lands in a coherent 0–100% range):

    optimize_usd              0 →  5
    identify_asset            5 → 10
    build_dataset_usd        10 → 50
    build_dataset_prepare    50 → 60
    predict                  60 → 90   <- stops at 90 on purpose
    apply_physics            90 → 100

predict stops at 90 so the auto "status = completed" branch in
EventBus._update_overall_progress_on_completion only fires when
apply_physics finishes and overall percent actually reaches 100.
"""

from __future__ import annotations

# Visible steps in user-facing order for the service's default pipeline
# (optimize_usd is off by default).
VISIBLE_STEP_ORDER: list[str] = [
    "identify_asset",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    "predict",
    "apply_physics",
]

TOTAL_VISIBLE_STEPS: int = len(VISIBLE_STEP_ORDER)

STEP_DISPLAY_NAMES: dict[str, str] = {
    "optimize_usd": "Optimizing USD",
    "identify_asset": "Identifying Asset",
    "build_dataset_usd": "Rendering USD Scene",
    "build_dataset_prepare_dataset": "Preparing Dataset",
    "predict": "Running VLM Predictions",
    "apply_physics": "Applying Physics Schemas",
}

# Overall-progress weight ranges per step, used while a step is running.
# (start, end) — an in-flight step at percent p contributes
# `start + (end - start) * p / 100` to the overall percentage.
STEP_WEIGHTS: dict[str, tuple[int, int]] = {
    "optimize_usd": (0, 5),
    "identify_asset": (5, 10),
    "build_dataset_usd": (10, 50),
    "build_dataset_prepare_dataset": (50, 60),
    "predict": (60, 90),
    "apply_physics": (90, 100),
}

# Overall percent to snap to when a step completes.
STEP_COMPLETION_PERCENT: dict[str, int] = {
    "optimize_usd": 5,
    "identify_asset": 10,
    "build_dataset_usd": 50,
    "build_dataset_prepare_dataset": 60,
    "predict": 90,  # keep below 100 so apply_physics drives the final flip
    "apply_physics": 100,
}

# 1-indexed position among visible steps (for UIs that show "step N of M").
# optimize_usd collapses onto the same slot as identify_asset — when the
# user enables optimize_usd as well, the UI still advances naturally.
STEP_NUMBER: dict[str, int] = {
    "optimize_usd": 1,
    "identify_asset": 1,
    "build_dataset_usd": 2,
    "build_dataset_prepare_dataset": 3,
    "predict": 4,
    "apply_physics": 5,
}
