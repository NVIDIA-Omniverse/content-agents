# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline task: harmonize material predictions via multi-signal grouping.

Within a single asset run, identifies meshes that likely represent the same
physical part using geometry fingerprints, naming patterns, and prim-path
signatures.  When grouped prims received different material predictions, an LLM
reads all reasonings and decides whether to unify them or keep as-is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class HarmonizePredictionsTask(Task):
    """Harmonize predictions within a single asset using multi-signal grouping."""

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        predictions_path = context.get("predictions_path")
        if not predictions_path:
            raise ValueError("predictions_path is required")

        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        llm_config = context.get("llm_config")
        optimized_usd_path = context.get("optimized_usd_path")

        from ..scene.harmonize import harmonize_asset_predictions

        result_path, remap = harmonize_asset_predictions(
            predictions_path=predictions_path,
            llm_config=llm_config,
            optimized_usd_path=optimized_usd_path,
        )

        context["predictions_path"] = str(result_path)
        context["harmonized_count"] = len(remap)
        context["remap"] = remap
        return context
