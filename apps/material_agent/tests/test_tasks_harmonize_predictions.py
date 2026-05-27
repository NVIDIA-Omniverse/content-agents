# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for HarmonizePredictionsTask."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from material_agent.tasks.harmonize_predictions import HarmonizePredictionsTask


def _write_predictions(path: Path, predictions: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")


@pytest.fixture(autouse=True)
def _fake_scene_harmonize():
    """Inject a fake material_agent.scene.harmonize module so the lazy import works."""
    harmonize_mod = types.ModuleType("material_agent.scene.harmonize")
    harmonize_mod.harmonize_asset_predictions = MagicMock()  # type: ignore[attr-defined]

    scene_mod = types.ModuleType("material_agent.scene")
    scene_mod.harmonize = harmonize_mod  # type: ignore[attr-defined]

    with patch.dict(
        sys.modules,
        {
            "material_agent.scene": scene_mod,
            "material_agent.scene.harmonize": harmonize_mod,
        },
    ):
        yield harmonize_mod.harmonize_asset_predictions


class TestHarmonizePredictionsTask:
    def test_missing_predictions_path_raises(self):
        task = HarmonizePredictionsTask()
        with pytest.raises(ValueError, match="predictions_path"):
            task.run({})

    def test_nonexistent_file_raises(self, tmp_path):
        task = HarmonizePredictionsTask()
        ctx = {"predictions_path": str(tmp_path / "does_not_exist.jsonl")}
        with pytest.raises(FileNotFoundError, match="Predictions file not found"):
            task.run(ctx)

    def test_successful_run_updates_context(self, tmp_path, _fake_scene_harmonize):
        mock_harmonize = _fake_scene_harmonize
        predictions_path = tmp_path / "predictions.jsonl"
        _write_predictions(
            predictions_path,
            [{"id": "/a", "materials": {"material": "Gold"}}],
        )

        harmonized_path = tmp_path / "predictions_harmonized.jsonl"
        harmonized_path.touch()
        remap = {"/a": "Silver"}
        mock_harmonize.return_value = (harmonized_path, remap)

        ctx = {
            "predictions_path": str(predictions_path),
            "llm_config": {"backend": "openai", "model": "gpt-4"},
        }
        task = HarmonizePredictionsTask()
        result = task.run(ctx)

        assert result["predictions_path"] == str(harmonized_path)
        assert result["harmonized_count"] == 1
        assert result["remap"] == remap
        mock_harmonize.assert_called_once_with(
            predictions_path=predictions_path,
            llm_config={"backend": "openai", "model": "gpt-4"},
            optimized_usd_path=None,
            trusted_root=tmp_path,
        )

    def test_llm_config_read_from_context(self, tmp_path, _fake_scene_harmonize):
        mock_harmonize = _fake_scene_harmonize
        predictions_path = tmp_path / "predictions.jsonl"
        _write_predictions(
            predictions_path,
            [{"id": "/a", "materials": {"material": "Gold"}}],
        )

        mock_harmonize.return_value = (predictions_path, {})

        llm_config = {"backend": "azure", "model": "gpt-5", "temperature": 0.5}
        ctx = {
            "predictions_path": str(predictions_path),
            "llm_config": llm_config,
            "optimized_usd_path": "/some/path.usd",
        }
        task = HarmonizePredictionsTask()
        task.run(ctx)

        mock_harmonize.assert_called_once_with(
            predictions_path=predictions_path,
            llm_config=llm_config,
            optimized_usd_path="/some/path.usd",
            trusted_root=tmp_path,
        )
