# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validate_predictions and config_validate_predictions tasks."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from material_agent.tasks.config_validate_predictions import (
    ValidatePredictionsConfigTask,
)
from material_agent.tasks.validate_predictions import (
    ValidatePredictionsTask,
    _best_fuzzy_match,
    _extract_material_name,
    _set_material_name,
)

# ---------------------------------------------------------------------------
# _best_fuzzy_match
# ---------------------------------------------------------------------------


class TestBestFuzzyMatch:
    VALID_NAMES = [
        "Stainless Steel Polished",
        "Gold Polished",
        "Plastic White",
        "Copper Brushed",
        "Rubber Black",
    ]

    def test_exact_match_returns_score_one(self):
        match, score = _best_fuzzy_match("Stainless Steel Polished", self.VALID_NAMES)
        assert match == "Stainless Steel Polished"
        assert score == pytest.approx(1.0 + 0.1, abs=0.01)  # 1.0 ratio + 0.1 bonus

    def test_close_match_returns_high_score(self):
        match, score = _best_fuzzy_match("Stainles Steel Polished", self.VALID_NAMES)
        assert match == "Stainless Steel Polished"
        assert score > 0.8

    def test_token_containment_bonus(self):
        # "Steel Polished" tokens are a subset of "Stainless Steel Polished" tokens,
        # so "Stainless Steel Polished" should get a +0.1 bonus and beat "Gold Polished".
        match, score = _best_fuzzy_match("Steel Polished", self.VALID_NAMES)
        assert match == "Stainless Steel Polished"

    def test_no_match_returns_low_score(self):
        match, score = _best_fuzzy_match("Xylophone Unicorn", self.VALID_NAMES)
        assert score < 0.4


# ---------------------------------------------------------------------------
# _extract_material_name / _set_material_name
# ---------------------------------------------------------------------------


class TestExtractMaterialName:
    def test_extracts_from_dict(self):
        pred = {"materials": {"material": "Gold Polished"}}
        assert _extract_material_name(pred) == "Gold Polished"

    def test_returns_none_for_missing_materials(self):
        assert _extract_material_name({}) is None

    def test_returns_none_for_non_dict_materials(self):
        assert _extract_material_name({"materials": "string_value"}) is None

    def test_returns_none_for_missing_material_key(self):
        assert _extract_material_name({"materials": {"other": "value"}}) is None


class TestSetMaterialName:
    def test_sets_correctly(self):
        pred = {"materials": {"material": "old"}}
        _set_material_name(pred, "new")
        assert pred["materials"]["material"] == "new"

    def test_creates_materials_dict_if_missing(self):
        pred = {}
        _set_material_name(pred, "Copper Brushed")
        assert pred["materials"]["material"] == "Copper Brushed"


# ---------------------------------------------------------------------------
# ValidatePredictionsTask.run()
# ---------------------------------------------------------------------------


def _write_predictions(path: Path, predictions: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")


def _read_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


VALID_NAMES = [
    "Stainless Steel Polished",
    "Gold Polished",
    "Plastic White",
    "Copper Brushed",
    "Rubber Black",
]


class TestValidatePredictionsTaskRun:
    def _make_context(self, predictions_path: Path) -> dict:
        return {
            "predictions_path": str(predictions_path),
            "material_names": VALID_NAMES,
        }

    def test_all_valid_predictions_no_changes(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "Gold Polished"}},
            {"id": "/b", "materials": {"material": "Rubber Black"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["valid"] == 2
        assert stats["auto_corrected"] == 0
        assert stats["no_material"] == 0

        # File unchanged (names still match)
        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == "Gold Polished"
        assert reloaded[1]["materials"]["material"] == "Rubber Black"

    def test_auto_correctable_prediction(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "Stainles Steel Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["auto_corrected"] == 1

        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == "Stainless Steel Polished"

    def test_no_material_field_counted(self, tmp_path):
        preds = [
            {"id": "/a"},  # no materials at all
            {"id": "/b", "materials": {"material": "Gold Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["no_material"] == 1
        assert stats["valid"] == 1

    def test_predictions_file_rewritten_in_place(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "Stainles Steel Polished"}},
            {"id": "/b", "materials": {"material": "Gold Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        task.run(ctx)

        # File should exist at same path and content updated
        assert path.exists()
        reloaded = _read_predictions(path)
        assert len(reloaded) == 2
        assert reloaded[0]["materials"]["material"] == "Stainless Steel Polished"
        assert reloaded[1]["materials"]["material"] == "Gold Polished"

    def test_validation_report_written(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "Stainles Steel Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        task.run(ctx)

        report_path = tmp_path / "validate_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "stats" in report
        assert "auto_corrected" in report
        assert report["stats"]["auto_corrected"] == 1

    def test_missing_predictions_file_skips(self, tmp_path):
        path = tmp_path / "does_not_exist.jsonl"
        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        result = task.run(ctx)
        assert result["validation_stats"]["skipped"] is True


# ---------------------------------------------------------------------------
# ValidatePredictionsConfigTask.run()
# ---------------------------------------------------------------------------


class TestValidatePredictionsConfigTaskRun:
    def test_loads_config_sets_context(self, tmp_path):
        config = {
            "predictions_path": "/some/path/predictions.jsonl",
            "material_names": ["Mat A", "Mat B"],
            "llm": {"backend": "openai", "model": "gpt-4"},
        }
        cfg_path = tmp_path / "config.yaml"
        import yaml

        cfg_path.write_text(yaml.dump(config))

        ctx = {"config_path": str(cfg_path)}
        task = ValidatePredictionsConfigTask()
        result = task.run(ctx)

        assert result["predictions_path"] == "/some/path/predictions.jsonl"
        assert result["material_names"] == ["Mat A", "Mat B"]
        assert result["llm_config"]["backend"] == "openai"

    def test_raises_on_missing_predictions_path(self, tmp_path):
        config = {"material_names": ["Mat A"]}
        cfg_path = tmp_path / "config.yaml"
        import yaml

        cfg_path.write_text(yaml.dump(config))

        ctx = {"config_path": str(cfg_path)}
        task = ValidatePredictionsConfigTask()
        with pytest.raises(ValueError, match="predictions_path"):
            task.run(ctx)

    def test_raises_on_missing_material_names(self, tmp_path):
        config = {"predictions_path": "/some/path.jsonl"}
        cfg_path = tmp_path / "config.yaml"
        import yaml

        cfg_path.write_text(yaml.dump(config))

        ctx = {"config_path": str(cfg_path)}
        task = ValidatePredictionsConfigTask()
        with pytest.raises(ValueError, match="material_names"):
            task.run(ctx)

    def test_raises_on_missing_config_path(self):
        ctx = {}
        task = ValidatePredictionsConfigTask()
        with pytest.raises(ValueError, match="config_path"):
            task.run(ctx)

    def test_raises_on_nonexistent_config_file(self, tmp_path):
        ctx = {"config_path": str(tmp_path / "nope.yaml")}
        task = ValidatePredictionsConfigTask()
        with pytest.raises(FileNotFoundError):
            task.run(ctx)
