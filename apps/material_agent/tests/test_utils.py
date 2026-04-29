# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material_agent.utils."""

from __future__ import annotations

from unittest.mock import Mock

from material_agent import utils


def test_get_version_returns_package_version_or_dev_fallback(monkeypatch) -> None:
    monkeypatch.setattr(utils, "version", lambda _name: "1.2.3")
    assert utils.get_version() == "1.2.3"

    def _raise(_name: str) -> str:
        raise utils.PackageNotFoundError

    monkeypatch.setattr(utils, "version", _raise)
    assert utils.get_version() == "0.0.1-dev"


def test_calculate_metrics_handles_empty_scores() -> None:
    assert utils.calculate_metrics([], []) == {
        "functional_correctness_score": 0,
        "success_rate": 0,
        "exact_match_rate": 0,
        "total_cases": 0,
        "successful_cases": 0,
        "exact_matches": 0,
        "score_distribution": {},
        "failure_count": 0,
    }


def test_calculate_metrics_computes_distribution_and_failures() -> None:
    metrics = utils.calculate_metrics(
        [5, 4, 0, 2],
        [
            {"score": 5, "exact_match": True},
            {"score": 4},
            {"score": 0},
            {"score": 2},
        ],
        success_threshold=4.0,
    )

    assert metrics == {
        "functional_correctness_score": 3.67,
        "success_rate": 66.7,
        "exact_match_rate": 25.0,
        "total_cases": 4,
        "valid_cases": 3,
        "successful_cases": 2,
        "exact_matches": 1,
        "score_distribution": {1: 0, 2: 1, 3: 0, 4: 1, 5: 1},
        "failure_count": 2,
    }


def test_display_results_and_format_prediction_output(monkeypatch) -> None:
    fake_console = Mock()
    monkeypatch.setattr(utils, "console", fake_console)

    utils.display_results(
        {
            "functional_correctness_score": 4.5,
            "success_rate": 75.0,
            "exact_match_rate": 50.0,
            "total_cases": 4,
            "valid_cases": 3,
            "successful_cases": 3,
            "exact_matches": 2,
            "failure_count": 1,
            "score_distribution": {1: 0, 2: 1, 3: 0, 4: 1, 5: 2},
        },
        title="Demo",
    )

    printed_text = "\n".join(
        str(call.args[0]) for call in fake_console.print.call_args_list
    )
    assert "Demo" in printed_text
    assert "Score Distribution" in printed_text
    assert "Score 5" in printed_text

    assert utils.format_prediction_output(
        {
            "id": "mesh-1",
            "image_path": "img.png",
            "vlm_response": "Steel",
            "confidence": 0.9,
        }
    ) == {
        "id": "mesh-1",
        "image_path": "img.png",
        "materials": "Steel",
        "confidence": 0.9,
    }
    assert utils.format_prediction_output(
        {"id": "mesh-2"}, include_confidence=False
    ) == {
        "id": "mesh-2",
        "image_path": "",
        "materials": "",
    }
