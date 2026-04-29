# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import ValidationError

from material_agent.workflows.models import (
    DatasetEntry,
    Evaluation,
    Metrics,
    Prediction,
)


def test_workflow_models_validate_expected_fields():
    entry = DatasetEntry(
        id="entry-1",
        image_path="images/a.png",
        ground_truth="metal",
        metadata={"source": "fixture"},
    )
    prediction = Prediction(
        id="entry-1",
        image_path="images/a.png",
        materials="metal",
        confidence=0.91,
        ground_truth="metal",
    )
    evaluation = Evaluation(
        id="entry-1",
        materials="metal",
        ground_truth="metal",
        score=5,
        explanation="correct",
    )
    metrics = Metrics(
        functional_correctness_score=4.5,
        success_rate=75.0,
        total_cases=4,
        valid_cases=4,
        successful_cases=3,
        score_distribution={1: 0, 2: 0, 3: 1, 4: 1, 5: 2},
        failure_count=1,
    )

    assert entry.metadata == {"source": "fixture"}
    assert prediction.confidence == 0.91
    assert evaluation.score == 5
    assert metrics.successful_cases == 3


def test_workflow_models_enforce_score_range():
    try:
        Evaluation(
            id="entry-1",
            materials="metal",
            ground_truth="metal",
            score=6,
            explanation="invalid",
        )
    except ValidationError as exc:
        assert "less than or equal to 5" in str(exc)
    else:
        raise AssertionError("expected pydantic ValidationError")
