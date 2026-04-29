# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prediction symmetry and consistency analysis."""

from material_agent.tasks.prediction_analyzer import PredictionAnalyzer


def _prediction(prim_id: str, material: str) -> dict:
    return {"id": prim_id, "materials": {"material": material}}


def test_detects_g1_left_right_path_symmetry_violations() -> None:
    """G1 side names start with left_/right_ and end in generic mesh leaves."""
    predictions = [
        _prediction(
            "/g1/humanoid__unitree__g1__left_shoulder_roll_link/visuals/"
            "left_shoulder_roll_link/mesh",
            "Aluminum",
        ),
        _prediction(
            "/g1/humanoid__unitree__g1__right_shoulder_roll_link/visuals/"
            "right_shoulder_roll_link/mesh",
            "Plastic Black",
        ),
        _prediction(
            "/g1/humanoid__unitree__g1__left_hip_pitch_link/visuals/"
            "left_hip_pitch_link/mesh",
            "Aluminum",
        ),
        _prediction(
            "/g1/humanoid__unitree__g1__right_hip_pitch_link/visuals/"
            "right_hip_pitch_link/mesh",
            "Plastic Black",
        ),
    ]

    result = PredictionAnalyzer(predictions).analyze()

    assert len(result.symmetry_pairs) == 2
    assert len(result.symmetry_violations) == 2
    assert {v.detection_method for v in result.symmetry_violations} == {"path"}
    assert {
        (v.prim_a, v.prim_b, v.material_a, v.material_b)
        for v in result.symmetry_violations
    } == {
        (
            "/g1/humanoid__unitree__g1__left_hip_pitch_link/visuals/"
            "left_hip_pitch_link/mesh",
            "/g1/humanoid__unitree__g1__right_hip_pitch_link/visuals/"
            "right_hip_pitch_link/mesh",
            "Aluminum",
            "Plastic Black",
        ),
        (
            "/g1/humanoid__unitree__g1__left_shoulder_roll_link/visuals/"
            "left_shoulder_roll_link/mesh",
            "/g1/humanoid__unitree__g1__right_shoulder_roll_link/visuals/"
            "right_shoulder_roll_link/mesh",
            "Aluminum",
            "Plastic Black",
        ),
    }


def test_detects_g1_geometry_leaf_symmetry_with_readable_names() -> None:
    predictions = [
        _prediction(
            "/visuals/left_ankle_roll_link/left_ankle_roll_link/mesh/Geometry",
            "Plastic Black",
        ),
        _prediction(
            "/visuals/right_ankle_roll_link/right_ankle_roll_link/mesh/Geometry",
            "Aluminum",
        ),
    ]

    result = PredictionAnalyzer(predictions).analyze()

    assert len(result.symmetry_pairs) == 1
    assert len(result.symmetry_violations) == 1
    assert result.symmetry_violations[0].detection_method == "path"
    assert "left_ankle_roll_link" in result.critique
    assert "right_ankle_roll_link" in result.critique
    assert "mesh" not in result.critique


def test_left_right_token_matching_does_not_match_unrelated_substrings() -> None:
    predictions = [
        _prediction("/robot/highlight_panel/mesh", "Aluminum"),
        _prediction("/robot/highright_panel/mesh", "Plastic Black"),
    ]

    result = PredictionAnalyzer(predictions).analyze()

    assert result.symmetry_pairs == []
    assert result.symmetry_violations == []


def test_symmetry_resolution_can_be_left_to_vlm_feedback() -> None:
    predictions = [
        _prediction("/robot/left_arm/mesh", "Aluminum"),
        _prediction("/robot/right_arm/mesh", "Plastic Black"),
    ]

    result = PredictionAnalyzer(
        predictions,
        resolve_symmetry_directly=False,
        resolve_consistency_directly=False,
    ).analyze()

    assert len(result.symmetry_violations) == 1
    assert result.resolved_assignments == {}
    assert set(result.prim_feedback) == {
        "/robot/left_arm/mesh",
        "/robot/right_arm/mesh",
    }
    assert "Recommended:" not in result.critique
    assert "Both should use" not in result.critique
    assert "reference image" in result.critique


def test_consistency_resolution_can_be_left_to_vlm_feedback() -> None:
    predictions = [
        _prediction("/robot/visuals/panel_a/mesh", "Aluminum"),
        _prediction("/robot/visuals/panel_b/mesh", "Plastic Black"),
    ]

    result = PredictionAnalyzer(
        predictions,
        resolve_consistency_directly=False,
    ).analyze()

    assert len(result.consistency_violations) == 1
    assert result.resolved_assignments == {}
    assert set(result.prim_feedback) == {
        "/robot/visuals/panel_a/mesh",
        "/robot/visuals/panel_b/mesh",
    }
    assert "Recommended:" not in result.critique
    assert "Use '" not in result.critique
    assert "reference image" in result.critique
