# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from physics_agent.functions.mass_scale_quality import (
    HIGH_MASS_KG,
    build_mass_scale_quality_warnings,
    extract_bbox_metrics_meters,
    has_mass_scale_suspicious_warning,
    merge_quality_warnings,
)


def test_mass_scale_quality_warns_for_high_mass_with_large_bbox():
    prediction = {
        "id": "/World/Robot/oversized_link",
        "classification": {
            "physical_properties": {
                "density": 2700,
                "estimated_mass_kg": 25000,
            }
        },
    }
    dataset_entry = {
        "id": "/World/Robot/oversized_link",
        "metadata": {
            "world_bbox_meters": {
                "size": [8.0, 0.8, 0.8],
            }
        },
    }

    warnings = build_mass_scale_quality_warnings(prediction, dataset_entry)

    assert [warning["code"] for warning in warnings] == ["mass_scale_suspicious"]
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["details"]["max_dimension_m"] == 8.0
    assert warnings[0]["details"]["estimated_mass_kg"] == 25000
    assert warnings[0]["details"]["thresholds"]["high_mass_kg"] == HIGH_MASS_KG


def test_mass_scale_quality_parses_bbox_metrics_from_prompt_fallback():
    dataset_entry = {
        "id": "prim-1",
        "user_prompt": (
            "Context:\n"
            "Geometric info:\n"
            "  - Dimensions (meters): width=1.500m, height=2.000m, depth=3.000m\n"
            "  - Bounding box volume: 9.000000 m^3"
        ),
    }

    metrics = extract_bbox_metrics_meters(dataset_entry)

    assert metrics["size_m"] == [1.5, 2.0, 3.0]
    assert metrics["max_dimension_m"] == 3.0
    assert metrics["volume_m3"] == 9.0


def test_mass_scale_quality_parses_plain_and_unicode_volume_units():
    for volume_text in (
        "Bounding box volume: 9.000000 m3",
        "Bounding box volume: 9.000000 m³",
    ):
        metrics = extract_bbox_metrics_meters({"user_prompt": volume_text})

        assert metrics["volume_m3"] == 9.0


def test_mass_scale_quality_ignores_normal_handheld_part():
    prediction = {
        "id": "prim-1",
        "classification": {
            "physical_properties": {
                "density": 1200,
                "estimated_mass_kg": 0.25,
            }
        },
    }
    dataset_entry = {
        "id": "prim-1",
        "metadata": {
            "world_bbox_meters": {
                "size": [0.12, 0.04, 0.03],
            }
        },
    }

    assert build_mass_scale_quality_warnings(prediction, dataset_entry) == []


def test_mass_scale_quality_helpers_filter_and_dedupe_warnings():
    warning = {
        "code": "mass_scale_suspicious",
        "severity": "warning",
        "message": "synthetic",
        "details": {"source": "existing"},
    }
    existing = [warning, "not-a-warning", {"code": "other", "severity": "info"}]
    generated = [
        {**warning, "details": {"source": "generated"}},
        {"code": "other", "severity": "info"},
    ]

    assert has_mass_scale_suspicious_warning({"quality_warnings": [warning]})
    assert not has_mass_scale_suspicious_warning(
        {"quality_warnings": [{**warning, "severity": "info"}]}
    )
    merged = merge_quality_warnings(existing, generated)

    assert [item["code"] for item in merged] == [
        "mass_scale_suspicious",
        "other",
    ]
    assert merged[0]["details"] == {"source": "existing"}
