# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared material-name helpers for material-agent tasks."""

from __future__ import annotations

UNKNOWN_MATERIAL_SENTINEL = "__UNKNOWN__"
DISALLOWED_UNKNOWN_VALIDATION_STATUS = "disallowed_unknown"
PREDICTION_CONTAINER_KEYS = ("predictions", "results", "items", "objects")
PREDICTION_ID_KEYS = ("id", "object_id", "prim_path", "path")
# Top-level prediction material fields. The singular "materials" container is
# handled separately because it may be either a string or a structured dict.
PREDICTION_MATERIAL_KEYS = ("material", "predicted_material")
PREDICTION_VALIDATION_STATUS_KEYS = ("validation_status", "material_validation_status")
_UNKNOWN_MATERIAL_SENTINEL_NORMALIZED = UNKNOWN_MATERIAL_SENTINEL.lower()
_DISALLOWED_UNKNOWN_VALIDATION_STATUS_NORMALIZED = (
    DISALLOWED_UNKNOWN_VALIDATION_STATUS.lower()
)


def normalize_material_name(name: str) -> str:
    """Normalize whitespace around a material name while preserving display case."""
    return name.strip()


def is_unknown_material_name(name: object) -> bool:
    """Return True when a material value is the supported unknown sentinel."""
    return (
        isinstance(name, str)
        and normalize_material_name(name).lower()
        == _UNKNOWN_MATERIAL_SENTINEL_NORMALIZED
    )


def is_disallowed_unknown_validation_status(status: object) -> bool:
    """Return True when validation recorded a cleared unknown sentinel."""
    return (
        isinstance(status, str)
        and normalize_material_name(status).lower()
        == _DISALLOWED_UNKNOWN_VALIDATION_STATUS_NORMALIZED
    )


def is_actionable_material_name(name: object) -> bool:
    """Return True when a material should be resolved and applied."""
    return (
        isinstance(name, str)
        and bool(normalize_material_name(name))
        and not is_unknown_material_name(name)
    )
