# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared material-name helpers."""

from __future__ import annotations

import pytest

from material_agent.materials import (
    is_actionable_material_name,
    is_unknown_material_name,
    normalize_material_name,
)


@pytest.mark.parametrize("name", ["__UNKNOWN__", "__unknown__", " __UNKNOWN__ "])
def test_is_unknown_material_name_accepts_exact_sentinel_variants(name: str) -> None:
    assert is_unknown_material_name(name)
    assert not is_actionable_material_name(name)


@pytest.mark.parametrize(
    "name", [None, "", "   ", "unknown", "Unknown", "unknown material", 123]
)
def test_is_unknown_material_name_rejects_non_sentinel_values(name: object) -> None:
    assert not is_unknown_material_name(name)


def test_is_actionable_material_name_requires_non_unknown_text() -> None:
    assert is_actionable_material_name(" Steel ")
    assert is_actionable_material_name("Unknown")
    assert normalize_material_name(" Steel ") == "Steel"
    assert not is_actionable_material_name("")
