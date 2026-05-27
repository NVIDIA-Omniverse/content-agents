# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD stage data URI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from world_understanding.utils.usd.stage import create_data_uri_from_file


def test_create_data_uri_rejects_invalid_mime_type(tmp_path: Path) -> None:
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    with pytest.raises(ValueError, match="Invalid MIME type"):
        create_data_uri_from_file(
            usd_path,
            mime_type='model/vnd.usd";base64,evil=',
        )


def test_create_data_uri_accepts_standard_mime_type(tmp_path: Path) -> None:
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    data_uri = create_data_uri_from_file(usd_path, mime_type="model/vnd.usd")

    assert data_uri.startswith("data:model/vnd.usd;name=scene.usda;base64,")
