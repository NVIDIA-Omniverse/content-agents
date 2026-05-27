# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validation_agent.utils."""

from __future__ import annotations

from validation_agent import utils


def test_get_version_returns_package_version_or_dev_fallback(monkeypatch) -> None:
    monkeypatch.setattr(utils, "version", lambda _name: "1.2.3")
    assert utils.get_version() == "1.2.3"

    def _raise(_name: str) -> str:
        raise utils.PackageNotFoundError

    monkeypatch.setattr(utils, "version", _raise)
    assert utils.get_version() == "0.0.1-dev"
