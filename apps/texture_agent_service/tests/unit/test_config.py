# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from ...service.config import ServiceConfig


def test_service_config_reads_prefixed_or_unprefixed_api_key(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TA_NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "fallback-key")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.nvidia_api_key == "fallback-key"
    assert config.session_storage_path == str(sessions)
    assert config.description == "desc"


def test_service_config_falls_back_to_local_sessions_path(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_NVIDIA_API_KEY", "prefixed-key")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    config = ServiceConfig(session_storage_path=str(tmp_path / "missing"))

    assert config.nvidia_api_key == "prefixed-key"
    assert config.session_storage_path.endswith("apps/texture_agent_service/sessions")
    assert config.description == "desc"


def test_service_config_reads_cancel_drain_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_CANCEL_DRAIN_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.cancel_drain_timeout_seconds == 2.5


def test_load_description_reads_repo_readme() -> None:
    description = ServiceConfig._load_description()

    assert "Texture Agent" in description
