# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from texture_agent.config.unified_config import config_to_context

from ...service.config import ServiceConfig
from ...service.routers.pipeline_router import build_default_pipeline_config


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


def test_service_config_passes_s3_connection_pool_size(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    config = ServiceConfig(
        session_storage_path=str(sessions),
        storage_kind="s3",
        storage_s3_bucket="bucket",
        storage_s3_max_pool_connections=123,
    )
    store = config.build_session_store()

    assert store._max_pool_connections == 123


def test_service_config_uses_wu_s3_fallbacks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TA_STORAGE_S3_BUCKET", raising=False)
    monkeypatch.delenv("TA_STORAGE_S3_REGION", raising=False)
    monkeypatch.delenv("TA_STORAGE_S3_PROFILE", raising=False)
    monkeypatch.setenv("WU_S3_BUCKET", "wu-bucket")
    monkeypatch.setenv("WU_S3_REGION", "us-east-2")
    monkeypatch.setenv("WU_S3_PROFILE", "wu-profile")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    config = ServiceConfig(
        session_storage_path=str(sessions),
        storage_kind="s3",
    )
    store = config.build_session_store()

    assert config.storage_s3_bucket == "wu-bucket"
    assert config.storage_s3_region == "us-east-2"
    assert config.storage_s3_profile == "wu-profile"
    assert store.bucket == "wu-bucket"
    assert store._region == "us-east-2"
    assert store._profile == "wu-profile"


def test_load_description_reads_repo_readme() -> None:
    description = ServiceConfig._load_description()

    assert "Texture Agent" in description


def test_default_pipeline_config_fails_on_any_texture_generation_error(
    tmp_path: Path,
) -> None:
    """Service-created runs must not silently complete with dropped materials."""
    config = build_default_pipeline_config(
        session_id="session-1",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Steel": {"prompt": "brushed steel"}},
    )

    context = config_to_context(config)

    assert context["texture_config"]["failure_threshold"] == 0.0
