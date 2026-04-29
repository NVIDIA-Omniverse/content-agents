# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD asset-identification task backend provisioning."""

from typing import Any

import pytest

from world_understanding.agentic.usd_tasks.identify_asset import IdentifyAssetTask


class _FakeVLM:
    model_name = "fake-vlm"

    def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
        return "{}"


@pytest.mark.parametrize(
    ("backend", "env_name", "env_value"),
    [
        ("nim", "NVIDIA_API_KEY", "nvapi-test"),
        ("openai", "OPENAI_API_KEY", "openai-test"),
        ("anthropic", "ANTHROPIC_API_KEY", "anthropic-test"),
        ("gemini", "GOOGLE_API_KEY", "google-test"),
    ],
)
def test_identify_asset_passes_backend_api_key_from_env(
    monkeypatch,
    tmp_path,
    backend: str,
    env_name: str,
    env_value: str,
):
    captured: dict[str, Any] = {}

    def fake_create_vlm(actual_backend: str, **kwargs: Any) -> _FakeVLM:
        captured["backend"] = actual_backend
        captured["kwargs"] = kwargs
        return _FakeVLM()

    monkeypatch.setenv(env_name, env_value)
    monkeypatch.setattr(
        "world_understanding.functions.models.vision_language_models.create_vlm",
        fake_create_vlm,
    )

    context = {
        "vlm_config": {"backend": backend, "model": "model-test"},
        "output_dir": str(tmp_path),
    }

    IdentifyAssetTask().run(context)

    assert captured == {
        "backend": backend,
        "kwargs": {"model": "model-test", "api_key": env_value},
    }
    assert context["identification"]["asset_type"] == "unknown"
