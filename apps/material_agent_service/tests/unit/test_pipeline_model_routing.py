# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for service model routing configuration."""

from __future__ import annotations

from ...service.routers import pipeline_router


def test_predict_model_routing_applies_service_token_limits(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_router.config, "vlm_temperature", 0.2)
    monkeypatch.setattr(pipeline_router.config, "vlm_max_tokens", 512)
    monkeypatch.setattr(pipeline_router.config, "llm_temperature", 0.1)
    monkeypatch.setattr(pipeline_router.config, "llm_max_tokens", 256)

    config = {
        "steps": {
            "predict": {
                "vlm": {"max_tokens": 9999},
                "llm": {"max_tokens": 9999},
            }
        }
    }
    routing = pipeline_router._ModelRouting(
        vlm_backend="nim",
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_nim_base_url="http://vlm-nim:8000/v1",
        llm_backend="nim",
        llm_model="Qwen/Qwen2.5-VL-7B-Instruct",
        llm_nim_base_url="http://vlm-nim:8000/v1",
        llm_uses_vlm_sidecar=True,
    )

    pipeline_router._configure_predict_model_routing(config, routing)

    predict_config = config["steps"]["predict"]
    assert predict_config["vlm"]["temperature"] == 0.2
    assert predict_config["vlm"]["max_tokens"] == 512
    assert predict_config["llm"]["temperature"] == 0.1
    assert predict_config["llm"]["max_tokens"] == 256
    assert predict_config["llm"]["base_url"] == "http://vlm-nim:8000/v1"
