# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material pipeline render concurrency limits."""

from ...service.routers import pipeline_router


def test_restore_usd_is_injected_before_apply_when_optimization_runs():
    steps = [
        "optimize_usd",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "apply",
        "render",
    ]

    result = pipeline_router._inject_restore_usd_step(
        steps,
        optimize_usd_enabled=True,
    )

    assert result == [
        "optimize_usd",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "restore_usd",
        "apply",
        "render",
    ]


def test_restore_usd_is_not_injected_without_optimization():
    steps = [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "apply",
        "render",
    ]

    result = pipeline_router._inject_restore_usd_step(
        steps,
        optimize_usd_enabled=False,
    )

    assert result == steps


def test_restore_usd_is_not_injected_without_apply():
    steps = [
        "optimize_usd",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
    ]

    result = pipeline_router._inject_restore_usd_step(
        steps,
        optimize_usd_enabled=True,
    )

    assert result == steps


def test_restore_usd_injection_keeps_existing_restore_step():
    steps = [
        "optimize_usd",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "restore_usd",
        "apply",
    ]

    result = pipeline_router._inject_restore_usd_step(
        steps,
        optimize_usd_enabled=True,
    )

    assert result == steps


def test_default_render_request_limit_is_preserved_without_caps(monkeypatch):
    monkeypatch.delenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.delenv("MA_RENDER_GLOBAL_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.setattr(pipeline_router.config, "max_render_num_workers", 32)
    pipeline_config = {
        "steps": {
            "build_dataset_usd": {
                "num_workers": 32,
                "max_concurrent_requests": 128,
            }
        }
    }

    pipeline_router._apply_build_dataset_render_worker_limit(pipeline_config, None)

    build_dataset_config = pipeline_config["steps"]["build_dataset_usd"]
    assert build_dataset_config["num_workers"] == 32
    assert build_dataset_config["max_concurrent_requests"] == 128


def test_render_worker_override_caps_async_request_limit(monkeypatch):
    monkeypatch.delenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.delenv("MA_RENDER_GLOBAL_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.setattr(pipeline_router.config, "max_render_num_workers", 32)
    pipeline_config = {
        "steps": {
            "build_dataset_usd": {
                "num_workers": 32,
                "max_concurrent_requests": 32,
            }
        }
    }

    pipeline_router._apply_build_dataset_render_worker_limit(pipeline_config, 1)

    build_dataset_config = pipeline_config["steps"]["build_dataset_usd"]
    assert build_dataset_config["num_workers"] == 1
    assert build_dataset_config["max_concurrent_requests"] == 1


def test_global_render_cap_limits_async_request_limit(monkeypatch):
    monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "8")
    monkeypatch.delenv("MA_RENDER_GLOBAL_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.setattr(pipeline_router.config, "max_render_num_workers", 32)
    pipeline_config = {
        "steps": {
            "build_dataset_usd": {
                "num_workers": 32,
                "max_concurrent_requests": 128,
            }
        }
    }

    pipeline_router._apply_build_dataset_render_worker_limit(pipeline_config, None)

    build_dataset_config = pipeline_config["steps"]["build_dataset_usd"]
    assert build_dataset_config["num_workers"] == 32
    assert build_dataset_config["max_concurrent_requests"] == 8


def test_large_scene_render_batch_size_is_capped(monkeypatch):
    monkeypatch.setattr(pipeline_router.config, "scene_render_batch_size", 16)
    pipeline_config = {
        "steps": {
            "build_dataset_usd": {
                "batch_size": 64,
            }
        }
    }

    pipeline_router._apply_large_scene_render_batch_limit(pipeline_config)

    assert pipeline_config["steps"]["build_dataset_usd"]["batch_size"] == 16


def test_large_scene_render_batch_size_keeps_lower_value(monkeypatch):
    monkeypatch.setattr(pipeline_router.config, "scene_render_batch_size", 16)
    pipeline_config = {
        "steps": {
            "build_dataset_usd": {
                "batch_size": 8,
            }
        }
    }

    pipeline_router._apply_large_scene_render_batch_limit(pipeline_config)

    assert pipeline_config["steps"]["build_dataset_usd"]["batch_size"] == 8
