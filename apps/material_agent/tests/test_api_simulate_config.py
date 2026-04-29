# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from material_agent.api.simulate_config import patch_config_for_simulate


def test_patch_config_for_simulate_patches_expected_backends_without_mutating_input():
    original = {
        "steps": {
            "predict": {
                "vlm": {"backend": "nim"},
                "llm": {"backend": "nim"},
            },
            "validate_predictions": {"llm": {"backend": "nim"}},
            "harmonize_predictions": {"llm": {"backend": "nim"}},
            "build_dataset_usd": {"renderer": {"backend": "remote"}},
            "render": {"backend": "remote"},
            "cluster_prims": {"embedding_service": "nim"},
            "optimize_usd": {"backend": "remote"},
        },
        "scene": {
            "analyze": {"llm": {"backend": "real"}},
            "reconcile": {"llm": {"backend": "real"}},
            "harmonize": {"llm": {"backend": "real"}},
        },
    }

    patched = patch_config_for_simulate(original)

    assert patched is not original
    assert original["steps"]["predict"]["vlm"]["backend"] == "nim"
    assert patched["steps"]["predict"]["vlm"]["backend"] == "mock"
    assert patched["steps"]["predict"]["vlm"]["api_key"] == "not-used"
    assert patched["steps"]["predict"]["llm"]["backend"] == "mock"
    assert patched["steps"]["validate_predictions"]["llm"]["backend"] == "mock"
    assert patched["steps"]["harmonize_predictions"]["llm"]["backend"] == "mock"
    assert patched["steps"]["build_dataset_usd"]["renderer"]["backend"] == "mock"
    assert patched["steps"]["render"]["backend"] == "mock"
    assert patched["steps"]["cluster_prims"]["embedding_service"] == "mock"
    assert patched["scene"]["reconcile"]["llm"]["backend"] == "mock"
    assert patched["scene"]["harmonize"]["llm"]["backend"] == "mock"
    assert patched["scene"]["analyze"]["llm"]["backend"] == "real"


def test_patch_config_for_simulate_can_mock_scene_analyze():
    patched = patch_config_for_simulate(
        {"scene": {"analyze": {"llm": {"backend": "real"}}}},
        mock_analyze=True,
    )

    assert patched["scene"]["analyze"]["llm"]["backend"] == "mock"
    assert patched["scene"]["analyze"]["llm"]["api_key"] == "not-used"
