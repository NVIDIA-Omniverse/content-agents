# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request model validation tests for the texture-agent service."""

import pytest
from pydantic import ValidationError

from ...service.models.requests import (
    MaterialTextures,
    RegenerateRequest,
    TexturePipelineStep,
)


def test_material_textures_strips_nested_per_prim_prompt() -> None:
    payload = MaterialTextures(
        root={
            "Steel": {
                "prompt": " weathered steel ",
                "per_prim": {
                    "/World/Rung_01": {"prompt": " scrape marks "},
                },
            }
        }
    )

    assert payload.as_config() == {
        "Steel": {
            "prompt": "weathered steel",
            "per_prim": {
                "/World/Rung_01": {"prompt": "scrape marks"},
            },
        }
    }


def test_material_textures_rejects_empty_per_prim_override() -> None:
    with pytest.raises(
        ValidationError,
        match="Per-prim override must include prompt or opacity",
    ):
        MaterialTextures(
            root={
                "Steel": {
                    "prompt": "weathered steel",
                    "per_prim": {"/World/Rung_01": {}},
                }
            }
        )


def test_regenerate_request_without_material_textures_has_no_override() -> None:
    request = RegenerateRequest(steps=[TexturePipelineStep.GENERATE_TEXTURES])

    assert request.material_textures_config() is None
