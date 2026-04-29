# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request models for Texture Agent Service API."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TexturePipelineStep(StrEnum):
    """Available pipeline steps."""

    PREPARE_UVS = "prepare_uvs"
    DISCOVER_MATERIALS = "discover_materials"
    GENERATE_PROMPTS = "generate_prompts"
    RENDER_PREVIEWS = "render_previews"
    GENERATE_TEXTURES = "generate_textures"
    BLEND_TEXTURES = "blend_textures"
    APPLY_TEXTURES = "apply_textures"
    RENDER = "render"


class RegenerateRequest(BaseModel):
    """Request to regenerate specific steps from cache."""

    steps: list[TexturePipelineStep] = Field(description="Steps to re-run from cache")

    material_textures: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Override per-material prompt/opacity for regeneration",
    )
