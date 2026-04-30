# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request models for Texture Agent Service API."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class PrimTextureOverride(BaseModel):
    """Per-prim prompt/opacity override nested under a material override."""

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(default=None, min_length=1)
    opacity: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Prompt must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def _requires_one_override(self) -> "PrimTextureOverride":
        if self.prompt is None and self.opacity is None:
            raise ValueError("Per-prim override must include prompt or opacity")
        return self


class MaterialTextureOverride(BaseModel):
    """Per-material texture prompt/opacity override accepted by the API."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    opacity: float | None = Field(default=None, ge=0.0, le=1.0)
    per_prim: dict[str, PrimTextureOverride] | None = None

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Prompt must be a non-empty string")
        return stripped

    @field_validator("per_prim")
    @classmethod
    def _validate_per_prim_keys(
        cls, value: dict[str, PrimTextureOverride] | None
    ) -> dict[str, PrimTextureOverride] | None:
        if value is None:
            return None
        for key in value:
            if not key or not key.strip():
                raise ValueError("Per-prim override keys must be non-empty")
        return value


class MaterialTextures(BaseModel):
    """Root model for API material texture overrides."""

    model_config = ConfigDict(extra="forbid")

    root: dict[str, MaterialTextureOverride] = Field(default_factory=dict)

    @field_validator("root")
    @classmethod
    def _validate_material_keys(
        cls, value: dict[str, MaterialTextureOverride]
    ) -> dict[str, MaterialTextureOverride]:
        for key in value:
            if not key or not key.strip():
                raise ValueError("Material override keys must be non-empty")
        return value

    def as_config(self) -> dict[str, dict[str, Any]]:
        """Return the plain dict shape consumed by the texture pipeline."""
        return {
            material: override.model_dump(exclude_none=True)
            for material, override in self.root.items()
        }


class RegenerateRequest(BaseModel):
    """Request to regenerate specific steps from cache."""

    steps: list[TexturePipelineStep] = Field(
        min_length=1,
        description="Steps to re-run from cache (at least one)",
    )

    material_textures: dict[str, MaterialTextureOverride] | None = Field(
        default=None,
        description="Override per-material prompt/opacity for regeneration",
    )

    @field_validator("material_textures")
    @classmethod
    def _validate_material_texture_keys(
        cls, value: dict[str, MaterialTextureOverride] | None
    ) -> dict[str, MaterialTextureOverride] | None:
        if value is None:
            return None
        for key in value:
            if not key or not key.strip():
                raise ValueError("Material override keys must be non-empty")
        return value

    def material_textures_config(self) -> dict[str, dict[str, Any]] | None:
        """Return material overrides in the plain dict format used by YAML config."""
        if self.material_textures is None:
            return None
        return {
            material: override.model_dump(exclude_none=True)
            for material, override in self.material_textures.items()
        }
