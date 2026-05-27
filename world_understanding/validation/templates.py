# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation Agent V1 template definitions and registry helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import Field

from world_understanding.validation.models import ValidationModel, field_validator

V1_TEMPLATE_NAMES = (
    "look_right",
    "render_valid",
    "physics_sane",
    "physical_behavior",
)


class ValidationContractError(ValueError):
    """Raised when validation contract definitions are inconsistent."""


class ValidationTemplateDefinition(ValidationModel):
    """Static contract metadata for a Validation Agent template."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required_input_kinds: tuple[str, ...] = Field(default_factory=tuple)
    optional_input_kinds: tuple[str, ...] = Field(default_factory=tuple)
    required_capabilities: tuple[str, ...] = Field(default_factory=tuple)
    issue_code_namespaces: tuple[str, ...] = Field(default_factory=tuple)
    output_evidence: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator(
        "required_input_kinds",
        "optional_input_kinds",
        "required_capabilities",
        "issue_code_namespaces",
        "output_evidence",
        mode="before",
    )
    @classmethod
    def _normalize_string_tuple(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Sequence):
            return tuple(str(item) for item in value)
        raise ValueError("Expected a string or sequence of strings")


DEFAULT_TEMPLATE_DEFINITIONS = (
    ValidationTemplateDefinition(
        name="look_right",
        description="Prompt/reference visual validation over rendered or supplied imagery.",
        required_input_kinds=("images_or_usd",),
        optional_input_kinds=(
            "reference_image",
            "sampled_video_frame",
            "focused_render",
        ),
        required_capabilities=("vlm",),
        issue_code_namespaces=("visual",),
        output_evidence=("judge_plan", "image_caption_pairs", "judgment"),
    ),
    ValidationTemplateDefinition(
        name="render_valid",
        description="Render evidence preflight and render-artifact detection.",
        required_input_kinds=("images_or_render_bundle_or_usd",),
        optional_input_kinds=("animation_frame", "render_response"),
        required_capabilities=(),
        issue_code_namespaces=("render", "ovrtx"),
        output_evidence=("image_paths", "animation_frames", "render_response"),
    ),
    ValidationTemplateDefinition(
        name="physics_sane",
        description="Deterministic USD physics authoring sanity checks.",
        required_input_kinds=("usd",),
        optional_input_kinds=("asset_validator_report",),
        required_capabilities=(),
        issue_code_namespaces=("physics", "physics_sane"),
        output_evidence=("usd_paths", "physics_summary"),
    ),
    ValidationTemplateDefinition(
        name="physical_behavior",
        description=(
            "Evidence-backed physical behavior validation from motion and "
            "Physics Agent refine artifacts."
        ),
        required_input_kinds=("video_or_animation_or_policy",),
        optional_input_kinds=(
            "time_sampled_usd",
            "animation_usd",
            "video",
            "sampled_video_frame",
            "simulation_json",
            "trajectory_metrics",
            "refine_summary",
        ),
        required_capabilities=(),
        issue_code_namespaces=("physics", "physical_behavior"),
        output_evidence=(
            "resolution",
            "available_evidence",
            "refine_summaries",
            "behavior_summary",
        ),
    ),
)


class ValidationTemplateRegistry:
    """Allowlist registry for stable Validation Agent V1 template definitions."""

    def __init__(
        self,
        definitions: Iterable[
            ValidationTemplateDefinition
        ] = DEFAULT_TEMPLATE_DEFINITIONS,
    ) -> None:
        self._definitions: dict[str, ValidationTemplateDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ValidationTemplateDefinition) -> None:
        """Register one template definition."""

        if definition.name not in V1_TEMPLATE_NAMES:
            raise ValidationContractError(
                f"Unknown validation template {definition.name!r}. "
                f"Known templates: {', '.join(V1_TEMPLATE_NAMES)}"
            )
        if definition.name in self._definitions:
            raise ValidationContractError(
                f"Validation template {definition.name!r} is already registered"
            )
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ValidationTemplateDefinition:
        """Return the template definition for ``name``."""

        self.validate_template_names((name,))
        return self._definitions[name]

    def names(self) -> tuple[str, ...]:
        """Return registered names in canonical V1 order."""

        return tuple(name for name in V1_TEMPLATE_NAMES if name in self._definitions)

    def definitions(self) -> tuple[ValidationTemplateDefinition, ...]:
        """Return registered definitions in canonical V1 order."""

        return tuple(self.get(name) for name in self.names())

    def validate_template_names(self, names: Iterable[str]) -> None:
        """Validate that template names are in the V1 allowlist and registered."""

        for name in names:
            if name not in V1_TEMPLATE_NAMES:
                raise ValidationContractError(
                    f"Unknown validation template {name!r}. "
                    f"Known templates: {', '.join(V1_TEMPLATE_NAMES)}"
                )
            if name not in self._definitions:
                raise ValidationContractError(
                    f"Validation template {name!r} is not registered"
                )


def create_default_template_registry() -> ValidationTemplateRegistry:
    """Return the default V1 template-definition registry."""

    return ValidationTemplateRegistry(DEFAULT_TEMPLATE_DEFINITIONS)
