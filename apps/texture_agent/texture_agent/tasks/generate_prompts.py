# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: expand configured texture prompts and optionally auto-generate more.

``material_textures`` is strict by default: only materials listed there are
expanded into texture units. Set ``auto_prompt.enabled: true`` to generate
prompts for discovered materials that do not have explicit specs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

from texture_agent.api.defaults import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
)
from texture_agent.functions.material_discovery import (
    MaterialInfo,
    expand_to_prim_units,
)
from texture_agent.functions.prompt_generation import (
    _fallback_prompts,
    generate_texture_prompts,
)

logger = logging.getLogger(__name__)


class GeneratePromptsTask(Task):
    """Generate texture prompts for materials missing explicit specs.

    If material_textures config already provides prompts for all selected
    materials, this step is a no-op (no LLM call).

    If auto_prompt.enabled is true and material_textures is empty or some
    materials lack specs, calls an LLM to generate prompts for uncovered
    materials.

    After prompt generation, expands materials into PrimTextureUnits.

    Context keys read:
        discovered_materials (list[MaterialInfo]): From DiscoverMaterialsTask.
        material_textures (dict): Per-material specs from config.
        auto_prompt_config (dict): LLM config and user_prompt.
        texture_config (dict): For mode (per_material / per_prim).
        working_dir (str): Working directory.

    Context keys written:
        material_textures (dict): Updated with auto-generated prompts.
        auto_prompt_additions (dict): Specs added by auto-prompt generation.
        prim_texture_units (list[PrimTextureUnit]): Expanded generation units.
    """

    def __init__(self) -> None:
        self.name = "GeneratePrompts"
        self.description = "Generate texture prompts for materials via LLM"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        materials: list[MaterialInfo] = context.get("discovered_materials", [])
        material_textures: dict = context.get("material_textures", {})
        auto_prompt_config: dict = context.get("auto_prompt_config", {})
        texture_config: dict = context.get("texture_config", {})
        working_dir = context.get("working_dir")

        if not materials:
            logger.info("No materials discovered -- skipping prompt generation")
            context["prim_texture_units"] = []
            return context

        # Determine which materials need auto-prompts
        needs_prompt = [m for m in materials if m.name not in material_textures]
        nested_auto_prompt = texture_config.get("auto_prompt", {})
        auto_prompt_enabled = bool(
            auto_prompt_config.get("enabled", nested_auto_prompt.get("enabled", False))
        )

        auto_specs: dict[str, dict[str, Any]] = {}
        if needs_prompt and auto_prompt_enabled:
            user_prompt = auto_prompt_config.get("user_prompt", "")
            llm_config = auto_prompt_config.get("llm", {})
            default_opacity = auto_prompt_config.get(
                "default_opacity",
                context.get("blend_config", {}).get("default_opacity", 0.80),
            )

            from world_understanding.functions.models.chat_models import (
                create_chat_model_from_config,
            )

            try:
                llm = create_chat_model_from_config(
                    llm_config,
                    defaults={
                        "backend": DEFAULT_LLM_BACKEND,
                        "model": DEFAULT_LLM_MODEL,
                        "max_tokens": DEFAULT_LLM_MAX_TOKENS,
                        "temperature": DEFAULT_LLM_TEMPERATURE,
                    },
                )
            except Exception as err:
                logger.warning(
                    "Auto-prompt LLM could not be created (%s) -- using "
                    "fallback prompts composed from user_prompt + material name",
                    err,
                )
                llm = None

            if llm is None:
                # create_chat_model_from_config returns None (no warning
                # above) when the backend has no API key available; the
                # try/except handles the other failure modes.
                auto_specs = _fallback_prompts(
                    needs_prompt, user_prompt, default_opacity
                )
            else:
                auto_specs = generate_texture_prompts(
                    materials=needs_prompt,
                    llm=llm,
                    user_prompt=user_prompt,
                    default_opacity=default_opacity,
                )

            # Merge auto-generated specs into material_textures
            # (explicit configs take precedence -- they're already in the dict)
            material_textures.update(auto_specs)
            context["material_textures"] = material_textures
            context["auto_prompt_additions"] = auto_specs

            logger.info(
                "Auto-generated prompts for %d materials "
                "(%d explicit + %d auto = %d total)",
                len(auto_specs),
                len(material_textures) - len(auto_specs),
                len(auto_specs),
                len(material_textures),
            )

            for name, spec in auto_specs.items():
                prompt = spec["prompt"]
                display = prompt[:60] + "..." if len(prompt) > 60 else prompt
                logger.info(
                    "  [auto] %-30s prompt=%r opacity=%.2f",
                    name,
                    display,
                    spec["opacity"],
                )
        elif needs_prompt:
            context["auto_prompt_additions"] = {}
            logger.info(
                "Auto-prompt disabled; %d discovered materials without explicit "
                "material_textures specs will be skipped",
                len(needs_prompt),
            )
        else:
            context["auto_prompt_additions"] = {}
            logger.info(
                "All %d materials have explicit prompts -- skipping LLM",
                len(materials),
            )

        # Save prompts to working dir
        if working_dir:
            out_dir = Path(working_dir) / "prompts"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "material_prompts.json").write_text(
                json.dumps(material_textures, indent=2)
            )

        # Expand to prim texture units
        mode = texture_config.get("mode", "per_material")
        units = expand_to_prim_units(materials, material_textures, mode)
        context["prim_texture_units"] = units

        logger.info("Expanded to %d texture units (mode=%s)", len(units), mode)
        for u in units:
            logger.info("  %-40s prim=%s", u.key, u.prim_path or "(all)")

        return context
