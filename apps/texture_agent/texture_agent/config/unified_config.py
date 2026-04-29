# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified config loader for the texture agent pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from texture_agent.config.schema import DEFAULTS, STEP_ORDER, STEP_OUTPUT_DIRS

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a texture pipeline config file.

    Resolves relative paths against the config file's directory,
    applies defaults, and creates the working directory structure.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated and resolved config dict.
    """
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got {type(config).__name__}: {config_path}"
        )

    # Project defaults
    project = config.setdefault("project", {})
    project.setdefault("name", config_path.stem)
    project.setdefault("session_id", project["name"])

    # Resolve working directory
    working_dir = project.get("working_dir")
    if working_dir:
        working_dir = (config_dir / working_dir).resolve()
    else:
        working_dir = config_dir / f".{project['session_id']}"
    project["working_dir"] = str(working_dir)

    # Resolve input paths
    input_cfg = config.get("input", {})
    if "usd_path" in input_cfg:
        usd_path = Path(input_cfg["usd_path"])
        if not usd_path.is_absolute():
            usd_path = (config_dir / usd_path).resolve()
        input_cfg["usd_path"] = str(usd_path)

    # Apply defaults for texture config
    texture = config.setdefault("texture", {})
    for key, val in DEFAULTS["texture"].items():
        texture.setdefault(key, val)

    # Apply defaults for variations
    variations = config.setdefault("variations", {})
    for key, val in DEFAULTS["variations"].items():
        variations.setdefault(key, val)

    # Apply defaults for steps
    steps = config.setdefault("steps", {})
    for step_name in STEP_ORDER:
        step_cfg = steps.setdefault(step_name, {})
        defaults = DEFAULTS["steps"].get(step_name, {})
        for key, val in defaults.items():
            step_cfg.setdefault(key, val)

    # Validate required fields
    if not input_cfg.get("usd_path"):
        raise ValueError("Config missing required field: input.usd_path")

    # Create working directory structure
    working_dir_path = Path(working_dir)
    working_dir_path.mkdir(parents=True, exist_ok=True)
    for _step_name, dir_name in STEP_OUTPUT_DIRS.items():
        (working_dir_path / dir_name).mkdir(parents=True, exist_ok=True)

    logger.info("Loaded config: %s", config_path)
    logger.info("  Project: %s", project["name"])
    logger.info("  Input: %s", input_cfg.get("usd_path"))
    logger.info("  Working dir: %s", working_dir)

    return config


def config_to_context(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a loaded config dict into a pipeline context dict.

    Maps config sections to the context keys expected by pipeline tasks.

    Args:
        config: Loaded config from load_config().

    Returns:
        Context dict for pipeline execution.
    """
    return {
        "usd_path": config["input"]["usd_path"],
        "prim_paths": config["input"].get("prim_paths"),
        "working_dir": config["project"]["working_dir"],
        "texture_config": config.get("texture", {}),
        "material_textures": config.get("material_textures", {}),
        "blend_config": config["steps"].get("blend_textures", {}),
        "render_preview_config": config["steps"].get("render_previews", {}),
        "render_config": config["steps"].get("render", {}),
        "auto_prompt_config": config.get("auto_prompt", {}),
        "variations_config": config.get("variations", {}),
        "steps": config.get("steps", {}),
        "config": config,
    }
