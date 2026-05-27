# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply Physics configuration task."""

import logging
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from physics_agent.config.validator import VALID_COLLISION_APPROX
from physics_agent.functions.mass_scale_quality import VALID_MASS_SCALE_POLICIES

logger = logging.getLogger(__name__)


class ApplyPhysicsConfigTask(Task):
    """Load and validate apply_physics step configuration.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - usd_path: Input USD file path
        - predictions_path: Path to predictions JSONL
        - output_usd_path: Output path for the physics-augmented USD
        - collision_approx: Collision approximation method
        - mass_scale_policy: warn | skip_mass | fail for mass/scale QA warnings
        - allow_empty_predictions: Allow empty prediction files to produce a
          rigid-body-only USD (default: False)
    """

    def __init__(self) -> None:
        self.name = "ApplyPhysicsConfig"
        self.description = "Load apply physics step configuration"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        config = self._load_config(context)

        # Anchor relative paths on the YAML file only when we actually loaded
        # from it; if the caller passed config_dict directly, relative paths
        # are resolved against the cwd.
        if "config_dict" not in context and context.get("config_path"):
            config_dir = Path(context["config_path"]).parent
        else:
            config_dir = Path.cwd()

        usd_path_str = config.get("usd_path")
        predictions_path_str = config.get("predictions_path")
        output_usd_path_str = config.get("output_usd_path")
        if not usd_path_str:
            raise ValueError("apply_physics: missing required 'usd_path' in config")
        if not predictions_path_str:
            raise ValueError(
                "apply_physics: missing required 'predictions_path' in config"
            )
        if not output_usd_path_str:
            raise ValueError(
                "apply_physics: missing required 'output_usd_path' in config"
            )

        usd_path = self._resolve_path(usd_path_str, config_dir)
        predictions_path = self._resolve_path(predictions_path_str, config_dir)
        output_usd_path = self._resolve_path(output_usd_path_str, config_dir)
        collision_approx = config.get("collision_approx", "convexHull")
        if collision_approx not in VALID_COLLISION_APPROX:
            raise ValueError(
                "apply_physics.collision_approx must be one of "
                f"{sorted(VALID_COLLISION_APPROX)}, got '{collision_approx}'"
            )
        output_key = config.get("output_key", "classification")
        mass_scale_policy = config.get("mass_scale_policy", "skip_mass")
        if mass_scale_policy not in VALID_MASS_SCALE_POLICIES:
            raise ValueError(
                "apply_physics.mass_scale_policy must be one of "
                f"{sorted(VALID_MASS_SCALE_POLICIES)}, got '{mass_scale_policy}'"
            )
        allow_empty_predictions = config.get("allow_empty_predictions", False)
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "apply_physics.allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )

        context.update(
            {
                "usd_path": str(usd_path),
                "predictions_path": str(predictions_path),
                "output_usd_path": str(output_usd_path),
                "collision_approx": collision_approx,
                "output_key": output_key,
                "mass_scale_policy": mass_scale_policy,
                "allow_empty_predictions": allow_empty_predictions,
            }
        )

        logger.info("Input USD: %s", usd_path)
        logger.info("Predictions: %s", predictions_path)
        logger.info("Output USD: %s", output_usd_path)
        logger.info("Collision approx: %s", collision_approx)
        logger.info("Output key: %s", output_key)
        logger.info("Mass scale policy: %s", mass_scale_policy)
        logger.info("Allow empty predictions: %s", allow_empty_predictions)

        return context

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        if "config_dict" in context:
            config_dict = context["config_dict"]
            if not isinstance(config_dict, dict):
                raise ValueError(
                    "apply_physics config_dict must be a mapping, got "
                    f"{type(config_dict).__name__}"
                )
            return config_dict
        config_path = context.get("config_path")
        if not config_path:
            raise ValueError("No config_path or config_dict in context")
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"apply_physics config must be a YAML mapping, got "
                f"{type(loaded).__name__}: {config_path}"
            )
        return loaded

    def _resolve_path(self, path: str, config_dir: Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (config_dir / p).resolve()
