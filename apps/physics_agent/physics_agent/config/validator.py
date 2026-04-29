# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration validator for Physics Agent."""

import logging
from typing import Any

from physics_agent.config.schema import (
    REQUIRED_FIELDS,
    REQUIRED_SECTIONS,
)

logger = logging.getLogger(__name__)

# Allowed values for apply_physics.collision_approx. Shared with the
# per-step ConfigTask so both validation paths stay in sync.
VALID_COLLISION_APPROX = frozenset(
    {
        "convexHull",
        "convexDecomposition",
        "boundingCube",
        "boundingSphere",
        "meshSimplification",
        "none",
    }
)


class ConfigValidator:
    """Validator for Physics Agent configuration."""

    def validate(self, config: dict[str, Any]) -> None:
        """Validate the configuration structure.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        for section in REQUIRED_SECTIONS:
            if section not in config:
                raise ValueError(f"Missing required section: '{section}'")

        # Check required fields in each section
        for section, fields in REQUIRED_FIELDS.items():
            if section not in config:
                continue
            section_config = config[section]
            if section_config is None:
                section_config = {}
            for field in fields:
                if field not in section_config or section_config[field] is None:
                    raise ValueError(f"Missing required field: '{section}.{field}'")

        # Validate steps section if present
        steps = config.get("steps", {})
        if steps:
            self._validate_steps(steps)

    def _validate_steps(self, steps: dict[str, Any]) -> None:
        """Validate steps configuration.

        Args:
            steps: Steps configuration dictionary
        """
        valid_steps = {
            "optimize_usd",
            "identify_asset",
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "predict",
            "restore_usd",
            "apply_physics",
        }

        for step_name in steps.keys():
            if step_name not in valid_steps:
                logger.warning(
                    "Unknown step '%s' in configuration. Valid steps: %s",
                    step_name,
                    ", ".join(sorted(valid_steps)),
                )

    def validate_step_requirements(
        self,
        step_name: str,
        step_config: dict[str, Any],
        full_config: dict[str, Any],
    ) -> None:
        """Validate requirements for a specific step.

        Args:
            step_name: Name of the step
            step_config: Step configuration
            full_config: Full configuration dictionary
        """
        # Step-specific validation
        if step_name == "predict":
            # Ensure VLM config is present
            if "vlm" not in step_config:
                logger.warning(
                    "predict step has no 'vlm' configuration - using defaults"
                )

            # Validate output_key if present
            output_key = step_config.get("output_key")
            if output_key and not isinstance(output_key, str):
                raise ValueError(
                    f"predict.output_key must be a string, got {type(output_key)}"
                )

        elif step_name == "apply_physics":
            collision_approx = step_config.get("collision_approx", "convexHull")
            if collision_approx not in VALID_COLLISION_APPROX:
                raise ValueError(
                    "apply_physics.collision_approx must be one of "
                    f"{sorted(VALID_COLLISION_APPROX)}, got '{collision_approx}'"
                )
