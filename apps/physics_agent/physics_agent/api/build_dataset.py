# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build dataset APIs for Physics Agent.

This module provides APIs for building datasets from USD files.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from physics_agent.api.types import APIResult

logger = logging.getLogger(__name__)


# ============================================================================
# Build Dataset USD API
# ============================================================================


@dataclass
class BuildDatasetUsdInput:
    """Input parameters for USD dataset building."""

    config: Path | dict[str, Any]
    """Path to config file or config dictionary"""

    source_override: Path | None = None
    """Override source USD path"""

    output_dir_override: Path | None = None
    """Override output directory"""

    extract_metadata: bool = False
    """Extract prim metadata"""

    verbose: bool = False
    """Enable verbose logging"""


@dataclass
class BuildDatasetUsdOutput(APIResult):
    """Output from USD dataset building."""

    dataset_path: Path | None = None
    """Path to dataset manifest"""

    num_prims: int = 0
    """Number of prims processed"""

    num_images: int = 0
    """Number of images generated"""

    batch_results: dict[str, Any] = field(default_factory=dict)
    """Results for batch processing"""


def build_dataset_usd(params: BuildDatasetUsdInput) -> BuildDatasetUsdOutput:
    """Build dataset from USD file(s).

    Args:
        params: Input parameters

    Returns:
        BuildDatasetUsdOutput with results
    """
    return asyncio.run(abuild_dataset_usd(params))


async def abuild_dataset_usd(params: BuildDatasetUsdInput) -> BuildDatasetUsdOutput:
    """Async version of build_dataset_usd.

    Args:
        params: Input parameters

    Returns:
        BuildDatasetUsdOutput with results
    """
    try:
        from physics_agent.workflows import (
            create_usd_data_preparation_workflow_from_config,
        )

        # Create workflow
        workflow = create_usd_data_preparation_workflow_from_config()

        # Prepare context
        context: dict[str, Any] = {
            "extract_metadata": params.extract_metadata,
            "verbose": params.verbose,
        }

        # Handle config (path or dict)
        if isinstance(params.config, dict):
            context["config_dict"] = params.config
        else:
            context["config_path"] = str(params.config)

        # Add overrides
        if params.source_override:
            context["source_override"] = str(params.source_override)
        if params.output_dir_override:
            context["output_dir_override"] = str(params.output_dir_override)

        # Run workflow
        result = workflow.run(context)

        # Check for errors
        if result.get("error") or result.get("workflow_terminated"):
            return BuildDatasetUsdOutput(
                success=False,
                error=result.get("error", "Workflow terminated unexpectedly"),
            )

        # Extract results
        dataset_path = result.get("dataset_path") or result.get("output_dir")
        num_prims = result.get("num_prims", 0)
        num_images = result.get("num_images", 0)
        batch_results = result.get("batch_results", {})

        return BuildDatasetUsdOutput(
            success=True,
            dataset_path=Path(dataset_path) if dataset_path else None,
            num_prims=num_prims,
            num_images=num_images,
            batch_results=batch_results,
        )

    except Exception as e:
        logger.error("USD dataset building failed: %s", e, exc_info=True)
        return BuildDatasetUsdOutput(success=False, error=str(e))


# ============================================================================
# Prepare Dataset API
# ============================================================================


@dataclass
class BuildDatasetPrepareDatasetInput:
    """Input parameters for dataset preparation."""

    config: Path | dict[str, Any]
    """Path to config file or config dictionary"""

    dataset_override: Path | None = None
    """Override dataset path"""

    verbose: bool = False
    """Enable verbose logging"""


@dataclass
class BuildDatasetPrepareDatasetOutput(APIResult):
    """Output from dataset preparation."""

    dataset_entries: list[dict[str, Any]] = field(default_factory=list)
    """Dataset entries"""

    dataset_jsonl_path: Path | None = None
    """Path to dataset JSONL file"""

    failed_models: list[str] = field(default_factory=list)
    """List of failed models"""


def build_dataset_prepare_dataset(
    params: BuildDatasetPrepareDatasetInput,
) -> BuildDatasetPrepareDatasetOutput:
    """Prepare dataset for predictions.

    Args:
        params: Input parameters

    Returns:
        BuildDatasetPrepareDatasetOutput with results
    """
    return asyncio.run(abuild_dataset_prepare_dataset(params))


async def abuild_dataset_prepare_dataset(
    params: BuildDatasetPrepareDatasetInput,
) -> BuildDatasetPrepareDatasetOutput:
    """Async version of build_dataset_prepare_dataset.

    Args:
        params: Input parameters

    Returns:
        BuildDatasetPrepareDatasetOutput with results
    """
    try:
        from physics_agent.workflows import (
            create_prepare_dataset_workflow_from_config,
        )

        # Create workflow
        workflow = create_prepare_dataset_workflow_from_config()

        # Prepare context
        context: dict[str, Any] = {
            "verbose": params.verbose,
        }

        # Handle config (path or dict)
        if isinstance(params.config, dict):
            context["config_dict"] = params.config
        else:
            context["config_path"] = str(params.config)

        # Add overrides
        if params.dataset_override:
            context["dataset_override"] = str(params.dataset_override)

        # Run workflow
        result = workflow.run(context)

        # Check for errors
        if result.get("error") or result.get("workflow_terminated"):
            return BuildDatasetPrepareDatasetOutput(
                success=False,
                error=result.get("error", "Workflow terminated unexpectedly"),
            )

        # Extract results
        dataset_entries = result.get("dataset_entries", [])
        dataset_jsonl_path = result.get("dataset_jsonl_path")
        failed_models = result.get("failed_models", [])

        return BuildDatasetPrepareDatasetOutput(
            success=True,
            dataset_entries=dataset_entries,
            dataset_jsonl_path=Path(dataset_jsonl_path) if dataset_jsonl_path else None,
            failed_models=failed_models,
        )

    except Exception as e:
        logger.error("Dataset preparation failed: %s", e, exc_info=True)
        return BuildDatasetPrepareDatasetOutput(success=False, error=str(e))
