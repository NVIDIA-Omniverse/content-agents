# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared USD workflow creation for World Understanding agents.

This module provides common workflow factories used by both material-agent and
physics-agent for USD dataset building. By sharing this code, both agents
benefit from bug fixes, performance improvements, and new features automatically.
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.usd_tasks.config import USDDataPrepConfigTask
from world_understanding.agentic.usd_tasks.dataset_manifest import (
    USDDatasetManifestTask,
)
from world_understanding.agentic.usd_tasks.prim_traversal import (
    USDPrimTraversalAndRenderingTask,
)
from world_understanding.agentic.usd_tasks.renderer import (
    USDRendererProvisioningTask,
)
from world_understanding.agentic.usd_tasks.usd_loader import USDLoadingTask
from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import InMemoryObjectStore

logger = logging.getLogger(__name__)


def create_usd_dataset_workflow(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    workflow_name: str = "USD Dataset Preparation",
    workflow_description: str = "Build dataset from USD file with multi-view rendering",
) -> Workflow:
    """Create a workflow for building datasets from USD files.

    This is the shared workflow used by both material-agent, physics-agent, and joint-agent
    for USD dataset preparation. It loads a USD file, traverses prims, renders
    multiple views of each prim, and creates a dataset manifest.

    The workflow consists of these tasks:
    1. USDDataPrepConfigTask - Load and validate configuration
    2. USDRendererProvisioningTask - Initialize renderer (remote, ovrtx, or warp)
    3. USDLoadingTask - Load USD stage
    4. USDPrimTraversalAndRenderingTask - Render prims from multiple viewpoints
    5. USDDatasetManifestTask - Create intermediate Phase 1 dataset files

    Args:
        config_path: Optional path to YAML configuration file
        overrides: Optional dictionary of configuration overrides (e.g., from CLI)
        workflow_name: Name for the workflow (for logging/display)
        workflow_description: Description of the workflow

    Returns:
        Configured Workflow instance ready to run

    Example:
        ```python
        # Create workflow with config file
        workflow = create_usd_dataset_workflow(
            config_path=Path("configs/data_prep.yaml"),
            overrides={"usd_path": Path("custom.usd")}
        )

        # Run workflow
        result = workflow.run({
            "config_path": config_path,
            **overrides
        })

        # Access results
        dataset_path = result["dataset_path"]
        num_prims = result["num_prims"]
        ```

    Initial Context Expected:
        The workflow expects certain keys in initial_context:
        - config_path: Path to YAML configuration file (required if not provided here)
        - source_override: Optional USD path override (overrides config)
        - output_dir_override: Optional output directory override
        - extract_prim_metadata: Optional metadata extraction flag
        - Various other configuration options (see USDDatasetConfig)

    Returns Context:
        The workflow returns a context with:
        - dataset_path: Path to generated dataset.json
        - num_prims: Number of prims processed
        - num_images: Number of images rendered
        - usd_stage: Loaded USD stage (in object_store)
        - Various other outputs from tasks

    Notes:
        - This is a shared implementation used by multiple agents
        - Task ordering is important - don't change without testing all agents
        - Object store is used for large data (USD stage, models)
        - Context is used for metadata and paths
    """
    # Create the workflow with standard USD dataset preparation tasks
    # Note: This workflow only creates Phase 1 intermediate files.
    # Consolidation to v0.2 format is done by the prepare-dataset step.
    tasks = [
        USDDataPrepConfigTask(),
        USDRendererProvisioningTask(),
        USDLoadingTask(),
        USDPrimTraversalAndRenderingTask(),
        USDDatasetManifestTask(),
    ]

    workflow = Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name=workflow_name,
        description=workflow_description,
    )

    logger.debug(
        f"Created USD dataset workflow with {len(tasks)} tasks: {workflow_name}"
    )

    return workflow


def run_usd_dataset_workflow(
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function to create and run USD dataset workflow.

    This is a simpler interface for agents that just want to run the workflow
    without needing to manage the workflow object themselves.

    Args:
        config_path: Path to YAML configuration file
        overrides: Optional configuration overrides

    Returns:
        Workflow execution result context

    Example:
        ```python
        result = run_usd_dataset_workflow(
            config_path=Path("config.yaml"),
            overrides={"batch_size": 20}
        )
        print(f"Dataset created: {result['dataset_path']}")
        ```
    """
    workflow = create_usd_dataset_workflow(
        config_path=config_path,
        overrides=overrides,
    )

    # Prepare initial context
    initial_context = {"config_path": config_path}
    if overrides:
        initial_context.update(overrides)

    # Run workflow
    result = workflow.run(initial_context)

    return result


# Convenience aliases for backward compatibility
create_data_preparation_workflow_from_config = create_usd_dataset_workflow
