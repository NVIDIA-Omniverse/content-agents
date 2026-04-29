# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified pipeline workflow using the new config system.

This workflow uses the UnifiedPipelineConfigTask to load and validate
configuration, then executes steps with auto-wired paths.
"""

import logging

from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import TempDirObjectStore

from material_agent.config import UnifiedPipelineConfigTask
from material_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask

logger = logging.getLogger(__name__)


def create_unified_pipeline_workflow() -> Workflow:
    """Create a unified pipeline workflow using the new config system.

    This workflow:
    1. Loads and validates the unified configuration
    2. Auto-derives all paths from project settings
    3. Executes steps with auto-wired configs
    4. No manual path management needed

    The workflow expects the following initial context:
        - config_path: Path to the unified YAML configuration file
        - skip_steps: Optional list of step names to skip
        - only_steps: Optional list of step names to run exclusively

    Returns:
        Configured Workflow instance for unified pipeline execution

    Example:
        workflow = create_unified_pipeline_workflow()
        result = workflow.run({
            "config_path": "configs/unified_ladder.yaml",
            "only_steps": ["predict", "apply"]
        })
    """
    tasks = [
        # Load and validate unified configuration
        # This task:
        # - Validates structure and conventions
        # - Creates ProjectPathResolver
        # - Parses materials
        # - Builds auto-wired step configs
        UnifiedPipelineConfigTask(),
        # Execute pipeline steps with auto-wired configs
        # This task:
        # - Executes each step in order
        # - Manages data flow between steps
        # - Handles errors and recovery
        UnifiedPipelineExecutorTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=TempDirObjectStore(),
        name="Unified Pipeline",
        description="Unified pipeline with auto-derived paths and single config format",
    )
