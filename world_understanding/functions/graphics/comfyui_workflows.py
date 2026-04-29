# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic ComfyUI workflow execution functions."""

import json
from pathlib import Path
from typing import Any

from world_understanding.utils.comfyui_client import ComfyUIClient


def execute_comfyui_workflow(
    workflow_name: str,
    inputs: dict[str, Any],
    output_nodes: list[str] | None = None,
    server_url: str | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Execute a named ComfyUI workflow with given inputs.

    Args:
        workflow_name: Name of the workflow file (without .json extension)
        inputs: Input parameters for the workflow
        output_nodes: List of node IDs to get outputs from
        server_url: ComfyUI server URL (uses COMFYUI_URL env var if not provided)
        timeout: Maximum time to wait for completion in seconds

    Returns:
        Dict containing:
            - images: Dict mapping node_id to PIL Image objects
            - execution_time: Time taken to execute in seconds
    """
    # Load workflow from data directory
    workflow_dir = Path(__file__).parent.parent.parent / "data" / "comfyui_workflows"
    workflow_path = workflow_dir / f"{workflow_name}.json"

    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow not found: {workflow_path}")

    with open(workflow_path, encoding="utf-8") as f:
        workflow = json.load(f)

    # Initialize client
    client = ComfyUIClient(server_url)

    # Start timing
    import time

    start_time = time.time()

    # Execute workflow
    images = client.execute_workflow(workflow, inputs, output_nodes, timeout)

    execution_time = time.time() - start_time

    return {"images": images, "execution_time": execution_time}
