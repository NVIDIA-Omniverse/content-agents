# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image editing functions using ComfyUI."""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from world_understanding.utils.comfyui_client import ComfyUIClient


def edit_image_with_comfyui(
    image: str | Path | Image.Image | np.ndarray,
    prompt: str,
    negative_prompt: str = "",
    return_rescaled_input: bool = False,
    server_url: str | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Edit an image using ComfyUI's text-guided editing (no mask required).

    Args:
        image: Input image (path, PIL Image, or numpy array)
        prompt: Text prompt describing the desired edit
        negative_prompt: What to avoid in the edit
        return_rescaled_input: Whether to return the rescaled input image
        server_url: ComfyUI server URL (uses COMFYUI_URL env var if not provided)
        timeout: Maximum time to wait for completion in seconds

    Returns:
        Dict containing:
            - edited_image: The edited PIL Image
            - rescaled_input: The rescaled input PIL Image (if return_rescaled_input=True)
            - image_size: (width, height) of the output
            - execution_time: Time taken to process
    """
    # Handle different image input types
    if isinstance(image, str | Path):
        image_path = str(image)
        temp_file_created = False
    elif isinstance(image, Image.Image):
        # Save PIL image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name
            temp_file_created = True
    elif isinstance(image, np.ndarray):
        # Convert numpy array to PIL and save
        pil_image = Image.fromarray(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp.name)
            image_path = tmp.name
            temp_file_created = True
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    try:
        # Initialize client and upload image
        client = ComfyUIClient(server_url)
        filename, subfolder, img_type = client.upload_image(image_path)

        # Load the workflow
        workflow_dir = (
            Path(__file__).parent.parent.parent / "data" / "comfyui_workflows"
        )
        workflow_path = workflow_dir / "qwen_image_edit.json"

        with open(workflow_path, encoding="utf-8") as f:
            workflow = json.load(f)

        # Update the workflow with our image and prompts
        for node_id, node in workflow.items():
            if node_id == "78" and node["class_type"] == "LoadImage":
                # Update the input image
                node["inputs"]["image"] = filename

            elif node_id == "76" and node["class_type"] == "TextEncodeQwenImageEdit":
                # Update the positive prompt
                node["inputs"]["prompt"] = prompt

            elif node_id == "77" and node["class_type"] == "TextEncodeQwenImageEdit":
                # Update the negative prompt
                node["inputs"]["prompt"] = negative_prompt

        # Determine which nodes to get outputs from
        output_nodes = ["60"]  # Main output image
        if return_rescaled_input:
            output_nodes.append("104")  # Rescaled input (SaveImage node)

        # Execute workflow
        import time

        start_time = time.time()
        images = client.execute_workflow(workflow, {}, output_nodes, timeout)
        execution_time = time.time() - start_time

        # Prepare result
        edited_image = images.get("60")

        result = {
            "edited_image": edited_image,
            "execution_time": execution_time,
        }

        if return_rescaled_input:
            result["rescaled_input"] = images.get("104")

        # Get image size from the edited image
        if result["edited_image"]:
            result["image_size"] = result["edited_image"].size

        return result

    finally:
        # Clean up temp files if created
        if temp_file_created and Path(image_path).exists():
            Path(image_path).unlink()
