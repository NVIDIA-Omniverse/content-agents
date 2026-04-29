# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image editing tool using ComfyUI."""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field
from rich.console import Console
from rich.panel import Panel

from world_understanding.functions.graphics.image_editing import edit_image_with_comfyui
from world_understanding.tools.base import (
    ExecutionPolicy,
    ToolInput,
    ToolOutput,
    register_tool,
)

logger = logging.getLogger(__name__)


class ImageEditInput(ToolInput):
    """Input for image editing tool."""

    image_path: str = Field(..., description="Path to the image to edit")
    prompt: str = Field(
        ...,
        description="Text prompt describing the desired edit (e.g., 'make it sunset', 'add flowers')",
    )
    negative_prompt: str = Field(
        default="",
        description="What to avoid in the edit (e.g., 'blurry, artifacts')",
    )
    return_rescaled_input: bool = Field(
        default=False,
        description="Whether to return the rescaled input image along with the edited image",
    )
    server_url: str | None = Field(
        default=None,
        description="ComfyUI server URL (uses COMFYUI_URL env var if not provided)",
    )


class ImageEditOutput(ToolOutput):
    """Output for image editing tool."""

    edited_image_path: str = Field(..., description="Path to the edited image")
    rescaled_input_path: str | None = Field(
        None, description="Path to the rescaled input image (if requested)"
    )
    image_width: int = Field(..., description="Width of the output images")
    image_height: int = Field(..., description="Height of the output images")
    execution_time: float = Field(..., description="Time taken to process in seconds")


def _display_image_edit_results(
    outputs: dict[str, Any], console: Console, indent: str = ""
) -> None:
    """Display image editing results."""
    console.print(f"\n{indent}[bold]Image Editing Results[/bold]")

    # Create a summary panel
    summary_lines = [
        f"[green]✓[/green] Edited image: {outputs['edited_image_path']}",
        f"Size: {outputs['image_width']}x{outputs['image_height']}",
        f"Execution time: {outputs['execution_time']:.2f}s",
    ]

    if outputs.get("rescaled_input_path"):
        summary_lines.insert(
            1, f"[green]✓[/green] Rescaled input: {outputs['rescaled_input_path']}"
        )

    panel = Panel(
        "\n".join(summary_lines),
        title="Image Edit Complete",
        border_style="green",
        expand=False,
    )
    console.print(panel)


@register_tool(
    name="image_edit",
    version="0.1.0",
    description="Edit images using text-guided AI without masks",
    tags=["graphics", "editing", "ai", "gpu", "comfyui"],
    input_model=ImageEditInput,
    output_model=ImageEditOutput,
    policy=ExecutionPolicy(timeout_s=300.0, device="cuda"),
)
def image_edit_tool(inputs: ImageEditInput) -> ImageEditOutput:
    """Edit images using ComfyUI's text-guided editing."""
    try:
        # Call the function
        result = edit_image_with_comfyui(
            image=inputs.image_path,
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
            return_rescaled_input=inputs.return_rescaled_input,
            server_url=inputs.server_url,
        )

        # Save the edited image
        output_dir = Path(inputs.image_path).parent
        output_name = Path(inputs.image_path).stem + "_edited.png"
        edited_image_path = output_dir / output_name

        if result["edited_image"]:
            result["edited_image"].save(edited_image_path)
        else:
            raise ValueError("No edited image returned from ComfyUI")

        # Save rescaled input if requested and available
        rescaled_input_path = None
        if inputs.return_rescaled_input and result.get("rescaled_input"):
            rescaled_name = Path(inputs.image_path).stem + "_rescaled.png"
            rescaled_input_path = output_dir / rescaled_name
            result["rescaled_input"].save(rescaled_input_path)

        return ImageEditOutput(
            edited_image_path=str(edited_image_path),
            rescaled_input_path=str(rescaled_input_path)
            if rescaled_input_path
            else None,
            image_width=result["image_size"][0],
            image_height=result["image_size"][1],
            execution_time=result["execution_time"],
        )

    except Exception as e:
        logger.error(f"Image editing failed: {e}")
        raise


# Attach display function to the tool
image_edit_tool._display_function = _display_image_edit_results
