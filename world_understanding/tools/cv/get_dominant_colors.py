# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Get dominant colors tool using k-means clustering."""

from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console

from world_understanding.functions.cv.get_dominant_colors import get_dominant_colors
from world_understanding.tools.base import (
    ExecutionPolicy,
    ToolInput,
    ToolOutput,
    register_tool,
)


class GetDominantColorsInput(ToolInput):
    """Input for get dominant colors tool."""

    image_path: str = Field(..., description="Path to the image to analyze")
    n_colors: int = Field(
        default=5, ge=1, le=20, description="Number of dominant colors to extract"
    )
    analyze_brightness: bool = Field(
        default=True, description="Whether to calculate average brightness"
    )


class ColorInfo(BaseModel):
    """Information about a single color."""

    rgb: list[int] = Field(..., description="RGB values [0-255]")
    hex: str = Field(..., description="Hex color code")
    percentage: float = Field(..., description="Percentage of image with this color")


class GetDominantColorsOutput(ToolOutput):
    """Output for get dominant colors tool."""

    dominant_colors: list[ColorInfo] = Field(
        ..., description="List of dominant colors sorted by percentage"
    )
    average_brightness: float = Field(..., description="Average brightness (0-255)")
    color_diversity: float = Field(..., description="Color diversity measure")
    n_clusters: int = Field(..., description="Number of color clusters used")


def _display_color_analysis(
    outputs: dict[str, Any], console: Console, indent: str = ""
) -> None:
    """Display color analysis results in a formatted way."""
    console.print(f"{indent}[bold]Color Analysis Results:[/bold]")
    console.print(
        f"{indent}Average Brightness: {outputs.get('average_brightness', 0):.1f}/255"
    )
    console.print(f"{indent}Color Diversity: {outputs.get('color_diversity', 0):.3f}")
    console.print(f"{indent}[bold]Dominant Colors:[/bold]")

    for i, color in enumerate(outputs.get("dominant_colors", []), 1):
        rgb = color["rgb"]
        hex_code = color["hex"]
        percentage = color["percentage"] * 100
        console.print(
            f"{indent}  {i}. {hex_code} "
            f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) - "
            f"{percentage:.1f}%",
            style=f"on {hex_code}",
        )


@register_tool(
    name="get_dominant_colors",
    version="0.1.0",
    description="Get dominant colors from an image using k-means clustering",
    tags=["color", "dominant", "extraction", "cpu"],
    input_model=GetDominantColorsInput,
    output_model=GetDominantColorsOutput,
    policy=ExecutionPolicy(timeout_s=30.0),
)
def get_dominant_colors_tool(inputs: GetDominantColorsInput) -> GetDominantColorsOutput:
    """Execute color analysis on the image."""
    # Call the portable function - it handles both paths and PIL images
    result = get_dominant_colors(
        image=inputs.image_path,
        n_colors=inputs.n_colors,
        analyze_brightness=inputs.analyze_brightness,
    )

    # Convert result to ColorInfo objects
    dominant_colors = []
    for color_data in result["dominant_colors"]:
        dominant_colors.append(
            ColorInfo(
                rgb=color_data["rgb"],
                hex=color_data["hex"],
                percentage=color_data["percentage"] * 100,  # Convert to percentage
            )
        )

    return GetDominantColorsOutput(
        dominant_colors=dominant_colors,
        average_brightness=result["average_brightness"],
        color_diversity=result["color_diversity"],
        n_clusters=result["n_clusters"],
    )


# Attach display function to the tool
get_dominant_colors_tool._display_function = _display_color_analysis
