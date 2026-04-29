# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Color matcher tool to check if an image contains a specific color."""

from typing import Any

from pydantic import Field, field_validator
from rich.console import Console

from world_understanding.functions.cv.find_similar_color import find_similar_color
from world_understanding.tools.base import (
    ExecutionPolicy,
    ToolInput,
    ToolOutput,
    register_tool,
)


class FindSimilarColorInput(ToolInput):
    """Input for color matcher tool."""

    image_path: str = Field(..., description="Path to the image to analyze")
    target_color: list[int] = Field(
        ...,
        description="Target RGB color to search for [R, G, B] values (0-255)",
        min_length=3,
        max_length=3,
    )
    color_tolerance: int = Field(
        default=50,
        ge=0,
        le=255,
        description=(
            "Tolerance for color matching (0-255). "
            "Higher values match more similar colors"
        ),
    )
    min_percentage: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description=("Minimum percentage of pixels that must match the target color"),
    )

    @field_validator("target_color")
    @classmethod
    def validate_color(cls, v: list[int]) -> list[int]:
        """Validate RGB color values."""
        for value in v:
            if not 0 <= value <= 255:
                raise ValueError(f"RGB values must be between 0 and 255, got {value}")
        return v


class FindSimilarColorOutput(ToolOutput):
    """Output for color matcher tool."""

    contains_color: bool = Field(
        ..., description="Whether the image contains the target color"
    )
    matching_percentage: float = Field(
        ..., description="Percentage of pixels matching the target color"
    )
    pixel_count: int = Field(
        ..., description="Number of pixels matching the target color"
    )
    total_pixels: int = Field(..., description="Total number of pixels in the image")
    target_color_rgb: list[int] = Field(
        ..., description="The target color that was searched for"
    )
    target_color_hex: str = Field(
        ..., description="Hex representation of the target color"
    )
    closest_colors: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of closest colors found in the image with their percentages"
        ),
    )


def _display_color_match_results(
    outputs: dict[str, Any], console: Console, indent: str = ""
) -> None:
    """Display color match results in a formatted way."""
    target_hex = outputs.get("target_color_hex", "#000000")
    target_rgb = outputs.get("target_color_rgb", [0, 0, 0])
    contains = outputs.get("contains_color", False)
    percentage = outputs.get("matching_percentage", 0.0)

    console.print(f"{indent}[bold]Color Match Results:[/bold]")
    console.print(
        f"{indent}Target Color: {target_hex} "
        f"RGB({target_rgb[0]}, {target_rgb[1]}, {target_rgb[2]})",
        style=f"on {target_hex}",
    )

    status = "✓ Found" if contains else "✗ Not Found"
    status_color = "green" if contains else "red"
    console.print(f"{indent}Status: [{status_color}]{status}[/{status_color}]")

    console.print(f"{indent}Matching Pixels: {percentage:.2f}%")

    if "closest_colors" in outputs and outputs["closest_colors"]:
        console.print(f"{indent}[bold]Closest Colors in Image:[/bold]")
        for i, color in enumerate(outputs["closest_colors"][:5], 1):
            hex_code = color.get("hex", "#000000")
            rgb = color.get("rgb", [0, 0, 0])
            pct = color.get("percentage", 0.0) * 100
            console.print(
                f"{indent}  {i}. {hex_code} "
                f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) - "
                f"{pct:.1f}%",
                style=f"on {hex_code}",
            )


@register_tool(
    name="find_similar_color",
    version="0.1.0",
    description="Check if an image contains a specific color within tolerance",
    tags=["color", "matching", "detection", "cpu"],
    input_model=FindSimilarColorInput,
    output_model=FindSimilarColorOutput,
    policy=ExecutionPolicy(timeout_s=30.0),
)
def find_similar_color_tool(
    inputs: FindSimilarColorInput,
) -> FindSimilarColorOutput:
    """Check if an image contains a specific color within tolerance."""
    # Call the portable function - it handles both paths and PIL images
    result = find_similar_color(
        image=inputs.image_path,
        target_color=inputs.target_color,
        color_tolerance=inputs.color_tolerance,
        min_percentage=inputs.min_percentage,
    )

    # Create output
    return FindSimilarColorOutput(
        contains_color=result["contains_color"],
        matching_percentage=result["matching_percentage"],
        pixel_count=result["pixel_count"],
        total_pixels=result["total_pixels"],
        target_color_rgb=result["target_color_rgb"],
        target_color_hex=result["target_color_hex"],
        closest_colors=result.get("closest_colors", []),
    )
