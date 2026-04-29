# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Color matcher function to check if an image contains a specific color."""

from typing import Any

import numpy as np
from PIL import Image as PILImage


def find_similar_color(
    image: str | PILImage.Image,
    target_color: list[int] | tuple[int, int, int],
    color_tolerance: int = 50,
    min_percentage: float = 1.0,
) -> dict[str, Any]:
    """
    Check if an image contains a specific color within a given tolerance.

    Args:
        image: Path to the image file or PIL Image object
        target_color: Target RGB color to search for [R, G, B] values (0-255)
        color_tolerance: Tolerance for color matching (0-255).
                        Higher values match more similar colors. Default: 50
        min_percentage: Minimum percentage of pixels that must match
                       the target color. Default: 1.0

    Returns:
        Dict containing:
            - contains_color: Whether the image contains the target color
            - matching_percentage: Percentage of pixels matching the target
            - pixel_count: Number of pixels matching the target color
            - total_pixels: Total number of pixels in the image
            - target_color_rgb: The target color that was searched for
            - target_color_hex: Hex representation of the target color
            - closest_colors: List of closest colors found in the image

    Raises:
        ValueError: If RGB values are not between 0-255
        FileNotFoundError: If image file doesn't exist
        IOError: If image cannot be loaded
    """
    # Validate RGB values
    if isinstance(target_color, list | tuple) and len(target_color) == 3:
        target_color = list(target_color)
    else:
        raise ValueError("target_color must be a list or tuple of 3 values")

    for value in target_color:
        if not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError(f"RGB values must be integers between 0-255, got {value}")

    # Load image
    if isinstance(image, str):
        # Load from file path
        try:
            img = PILImage.open(image)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file not found: {image}") from e
        except Exception as e:
            raise OSError(f"Failed to load image: {e}") from e
    elif isinstance(image, PILImage.Image):
        # Use provided PIL Image
        img = image if image.mode == "RGB" else image.convert("RGB")
    else:
        raise TypeError(
            f"image must be a file path (str) or PIL Image, got {type(image)}"
        )

    # Convert to numpy array
    img_array = np.array(img)
    height, width, _ = img_array.shape
    total_pixels = height * width

    # Reshape to list of pixels
    pixels = img_array.reshape(-1, 3)

    # Calculate color distances using Euclidean distance in RGB space
    target_color_np = np.array(target_color)
    distances = np.sqrt(np.sum((pixels - target_color_np) ** 2, axis=1))

    # Find pixels within tolerance
    matching_mask = distances <= color_tolerance
    matching_count = np.sum(matching_mask)
    matching_percentage = (matching_count / total_pixels) * 100

    # Determine if image contains the color
    contains_color = bool(matching_percentage >= min_percentage)

    # Find closest colors in the image (for additional context)
    unique_colors, color_counts = np.unique(pixels, axis=0, return_counts=True)

    # Calculate distances for all unique colors
    unique_distances = np.sqrt(np.sum((unique_colors - target_color_np) ** 2, axis=1))

    # Get top 5 closest colors
    closest_indices = np.argsort(unique_distances)[:5]
    closest_colors = []

    for idx in closest_indices:
        color = unique_colors[idx]
        count = color_counts[idx]
        distance = unique_distances[idx]
        percentage = (count / total_pixels) * 100

        closest_colors.append(
            {
                "rgb": color.tolist(),
                "hex": "#{:02x}{:02x}{:02x}".format(*color),
                "distance": float(distance),
                "percentage": float(percentage),
            }
        )

    # Format target color as hex
    target_hex = "#{:02x}{:02x}{:02x}".format(*target_color)

    return {
        "contains_color": contains_color,
        "matching_percentage": float(matching_percentage),
        "pixel_count": int(matching_count),
        "total_pixels": total_pixels,
        "target_color_rgb": target_color,
        "target_color_hex": target_hex,
        "closest_colors": closest_colors,
    }
