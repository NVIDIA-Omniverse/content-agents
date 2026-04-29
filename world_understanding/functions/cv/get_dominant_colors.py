# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract dominant colors from images using k-means clustering."""

from typing import Any

import numpy as np
from PIL import Image as PILImage
from sklearn.cluster import KMeans


def get_dominant_colors(
    image: str | PILImage.Image, n_colors: int = 5, analyze_brightness: bool = True
) -> dict[str, Any]:
    """
    Extract dominant colors from an image using k-means clustering.

    Args:
        image: Path to the image file or PIL Image object
        n_colors: Number of dominant colors to extract (1-20)
        analyze_brightness: Whether to calculate average brightness

    Returns:
        Dict containing:
            - dominant_colors: List of color info dicts with rgb, hex, %
            - average_brightness: Average brightness of the image (0-255)
            - color_diversity: Measure of color variation (0-1)
            - n_clusters: Number of color clusters used

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If n_colors is out of valid range
        IOError: If image cannot be loaded
    """
    # Validate n_colors
    if not 1 <= n_colors <= 20:
        raise ValueError(f"n_colors must be between 1 and 20, got {n_colors}")

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

    # Convert to numpy array and reshape to pixels
    img_array = np.array(img)
    height, width, _ = img_array.shape
    pixels = img_array.reshape(-1, 3)

    # K-means clustering for dominant colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Calculate color percentages
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels)

    # Format results
    dominant_colors = []
    for _i, (center, pct) in enumerate(
        zip(kmeans.cluster_centers_, percentages, strict=False)
    ):
        rgb = center.astype(int).tolist()
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        dominant_colors.append({"rgb": rgb, "hex": hex_color, "percentage": float(pct)})

    # Sort by percentage (highest first)
    dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)

    # Calculate brightness if requested
    average_brightness = float(np.mean(img_array)) if analyze_brightness else 0.0

    # Calculate color diversity (standard deviation of pixel values)
    color_diversity = float(np.std(pixels) / 255.0)  # Normalize to 0-1

    return {
        "dominant_colors": dominant_colors,
        "average_brightness": average_brightness,
        "color_diversity": color_diversity,
        "n_clusters": n_colors,
    }
