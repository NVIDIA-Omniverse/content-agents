# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small image blankness checks for render-output guardrails."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

type ImageBlanknessInput = str | Path | bytes | bytearray | memoryview | PILImage.Image
type RGBArray = NDArray[np.uint8]
type FloatArray = NDArray[np.float32]

_ALPHA_VISIBLE_THRESHOLD = 8
_MIN_STRONG_MINOR_PIXELS = 16
_MIN_STRONG_MINOR_RATIO = 0.0001
_STRONG_COLOR_DISTANCE = 24.0
_NEAR_UNIFORM_RGB_RANGE_THRESHOLD = 3.0
_MAX_ALPHA_CONTENT_RATIO = 0.95
_COUNT_CHUNK_PIXELS = 1_000_000


@dataclass(frozen=True)
class ImageBlanknessStats:
    """Lightweight metrics for uniform or near-uniform rendered images."""

    blank: bool
    reason: str | None
    width: int
    height: int
    mode: str
    sampled_pixels: int
    unique_colors: int
    dominant_color_ratio: float
    luma_std: float
    luma_dynamic_range: float
    rgb_dynamic_range: float
    strong_minority_pixel_ratio: float
    alpha_visible_ratio: float | None

    def to_dict(self) -> dict[str, int | float | str | bool | None]:
        """Return a JSON-serializable stats dictionary."""
        return {
            "blank": self.blank,
            "reason": self.reason,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "sampled_pixels": self.sampled_pixels,
            "unique_colors": self.unique_colors,
            "dominant_color_ratio": self.dominant_color_ratio,
            "luma_std": self.luma_std,
            "luma_dynamic_range": self.luma_dynamic_range,
            "rgb_dynamic_range": self.rgb_dynamic_range,
            "strong_minority_pixel_ratio": self.strong_minority_pixel_ratio,
            "alpha_visible_ratio": self.alpha_visible_ratio,
        }


def image_is_blank(
    path_or_bytes: ImageBlanknessInput,
    *,
    min_unique: int = 4,
    max_uniformity: float = 0.99,
    blank_std_threshold: float = 1.0,
    blank_dynamic_range_threshold: float = 2.0,
    max_analysis_pixels: int = 1_000_000,
) -> bool:
    """Return True when an image is uniform or near-uniform."""
    return analyze_image_blankness(
        path_or_bytes,
        min_unique=min_unique,
        max_uniformity=max_uniformity,
        blank_std_threshold=blank_std_threshold,
        blank_dynamic_range_threshold=blank_dynamic_range_threshold,
        max_analysis_pixels=max_analysis_pixels,
    ).blank


def analyze_image_blankness(
    image: ImageBlanknessInput,
    *,
    min_unique: int = 4,
    max_uniformity: float = 0.99,
    blank_std_threshold: float = 1.0,
    blank_dynamic_range_threshold: float = 2.0,
    max_analysis_pixels: int = 1_000_000,
) -> ImageBlanknessStats:
    """Compute simple blankness metrics for a PNG/JPEG/path/PIL image."""
    if min_unique < 1:
        raise ValueError("min_unique must be at least 1")
    if not 0.0 < max_uniformity <= 1.0:
        raise ValueError("max_uniformity must be in the range (0, 1]")
    if max_analysis_pixels < 1:
        raise ValueError("max_analysis_pixels must be at least 1")

    loaded_image = _load_image(image)
    width, height = loaded_image.size
    alpha_visible_ratio = _alpha_visible_ratio(loaded_image)
    if alpha_visible_ratio == 0.0:
        return ImageBlanknessStats(
            blank=True,
            reason="transparent",
            width=width,
            height=height,
            mode=loaded_image.mode,
            sampled_pixels=0,
            unique_colors=0,
            dominant_color_ratio=1.0,
            luma_std=0.0,
            luma_dynamic_range=0.0,
            rgb_dynamic_range=0.0,
            strong_minority_pixel_ratio=0.0,
            alpha_visible_ratio=alpha_visible_ratio,
        )

    pixels = _sample_rgb_pixels(loaded_image, max_analysis_pixels)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    unique_colors = int(unique.shape[0])
    dominant_index = int(np.argmax(counts))
    dominant_color_ratio = float(counts[dominant_index] / len(pixels))
    dominant_color = unique[dominant_index]
    strong_minority_count = _count_strong_minority_pixels(pixels, dominant_color)
    strong_minority_pixel_ratio = strong_minority_count / len(pixels)
    has_strong_minority = (
        strong_minority_count >= _MIN_STRONG_MINOR_PIXELS
        and strong_minority_pixel_ratio >= _MIN_STRONG_MINOR_RATIO
    )
    luma = _rgb_to_luma(pixels)
    luma_std = float(np.std(luma))
    luma_dynamic_range = float(np.ptp(luma))
    rgb_dynamic_range = float(np.max(np.ptp(pixels, axis=0)))
    has_alpha_content = (
        alpha_visible_ratio is not None
        and 0.0 < alpha_visible_ratio <= _MAX_ALPHA_CONTENT_RATIO
        and int(round(alpha_visible_ratio * width * height)) >= _MIN_STRONG_MINOR_PIXELS
    )

    reason: str | None = None
    if unique_colors == 1 and not has_strong_minority:
        reason = "solid_color"
    elif (
        unique_colors < min_unique
        and not has_strong_minority
        and (
            rgb_dynamic_range <= _NEAR_UNIFORM_RGB_RANGE_THRESHOLD
            or dominant_color_ratio >= max_uniformity
        )
    ):
        reason = "too_few_unique_colors"
    elif dominant_color_ratio >= max_uniformity and not has_strong_minority:
        reason = "dominant_color"
    elif (
        luma_std <= blank_std_threshold
        and luma_dynamic_range <= blank_dynamic_range_threshold
        and rgb_dynamic_range <= _NEAR_UNIFORM_RGB_RANGE_THRESHOLD
    ):
        reason = "near_uniform_luminance"

    if has_alpha_content and reason in {
        "solid_color",
        "too_few_unique_colors",
        "dominant_color",
        "near_uniform_luminance",
    }:
        reason = None

    return ImageBlanknessStats(
        blank=reason is not None,
        reason=reason,
        width=width,
        height=height,
        mode=loaded_image.mode,
        sampled_pixels=int(len(pixels)),
        unique_colors=unique_colors,
        dominant_color_ratio=dominant_color_ratio,
        luma_std=luma_std,
        luma_dynamic_range=luma_dynamic_range,
        rgb_dynamic_range=rgb_dynamic_range,
        strong_minority_pixel_ratio=strong_minority_pixel_ratio,
        alpha_visible_ratio=alpha_visible_ratio,
    )


def _load_image(image: ImageBlanknessInput) -> PILImage.Image:
    if isinstance(image, str | Path):
        with PILImage.open(image) as opened:
            opened.load()
            return _normalize_image_mode(opened, copy_if_same=True)

    if isinstance(image, bytes | bytearray | memoryview):
        with PILImage.open(io.BytesIO(bytes(image))) as opened:
            opened.load()
            return _normalize_image_mode(opened, copy_if_same=True)

    if isinstance(image, PILImage.Image):
        image.load()
        return _normalize_image_mode(image, copy_if_same=False)

    raise TypeError(f"Unsupported image input type: {type(image)}")


def _normalize_image_mode(
    image: PILImage.Image,
    *,
    copy_if_same: bool,
) -> PILImage.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        if image.mode == "RGBA" and not copy_if_same:
            return image
        return image.convert("RGBA")
    if image.mode == "RGB" and not copy_if_same:
        return image
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")


def _alpha_visible_ratio(image: PILImage.Image) -> float | None:
    if image.mode != "RGBA":
        return None
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    return float(np.count_nonzero(alpha > _ALPHA_VISIBLE_THRESHOLD) / alpha.size)


def _sample_rgb_pixels(image: PILImage.Image, max_pixels: int) -> RGBArray:
    if image.mode == "RGBA":
        array = np.asarray(image, dtype=np.uint8)
        sampled = _sample_pixels_from_array(array, max_pixels)
        rgb = sampled[:, :3]
        transparent_pixels = sampled[:, 3] <= _ALPHA_VISIBLE_THRESHOLD
        if np.any(transparent_pixels):
            rgb = rgb.copy()
            rgb[transparent_pixels] = 0
        return cast(RGBArray, rgb)

    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    return cast(RGBArray, _sample_pixels_from_array(array, max_pixels))


def _sample_pixels_from_array(
    array: NDArray[np.uint8],
    max_pixels: int,
) -> NDArray[np.uint8]:
    pixels = array.reshape(-1, array.shape[-1])
    if len(pixels) <= max_pixels:
        return pixels

    height, width = array.shape[:2]
    flat_indices = _halton_flat_indices(height, width, max_pixels)
    return pixels[flat_indices]


def _halton_flat_indices(height: int, width: int, count: int) -> NDArray[np.int64]:
    pixel_count = height * width
    count = min(count, pixel_count)
    sequence: NDArray[np.uint64] = np.arange(1, count + 1, dtype=np.uint64)
    y_indices = np.minimum(
        (_radical_inverse(sequence, base=2) * height).astype(np.int64),
        height - 1,
    )
    x_indices = np.minimum(
        (_radical_inverse(sequence, base=3) * width).astype(np.int64),
        width - 1,
    )
    return cast(NDArray[np.int64], y_indices * width + x_indices)


def _radical_inverse(
    indices: NDArray[np.uint64],
    *,
    base: int,
) -> NDArray[np.float64]:
    values = indices.copy()
    result = np.zeros(values.shape, dtype=np.float64)
    factor = 1.0 / base
    while np.any(values):
        result += factor * (values % base)
        values //= base
        factor /= base
    return result


def _count_strong_minority_pixels(
    pixels: RGBArray,
    dominant_color: RGBArray,
) -> int:
    dominant: NDArray[np.int16] = dominant_color.astype(np.int16, copy=False)
    count = 0
    min_distance_sq = int(_STRONG_COLOR_DISTANCE * _STRONG_COLOR_DISTANCE)
    for start in range(0, len(pixels), _COUNT_CHUNK_PIXELS):
        chunk: NDArray[np.int16] = pixels[start : start + _COUNT_CHUNK_PIXELS].astype(
            np.int16
        )
        diff: NDArray[np.int32] = (chunk - dominant).astype(np.int32, copy=False)
        distance_sq = cast(NDArray[Any], np.sum(diff * diff, axis=1))
        count += int(np.count_nonzero(distance_sq >= min_distance_sq))
    return count


def _rgb_to_luma(pixels: RGBArray) -> FloatArray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return cast(FloatArray, pixels @ weights)
