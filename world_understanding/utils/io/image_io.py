# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image I/O utilities."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_to_array(path: str | Path) -> np.ndarray:
    """Load an image from disk as numpy array."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def save_image_from_array(img: np.ndarray, path: str | Path) -> None:
    """Save a numpy array as an image."""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
