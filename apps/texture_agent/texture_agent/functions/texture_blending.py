# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Texture blending: composite generated textures onto constant base colors.

When an OpenPBR material has a constant base_color (e.g., steel gray) and
we want to add a texture effect (e.g., rust), we blend the generated texture
onto a flat image of the base color. This ensures that untextured areas
retain the original material color.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def blend_texture_onto_constant(
    base_color: tuple[float, float, float],
    texture: Image.Image,
    mask: Image.Image | None = None,
    output_size: tuple[int, int] = (1024, 1024),
    opacity: float = 1.0,
) -> Image.Image:
    """Blend a texture onto a flat base color using alpha compositing.

    Args:
        base_color: RGB values in [0, 1] (linear). The material's constant
            base_color from the OpenPBR definition.
        texture: The texture to overlay (e.g., rust, scratches, dirt).
            If RGBA, the alpha channel is used as the blend mask.
        mask: Optional explicit mask image (white = texture, black = base).
            Overrides the texture's own alpha channel.
        output_size: Resolution of the output image (width, height).
        opacity: Global opacity multiplier for the texture layer [0, 1].
            0.0 = pure base color, 1.0 = full texture (or texture alpha).

    Returns:
        Blended RGB image at the requested output_size.
    """
    # Create flat base image from constant color
    base_rgb = tuple(int(c * 255) for c in base_color)
    base = Image.new("RGB", output_size, base_rgb)

    # Resize texture to match output
    texture = texture.resize(output_size, Image.Resampling.LANCZOS)

    # Determine the blend mask
    if mask is not None:
        alpha = mask.resize(output_size, Image.Resampling.LANCZOS).convert("L")
    elif texture.mode == "RGBA":
        alpha = texture.split()[3]
    else:
        # No mask and no alpha -- full opacity (texture fully replaces base)
        alpha = Image.new("L", output_size, 255)

    # Apply global opacity multiplier
    if opacity < 1.0:
        alpha_arr = np.array(alpha, dtype=np.float32) * opacity
        alpha = Image.fromarray(alpha_arr.clip(0, 255).astype(np.uint8))

    # Composite texture onto base
    texture_rgb = texture.convert("RGB")
    base.paste(texture_rgb, mask=alpha)
    return base
