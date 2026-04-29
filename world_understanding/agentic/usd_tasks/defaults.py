# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default configurations for USD data preparation tasks.

This module provides centralized defaults for USD rendering and dataset building
to ensure consistency across the world_understanding library.
"""

# ============================================================================
# Camera Configuration Defaults
# ============================================================================

# Default camera directions for corner view (2 opposite corners for efficiency)
DEFAULT_CAMERA_DIRECTIONS = ["+x+y+z", "-x-y-z"]

# Default camera directions for side view (all 6 cardinal directions)
DEFAULT_SIDE_VIEW_DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]

# Default camera directions for all corners (8 corners of a cube)
ALL_CORNER_DIRECTIONS = [
    "+x+y+z",
    "-x+y+z",
    "-x-y+z",
    "+x-y+z",
    "+x+y-z",
    "-x+y-z",
    "-x-y-z",
    "+x-y-z",
]


# ============================================================================
# Rendering Defaults
# ============================================================================

DEFAULT_RENDER_BACKEND = "remote"
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_CAMERA_VIEW_TYPE = "corner"
DEFAULT_CULL_STYLE = "back"
DEFAULT_SHOULD_HIGHLIGHT_PRIM = False
DEFAULT_SHOULD_ASSIGN_RANDOM_COLORS = True
DEFAULT_SHOULD_RESET_MATERIALS = True


# ============================================================================
# USD Dataset Building Defaults
# ============================================================================

USD_RENDERING_DEFAULTS = {
    "backend": DEFAULT_RENDER_BACKEND,
    "image_width": DEFAULT_IMAGE_WIDTH,
    "image_height": DEFAULT_IMAGE_HEIGHT,
    "camera_view_type": DEFAULT_CAMERA_VIEW_TYPE,
    # Don't set camera_directions here - let renderer.py default based on camera_view_type
    # This allows corner view -> 8 directions, side view -> 6 directions
    "cull_style": DEFAULT_CULL_STYLE,
    "should_highlight_prim": DEFAULT_SHOULD_HIGHLIGHT_PRIM,
    "should_assign_random_colors": DEFAULT_SHOULD_ASSIGN_RANDOM_COLORS,
    "should_reset_materials": DEFAULT_SHOULD_RESET_MATERIALS,
}

USD_PROCESSING_DEFAULTS = {
    "skip_existing": False,
    "batch_size": 10,
    "num_workers": 1,
    "rendering_modes": ["prim_with_stage", "prim_only"],
}

USD_METADATA_DEFAULTS = {
    "extract_metadata": False,
    "extract_display_color": False,
    "extract_material_bindings": True,
    "extract_hierarchy": True,
    "include_display_color_statistics": False,
}

USD_MODEL_DEFAULTS = {
    "build_usd_model": True,
    "export_usd_model": True,
}

# Combined defaults for all USD dataset building configuration
USD_DATASET_DEFAULTS = {
    **USD_RENDERING_DEFAULTS,
    **USD_PROCESSING_DEFAULTS,
    **USD_METADATA_DEFAULTS,
    **USD_MODEL_DEFAULTS,
}
