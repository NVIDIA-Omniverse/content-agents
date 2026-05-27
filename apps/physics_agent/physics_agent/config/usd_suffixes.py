# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD suffix helpers shared by Physics Agent config and services."""

USD_ARTIFACT_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")


def default_apply_physics_output_suffix(input_suffix: str) -> str:
    """Return the default apply_physics output suffix for an input suffix."""
    suffix = input_suffix.lower()
    if suffix not in USD_ARTIFACT_EXTENSIONS:
        return ".usd"
    if suffix == ".usdz":
        return ".usda"
    return suffix
