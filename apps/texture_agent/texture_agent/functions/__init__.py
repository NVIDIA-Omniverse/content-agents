# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from .material_discovery import (
    MaterialInfo,
    PrimTextureUnit,
    discover_materials,
    expand_to_prim_units,
)
from .rest_client import (
    RestTextureVariationClient,
)
from .texture_blending import blend_texture_onto_constant
from .texture_generation import (
    # Engine abstraction
    BaseTextureEngine,
    # Legacy interfaces (backward compat)
    BaseTextureGenerator,
    # API data models (from texture_variation_api.md)
    Conditioning,
    GeneratedTextures,
    GenerationResult,
    ImageGenEngine,
    ImageGenTextureGenerator,
    JobStatus,
    TextureRequest,
    TextureResult,
    TextureVariationClient,
    TextureVariationConfig,
    create_texture_generator,
)
