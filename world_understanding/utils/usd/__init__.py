# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD utilities submodule for working with Universal Scene Description files.

This module provides utilities for:
- Stage manipulation (creation, loading, saving, merging)
- Prim operations (traversal, material handling, visibility control)
- Camera setup and positioning
- Material creation and binding (MDL materials)
- Scene graph operations
- Rendering and visualization
- Asset management
"""

from . import camera, composition, material, prim, stage

__all__ = ["camera", "composition", "material", "prim", "stage"]
