# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material Agent inference functions."""

from .inference import assign_material, batch_assign_materials

__all__ = ["assign_material", "batch_assign_materials"]
