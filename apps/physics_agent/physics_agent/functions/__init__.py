# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics Agent Functions."""

from physics_agent.functions.inference import batch_classify_assets, classify_asset

__all__ = [
    "classify_asset",
    "batch_classify_assets",
]
