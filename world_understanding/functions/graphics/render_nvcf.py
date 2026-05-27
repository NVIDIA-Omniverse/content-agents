# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated compatibility module for REST API-based rendering.

Use :mod:`world_understanding.functions.graphics.render_remote` for new code.
This module name is kept because the REST renderer was originally NVCF-only.
"""

# ruff: noqa: F401,F403

from world_understanding.functions.graphics.render_remote import *
from world_understanding.functions.graphics.render_remote import (
    _bundle_stage_with_local_assets,
    _convert_v2_sensor,
    _convert_v2_to_v1,
    _export_stage_and_get_url,
    _is_v2_response,
)
