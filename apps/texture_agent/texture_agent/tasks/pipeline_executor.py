# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline executor for the texture agent.

Re-exports the run_pipeline function from the workflows module
for convenience.
"""

from texture_agent.workflows.factory import run_pipeline

__all__ = ["run_pipeline"]
