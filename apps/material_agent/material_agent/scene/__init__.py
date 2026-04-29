# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large scene multi-asset pipeline for material agent.

Orchestrates: analyze scene -> extract sub-assets -> run pipeline on each
-> collect results and compose material layers onto the original scene.
"""
