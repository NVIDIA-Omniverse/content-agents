# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for physics_agent_service."""

# Exclude internal manual utilities from test collection — these are
# not pytest tests, and their module-level imports break collection.
collect_ignore_glob = ["internal/scripts/*"]
