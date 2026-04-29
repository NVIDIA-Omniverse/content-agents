# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NAT graphics tool registration."""

import pytest


def test_load_usd_stage_config():
    """Test that LoadUsdStageConfig can be created."""
    try:
        from world_understanding.nat.register_graphics import (
            LoadUsdStageConfig,
        )
    except ImportError:
        pytest.skip("NAT not installed")

    # Verify config can be created
    config = LoadUsdStageConfig()
    assert config is not None


def test_save_usd_stage_config():
    """Test that SaveUsdStageConfig can be created."""
    try:
        from world_understanding.nat.register_graphics import (
            SaveUsdStageConfig,
        )
    except ImportError:
        pytest.skip("NAT not installed")

    # Verify config can be created
    config = SaveUsdStageConfig()
    assert config is not None


def test_render_single_camera_config():
    """Test that RenderSingleCameraConfig can be created."""
    try:
        from world_understanding.nat.register_graphics import (
            RenderSingleCameraConfig,
        )
    except ImportError:
        pytest.skip("NAT not installed")

    # Verify config can be created
    config = RenderSingleCameraConfig()
    assert config is not None


def test_render_all_cameras_config():
    """Test that RenderAllCamerasConfig can be created."""
    try:
        from world_understanding.nat.register_graphics import (
            RenderAllCamerasConfig,
        )
    except ImportError:
        pytest.skip("NAT not installed")

    # Verify config can be created
    config = RenderAllCamerasConfig()
    assert config is not None
