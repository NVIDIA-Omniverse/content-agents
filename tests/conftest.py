# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for root test modules."""

import logging

import pytest


@pytest.fixture(autouse=True)
def reset_world_understanding_log_propagation():
    """Keep CLI logging setup from leaking into later caplog assertions."""
    logger = logging.getLogger("world_understanding")
    saved = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = saved
