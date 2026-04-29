# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures and utilities for Material Agent tests."""

import logging
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from PIL import Image


# TODO(ISSUE-ID): Remove this fixture (preload_unified_config) and the related
# ad-hoc imports in the tests below once the circular import chain rooted at
# material_agent.config.unified_config is resolved:
#   config.schema → api.defaults → api.__init__ → api.builders → config.schema
# Fix options: break the cycle in api.builders/config.schema, or move defaults
# out of import-time code in api.__init__. Tests that currently rely on this
# workaround and will need updating:
#   - test_validate_usd_config.py
#   - test_schema_step_order.py
@pytest.fixture(autouse=True, scope="session")
def preload_unified_config() -> None:
    """Seed the unified_config import chain at session start.

    material_agent.config.unified_config has a circular import (schema <->
    api.builders) that causes an ImportError on the very first import attempt.
    The second attempt succeeds because sys.modules has the partially-loaded
    entries from the first try.  Importing once here — before any test runs —
    ensures all subsequent imports in the session succeed reliably.
    """
    try:
        import material_agent.config.unified_config  # noqa: F401
    except ImportError:
        pass  # expected on first attempt; sys.modules is now seeded


@pytest.fixture(autouse=True)
def reset_log_propagation():
    """Restore logger propagation before each test.

    setup_logging() sets propagate=False on these loggers for CLI use.
    Resetting to True ensures pytest's caplog fixture can capture records.
    """
    logger_names = ["world_understanding", "material_agent"]
    saved = {n: logging.getLogger(n).propagate for n in logger_names}
    for n in logger_names:
        logging.getLogger(n).propagate = True
    yield
    for n, v in saved.items():
        logging.getLogger(n).propagate = v


@pytest.fixture
def mock_vlm():
    """Create a mock Vision-Language Model."""
    vlm = Mock()
    vlm.generate = MagicMock(
        return_value=(
            "Looking at the images, I can see a black tire with visible tread "
            "patterns. The material appears to be rubber based on its matte "
            "finish and flexible sidewalls."
        )
    )
    vlm.model_name = "mock-vlm"
    vlm.backend_name = "mock-service"
    return vlm


@pytest.fixture
def mock_llm():
    """Create a mock Language Model for parsing."""
    llm = Mock()
    response = Mock()
    response.content = (
        '{"material": "matt black rubber", '
        '"reasoning": "The tire shows characteristic black color with visible '
        "tread patterns. The flexibility evident in the sidewall and the matte "
        'finish indicate this is rubber material."}'
    )
    llm.invoke = MagicMock(return_value=response)
    return llm


@pytest.fixture
def sample_pil_images():
    """Create sample PIL Images for testing."""
    # Create small test images with different colors
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")
    img3 = Image.new("RGB", (100, 100), color="green")
    return [img1, img2, img3]


@pytest.fixture
def sample_image_files(tmp_path):
    """Create temporary image files for testing."""
    images = []
    colors = ["red", "blue", "green"]

    for i, color in enumerate(colors):
        img = Image.new("RGB", (100, 100), color=color)
        img_path = tmp_path / f"test_image_{i}.png"
        img.save(img_path)
        images.append(img_path)

    return images


@pytest.fixture
def sample_entries(sample_image_files, sample_pil_images):
    """Create sample entries for batch processing."""
    return [
        {
            "id": "entry_001",
            "text": "This is a car wheel. Materials: steel, rubber, plastic",
            "images": [str(sample_image_files[0]), str(sample_image_files[1])],
        },
        {
            "id": "entry_002",
            "text": "This is a car door. Materials: steel, glass, plastic",
            "images": sample_pil_images[:2],  # Use PIL Images
        },
        {
            "id": "entry_003",
            "text": "This is a car seat. Materials: leather, fabric, plastic",
            "images": [
                str(sample_image_files[2]),
                sample_pil_images[0],
            ],  # Mixed types
        },
    ]


@pytest.fixture
def mock_vlm_with_varied_responses():
    """Create a mock VLM that returns different responses for each call."""
    vlm = Mock()
    responses = [
        "The wheel appears to be made of rubber with visible tread patterns.",
        "The door shows a metallic surface consistent with painted steel.",
        "The seat material looks like leather based on its texture and sheen.",
    ]
    vlm.generate = MagicMock(side_effect=responses)
    vlm.model_name = "mock-vlm"
    vlm.backend_name = "mock-service"
    return vlm


@pytest.fixture
def mock_llm_with_varied_responses():
    """Create a mock LLM that returns different parsed responses."""
    llm = Mock()
    responses = [
        Mock(
            content='{"material": "rubber", "reasoning": "Visible tread patterns and black color indicate rubber tire."}'
        ),
        Mock(
            content='{"material": "steel", "reasoning": "Metallic surface with paint indicates steel construction."}'
        ),
        Mock(
            content='{"material": "leather", "reasoning": "Texture and sheen are characteristic of leather material."}'
        ),
    ]
    llm.invoke = MagicMock(side_effect=responses)
    return llm


@pytest.fixture
def mock_vlm_with_error():
    """Create a mock VLM that raises an error."""
    vlm = Mock()
    vlm.generate = MagicMock(side_effect=Exception("VLM inference failed"))
    vlm.model_name = "mock-vlm"
    vlm.backend_name = "mock-service"
    return vlm


@pytest.fixture
def mock_llm_with_invalid_json():
    """Create a mock LLM that returns invalid JSON."""
    llm = Mock()
    response = Mock()
    response.content = "This is not valid JSON: {material: missing quotes}"
    llm.invoke = MagicMock(return_value=response)
    return llm
