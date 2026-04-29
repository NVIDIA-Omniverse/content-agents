# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for image generation models."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

from world_understanding.functions.models.image_generation_models import (
    BaseImageGenerationModel,
    GeminiImageGenerationModel,
    NIMImageGenerationModel,
    OpenAIImageGenerationModel,
    create_image_generation_model,
)

# Check if google-genai is available
try:
    import google.genai  # noqa: F401

    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

# Check if openai is available
try:
    import openai  # noqa: F401

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_create_image_generation_model_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test creating Gemini image generation model."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        create_image_generation_model("gemini")


def test_create_image_generation_model_unknown_backend() -> None:
    """Test creating image generation model with unknown backend."""
    with pytest.raises(ValueError, match="Unknown image generation backend"):
        create_image_generation_model("unknown_backend")


def test_create_image_generation_model_not_implemented() -> None:
    """Test creating image generation model with unsupported backends."""
    with pytest.raises(ValueError, match="Unknown image generation backend"):
        create_image_generation_model("openai_dalle")

    with pytest.raises(ValueError, match="Unknown image generation backend"):
        create_image_generation_model("stability")


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_create_image_generation_model_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test creating OpenAI image generation model raises without API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        create_image_generation_model("openai")


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_initialization_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that OpenAIImageGenerationModel requires an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        OpenAIImageGenerationModel()


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_with_api_key() -> None:
    """Test that OpenAIImageGenerationModel can be initialized with an API key."""
    model = OpenAIImageGenerationModel(api_key="test-key")
    assert model.model_name == "gpt-image-1"
    assert model.backend_name == "openai"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_custom_model_name() -> None:
    """Test that OpenAIImageGenerationModel accepts custom model names."""
    model = OpenAIImageGenerationModel(api_key="test-key", model="custom-model")
    assert model.model_name == "custom-model"


def test_base_image_generation_model_interface() -> None:
    """Test that BaseImageGenerationModel defines the correct interface."""

    class MockImageGenModel(BaseImageGenerationModel):
        def generate(
            self,
            prompt: str,
            images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (100, 100))

        def generate_with_image_prompt_pairs(
            self,
            image_prompt_pairs: list[
                tuple[str, str | Path | PILImage.Image | np.ndarray]
            ],
            final_prompt: str,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (100, 100))

        @property
        def model_name(self) -> str:
            return "mock-model"

        @property
        def backend_name(self) -> str:
            return "mock"

    model = MockImageGenModel()
    assert model.model_name == "mock-model"
    assert model.backend_name == "mock"

    # Test generate
    result = model.generate("test prompt")
    assert isinstance(result, PILImage.Image)
    assert result.size == (100, 100)

    # Test generate_with_image_prompt_pairs
    result = model.generate_with_image_prompt_pairs([], "test prompt")
    assert isinstance(result, PILImage.Image)


def test_load_image_from_different_formats() -> None:
    """Test loading images from various formats."""

    class MockImageGenModel(BaseImageGenerationModel):
        def generate(
            self,
            prompt: str,
            images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (100, 100))

        def generate_with_image_prompt_pairs(
            self,
            image_prompt_pairs: list[
                tuple[str, str | Path | PILImage.Image | np.ndarray]
            ],
            final_prompt: str,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (100, 100))

        @property
        def model_name(self) -> str:
            return "mock"

        @property
        def backend_name(self) -> str:
            return "mock"

    model = MockImageGenModel()

    # Test PIL Image
    pil_img = PILImage.new("RGB", (50, 50))
    loaded = model._load_image(pil_img)
    assert isinstance(loaded, PILImage.Image)
    assert loaded.size == (50, 50)

    # Test numpy array
    np_img = np.zeros((50, 50, 3), dtype=np.uint8)
    loaded = model._load_image(np_img)
    assert isinstance(loaded, PILImage.Image)
    assert loaded.size == (50, 50)

    # Test unsupported type
    with pytest.raises(ValueError, match="Unsupported image type"):
        model._load_image(123)  # type: ignore[arg-type]


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_initialization_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that GeminiImageGenerationModel requires an API key."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        GeminiImageGenerationModel()


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_with_api_key() -> None:
    """Test that GeminiImageGenerationModel can be initialized with API key."""
    # Just test initialization, not actual API calls
    model = GeminiImageGenerationModel(api_key="test-key")
    assert model.model_name == "gemini-3-pro-image-preview"
    assert model.backend_name == "gemini"


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_custom_model_name() -> None:
    """Test that GeminiImageGenerationModel accepts custom model names."""
    model = GeminiImageGenerationModel(api_key="test-key", model="custom-model")
    assert model.model_name == "custom-model"


def test_create_image_generation_model_nim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test creating NIM image generation model raises without API key."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        create_image_generation_model("nim")


def test_nim_model_initialization_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that NIMImageGenerationModel requires an API key."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        NIMImageGenerationModel()


def test_nim_model_with_api_key() -> None:
    """Test that NIMImageGenerationModel can be initialized with an API key."""
    model = NIMImageGenerationModel(api_key="test-key")
    assert model.model_name == "black-forest-labs/flux_2-klein-4b"
    assert model.backend_name == "nim"


def test_nim_model_custom_model_name() -> None:
    """Test that NIMImageGenerationModel accepts custom model names."""
    model = NIMImageGenerationModel(api_key="test-key", model="org/my_model-v1")
    assert model.model_name == "org/my_model-v1"


def test_nim_model_url_slug_conversion() -> None:
    """Test that NIMImageGenerationModel converts model name to URL slug correctly."""
    assert (
        NIMImageGenerationModel._model_to_url_slug("black-forest-labs/flux_2-klein-4b")
        == "black-forest-labs/flux.2-klein-4b"
    )
    assert (
        NIMImageGenerationModel._model_to_url_slug("org/my_model_v2")
        == "org/my.model_v2"
    )
    assert (
        NIMImageGenerationModel._model_to_url_slug("no_slash_model") == "no.slash_model"
    )
