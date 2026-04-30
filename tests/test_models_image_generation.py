# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for image generation models."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

from world_understanding.functions.models.image_generation_models import (
    BaseImageGenerationModel,
    GeminiImageGenerationModel,
    NIMImageGenerationModel,
    NvidiaInferenceImageGenerationModel,
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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
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
def test_openai_model_rejects_remote_base_url_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test OpenAI image generation requires a key for remote base_url."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(
            base_url="https://api.openai-compatible.example/v1",
        )


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_rejects_placeholder_env_key_for_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(
            base_url="https://api.openai-compatible.example/v1",
        )


def test_openai_model_rejects_env_key_for_custom_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom OpenAI-compatible endpoints must use an explicit endpoint key."""

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "hosted-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(
            base_url="https://api.openai-compatible.example/v1",
        )


def test_openai_model_accepts_explicit_key_for_custom_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "hosted-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    OpenAIImageGenerationModel(
        api_key="endpoint-openai-key",
        base_url="https://api.openai-compatible.example/v1",
    )

    assert captured["api_key"] == "endpoint-openai-key"
    assert captured["base_url"] == "https://api.openai-compatible.example/v1"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_rejects_scheme_less_remote_base_url_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(base_url="api.openai-compatible.example:443/v1")


def test_openai_model_uses_dummy_key_for_local_base_url_before_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local OpenAI-compatible image endpoints must not receive hosted keys."""
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "real-hosted-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    OpenAIImageGenerationModel(
        api_key="not-used",
        base_url="http://localhost:8000/v1",
    )

    assert captured["api_key"] == "not-used"
    assert captured["base_url"] == "http://localhost:8000/v1"


def test_openai_model_rejects_env_key_for_local_base_url_without_explicit_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted ``OPENAI_API_KEY`` must not silently flow to a local
    OpenAI-compatible endpoint. Local URLs are non-provider trust boundaries
    and require an explicit endpoint-scoped ``api_key`` (or the documented
    ``not-used`` no-auth placeholder)."""
    monkeypatch.setenv("OPENAI_API_KEY", "real-hosted-openai-key")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(base_url="http://localhost:8000/v1")


def test_openai_model_rejects_local_base_url_without_explicit_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(base_url="http://localhost:8000/v1")


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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        GeminiImageGenerationModel()


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_accepts_gemini_api_key_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that GeminiImageGenerationModel accepts GEMINI_API_KEY."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    model = GeminiImageGenerationModel()

    assert model.model_name == "gemini-3-pro-image-preview"
    assert model.backend_name == "gemini"


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_replaces_placeholder_config_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "real-gemini-key")

    model = GeminiImageGenerationModel(api_key="YOUR_GOOGLE_API_KEY")

    assert model.model_name == "gemini-3-pro-image-preview"
    assert model.backend_name == "gemini"


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
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        create_image_generation_model("nim")


def test_nim_model_initialization_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that NIMImageGenerationModel requires an API key."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        NIMImageGenerationModel()


def test_nim_model_rejects_placeholder_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMImageGenerationModel()


def test_nim_model_rejects_ma_nim_key_for_hosted_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "local-sidecar-key")

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMImageGenerationModel(base_url="https://ai.api.nvidia.com/v1/genai")


def test_nim_model_rejects_nvidia_key_for_custom_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMImageGenerationModel(base_url="https://nim.example.com/v1/genai")


def test_nim_model_accepts_explicit_key_for_custom_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    model = NIMImageGenerationModel(
        api_key="endpoint-nim-key",
        base_url="https://nim.example.com/v1/genai",
    )

    assert model.model_name == "black-forest-labs/flux_2-klein-4b"
    assert model.backend_name == "nim"


def test_nim_model_accepts_explicit_local_sidecar_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")

    model = NIMImageGenerationModel(base_url="http://image-gen-nim:8000/v1")

    assert model.model_name == "black-forest-labs/flux_2-klein-4b"
    assert model.backend_name == "nim"


def test_nim_model_rejects_global_nvidia_key_for_local_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMImageGenerationModel(base_url="http://image-gen-nim:8000/v1")


def test_nim_model_with_api_key() -> None:
    """Test that NIMImageGenerationModel can be initialized with an API key."""
    model = NIMImageGenerationModel(api_key="test-key")
    assert model.model_name == "black-forest-labs/flux_2-klein-4b"
    assert model.backend_name == "nim"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_nvidia_inference_model_rejects_placeholder_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY")

    with pytest.raises(ValueError, match="INFERENCE_NVIDIA_API_KEY"):
        NvidiaInferenceImageGenerationModel()


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
