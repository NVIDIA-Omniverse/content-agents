# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image embedding model implementations."""

# For type hints
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from world_understanding.functions.models.base_embedding_model import BaseEmbeddingModel
from world_understanding.utils.image_utils import image_to_base64


class BaseImageEmbeddingModel(BaseEmbeddingModel):
    """Base image embedding model."""

    pass


class OpenAIImageEmbeddingModel(BaseImageEmbeddingModel):
    """OpenAI-compatible image embedding model."""

    # Available OpenAI models that can be used for image embeddings
    AVAILABLE_MODELS = []  # OpenAI doesn't currently provide image embedding models

    DEFAULT_MODEL = ""  # No default since no models available

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ):
        """Initialize OpenAI image embedding model.

        Args:
            api_key: API key for the service
            model: Model ID to use for embeddings
            base_url: Base URL for the API (optional)
            **kwargs: Additional configuration options
        """
        if not self.AVAILABLE_MODELS:
            raise NotImplementedError(
                "Currently provide no dedicated image embedding models. "
            )

        # Validate model is supported
        self._validate_model(model, self.AVAILABLE_MODELS)

        # Initialize parent class
        super().__init__(
            api_key=api_key,
            model=model,
            **kwargs,
        )

        self._init_embedding_dimension()

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available OpenAI models that can be used for image embeddings.

        Returns:
            List of available OpenAI model names
        """
        return cls.AVAILABLE_MODELS.copy()

    def embed_image(
        self, image: str | Path | PILImage.Image | np.ndarray
    ) -> np.ndarray:
        """Generate embedding for a single image."""
        return self.embed_images([image])[0]

    def embed_images(
        self,
        images: list[str | Path | PILImage.Image | np.ndarray],
    ) -> list[np.ndarray]:
        """Generate embeddings for multiple images."""
        if not images:
            return []

        # Process all images to base64
        image_base64_list = []
        for image in images:
            pil_image = self._load_image(image)
            image_base64 = image_to_base64(pil_image)
            image_base64_list.append(f"data:image/png;base64,{image_base64}")

        # Call embeddings API for all images at once
        response = self.client.embeddings.create(
            input=image_base64_list,
            model=self.model,
            encoding_format="float",
        )

        return self._get_embedding_vectors_from_response(response)


class NIMImageEmbeddingModel(BaseImageEmbeddingModel):
    """NVIDIA NIM image embedding model."""

    # Available NIM models for image embeddings
    AVAILABLE_MODELS = [
        "nvidia/nvclip",  # Multimodal model
        "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",  # Multimodal model
        "nvidia/llama-nemotron-embed-vl-1b-v2",  # Multimodal model
        # "nv-dinov2",  # image embedding model
    ]

    DEFAULT_MODEL = "nvidia/nvclip"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ):
        """Initialize NIM image embedding model.

        Args:
            api_key: NVIDIA API key
            model: Model ID to use for embeddings
            **kwargs: Additional configuration options
        """
        if not self.AVAILABLE_MODELS:
            raise NotImplementedError(
                "Currently provide no dedicated image embedding models. "
            )

        # Validate model is supported
        self._validate_model(model, self.AVAILABLE_MODELS)

        # Override base_url if provided in kwargs, otherwise use NIM default
        base_url = kwargs.pop("base_url", "https://integrate.api.nvidia.com/v1")

        # Call parent constructor with NIM-specific defaults
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs,
        )

        self._init_embedding_dimension()

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available NIM models for image embeddings.

        Returns:
            List of available NIM model names
        """
        return cls.AVAILABLE_MODELS.copy()

    def _init_embedding_dimension(self) -> None:
        """Initialize the embedding dimension based on the model."""
        # Create a dummy PIL image (e.g., 1x1 white pixel)
        dummy_image = PILImage.new("RGB", (1, 1), color=(255, 255, 255))
        embedding_vectors = self.embed_image(dummy_image)
        self._update_embedding_dimension(embedding_vectors.shape[0])

    def embed_images(
        self,
        images: list[str | Path | PILImage.Image | np.ndarray],
        input_type: str = "passage",
    ) -> list[np.ndarray]:
        """Generate embeddings for multiple images using NIM-specific parameters."""
        if not images:
            return []

        # Process all images to base64
        image_base64_list = []
        for image in images:
            pil_image = self._load_image(image)
            image_base64 = image_to_base64(pil_image)
            image_base64_list.append(f"data:image/png;base64,{image_base64}")

        # Call embeddings API with NIM-specific parameters
        if "nvclip" in self.model:
            response = self.client.embeddings.create(
                input=image_base64_list,
                model=self.model,
                encoding_format="float",
            )
        elif "vlm-embed" in self.model or "embed-vl" in self.model:
            # VLM models require modality parameter - must match input length
            modalities = ["image"] * len(image_base64_list)
            response = self.client.embeddings.create(
                input=image_base64_list,
                model=self.model,
                encoding_format="float",
                extra_body={
                    "modality": modalities,
                    "input_type": input_type,
                    "truncate": "NONE",
                },
            )
        else:
            raise ValueError(
                f"Unsupported NIM model: {self.model}. Available models: {self.AVAILABLE_MODELS}"
            )

        return self._get_embedding_vectors_from_response(response)


class LocalVisualImageEmbeddingModel(BaseImageEmbeddingModel):
    """Deterministic local image embedding based on downsampled RGB pixels."""

    AVAILABLE_MODELS: tuple[str, ...] = ("local_visual",)
    DEFAULT_MODEL = "local_visual"
    EMBEDDING_DIMENSION = 768
    _DESCRIPTOR_WEIGHT = 8.0
    _COLOR_ANCHOR_SIGMA = 0.08
    _COLOR_ANCHORS_RGB: tuple[tuple[int, int, int], ...] = (
        (0, 0, 0),
        (25, 25, 25),
        (80, 80, 80),
        (128, 128, 128),
        (160, 160, 150),
        (192, 192, 192),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 220, 70),
        (190, 140, 40),
        (184, 115, 51),
        (120, 70, 35),
        (40, 80, 50),
        (220, 180, 120),
    )

    def __init__(self, **kwargs: Any):
        # Skip BaseEmbeddingModel client initialization; this backend is local.
        self.api_key = "not-used"
        self.model = self.DEFAULT_MODEL
        self.base_url = None
        self.timeout = 120.0
        self._embedding_dim = self.EMBEDDING_DIMENSION

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List local visual embedding model names."""
        return list(cls.AVAILABLE_MODELS)

    def embed_image(
        self, image: str | Path | PILImage.Image | np.ndarray
    ) -> np.ndarray:
        """Generate embedding for a single image."""
        return self.embed_images([image])[0]

    def embed_images(
        self,
        images: list[str | Path | PILImage.Image | np.ndarray],
    ) -> list[np.ndarray]:
        """Generate deterministic local visual embeddings for images."""
        vectors: list[np.ndarray] = []
        for image in images:
            pil_image = self._load_image(image).convert("RGB").resize((16, 16))
            pixels = np.asarray(pil_image, dtype=np.float32) / 255.0
            raw_vec = pixels.reshape(-1)
            vec = raw_vec.copy()

            # Preserve absolute color/luminance for flat material patches.
            # A plain unit-normalized RGB raster makes uniform grayscale patches
            # at different brightness nearly collinear, which is a poor fit for
            # material clustering.
            luma = (
                pixels[..., 0] * 0.2126
                + pixels[..., 1] * 0.7152
                + pixels[..., 2] * 0.0722
            )
            mean_rgb = pixels.mean(axis=(0, 1))
            std_rgb = pixels.std(axis=(0, 1))
            luma_mean = float(luma.mean())
            luma_std = float(luma.std())
            descriptors = np.array(
                [
                    *mean_rgb,
                    *std_rgb,
                    luma_mean,
                    luma_std,
                    *(mean_rgb**2),
                    luma_mean**2,
                    1.0,
                ],
                dtype=np.float32,
            )
            anchors = np.asarray(self._COLOR_ANCHORS_RGB, dtype=np.float32) / 255.0
            anchor_distances = np.linalg.norm(anchors - mean_rgb[None, :], axis=1)
            color_anchor_responses = np.exp(
                -(anchor_distances**2) / (2 * self._COLOR_ANCHOR_SIGMA**2)
            ).astype(np.float32)

            features = np.concatenate([descriptors, color_anchor_responses])
            vec[: len(features)] = features * self._DESCRIPTOR_WEIGHT
            vec[-1] = self._DESCRIPTOR_WEIGHT
            norm = np.linalg.norm(vec)
            if norm == 0:
                vec = np.zeros(self.EMBEDDING_DIMENSION, dtype=np.float32)
                vec[0] = 1.0
                norm = 1.0
            vec = vec / norm
            vectors.append(vec.astype(np.float32))
        return vectors


def create_nim_image_embedding_model(
    api_key: str,
    model: str = NIMImageEmbeddingModel.DEFAULT_MODEL,
    **kwargs: Any,
) -> NIMImageEmbeddingModel:
    """Create NVIDIA NIM image embedding model.

    Args:
        api_key: NVIDIA API key
        model: Model ID to use for embeddings
        **kwargs: Additional arguments to pass to NIMImageEmbeddingModel

    Returns:
        Configured NIMImageEmbeddingModel instance
    """
    return NIMImageEmbeddingModel(api_key=api_key, model=model, **kwargs)


def list_nim_image_models() -> list[str]:
    """List all available NIM models for image embeddings.

    Returns:
        List of available NIM model names
    """
    return NIMImageEmbeddingModel.list_available_models()


def create_openai_image_embedding_model(
    api_key: str,
    model: str = OpenAIImageEmbeddingModel.DEFAULT_MODEL,
    **kwargs: Any,
) -> OpenAIImageEmbeddingModel:
    """Create OpenAI image embedding model.

    Args:
        api_key: OpenAI API key
        model: Model ID to use for embeddings
        base_url: Base URL for the API (optional)
        **kwargs: Additional arguments to pass to OpenAIImageEmbeddingModel

    Returns:
        Configured OpenAIImageEmbeddingModel instance
    """
    return OpenAIImageEmbeddingModel(
        api_key=api_key,
        model=model,
        **kwargs,
    )


def list_openai_image_models() -> list[str]:
    """List all available OpenAI models that can be used for image embeddings.

    Returns:
        List of available OpenAI model names
    """
    return OpenAIImageEmbeddingModel.list_available_models()


def create_image_embedding_model(
    backend: str, api_key: str | None = None, model: str | None = None, **kwargs: Any
) -> BaseImageEmbeddingModel:
    """Create an image embedding model for the specified backend.

    Args:
        backend: Backend name ('nim', 'openai')
        api_key: API key for the backend
        model: Model ID (optional, defaults used if not specified)
        embedding_dimension: Dimension of embeddings (backend-specific
            defaults)
        base_url: Base URL for the API (for openai backend)
        **kwargs: Additional backend-specific arguments

    Returns:
        Configured image embedding model instance

    Raises:
        ValueError: If backend is not supported or required parameters missing
    """
    if backend == "nim":
        if api_key is None:
            api_key = os.getenv("NVIDIA_API_KEY")
        if api_key is None:
            raise ValueError("API key is required for NIM backend")
        return create_nim_image_embedding_model(
            api_key=api_key,
            model=model or NIMImageEmbeddingModel.DEFAULT_MODEL,
            **kwargs,
        )

    elif backend == "openai":
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key is required for OpenAI backend")
        return create_openai_image_embedding_model(
            api_key=api_key,
            model=model or OpenAIImageEmbeddingModel.DEFAULT_MODEL,
            **kwargs,
        )

    elif backend == "mock":
        from world_understanding.functions.models.backends.public.mock import (
            MockImageEmbeddingModel,
        )

        return MockImageEmbeddingModel()

    elif backend in {"local", "local_visual"}:
        return LocalVisualImageEmbeddingModel()

    else:
        raise ValueError(
            "Unknown backend: "
            f"{backend}. Available backends: nim, openai, mock, local_visual"
        )
