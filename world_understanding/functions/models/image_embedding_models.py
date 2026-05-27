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

    MODEL_CONFIGS: dict[str, dict[str, Any]] = {
        "nvidia/llama-nemotron-embed-vl-1b-v2": {"family": "vlm"},
        "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1": {"family": "vlm"},
    }
    AVAILABLE_MODELS = list(MODEL_CONFIGS)

    DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"

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

        model_config = self.MODEL_CONFIGS.get(self.model)
        if model_config is None:
            raise ValueError(
                f"Unsupported NIM model: {self.model}. Available models: {self.AVAILABLE_MODELS}"
            )

        if model_config["family"] == "nvclip":
            response = self.client.embeddings.create(
                input=image_base64_list,
                model=self.model,
                encoding_format="float",
            )
        elif model_config["family"] == "vlm":
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
                f"Unsupported NIM model family for {self.model}: {model_config['family']}"
            )

        return self._get_embedding_vectors_from_response(response)


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

    else:
        raise ValueError(
            f"Unknown backend: {backend}. Available backends: nim, openai, mock"
        )
