# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal embedding model implementations."""

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from world_understanding.functions.models.base_embedding_model import BaseEmbeddingModel
from world_understanding.utils.image_utils import image_to_base64


class BaseMultimodalEmbeddingModel(BaseEmbeddingModel):
    """Base multimodal embedding model."""

    pass


class NIMMultimodalEmbeddingModel(BaseMultimodalEmbeddingModel):
    """NVIDIA NIM multimodal embedding model."""

    # Available NIM models for multimodal embeddings
    AVAILABLE_MODELS = [
        "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        "nvidia/nvclip",
    ]

    DEFAULT_MODEL = "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ):
        """Initialize NIM multimodal embedding model.

        Args:
            api_key: NVIDIA API key
            model: Model ID to use for embeddings
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        if not self.AVAILABLE_MODELS:
            raise NotImplementedError(
                "Currently provide no dedicated multimodal embedding models. "
            )

        # Validate model is supported
        self._validate_model(model, self.AVAILABLE_MODELS)

        # Override base_url if provided in kwargs, otherwise use NIM default
        base_url = kwargs.pop("base_url", "https://integrate.api.nvidia.com/v1")

        # Initialize parent class
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs,
        )

        self._init_embedding_dimension()

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available NIM models for multimodal embeddings.

        Returns:
            List of available NIM model names
        """
        return cls.AVAILABLE_MODELS.copy()

    def embed_texts(
        self,
        texts: list[str | Path],
        input_type: str = "passage",
    ) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of text strings or file paths to embed
            input_type: Type of input - 'passage' or 'query' (required by NIM models)
        """
        texts = [self._load_text(text) for text in texts]

        # Handle different NIM model types
        if "nvclip" in self.model:
            # nvclip is multimodal and doesn't require extra_body
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format="float",
            )
        elif "vlm-embed" in self.model:
            # VLM models require modality parameter - must match input length
            modalities = ["text"] * len(texts)
            response = self.client.embeddings.create(
                input=texts,
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
        elif "vlm-embed" in self.model:
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


def create_nim_multimodal_embedding_model(
    api_key: str,
    model: str = NIMMultimodalEmbeddingModel.DEFAULT_MODEL,
    **kwargs: Any,
) -> NIMMultimodalEmbeddingModel:
    """Create NVIDIA NIM multimodal embedding model.

    Args:
        api_key: NVIDIA API key
        model: Model ID to use for embeddings
        **kwargs: Additional arguments to pass to NIMMultimodalEmbeddingModel

    Returns:
        Configured NIMMultimodalEmbeddingModel instance
    """
    return NIMMultimodalEmbeddingModel(
        api_key=api_key,
        model=model,
        **kwargs,
    )


def list_nim_multimodal_models() -> list[str]:
    """List all available NIM models for multimodal embeddings.

    Returns:
        List of available NIM model names
    """
    return NIMMultimodalEmbeddingModel.list_available_models()


def create_multimodal_embedding_model(
    backend: str, api_key: str | None = None, model: str | None = None, **kwargs: Any
) -> BaseEmbeddingModel:
    """Create a multimodal embedding model for the specified backend.

    Args:
        backend: Backend name ('nim')
        api_key: API key for the backend
        model: Model ID (optional, defaults used if not specified)
        **kwargs: Additional backend-specific arguments

    Returns:
        Configured multimodal embedding model instance

    Raises:
        ValueError: If backend is not supported or required parameters missing
    """
    if backend == "nim":
        if api_key is None:
            api_key = os.getenv("NVIDIA_API_KEY")
        if api_key is None:
            raise ValueError("API key is required for NIM backend")
        return create_nim_multimodal_embedding_model(
            api_key=api_key,
            model=model or NIMMultimodalEmbeddingModel.DEFAULT_MODEL,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}. Available backends: nim")
