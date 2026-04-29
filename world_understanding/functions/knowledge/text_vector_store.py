# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text vector store for similarity search using FAISS.

This module provides a vector store implementation for text-metadata pairs,
enabling efficient similarity search using FAISS (Facebook AI Similarity Search).
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from world_understanding.functions.knowledge.base_vector_store import (
    BaseSearchResult,
    BaseVectorStore,
)
from world_understanding.functions.models.text_embedding_models import (
    BaseTextEmbeddingModel,
    create_text_embedding_model,
)

# Configure logger
logger = logging.getLogger(__name__)


class TextVectorStore(BaseVectorStore):
    """Vector store for text embeddings using FAISS.

    This class provides functionality to:
    - Add texts with metadata to a vector index
    - Add images by captioning them with VLM and storing the caption
    - Search for similar texts using vector similarity
    - Save and load the index for persistence
    - Update or remove texts from the index

    Attributes:
        embedding_model: The model used to generate text embeddings
        dimension: The dimension of the embeddings
        index: The FAISS index for similarity search
        metadata_store: Storage for text metadata
    """

    def __init__(
        self,
        embedding_model: BaseTextEmbeddingModel,
        index_type: str = "IndexFlatL2",
        normalize_embeddings: bool = False,
        nlist: int = 100,
        M: int = 32,
    ):
        """Initialize the text vector store.

        Args:
            embedding_model: Model to generate text embeddings
            index_type: Type of FAISS index to use. Options:
                - "IndexFlatL2": Exact search using L2 distance
                - "IndexFlatIP": Exact search using inner product
                - "IndexIVFFlat": Approximate search with inverted file index
                - "IndexHNSWFlat": Approximate search with HNSW graph
            normalize_embeddings: Whether to normalize embeddings before indexing
            nlist: Number of clusters for IVF index
            M: Number of bi-directional links for HNSW index
        """
        super().__init__(
            embedding_model=embedding_model,
            index_type=index_type,
            normalize_embeddings=normalize_embeddings,
            nlist=nlist,
            M=M,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TextVectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory path containing the saved index

        Returns:
            Loaded TextVectorStore instance

        Raises:
            FileNotFoundError: If the saved files don't exist
            ValueError: If model compatibility validation fails
        """
        return cls._load(path, create_text_embedding_model)


def build_text_vector_store(
    text_source: str | Path | list[str | Path] | None = None,
    image_source: (
        str | Path | PILImage.Image | list[str | PILImage.Image] | None
    ) = None,
    embedding_model: BaseTextEmbeddingModel | None = None,
    index_type: str = "IndexFlatL2",
    normalize_embeddings: bool = False,
    metadata_extractor: Callable[[str | Path], dict[str, Any]] | None = None,
    text_extensions: tuple[str, ...] | None = (
        ".txt",
        ".md",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".json",
        ".xml",
        ".csv",
    ),
    image_extensions: tuple[str, ...] | None = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".gif",
        ".webp",
        ".svg",
    ),
    recursive: bool = True,
    caption_prompt: str = "Describe this image in detail.",
    system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
    vlm_backend: str = "nim",
    vlm_model: str | None = None,
    vlm_api_key: str | None = None,
) -> TextVectorStore:
    """Build a text vector store from texts, images, or both.

    This method provides a convenient way to create and populate a
    vector store from various sources:
    - Text strings or file paths
    - Image file paths or PIL Image objects
    - Mixed sources

    Args:
        embedding_model: Model to generate embeddings.
        text_source: Source of texts - can be:
            - str: Text string or file path
            - list: List of text strings or file paths
            - None: No text sources
        image_source: Source of images - can be:
            - str: Image file path
            - PILImage.Image: PIL Image object
            - list: List of image paths or PIL Images
            - None: No image sources
        embedding_model: Text embedding model to generate embeddings. If None, uses
            default NIM embedding model.
        index_type: Type of FAISS index (default: "IndexFlatL2")
        normalize_embeddings: Whether to normalize embeddings (default: False)
        metadata_extractor: Optional function to extract metadata from
            file paths. Should accept a path and return a dict.
        text_extensions: Tuple of valid text extensions (with dots).
        image_extensions: Tuple of valid image extensions (with dots).
        recursive: If True and sources are directories, scan recursively
            (default: True)
        caption_prompt: Prompt to use for image captioning
        system_prompt: System instructions for the VLM
        vlm_backend: VLM backend to use ("azure_openai" or "nim")
        vlm_model: Model to use (uses backend default if None)
        vlm_api_key: API key for the VLM backend (uses env var if None)

    Returns:
        TextVectorStore populated with the provided texts and image captions

    Raises:
        ValueError: If sources are invalid or no content found
        FileNotFoundError: If directories don't exist

    Example:
        >>> # From text directory
        >>> store = build_text_vector_store(text_source="path/to/texts/")
        >>>
        >>> # From list of texts
        >>> store = build_text_vector_store(text_source=[
        ...     "This is the first text.",
        ...     "This is the second text.",
        ...     "This is the third text."
        ... ])
        >>>
        >>> # From dictionary
        >>> store = build_text_vector_store(text_source={
        ...     "doc1": "First document content",
        ...     "doc2": "Second document content"
        ... })
        >>>
        >>> # From images (added directly to vector store)
        >>> store = build_text_vector_store(
        ...     image_source="path/to/images/"
        ... )
        >>>
        >>> # Mixed text and image sources
        >>> store = build_text_vector_store(
        ...     text_source="path/to/texts/",
        ...     image_source="path/to/images/"
        ... )
        >>>
        >>> # With custom metadata
        >>> def extract_metadata(path):
        ...     return {"filename": Path(path).name}
        >>> store = build_text_vector_store(
        ...     text_source="texts/",
        ...     metadata_extractor=extract_metadata
        ... )
    """
    # Create default embedding model if not provided
    if embedding_model is None:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable must be set when "
                "embedding_model is not provided"
            )
        embedding_model = create_text_embedding_model(backend="nim", api_key=api_key)
        if not embedding_model:
            raise ValueError(
                "Failed to create embedding model. "
                "Please check your API key and try again."
            )

    # Create the vector store
    return TextVectorStore.build_vector_store(
        embedding_model=embedding_model,
        text_source=text_source,
        image_source=image_source,
        index_type=index_type,
        normalize_embeddings=normalize_embeddings,
        metadata_extractor=metadata_extractor,
        text_extensions=text_extensions,
        image_extensions=image_extensions,
        recursive=recursive,
        image_embedding_type="text",
        caption_prompt=caption_prompt,
        system_prompt=system_prompt,
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
    )


def find_similar_texts_from_vector_store(
    query: str | np.ndarray | str | Path | PILImage.Image,
    query_type: str,
    store: str | Path | TextVectorStore,
    k: int = 5,
    filter_metadata: dict[str, Any] | None = None,
) -> list[BaseSearchResult]:
    """Find similar texts from a vector store.

    This function searches for texts similar to the query using vector
    similarity. It can work with either a saved vector store (by path) or
    an existing TextVectorStore instance.

    Args:
        query: Query to find similar texts for. Can be:
            - str: Text string to search for (when query_type="text")
            - np.ndarray: Pre-computed embedding vector (when query_type="embedding")
            - str/Path: Path to an image file (when query_type="image")
            - PILImage.Image: PIL Image object (when query_type="image")
        query_type: Type of query - "text", "image", or "embedding"
        store: Vector store to search in. Can be:
            - str/Path: Path to a saved vector store directory
            - TextVectorStore: An existing vector store instance
        k: Number of similar texts to return (default: 5)
        filter_metadata: Optional metadata filters for exact matching.
            Only texts with matching metadata will be returned.

    Returns:
        List of SearchResult objects ordered by similarity (most similar first)

    Raises:
        FileNotFoundError: If the vector store path doesn't exist
        ValueError: If embedding model not provided and NVIDIA_API_KEY not set
            (only when loading from path), or if query_type is invalid

    Example:
        >>> # Find similar texts using a saved store path
        >>> results = find_similar_texts_from_vector_store(
        ...     query="machine learning algorithms",
        ...     query_type="text",
        ...     store="my_store/",
        ...     k=10
        ... )
        >>>
        >>> # Using an existing store instance
        >>> store = build_text_vector_store(source="texts/")
        >>> results = find_similar_texts_from_vector_store(
        ...     query="machine learning algorithms",
        ...     query_type="text",
        ...     store=store,  # Pass the store directly
        ...     k=10
        ... )
        >>>
        >>> # Search using an image
        >>> results = find_similar_texts_from_vector_store(
        ...     query="path/to/image.jpg",
        ...     query_type="image",
        ...     store="my_store/",
        ...     k=10
        ... )
        >>>
        >>> # Filter by metadata
        >>> results = find_similar_texts_from_vector_store(
        ...     query="machine learning algorithms",
        ...     query_type="text",
        ...     store="my_store/",
        ...     filter_metadata={"category": "technical"}
        ... )
    """
    if isinstance(store, str | Path):
        vector_store = TextVectorStore.load(store)
    else:
        vector_store = store
    return vector_store.find_similar_documents(query, query_type, k, filter_metadata)
