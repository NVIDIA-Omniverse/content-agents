# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image vector store for similarity search using FAISS.

This module provides a vector store implementation for image-metadata pairs,
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
    BaseDocument,
    BaseSearchResult,
    BaseVectorStore,
)
from world_understanding.functions.models.image_embedding_models import (
    BaseImageEmbeddingModel,
    create_image_embedding_model,
)

# Configure logger
logger = logging.getLogger(__name__)


class ImageVectorStore(BaseVectorStore):
    """Vector store for image embeddings using FAISS.

    This class provides functionality to:
    - Add images with metadata to a vector index
    - Search for similar images using vector similarity
    - Save and load the index for persistence
    - Update or remove images from the index

    Attributes:
        embedding_model: The model used to generate image embeddings
        dimension: The dimension of the embeddings
        index: The FAISS index for similarity search
        metadata_store: Storage for image metadata
    """

    def __init__(
        self,
        embedding_model: BaseImageEmbeddingModel,
        index_type: str = "IndexFlatL2",
        normalize_embeddings: bool = False,
        nlist: int = 100,
        M: int = 32,
    ):
        """Initialize the image vector store.

        Args:
            embedding_model: Model to generate image embeddings
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

    def add_document(
        self, document: BaseDocument, embedding_type: str = "image"
    ) -> int:
        """Add a document to the vector store.

        Args:
            document: Document to add
            embedding_type: Type of embedding to generate ("text" or "image")
        """
        return super().add_document(document, embedding_type)

    def add_text(
        self,
        text: str | Path,
        text_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        raise NotImplementedError("This model doesn't support text embedding")

    def add_texts(
        self,
        texts: list[str | Path],
        text_ids: list[str] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        raise NotImplementedError("This model doesn't support text embedding")

    @classmethod
    def load(cls, path: str | Path) -> "ImageVectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory path containing the saved index

        Returns:
            Loaded ImageVectorStore instance

        Raises:
            FileNotFoundError: If the saved files don't exist
            ValueError: If model compatibility validation fails
        """
        return cls._load(path, create_image_embedding_model)


def build_image_vector_store(
    source: str | Path | PILImage.Image | list[str | Path | PILImage.Image],
    embedding_model: BaseImageEmbeddingModel | None = None,
    index_type: str = "IndexFlatL2",
    normalize_embeddings: bool = False,
    metadata_extractor: Callable[[str | Path], dict[str, Any]] | None = None,
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
) -> ImageVectorStore:
    """Build an image vector store from images or directory.

    This function provides a convenient way to create and populate an
    ImageVectorStore from various sources:
    - A single directory path containing image files
    - A list of image file paths
    - A list of PIL Image objects
    - A mixed list of paths and PIL Image objects

    Args:
        source: Source of images - can be:
            - str: Image file path
            - Path: Image file path
            - PILImage.Image: PIL Image object
            - list: List of image paths or PIL Images
        embedding_model: Model to generate embeddings. If None, uses
            default NIM embedding model.
        index_type: Type of FAISS index (default: "IndexFlatL2")
        normalize_embeddings: Whether to normalize embeddings (default: False)
        metadata_extractor: Optional function to extract metadata from
            image paths. Should accept a path and return a dict.
        image_extensions: Tuple of valid image extensions (with dots).
        recursive: If True and source is a directory, scan recursively
            (default: True)

    Returns:
        ImageVectorStore populated with the provided images

    Raises:
        ValueError: If source is invalid or no images found
        FileNotFoundError: If directory doesn't exist

    Example:
        >>> # From directory
        >>> store = build_image_vector_store("path/to/images/")
        >>>
        >>> # From list of paths
        >>> store = build_image_vector_store([
        ...     "image1.jpg",
        ...     "image2.png",
        ...     PILImage.open("image3.jpg")
        ... ])
        >>>
        >>> # With custom metadata
        >>> def extract_metadata(path):
        ...     return {"filename": Path(path).name}
        >>> store = build_image_vector_store(
        ...     "images/",
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
        embedding_model = create_image_embedding_model(backend="nim", api_key=api_key)
        if not embedding_model:
            raise ValueError(
                "Failed to create embedding model. "
                "Please check your API key and try again."
            )

    return ImageVectorStore.build_vector_store(
        embedding_model=embedding_model,
        image_source=source,
        index_type=index_type,
        normalize_embeddings=normalize_embeddings,
        metadata_extractor=metadata_extractor,
        image_extensions=image_extensions,
        recursive=recursive,
    )


def find_similar_images_from_vector_store(
    query_image: str | Path | PILImage.Image | np.ndarray,
    store: str | Path | ImageVectorStore,
    k: int = 5,
    filter_metadata: dict[str, Any] | None = None,
) -> list[BaseSearchResult]:
    """Find similar images from a vector store.

    This function searches for images similar to the query image using vector
    similarity. It can work with either a saved vector store (by path) or
    an existing ImageVectorStore instance.

    Args:
        query_image: Query image to find similar images for. Can be:
            - str/Path: Path to an image file
            - PILImage.Image: PIL Image object
            - np.ndarray: Pre-computed embedding vector
        store: Vector store to search in. Can be:
            - str/Path: Path to a saved vector store directory
            - ImageVectorStore: An existing vector store instance
        k: Number of similar images to return (default: 5)
        filter_metadata: Optional metadata filters for exact matching.
            Only images with matching metadata will be returned.

    Returns:
        List of SearchResult objects ordered by similarity (most similar first)

    Raises:
        FileNotFoundError: If the vector store path doesn't exist
        ValueError: If embedding model not provided and NVIDIA_API_KEY not set
            (only when loading from path)

        Example:
        >>> # Find similar images using a saved store path
        >>> results = find_similar_images_from_vector_store(
        ...     query_image="query.jpg",
        ...     store="my_store/",
        ...     k=10
        ... )
        >>>
        >>> # Using an existing store instance
        >>> store = build_image_vector_store("images/")
        >>> results = find_similar_images_from_vector_store(
        ...     query_image="query.jpg",
        ...     store=store,  # Pass the store directly
        ...     k=10
        ... )
        >>>
        >>> # Filter by metadata
        >>> results = find_similar_images_from_vector_store(
        ...     query_image="query.jpg",
        ...     store="my_store/",
        ...     filter_metadata={"category": "landscape"}
        ... )
    """
    if isinstance(store, str | Path):
        vector_store = ImageVectorStore.load(store)
    else:
        vector_store = store
    return vector_store.find_similar_documents(
        query=query_image,
        query_type="image",
        k=k,
        filter_metadata=filter_metadata,
    )
