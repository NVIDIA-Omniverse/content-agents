# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tool wrappers for image vector store functionality.

This module provides tool interfaces for building and searching image vector stores
using FAISS for similarity search.
"""

from typing import Any

from world_understanding.functions.knowledge.image_vector_store import (
    build_image_vector_store as build_image_vector_store_func,
)
from world_understanding.functions.knowledge.image_vector_store import (
    find_similar_images_from_vector_store as find_similar_images_func,
)
from world_understanding.tools.base import ToolInput, ToolOutput, register_tool


# Input/Output Models for build_image_vector_store
class BuildImageVectorStoreInput(ToolInput):
    """Input for building an image vector store."""

    source: str | list[str]  # Path(s) to images or directory
    index_type: str = "IndexFlatL2"  # Type of FAISS index to use
    normalize_embeddings: bool = False  # Whether to normalize embeddings
    recursive: bool = True  # If True and source is directory, scan recursively
    save_path: str | None = None  # Optional path to save the vector store


class BuildImageVectorStoreOutput(ToolOutput):
    """Output for building an image vector store."""

    success: bool  # Whether the store was built successfully
    num_images_indexed: int  # Number of images added to the store
    index_type: str  # Type of index used
    embedding_dimension: int  # Dimension of the embeddings
    save_path: str | None = None  # Path where store was saved (if applicable)
    errors: list[str] = []  # Any errors during processing


# Input/Output Models for find_similar_images
class FindSimilarImagesInput(ToolInput):
    """Input for finding similar images."""

    query_image: str  # Path to query image
    store_path: str  # Path to saved vector store
    k: int = 5  # Number of similar images to return
    filter_metadata: dict[str, Any] | None = None  # Optional metadata filters


class FindSimilarImagesOutput(ToolOutput):
    """Output for finding similar images."""

    results: list[dict[str, Any]]  # List of similar images with scores
    num_results: int  # Number of results returned
    query_image: str  # Path to the query image
    search_errors: list[str] = []  # Any errors during search


# Tool: Build Image Vector Store
@register_tool(
    name="build_image_vector_store",
    version="0.1.0",
    tags=["knowledge", "vector-store", "images", "faiss", "gpu"],
    description="Build a vector store from images for similarity search",
    input_model=BuildImageVectorStoreInput,
    output_model=BuildImageVectorStoreOutput,
)
def build_image_vector_store_tool(
    inputs: BuildImageVectorStoreInput,
) -> BuildImageVectorStoreOutput:
    """
    Build an image vector store from images or directory.

    This tool creates a FAISS-based vector store for image similarity search.
    It processes images from various sources and builds an index for efficient
    similarity search.

    Args:
        inputs: BuildImageVectorStoreInput containing:
            - source: Path(s) to images or directory to scan
            - index_type: Type of FAISS index (default: "IndexFlatL2")
            - normalize_embeddings: Whether to normalize embeddings
            - recursive: Whether to scan directories recursively
            - save_path: Optional path to save the vector store

    Returns:
        BuildImageVectorStoreOutput containing:
            - success: Whether the store was built successfully
            - num_images_indexed: Number of images added
            - index_type: Type of index used
            - embedding_dimension: Dimension of embeddings
            - save_path: Where store was saved (if applicable)
            - errors: Any errors encountered

    Note:
        Requires NVIDIA_API_KEY environment variable to be set for embeddings.
    """
    errors: list[str] = []

    try:
        # Convert source to appropriate format
        source = inputs.source
        if isinstance(source, list):
            # List of image paths
            source_for_func = source
        else:
            # Single path (file or directory)
            source_for_func = source

        # Build the vector store
        vector_store = build_image_vector_store_func(
            source=source_for_func,
            index_type=inputs.index_type,
            normalize_embeddings=inputs.normalize_embeddings,
            recursive=inputs.recursive,
        )

        # Get store information
        num_images = len(vector_store.metadata_store)
        embedding_dim = vector_store.dimension

        # Save if path provided
        saved_path = None
        if inputs.save_path:
            try:
                vector_store.save(inputs.save_path)
                saved_path = inputs.save_path
            except Exception as e:
                errors.append(f"Failed to save vector store: {str(e)}")

        # Create output
        output = BuildImageVectorStoreOutput(
            success=True,
            num_images_indexed=num_images,
            index_type=inputs.index_type,
            embedding_dimension=embedding_dim,
            save_path=saved_path,
            errors=errors,
        )

    except FileNotFoundError as e:
        output = BuildImageVectorStoreOutput(
            success=False,
            num_images_indexed=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"File not found: {str(e)}"],
        )

    except ValueError as e:
        output = BuildImageVectorStoreOutput(
            success=False,
            num_images_indexed=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"Invalid input: {str(e)}"],
        )

    except Exception as e:
        output = BuildImageVectorStoreOutput(
            success=False,
            num_images_indexed=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"Unexpected error: {str(e)}"],
        )

    return output


# Tool: Find Similar Images
@register_tool(
    name="find_similar_images",
    version="0.1.0",
    tags=["knowledge", "vector-store", "search", "images", "gpu"],
    description="Find similar images using a vector store",
    input_model=FindSimilarImagesInput,
    output_model=FindSimilarImagesOutput,
)
def find_similar_images_tool(
    inputs: FindSimilarImagesInput,
) -> FindSimilarImagesOutput:
    """
    Find similar images from a vector store.

    This tool searches for images similar to a query image using vector
    similarity in a pre-built FAISS index.

    Args:
        inputs: FindSimilarImagesInput containing:
            - query_image: Path to the query image
            - store_path: Path to the saved vector store
            - k: Number of similar images to return
            - filter_metadata: Optional metadata filters

    Returns:
        FindSimilarImagesOutput containing:
            - results: List of similar images with scores
            - num_results: Number of results returned
            - query_image: Path to the query image
            - search_errors: Any errors encountered

    Note:
        The vector store must be previously built and saved using
        build_image_vector_store_tool.
    """
    search_errors: list[str] = []

    try:
        # Find similar images
        search_results = find_similar_images_func(
            query_image=inputs.query_image,
            store=inputs.store_path,
            k=inputs.k,
            filter_metadata=inputs.filter_metadata,
        )

        # Convert results to dictionary format
        results = []
        for result in search_results:
            result_dict = {
                "doc_id": getattr(result, "doc_id", "unknown"),
                "score": float(result.score),
                "metadata": getattr(result, "metadata", {}),
            }
            # Add document content if available
            if hasattr(result, "document") and result.document:
                if result.document.image_path:
                    result_dict["image_path"] = str(result.document.image_path)
                if result.document.text_content:
                    result_dict["text_content"] = result.document.text_content
            results.append(result_dict)

        # Create output
        output = FindSimilarImagesOutput(
            results=results,
            num_results=len(results),
            query_image=inputs.query_image,
            search_errors=search_errors,
        )

    except FileNotFoundError as e:
        output = FindSimilarImagesOutput(
            results=[],
            num_results=0,
            query_image=inputs.query_image,
            search_errors=[f"File not found: {str(e)}"],
        )

    except ValueError as e:
        output = FindSimilarImagesOutput(
            results=[],
            num_results=0,
            query_image=inputs.query_image,
            search_errors=[f"Invalid input: {str(e)}"],
        )

    except Exception as e:
        output = FindSimilarImagesOutput(
            results=[],
            num_results=0,
            query_image=inputs.query_image,
            search_errors=[f"Unexpected error: {str(e)}"],
        )

    return output
