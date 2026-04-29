# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tool wrappers for text vector store functionality.

This module provides tool interfaces for building and searching text vector stores
using FAISS for similarity search.
"""

from typing import Any

from world_understanding.functions.knowledge.text_vector_store import (
    build_text_vector_store as build_text_vector_store_func,
)
from world_understanding.functions.knowledge.text_vector_store import (
    find_similar_texts_from_vector_store as find_similar_texts_func,
)
from world_understanding.tools.base import ToolInput, ToolOutput, register_tool


# Input/Output Models for build_text_vector_store
class BuildTextVectorStoreInput(ToolInput):
    """Input for building a text vector store."""

    text_source: str | list[str] | None = None  # Text sources
    image_source: str | list[str] | None = None  # Image sources (will be captioned)
    index_type: str = "IndexFlatL2"  # Type of FAISS index to use
    normalize_embeddings: bool = False  # Whether to normalize embeddings
    recursive: bool = True  # If True and source is directory, scan recursively
    save_path: str | None = None  # Optional path to save the vector store


class BuildTextVectorStoreOutput(ToolOutput):
    """Output for building a text vector store."""

    success: bool  # Whether the store was built successfully
    num_documents_indexed: int  # Number of documents added to the store
    num_texts: int  # Number of text documents
    num_images_captioned: int  # Number of images that were captioned
    index_type: str  # Type of index used
    embedding_dimension: int  # Dimension of the embeddings
    save_path: str | None = None  # Path where store was saved (if applicable)
    errors: list[str] = []  # Any errors during processing


# Input/Output Models for find_similar_texts
class FindSimilarTextsInput(ToolInput):
    """Input for finding similar texts in a text vector store."""

    query: str  # Query text or path to image
    query_type: str  # Type of query: "text", "image", or "embedding"
    store_path: str  # Path to saved vector store
    k: int = 5  # Number of similar texts to return
    filter_metadata: dict[str, Any] | None = None  # Optional metadata filters


class FindSimilarTextsOutput(ToolOutput):
    """Output for finding similar texts."""

    results: list[dict[str, Any]]  # List of similar texts with scores
    num_results: int  # Number of results returned
    query: str  # The original query
    query_type: str  # Type of query used
    search_errors: list[str] = []  # Any errors during search


# Tool: Build Text Vector Store
@register_tool(
    name="build_text_vector_store",
    version="0.1.0",
    tags=["knowledge", "vector-store", "text", "nlp", "faiss", "gpu"],
    description="Build a text vector store from text documents and/or images (via captioning)",
    input_model=BuildTextVectorStoreInput,
    output_model=BuildTextVectorStoreOutput,
)
def build_text_vector_store_tool(
    inputs: BuildTextVectorStoreInput,
) -> BuildTextVectorStoreOutput:
    """
    Build a text vector store from text sources and/or images.

    This tool creates a FAISS-based vector store for text similarity search.
    Images are automatically captioned and stored as text, making this ideal
    for unified text-based search across different content types.

    Args:
        inputs: BuildTextVectorStoreInput containing:
            - text_source: Text sources (files, directories, or strings)
            - image_source: Image sources (will be captioned and stored as text)
            - index_type: Type of FAISS index (default: "IndexFlatL2")
            - normalize_embeddings: Whether to normalize embeddings
            - recursive: Whether to scan directories recursively
            - save_path: Optional path to save the vector store

    Returns:
        BuildTextVectorStoreOutput containing:
            - success: Whether the store was built successfully
            - num_documents_indexed: Total number of documents
            - num_texts: Number of text documents
            - num_images_captioned: Number of images captioned
            - index_type: Type of index used
            - embedding_dimension: Dimension of embeddings
            - save_path: Where store was saved (if applicable)
            - errors: Any errors encountered

    Note:
        Requires NVIDIA_API_KEY environment variable to be set for embeddings.
        Images are captioned using a VLM before being added to the text store.
    """
    errors: list[str] = []

    try:
        # Convert sources to appropriate format
        text_source = inputs.text_source
        if isinstance(text_source, list):
            text_source_for_func = text_source
        elif text_source:
            text_source_for_func = text_source
        else:
            text_source_for_func = None

        image_source = inputs.image_source
        if isinstance(image_source, list):
            image_source_for_func = image_source
        elif image_source:
            image_source_for_func = image_source
        else:
            image_source_for_func = None

        # Build the text vector store
        vector_store = build_text_vector_store_func(
            text_source=text_source_for_func,
            image_source=image_source_for_func,
            index_type=inputs.index_type,
            normalize_embeddings=inputs.normalize_embeddings,
            recursive=inputs.recursive,
        )

        # Count documents by type
        num_texts = 0
        num_images_captioned = 0
        for _doc_id, metadata in vector_store.metadata_store.items():
            # Check if this was originally an image that got captioned
            if metadata.document.image_path:
                num_images_captioned += 1
            else:
                num_texts += 1

        num_documents = len(vector_store.metadata_store)
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
        output = BuildTextVectorStoreOutput(
            success=True,
            num_documents_indexed=num_documents,
            num_texts=num_texts,
            num_images_captioned=num_images_captioned,
            index_type=inputs.index_type,
            embedding_dimension=embedding_dim,
            save_path=saved_path,
            errors=errors,
        )

    except FileNotFoundError as e:
        output = BuildTextVectorStoreOutput(
            success=False,
            num_documents_indexed=0,
            num_texts=0,
            num_images_captioned=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"File not found: {str(e)}"],
        )

    except ValueError as e:
        output = BuildTextVectorStoreOutput(
            success=False,
            num_documents_indexed=0,
            num_texts=0,
            num_images_captioned=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"Invalid input: {str(e)}"],
        )

    except Exception as e:
        output = BuildTextVectorStoreOutput(
            success=False,
            num_documents_indexed=0,
            num_texts=0,
            num_images_captioned=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"Unexpected error: {str(e)}"],
        )

    return output


# Tool: Find Similar Texts
@register_tool(
    name="find_similar_texts",
    version="0.1.0",
    tags=["knowledge", "vector-store", "search", "text", "nlp", "gpu"],
    description="Find similar texts in a text vector store",
    input_model=FindSimilarTextsInput,
    output_model=FindSimilarTextsOutput,
)
def find_similar_texts_tool(
    inputs: FindSimilarTextsInput,
) -> FindSimilarTextsOutput:
    """
    Find similar texts from a text vector store.

    This tool searches for texts similar to a query using vector
    similarity in a pre-built text FAISS index. It can handle
    text queries, image queries (via captioning), or pre-computed embeddings.

    Args:
        inputs: FindSimilarTextsInput containing:
            - query: Query text or path to image
            - query_type: Type of query ("text", "image", or "embedding")
            - store_path: Path to the saved vector store
            - k: Number of similar texts to return
            - filter_metadata: Optional metadata filters

    Returns:
        FindSimilarTextsOutput containing:
            - results: List of similar texts with scores
            - num_results: Number of results returned
            - query: The original query
            - query_type: Type of query used
            - search_errors: Any errors encountered

    Note:
        The vector store must be previously built and saved using
        build_text_vector_store_tool. Image queries are automatically
        captioned before searching.
    """
    search_errors: list[str] = []

    try:
        # Find similar texts
        search_results = find_similar_texts_func(
            query=inputs.query,
            query_type=inputs.query_type,
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
                if result.document.text_content:
                    result_dict["text_content"] = result.document.text_content

                # Add source information
                if result.document.text_path:
                    result_dict["source_path"] = str(result.document.text_path)
                elif result.document.image_path:
                    result_dict["source_image"] = str(result.document.image_path)
                    result_dict["is_captioned"] = True

            results.append(result_dict)

        # Create output
        output = FindSimilarTextsOutput(
            results=results,
            num_results=len(results),
            query=inputs.query,
            query_type=inputs.query_type,
            search_errors=search_errors,
        )

    except FileNotFoundError as e:
        output = FindSimilarTextsOutput(
            results=[],
            num_results=0,
            query=inputs.query,
            query_type=inputs.query_type,
            search_errors=[f"File not found: {str(e)}"],
        )

    except ValueError as e:
        output = FindSimilarTextsOutput(
            results=[],
            num_results=0,
            query=inputs.query,
            query_type=inputs.query_type,
            search_errors=[f"Invalid input: {str(e)}"],
        )

    except Exception as e:
        output = FindSimilarTextsOutput(
            results=[],
            num_results=0,
            query=inputs.query,
            query_type=inputs.query_type,
            search_errors=[f"Unexpected error: {str(e)}"],
        )

    return output
