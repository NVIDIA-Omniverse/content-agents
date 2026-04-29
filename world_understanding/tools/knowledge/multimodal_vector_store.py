# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tool wrappers for multimodal vector store functionality.

This module provides tool interfaces for building and searching multimodal vector stores
that can handle both text and image content using unified embeddings.
"""

from typing import Any

from world_understanding.functions.knowledge.multimodal_vector_store import (
    build_multimodal_vector_store as build_multimodal_vector_store_func,
)
from world_understanding.functions.knowledge.multimodal_vector_store import (
    find_similar_documents_from_vector_store as find_similar_documents_func,
)
from world_understanding.tools.base import ToolInput, ToolOutput, register_tool


# Input/Output Models for build_multimodal_vector_store
class BuildMultimodalVectorStoreInput(ToolInput):
    """Input for building a multimodal vector store."""

    text_source: str | list[str] | None = None  # Text sources
    image_source: str | list[str] | None = None  # Image sources
    index_type: str = "IndexFlatL2"  # Type of FAISS index to use
    normalize_embeddings: bool = False  # Whether to normalize embeddings
    recursive: bool = True  # If True and source is directory, scan recursively
    image_embedding_type: str = (
        "image"  # How to handle images: "text" (caption) or "image" (direct)
    )
    save_path: str | None = None  # Optional path to save the vector store
    include_filename_metadata: bool = (
        False  # If True, include filename and path info in metadata
    )


class BuildMultimodalVectorStoreOutput(ToolOutput):
    """Output for building a multimodal vector store."""

    success: bool  # Whether the store was built successfully
    num_documents_indexed: int  # Number of documents added to the store
    num_texts: int  # Number of text documents
    num_images: int  # Number of image documents
    num_multimodal: int  # Number of multimodal documents
    index_type: str  # Type of index used
    embedding_dimension: int  # Dimension of the embeddings
    save_path: str | None = None  # Path where store was saved (if applicable)
    errors: list[str] = []  # Any errors during processing


# Input/Output Models for find_similar_documents
class FindSimilarDocumentsInput(ToolInput):
    """Input for finding similar documents in a multimodal store."""

    query: str  # Query text or path to image
    query_type: str  # Type of query: "text", "image", or "embedding"
    store_path: str  # Path to saved vector store
    k: int = 5  # Number of similar documents to return
    filter_metadata: dict[str, Any] | None = None  # Optional metadata filters
    embedding_type: str = (
        "image"  # For image queries: "text" (caption) or "image" (direct)
    )


class FindSimilarDocumentsOutput(ToolOutput):
    """Output for finding similar documents."""

    results: list[dict[str, Any]]  # List of similar documents with scores
    num_results: int  # Number of results returned
    query: str  # The original query
    query_type: str  # Type of query used
    search_errors: list[str] = []  # Any errors during search


# Tool: Build Multimodal Vector Store
@register_tool(
    name="build_multimodal_vector_store",
    version="0.1.0",
    tags=["knowledge", "vector-store", "multimodal", "text", "images", "faiss", "gpu"],
    description="Build a vector store from text and/or images for unified similarity search",
    input_model=BuildMultimodalVectorStoreInput,
    output_model=BuildMultimodalVectorStoreOutput,
)
def build_multimodal_vector_store_tool(
    inputs: BuildMultimodalVectorStoreInput,
) -> BuildMultimodalVectorStoreOutput:
    """
    Build a multimodal vector store from text and/or image sources.

    This tool creates a FAISS-based vector store that can handle both text
    and image content using unified multimodal embeddings. It's useful for
    building search systems that work across different content types.

    Args:
        inputs: BuildMultimodalVectorStoreInput containing:
            - text_source: Text sources (files, directories, or strings)
            - image_source: Image sources (files or directories)
            - index_type: Type of FAISS index (default: "IndexFlatL2")
            - normalize_embeddings: Whether to normalize embeddings
            - recursive: Whether to scan directories recursively
            - image_embedding_type: How to embed images ("text" or "image")
            - save_path: Optional path to save the vector store
            - include_filename_metadata: Whether to include filename info in metadata

    Returns:
        BuildMultimodalVectorStoreOutput containing:
            - success: Whether the store was built successfully
            - num_documents_indexed: Total number of documents
            - num_texts: Number of text documents
            - num_images: Number of image documents
            - num_multimodal: Number of multimodal documents
            - index_type: Type of index used
            - embedding_dimension: Dimension of embeddings
            - save_path: Where store was saved (if applicable)
            - errors: Any errors encountered

    Note:
        Requires NVIDIA_API_KEY environment variable to be set for embeddings.
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

        # Create metadata extractor if requested
        metadata_extractor = None
        if inputs.include_filename_metadata:
            from pathlib import Path

            def extract_filename_metadata(file_path: str | Path) -> dict[str, Any]:
                """Extract filename and path metadata from a file."""
                path = Path(file_path)
                metadata = {
                    "filename": path.name,
                    "file_stem": path.stem,
                    "extension": path.suffix,
                    "parent_dir": path.parent.name,
                    "relative_path": str(path),
                }

                # Extract additional info from filename pattern if available
                # Example: 080-2418-000_1_0001_structured.txt
                if "_" in path.stem:
                    parts = path.stem.split("_")
                    if len(parts) > 1:
                        metadata["document_id"] = parts[0]
                    if len(parts) > 2:
                        metadata["page_or_section"] = parts[2]
                    if len(parts) > 3:
                        metadata["content_type"] = parts[3]

                # Determine content type flags
                if "structured" in path.name:
                    metadata["is_structured"] = True
                elif "text" in path.name:
                    metadata["is_text"] = True

                return metadata

            metadata_extractor = extract_filename_metadata

        # Build the vector store
        vector_store = build_multimodal_vector_store_func(
            text_source=text_source_for_func,
            image_source=image_source_for_func,
            index_type=inputs.index_type,
            normalize_embeddings=inputs.normalize_embeddings,
            recursive=inputs.recursive,
            image_embedding_type=inputs.image_embedding_type,
            metadata_extractor=metadata_extractor,
        )

        # Count documents by type
        num_texts = 0
        num_images = 0
        num_multimodal = 0
        for _doc_id, metadata in vector_store.metadata_store.items():
            if metadata.document.get_content_type() == "text":
                num_texts += 1
            elif metadata.document.get_content_type() == "image":
                num_images += 1
            elif metadata.document.get_content_type() == "multimodal":
                num_multimodal += 1

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
        output = BuildMultimodalVectorStoreOutput(
            success=True,
            num_documents_indexed=num_documents,
            num_texts=num_texts,
            num_images=num_images,
            num_multimodal=num_multimodal,
            index_type=inputs.index_type,
            embedding_dimension=embedding_dim,
            save_path=saved_path,
            errors=errors,
        )

    except FileNotFoundError as e:
        output = BuildMultimodalVectorStoreOutput(
            success=False,
            num_documents_indexed=0,
            num_texts=0,
            num_images=0,
            num_multimodal=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"File not found: {str(e)}"],
        )

    except ValueError as e:
        output = BuildMultimodalVectorStoreOutput(
            success=False,
            num_documents_indexed=0,
            num_texts=0,
            num_images=0,
            num_multimodal=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"Invalid input: {str(e)}"],
        )

    except Exception as e:
        output = BuildMultimodalVectorStoreOutput(
            success=False,
            num_documents_indexed=0,
            num_texts=0,
            num_images=0,
            num_multimodal=0,
            index_type=inputs.index_type,
            embedding_dimension=0,
            errors=[f"Unexpected error: {str(e)}"],
        )

    return output


# Tool: Find Similar Documents
@register_tool(
    name="find_similar_documents",
    version="0.1.0",
    tags=["knowledge", "vector-store", "search", "multimodal", "gpu"],
    description="Find similar documents in a multimodal vector store",
    input_model=FindSimilarDocumentsInput,
    output_model=FindSimilarDocumentsOutput,
)
def find_similar_documents_tool(
    inputs: FindSimilarDocumentsInput,
) -> FindSimilarDocumentsOutput:
    """
    Find similar documents from a multimodal vector store.

    This tool searches for documents similar to a query using vector
    similarity in a pre-built multimodal FAISS index. It can handle
    both text and image queries.

    Args:
        inputs: FindSimilarDocumentsInput containing:
            - query: Query text or path to image
            - query_type: Type of query ("text", "image", or "embedding")
            - store_path: Path to the saved vector store
            - k: Number of similar documents to return
            - filter_metadata: Optional metadata filters
            - embedding_type: For images, how to embed ("text" or "image")

    Returns:
        FindSimilarDocumentsOutput containing:
            - results: List of similar documents with scores
            - num_results: Number of results returned
            - query: The original query
            - query_type: Type of query used
            - search_errors: Any errors encountered

    Note:
        The vector store must be previously built and saved using
        build_multimodal_vector_store_tool.
    """
    search_errors: list[str] = []

    try:
        # Find similar documents
        search_results = find_similar_documents_func(
            query=inputs.query,
            query_type=inputs.query_type,
            store=inputs.store_path,
            k=inputs.k,
            filter_metadata=inputs.filter_metadata,
            embedding_type=inputs.embedding_type,
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
                if result.document.image_path:
                    result_dict["image_path"] = str(result.document.image_path)

                # Add content type from metadata
                content_type = result.document.get_content_type()
                result_dict["content_type"] = content_type

            results.append(result_dict)

        # Create output
        output = FindSimilarDocumentsOutput(
            results=results,
            num_results=len(results),
            query=inputs.query,
            query_type=inputs.query_type,
            search_errors=search_errors,
        )

    except FileNotFoundError as e:
        output = FindSimilarDocumentsOutput(
            results=[],
            num_results=0,
            query=inputs.query,
            query_type=inputs.query_type,
            search_errors=[f"File not found: {str(e)}"],
        )

    except ValueError as e:
        output = FindSimilarDocumentsOutput(
            results=[],
            num_results=0,
            query=inputs.query,
            query_type=inputs.query_type,
            search_errors=[f"Invalid input: {str(e)}"],
        )

    except Exception as e:
        output = FindSimilarDocumentsOutput(
            results=[],
            num_results=0,
            query=inputs.query,
            query_type=inputs.query_type,
            search_errors=[f"Unexpected error: {str(e)}"],
        )

    return output
