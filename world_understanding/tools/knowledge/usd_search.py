# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tool wrapper for USD asset search functionality.

This module provides tool interface for searching USD assets
using the USD search API.
"""

import time
from typing import Any

from world_understanding.functions.knowledge.usd_search import USDSearchClient
from world_understanding.tools.base import ToolInput, ToolOutput, register_tool


# Input/Output Models for USD Search
class USDSearchInput(ToolInput):
    """Input for USD asset search with configuration options."""

    query: str  # Search query string
    limit: int = 10  # Maximum number of results
    file_extension_include: str | list[str] | None = None  # File extension filter
    api_host: str | None = None  # Custom API host URL (optional)
    return_metadata: bool = True  # Whether to include metadata
    return_images: bool = True  # Whether to include images


class USDSearchOutput(ToolOutput):
    """Output for USD asset search."""

    success: bool  # Whether the search was successful
    results: list[dict[str, Any]]  # List of search results
    num_results: int  # Number of results returned
    query: str  # The original query
    api_host: str | None = None  # API host used
    file_extensions: list[str] | None = None  # File extensions filtered
    processing_time_ms: float | None = None  # Search processing time
    errors: list[str] = []  # Any errors during search


@register_tool(
    name="usd_search",
    version="0.1.0",
    tags=["knowledge", "usd", "assets", "search", "3d"],
    description="Search for USD assets using semantic search capabilities",
    input_model=USDSearchInput,
    output_model=USDSearchOutput,
)
def usd_search_tool(inputs: USDSearchInput) -> USDSearchOutput:
    """
    Search for USD assets using the USD search API.

    This tool provides semantic search capabilities for USD assets,
    allowing you to find assets based on descriptions, properties,
    or visual similarity. It supports custom API endpoints for different
    USD asset databases or environments.

    Args:
        inputs: USDSearchInput containing:
            - query: Search query string (e.g., "metallic paint", "wood texture")
            - limit: Maximum number of results to return (default: 10)
            - file_extension_include: Filter by file extensions (e.g., ["mdl", "usd"])
            - api_host: Custom API host URL (optional, uses default if not provided)
            - return_metadata: Whether to include metadata in results (default: True)
            - return_images: Whether to include images in results (default: True)

    Returns:
        USDSearchOutput containing:
            - success: Whether the search was successful
            - results: List of search results with metadata and scores
            - num_results: Number of results returned
            - query: The original query
            - api_host: API host that was used
            - file_extensions: File extensions that were filtered (if any)
            - processing_time_ms: Search processing time in milliseconds
            - errors: Any errors encountered

    Example:
        Search for metallic assets:
        >>> inputs = USDSearchInput(
        ...     query="metallic paint",
        ...     limit=5,
        ...     file_extension_include=["mdl"]
        ... )
        >>> output = usd_search_tool(inputs)
        >>> for result in output.results:
        ...     print(f"Asset: {result.get('source', {}).get('name')}")

        Search with custom API host:
        >>> inputs = USDSearchInput(
        ...     query="wood texture",
        ...     api_host="http://custom-api.example.com",
        ...     limit=3
        ... )
        >>> output = usd_search_tool(inputs)

    Note:
        This tool uses the USD search API to perform semantic searches
        across a database of USD assets. The API supports both text
        and visual queries with vector embeddings. If no api_host is
        specified, it uses the default USD_SEARCH_API_HOST configured
        in the usd_search module.
    """
    errors: list[str] = []
    start_time = time.time()

    try:
        # Create client with custom host if provided
        client = USDSearchClient(host=inputs.api_host)

        # Perform the search
        results = client.search(
            query=inputs.query,
            limit=inputs.limit,
            return_metadata=inputs.return_metadata,
            return_images=inputs.return_images,
            file_extension_include=inputs.file_extension_include,
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Process file extensions
        file_extensions = None
        if inputs.file_extension_include:
            if isinstance(inputs.file_extension_include, list):
                file_extensions = inputs.file_extension_include
            else:
                file_extensions = [inputs.file_extension_include]

        # Create successful output
        output = USDSearchOutput(
            success=True,
            results=results,
            num_results=len(results),
            query=inputs.query,
            api_host=inputs.api_host,
            file_extensions=file_extensions,
            processing_time_ms=processing_time_ms,
            errors=errors,
        )

    except ImportError as e:
        output = USDSearchOutput(
            success=False,
            results=[],
            num_results=0,
            query=inputs.query,
            api_host=inputs.api_host,
            errors=[
                f"Import error: {str(e)}. Make sure usd_search_client is installed."
            ],
        )

    except ConnectionError as e:
        output = USDSearchOutput(
            success=False,
            results=[],
            num_results=0,
            query=inputs.query,
            api_host=inputs.api_host,
            errors=[f"Connection error: {str(e)}"],
        )

    except ValueError as e:
        output = USDSearchOutput(
            success=False,
            results=[],
            num_results=0,
            query=inputs.query,
            api_host=inputs.api_host,
            errors=[f"Invalid input: {str(e)}"],
        )

    except Exception as e:
        output = USDSearchOutput(
            success=False,
            results=[],
            num_results=0,
            query=inputs.query,
            api_host=inputs.api_host,
            errors=[f"Unexpected error: {str(e)}"],
        )

    return output
