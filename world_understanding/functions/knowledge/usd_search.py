# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD material search functionality using the USD search API."""

import asyncio
import os
from typing import Any

import usd_search_client
from usd_search_client import BasicSearchRequest
from usd_search_client.models.query import Query
from usd_search_client.models.vector_query import VectorQuery, VectorQueryType
from usd_search_client.rest import ApiException

# Global configuration for USD search API.
# No default — set USD_SEARCH_API_HOST env var to point at your search service.
USD_SEARCH_API_HOST = os.environ.get("USD_SEARCH_API_HOST", "")


class USDSearchClient:
    """Client for searching USD materials using the USD search API."""

    def __init__(self, host: str | None = None):
        """Initialize the USD search client.

        Args:
            host: The API host URL. If not provided, uses the default USD_SEARCH_API_HOST.
        """
        self.configuration = usd_search_client.Configuration(
            host=host or USD_SEARCH_API_HOST
        )

    async def _search_async(
        self,
        query: str,
        limit: int = 10,
        return_metadata: bool = True,
        return_images: bool = True,
        file_extension_include: str | list[str] | None = None,
    ) -> Any | None:
        """Internal async method to perform the search.

        Args:
            query: The search query string
            limit: Maximum number of results to return
            return_metadata: Whether to return metadata in results
            return_images: Whether to return images in results
            file_extension_include: File extension(s) to include - can be a string (e.g., "mdl")
                                   or list of strings (e.g., ["mdl", "usd"])

        Returns:
            The API response or None if an error occurred
        """
        async with usd_search_client.ApiClient(self.configuration) as api_client:
            # Build the basic search request
            request_args = {
                "hybrid_text_query": query,
                "vector_queries": [
                    VectorQuery(
                        field_name="clip-embedding.embedding",
                        query_type=VectorQueryType.TEXT,
                        query=Query(actual_instance=query),
                    )
                ],
                "return_metadata": return_metadata,
                "return_images": return_images,
                "limit": limit,
            }

            # Add file extension filter if provided
            # Convert list to comma-separated string if needed
            if file_extension_include:
                if isinstance(file_extension_include, list):
                    # Join list into comma-separated string
                    request_args["file_extension_include"] = ",".join(
                        file_extension_include
                    )
                else:
                    request_args["file_extension_include"] = file_extension_include

            request = BasicSearchRequest(**request_args)

            try:
                api_response = await usd_search_client.search_hybrid(
                    request, api_client=api_client
                )
                return api_response

            except ApiException as e:
                print(f"API Exception occurred: Status {e.status}, Reason: {e.reason}")
                if e.body:
                    print(f"Error Body: {e.body}")
                return None

            except Exception as e:
                print(f"Unexpected error occurred: {str(e)} (Type: {type(e).__name__})")
                return None

    async def search_async(
        self,
        query: str,
        limit: int = 10,
        return_metadata: bool = True,
        return_images: bool = True,
        file_extension_include: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Async search for USD materials using the given query.

        This is the async version for use in async contexts (e.g., FastAPI).

        Args:
            query: The search query string (e.g., "White Powder-Coated Steel")
            limit: Maximum number of results to return (default: 10)
            return_metadata: Whether to include metadata in results (default: True)
            return_images: Whether to include images in results (default: True)
            file_extension_include: File extension(s) to include - can be a string (e.g., "mdl")
                                   or list of strings (e.g., ["mdl", "usd"])

        Returns:
            A list of search result dictionaries, or empty list if no results

        Example:
            >>> client = USDSearchClient()
            >>> results = await client.search_async("metallic paint", limit=5)
        """
        api_response = await self._search_async(
            query, limit, return_metadata, return_images, file_extension_include
        )

        if api_response is None:
            return []

        # Extract results from the API response
        results = self._extract_results(api_response)

        # Convert results to dictionaries for easier use
        formatted_results = []
        for result in results:
            if hasattr(result, "to_dict"):
                formatted_results.append(result.to_dict())
            elif hasattr(result, "__dict__"):
                formatted_results.append(result.__dict__)
            elif isinstance(result, dict):
                formatted_results.append(result)
            else:
                formatted_results.append({"data": result})

        return formatted_results

    def search(
        self,
        query: str,
        limit: int = 10,
        return_metadata: bool = True,
        return_images: bool = True,
        file_extension_include: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for USD materials using the given query.

        This method works in both sync and async contexts by detecting the event loop.

        Args:
            query: The search query string (e.g., "White Powder-Coated Steel")
            limit: Maximum number of results to return (default: 10)
            return_metadata: Whether to include metadata in results (default: True)
            return_images: Whether to include images in results (default: True)
            file_extension_include: File extension(s) to include - can be a string (e.g., "mdl")
                                   or list of strings (e.g., ["mdl", "usd"])

        Returns:
            A list of search result dictionaries, or empty list if no results

        Example:
            >>> client = USDSearchClient()
            >>> results = client.search("metallic paint", limit=5, file_extension_include=["mdl"])
            >>> for result in results:
            ...     print(f"ID: {result.get('id')}, Score: {result.get('score')}")
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - this shouldn't be called from here
            # Raise an error with helpful message
            raise RuntimeError(
                "Cannot use synchronous search() from async context. "
                "Use 'await client.search_async(...)' instead."
            )
        except RuntimeError as e:
            # Check if this is our error or the "no running loop" error
            if "Cannot use synchronous search()" in str(e):
                raise
            # No running loop - we can safely use asyncio.run()
            api_response = asyncio.run(
                self._search_async(
                    query, limit, return_metadata, return_images, file_extension_include
                )
            )

        if api_response is None:
            return []

        # Extract results from the API response
        results = self._extract_results(api_response)

        # Convert results to dictionaries for easier use
        formatted_results = []
        for result in results:
            if hasattr(result, "to_dict"):
                formatted_results.append(result.to_dict())
            elif hasattr(result, "__dict__"):
                formatted_results.append(result.__dict__)
            elif isinstance(result, dict):
                formatted_results.append(result)
            else:
                formatted_results.append({"data": result})

        return formatted_results

    def _extract_results(self, api_response: Any) -> list[Any]:
        """Extract the results list from the API response.

        Args:
            api_response: The raw API response

        Returns:
            A list of results extracted from the response
        """
        # If it's already a list, return it
        if isinstance(api_response, list):
            return api_response

        # Check common field names for results
        if hasattr(api_response, "__dict__"):
            for field_name in [
                "results",
                "items",
                "data",
                "hits",
                "documents",
                "matches",
            ]:
                if hasattr(api_response, field_name):
                    results = getattr(api_response, field_name)
                    if results is not None:
                        return results

        # Try to convert to dict and extract results
        try:
            if hasattr(api_response, "to_dict"):
                response_dict = api_response.to_dict()
            elif hasattr(api_response, "__dict__"):
                response_dict = api_response.__dict__
            else:
                response_dict = dict(api_response)

            # Look for results in the dictionary
            for field_name in [
                "results",
                "items",
                "data",
                "hits",
                "documents",
                "matches",
            ]:
                if field_name in response_dict:
                    return response_dict[field_name]

            # If not found, look for the first list in the dict
            for value in response_dict.values():
                if isinstance(value, list) and len(value) > 0:
                    return value

        except Exception:
            pass

        # Return empty list if no results found
        return []


# Convenience function for simple searches
def search_usd_materials(
    query: str,
    limit: int = 10,
    host: str | None = None,
    file_extension_include: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convenience function to search for USD materials.

    This is a simple wrapper that creates a client and performs a search.

    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
        host: Optional API host URL. If not provided, uses USD_SEARCH_API_HOST.
        file_extension_include: File extension(s) to include - can be a string (e.g., "mdl")
                               or list of strings (e.g., ["mdl", "usd"])

    Returns:
        A list of search result dictionaries

    Example:
        >>> results = search_usd_materials("wood texture", limit=3, file_extension_include=["mdl"])
        >>> for result in results:
        ...     print(f"Material: {result.get('metadata', {}).get('name')}")
    """
    client = USDSearchClient(host) if host else USDSearchClient()
    return client.search(
        query, limit=limit, file_extension_include=file_extension_include
    )


# Async version for use in async contexts
async def search_usd_materials_async(
    query: str,
    limit: int = 10,
    host: str | None = None,
    return_metadata: bool = True,
    return_images: bool = True,
    file_extension_include: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Async version of search_usd_materials for use in async contexts.

    Args:
        query: The search query string
        limit: Maximum number of results to return
        host: Optional API host URL. If not provided, uses USD_SEARCH_API_HOST.
        return_metadata: Whether to include metadata in results
        return_images: Whether to include images in results
        file_extension_include: File extension(s) to include - can be a string (e.g., "mdl")
                               or list of strings (e.g., ["mdl", "usd"])

    Returns:
        A list of search result dictionaries

    Example:
        >>> results = await search_usd_materials_async("metal", limit=5, file_extension_include=["mdl"])
    """
    client = USDSearchClient(host) if host else USDSearchClient()
    api_response = await client._search_async(
        query, limit, return_metadata, return_images, file_extension_include
    )

    if api_response is None:
        return []

    results = client._extract_results(api_response)

    # Convert to dictionaries
    formatted_results = []
    for result in results:
        if hasattr(result, "to_dict"):
            formatted_results.append(result.to_dict())
        elif hasattr(result, "__dict__"):
            formatted_results.append(result.__dict__)
        elif isinstance(result, dict):
            formatted_results.append(result)
        else:
            formatted_results.append({"data": result})

    return formatted_results
