# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import Any

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class BuildImageVectorStoreConfig(FunctionBaseConfig, name="build_image_vector_store"):  # type: ignore[call-arg]
    pass


@register_function(config_type=BuildImageVectorStoreConfig)  # type: ignore[misc]
async def build_image_vector_store(
    config: BuildImageVectorStoreConfig, builder: Builder
) -> Any:
    import json
    from pathlib import Path

    from world_understanding.functions.knowledge.image_vector_store import (
        build_image_vector_store as build_image_vector_store_fn,
    )

    async def _build_image_vector_store(
        source_directory: str,
        output_path: str,
        index_type: str = "IndexFlatL2",
        normalize_embeddings: bool = False,
        recursive: bool = True,
        image_extensions: str = ".jpg,.jpeg,.png,.bmp,.gif,.tiff,.tif,.webp",
    ) -> str:
        """
        Build a vector store from a directory of images for similarity search.

        Args:
            source_directory: Directory containing images to index
            output_path: Path to save the vector store
            index_type: FAISS index type (IndexFlatL2, IndexFlatIP, IndexIVFFlat, IndexHNSWFlat)
            normalize_embeddings: Whether to normalize embeddings before indexing
            recursive: If True, scan subdirectories recursively
            image_extensions: Comma-separated list of image extensions to include

        Returns:
            JSON string with build results including number of images indexed
        """
        try:
            # Parse image extensions
            extensions_list = tuple(
                f".{ext.strip().lstrip('.')}" for ext in image_extensions.split(",")
            )

            # Build the vector store
            store = await asyncio.to_thread(
                build_image_vector_store_fn,
                source=source_directory,
                index_type=index_type,
                normalize_embeddings=normalize_embeddings,
                image_extensions=extensions_list,
                recursive=recursive,
            )

            # Save the store
            await asyncio.to_thread(store.save, output_path)

            # Prepare result
            result = {
                "status": "success",
                "num_documents": store.num_documents,
                "index_type": index_type,
                "dimension": store.dimension,
                "output_path": str(Path(output_path).absolute()),
                "source_directory": str(Path(source_directory).absolute()),
            }

            return json.dumps(result, indent=2)

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except Exception as e:
            return f"Failed to build vector store: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _build_image_vector_store,
        description=(
            "Build a searchable vector store from a collection of images. "
            "This creates an index that enables fast similarity search to find "
            "images that are visually similar to a query image. Useful for "
            "image retrieval, duplicate detection, and visual search applications."
        ),
    )


class FindSimilarImagesConfig(FunctionBaseConfig, name="find_similar_images"):  # type: ignore[call-arg]
    pass


@register_function(config_type=FindSimilarImagesConfig)  # type: ignore[misc]
async def find_similar_images(config: FindSimilarImagesConfig, builder: Builder) -> Any:
    import json
    from pathlib import Path

    from world_understanding.functions.knowledge.image_vector_store import (
        find_similar_images_from_vector_store as find_similar_images_fn,
    )

    async def _find_similar_images(
        query_image_path: str,
        vector_store_path: str,
        k: int = 5,
        filter_key: str = "",
        filter_value: str = "",
    ) -> str:
        """
        Find images similar to a query image using a pre-built vector store.

        Args:
            query_image_path: Path to the query image
            vector_store_path: Path to the saved vector store
            k: Number of similar images to return (default: 5)
            filter_key: Optional metadata key to filter results
            filter_value: Optional metadata value to filter results

        Returns:
            JSON string with similar images and their similarity scores
        """
        try:
            # Build filter metadata if provided
            filter_metadata = None
            if filter_key and filter_value:
                filter_metadata = {filter_key: filter_value}

            # Search for similar images
            results = await asyncio.to_thread(
                find_similar_images_fn,
                query_image=query_image_path,
                store=vector_store_path,
                k=k,
                filter_metadata=filter_metadata,
            )

            # Format results
            formatted_results = {
                "query_image": str(Path(query_image_path).absolute()),
                "vector_store": str(Path(vector_store_path).absolute()),
                "num_results": len(results),
                "results": [
                    {
                        "rank": result.rank + 1,  # Make 1-based for user display
                        "image_path": result.document.image_path,
                        "similarity_score": f"{result.score:.4f}",
                        "metadata": result.document.metadata,
                    }
                    for result in results
                ],
            }

            return json.dumps(formatted_results, indent=2)

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except Exception as e:
            return f"Failed to search for similar images: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _find_similar_images,
        description=(
            "Search for visually similar images using a pre-built vector store. "
            "Finds the most similar images to a query image based on visual features. "
            "Useful for finding duplicates, similar products, or related visual content. "
            "Optionally filter results by metadata attributes."
        ),
    )


class BuildTextVectorStoreConfig(FunctionBaseConfig, name="build_text_vector_store"):  # type: ignore[call-arg]
    pass


@register_function(config_type=BuildTextVectorStoreConfig)  # type: ignore[misc]
async def build_text_vector_store(
    config: BuildTextVectorStoreConfig, builder: Builder
) -> Any:
    import json
    from pathlib import Path

    from world_understanding.functions.knowledge.text_vector_store import (
        build_text_vector_store as build_text_vector_store_fn,
    )

    async def _build_text_vector_store(
        text_sources: str = "",
        image_sources: str = "",
        output_path: str = "",
        index_type: str = "IndexFlatL2",
        normalize_embeddings: bool = False,
        recursive: bool = True,
        text_extensions: str = ".txt,.md,.rst,.py,.js,.ts,.html,.css,.json,.xml,.csv",
        image_extensions: str = ".jpg,.jpeg,.png,.bmp,.gif,.tiff,.tif,.webp,.ico",
        image_embedding_type: str = "text",
        caption_prompt: str = "Describe this image in detail.",
        system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
        vlm_backend: str = "nim",
        vlm_model: str = "",
        vlm_api_key: str = "",
    ) -> str:
        """
        Build a vector store from texts and/or images for similarity search.

        Args:
            text_sources: Directory containing text files to index (optional)
            image_sources: Directory containing images to index (optional)
            output_path: Path to save the vector store
            index_type: FAISS index type (IndexFlatL2, IndexFlatIP, IndexIVFFlat, IndexHNSWFlat)
            normalize_embeddings: Whether to normalize embeddings before indexing
            recursive: If True, scan subdirectories recursively
            text_extensions: Comma-separated list of text extensions to include
            image_extensions: Comma-separated list of image extensions to include
            image_embedding_type: How to handle images ("text" for captioning, "image" for direct embedding)
            caption_prompt: Prompt to use for image captioning
            system_prompt: System instructions for the VLM
            vlm_backend: VLM backend to use ("azure_openai" or "nim")
            vlm_model: Model to use (uses backend default if empty)
            vlm_api_key: API key for the VLM backend (uses env var if empty)

        Returns:
            JSON string with build results including number of texts indexed
        """
        try:
            # Parse extensions
            text_extensions_list = tuple(
                f".{ext.strip().lstrip('.')}" for ext in text_extensions.split(",")
            )
            image_extensions_list = tuple(
                f".{ext.strip().lstrip('.')}" for ext in image_extensions.split(",")
            )

            # Prepare sources
            text_source = text_sources if text_sources else None
            image_source = image_sources if image_sources else None

            # Prepare VLM parameters
            vlm_params = {}
            if vlm_model:
                vlm_params["vlm_model"] = vlm_model
            if vlm_api_key:
                vlm_params["vlm_api_key"] = vlm_api_key

            # Build the vector store
            store = await asyncio.to_thread(
                build_text_vector_store_fn,
                text_source=text_source,
                image_source=image_source,
                index_type=index_type,
                normalize_embeddings=normalize_embeddings,
                text_extensions=text_extensions_list,
                image_extensions=image_extensions_list,
                recursive=recursive,
                image_embedding_type=image_embedding_type,
                caption_prompt=caption_prompt,
                system_prompt=system_prompt,
                vlm_backend=vlm_backend,
                **vlm_params,
            )

            # Save the store
            await asyncio.to_thread(store.save, output_path)

            # Prepare result
            result = {
                "status": "success",
                "num_documents": store.num_documents,
                "index_type": index_type,
                "dimension": store.dimension,
                "output_path": str(Path(output_path).absolute()),
                "text_sources": (
                    str(Path(text_sources).absolute()) if text_sources else None
                ),
                "image_sources": (
                    str(Path(image_sources).absolute()) if image_sources else None
                ),
            }

            return json.dumps(result, indent=2)

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except Exception as e:
            return f"Failed to build vector store: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _build_text_vector_store,
        description=(
            "Build a searchable vector store from texts and/or images. "
            "This creates an index that enables fast similarity search to find "
            "texts or image captions that are similar to a query. Useful for "
            "document retrieval, content search, and knowledge management applications."
        ),
    )


class FindSimilarTextsConfig(FunctionBaseConfig, name="find_similar_texts"):  # type: ignore[call-arg]
    pass


@register_function(config_type=FindSimilarTextsConfig)  # type: ignore[misc]
async def find_similar_texts(config: FindSimilarTextsConfig, builder: Builder) -> Any:
    import json
    from pathlib import Path

    from world_understanding.functions.knowledge.text_vector_store import (
        find_similar_texts_from_vector_store as find_similar_texts_fn,
    )

    async def _find_similar_texts(
        query: str,
        query_type: str,
        vector_store_path: str,
        k: int = 5,
        filter_key: str = "",
        filter_value: str = "",
    ) -> str:
        """
        Find texts similar to a query using a pre-built text vector store.

        Args:
            query: Query text or path to query image
            query_type: Type of query ("text", "image", or "embedding")
            vector_store_path: Path to the saved vector store
            k: Number of similar texts to return (default: 5)
            filter_key: Optional metadata key to filter results
            filter_value: Optional metadata value to filter results

        Returns:
            JSON string with similar texts and their similarity scores
        """
        try:
            # Build filter metadata if provided
            filter_metadata = None
            if filter_key and filter_value:
                filter_metadata = {filter_key: filter_value}

            # Search for similar texts
            results = await asyncio.to_thread(
                find_similar_texts_fn,
                query=query,
                query_type=query_type,
                store=vector_store_path,
                k=k,
                filter_metadata=filter_metadata,
            )

            # Format results
            formatted_results = {
                "query": query,
                "query_type": query_type,
                "vector_store": str(Path(vector_store_path).absolute()),
                "num_results": len(results),
                "results": [
                    {
                        "rank": result.rank + 1,  # Make 1-based for user display
                        "text_content": result.document.text_content,
                        "text_id": result.document.document_id,
                        "similarity_score": f"{result.score:.4f}",
                        "metadata": result.document.metadata,
                    }
                    for result in results
                ],
            }

            return json.dumps(formatted_results, indent=2)

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except Exception as e:
            return f"Failed to search for similar texts: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _find_similar_texts,
        description=(
            "Search for similar texts using a pre-built text vector store. "
            "Finds the most similar texts to a query based on semantic similarity. "
            "Supports text queries, image queries (via captioning), and embedding queries. "
            "Useful for document search, content discovery, and knowledge retrieval. "
            "Optionally filter results by metadata attributes."
        ),
    )


class BuildMultimodalVectorStoreConfig(
    FunctionBaseConfig, name="build_multimodal_vector_store"
):  # type: ignore[call-arg]
    pass


@register_function(config_type=BuildMultimodalVectorStoreConfig)  # type: ignore[misc]
async def build_multimodal_vector_store(
    config: BuildMultimodalVectorStoreConfig, builder: Builder
) -> Any:
    import json
    from pathlib import Path

    from world_understanding.functions.knowledge.multimodal_vector_store import (
        build_multimodal_vector_store as build_multimodal_vector_store_fn,
    )

    async def _build_multimodal_vector_store(
        text_sources: str = "",
        image_sources: str = "",
        output_path: str = "",
        index_type: str = "IndexFlatL2",
        normalize_embeddings: bool = False,
        recursive: bool = True,
        text_extensions: str = ".txt,.md,.rst,.py,.js,.ts,.html,.css,.json,.xml,.csv",
        image_extensions: str = ".jpg,.jpeg,.png,.bmp,.tiff,.gif,.webp,.svg",
        image_embedding_type: str = "image",
        caption_prompt: str = "Describe this image in detail.",
        system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
        vlm_backend: str = "nim",
        vlm_model: str = "",
        vlm_api_key: str = "",
    ) -> str:
        """
        Build a multimodal vector store from texts and/or images for similarity search.

        Args:
            text_sources: Directory containing text files to index (optional)
            image_sources: Directory containing images to index (optional)
            output_path: Path to save the vector store
            index_type: FAISS index type (IndexFlatL2, IndexFlatIP, IndexIVFFlat, IndexHNSWFlat)
            normalize_embeddings: Whether to normalize embeddings before indexing
            recursive: If True, scan subdirectories recursively
            text_extensions: Comma-separated list of text extensions to include
            image_extensions: Comma-separated list of image extensions to include
            image_embedding_type: How to handle images ("text" for captioning, "image" for direct embedding)
            caption_prompt: Prompt to use for image captioning
            system_prompt: System instructions for the VLM
            vlm_backend: VLM backend to use ("azure_openai" or "nim")
            vlm_model: Model to use (uses backend default if empty)
            vlm_api_key: API key for the VLM backend (uses env var if empty)

        Returns:
            JSON string with build results including number of documents indexed
        """
        try:
            # Parse extensions
            text_extensions_list = tuple(
                f".{ext.strip().lstrip('.')}" for ext in text_extensions.split(",")
            )
            image_extensions_list = tuple(
                f".{ext.strip().lstrip('.')}" for ext in image_extensions.split(",")
            )

            # Prepare sources
            text_source = text_sources if text_sources else None
            image_source = image_sources if image_sources else None

            # Prepare VLM parameters
            vlm_params = {}
            if vlm_model:
                vlm_params["vlm_model"] = vlm_model
            if vlm_api_key:
                vlm_params["vlm_api_key"] = vlm_api_key

            # Build the vector store
            store = await asyncio.to_thread(
                build_multimodal_vector_store_fn,
                text_source=text_source,
                image_source=image_source,
                index_type=index_type,
                normalize_embeddings=normalize_embeddings,
                text_extensions=text_extensions_list,
                image_extensions=image_extensions_list,
                recursive=recursive,
                image_embedding_type=image_embedding_type,
                caption_prompt=caption_prompt,
                system_prompt=system_prompt,
                vlm_backend=vlm_backend,
                **vlm_params,
            )

            # Save the store
            await asyncio.to_thread(store.save, output_path)

            # Prepare result
            result = {
                "status": "success",
                "num_documents": store.num_documents,
                "index_type": index_type,
                "dimension": store.dimension,
                "output_path": str(Path(output_path).absolute()),
                "text_sources": (
                    str(Path(text_sources).absolute()) if text_sources else None
                ),
                "image_sources": (
                    str(Path(image_sources).absolute()) if image_sources else None
                ),
            }

            return json.dumps(result, indent=2)

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except Exception as e:
            return f"Failed to build vector store: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _build_multimodal_vector_store,
        description=(
            "Build a searchable multimodal vector store from texts and/or images. "
            "This creates an index that enables fast similarity search across different "
            "content types using unified embeddings. Useful for cross-modal retrieval, "
            "content discovery, and knowledge management applications."
        ),
    )


class FindSimilarDocumentsConfig(FunctionBaseConfig, name="find_similar_documents"):  # type: ignore[call-arg]
    pass


@register_function(config_type=FindSimilarDocumentsConfig)  # type: ignore[misc]
async def find_similar_documents(
    config: FindSimilarDocumentsConfig, builder: Builder
) -> Any:
    import json
    from pathlib import Path

    from world_understanding.functions.knowledge.multimodal_vector_store import (
        find_similar_documents_from_vector_store as find_similar_documents_fn,
    )

    async def _find_similar_documents(
        query: str,
        query_type: str,
        vector_store_path: str,
        k: int = 5,
        filter_key: str = "",
        filter_value: str = "",
        embedding_type: str = "image",
    ) -> str:
        """
        Find documents similar to a query using a pre-built multimodal vector store.

        Args:
            query: Query text or path to query image
            query_type: Type of query ("text", "image", or "embedding")
            vector_store_path: Path to the saved vector store
            k: Number of similar documents to return (default: 5)
            filter_key: Optional metadata key to filter results
            filter_value: Optional metadata value to filter results
            embedding_type: Type of embedding to use for image queries ("text" or "image")

        Returns:
            JSON string with similar documents and their similarity scores
        """
        try:
            # Build filter metadata if provided
            filter_metadata = None
            if filter_key and filter_value:
                filter_metadata = {filter_key: filter_value}

            # Search for similar documents
            results = await asyncio.to_thread(
                find_similar_documents_fn,
                query=query,
                query_type=query_type,
                store=vector_store_path,
                k=k,
                filter_metadata=filter_metadata,
                embedding_type=embedding_type,
            )

            # Format results
            formatted_results = {
                "query": query,
                "query_type": query_type,
                "vector_store": str(Path(vector_store_path).absolute()),
                "num_results": len(results),
                "results": [
                    {
                        "rank": result.rank + 1,  # Make 1-based for user display
                        "document_id": result.document.document_id,
                        "content_type": result.document.get_content_type(),
                        "text_content": result.document.text_content,
                        "image_path": result.document.image_path,
                        "similarity_score": f"{result.score:.4f}",
                        "metadata": result.document.metadata,
                    }
                    for result in results
                ],
            }

            return json.dumps(formatted_results, indent=2)

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except Exception as e:
            return f"Failed to search for similar documents: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _find_similar_documents,
        description=(
            "Search for similar documents using a pre-built multimodal vector store. "
            "Finds the most similar documents to a query based on unified embeddings. "
            "Supports text queries, image queries, and embedding queries across different "
            "content types. Useful for cross-modal retrieval, content discovery, and "
            "knowledge management. Optionally filter results by metadata attributes."
        ),
    )


class CollectDocumentsConfig(FunctionBaseConfig, name="collect_documents"):  # type: ignore[call-arg]
    pass


@register_function(config_type=CollectDocumentsConfig)  # type: ignore[misc]
async def collect_documents(config: CollectDocumentsConfig, builder: Builder) -> Any:
    import json
    from pathlib import Path

    from world_understanding.functions.knowledge.multimodal_vector_store import (
        collect_documents_from_vector_store as collect_documents_fn,
    )

    async def _collect_documents(
        vector_store_path: str,
        filter_key: str = "",
        filter_value: str = "",
    ) -> str:
        """
        Collect documents from a multimodal vector store with optional metadata filtering.

        Args:
            vector_store_path: Path to the saved vector store
            filter_key: Optional metadata key to filter results
            filter_value: Optional metadata value to filter results

        Returns:
            JSON string with collected documents and their metadata
        """
        try:
            # Build filter metadata if provided
            filter_metadata = None
            if filter_key and filter_value:
                filter_metadata = {filter_key: filter_value}

            # Collect documents
            documents = await asyncio.to_thread(
                collect_documents_fn,
                store=vector_store_path,
                filter_metadata=filter_metadata,
            )

            # Format results
            formatted_results = {
                "vector_store": str(Path(vector_store_path).absolute()),
                "num_documents": len(documents),
                "filter_applied": filter_metadata is not None,
                "filter_metadata": filter_metadata,
                "documents": [
                    {
                        "document_id": doc.document_id,
                        "content_type": doc.get_content_type(),
                        "text_content": doc.text_content,
                        "image_path": doc.image_path,
                        "metadata": doc.metadata,
                    }
                    for doc in documents
                ],
            }

            return json.dumps(formatted_results, indent=2)

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except Exception as e:
            return f"Failed to collect documents: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _collect_documents,
        description=(
            "Collect documents from a multimodal vector store with optional metadata filtering. "
            "Returns all documents that match the specified metadata criteria, or all documents "
            "if no filter is provided. Supports both exact matching and case-insensitive string matching. "
            "Useful for document retrieval, content analysis, and knowledge management."
        ),
    )
