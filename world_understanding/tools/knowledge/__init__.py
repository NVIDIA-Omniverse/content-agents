# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Knowledge tools for vector stores and knowledge management."""

from . import (
    extract_document_content,
    image_vector_store,
    multimodal_vector_store,
    text_vector_store,
    usd_search,
)

# Import the tools to trigger their registration
from .extract_document_content import (
    ExtractDocumentContentInput,
    ExtractDocumentContentOutput,
    SplitDocumentContentInput,
    SplitDocumentContentOutput,
    extract_document_content_tool,
    split_document_content_tool,
)
from .image_vector_store import (
    BuildImageVectorStoreInput,
    BuildImageVectorStoreOutput,
    FindSimilarImagesInput,
    FindSimilarImagesOutput,
    build_image_vector_store_tool,
    find_similar_images_tool,
)
from .multimodal_vector_store import (
    BuildMultimodalVectorStoreInput,
    BuildMultimodalVectorStoreOutput,
    FindSimilarDocumentsInput,
    FindSimilarDocumentsOutput,
    build_multimodal_vector_store_tool,
    find_similar_documents_tool,
)
from .text_vector_store import (
    BuildTextVectorStoreInput,
    BuildTextVectorStoreOutput,
    FindSimilarTextsInput,
    FindSimilarTextsOutput,
    build_text_vector_store_tool,
    find_similar_texts_tool,
)
from .usd_search import (
    USDSearchInput,
    USDSearchOutput,
    usd_search_tool,
)

__all__ = [
    "extract_document_content",
    "image_vector_store",
    "multimodal_vector_store",
    "text_vector_store",
    "usd_search",
]
