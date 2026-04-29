# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tool wrappers for document content extraction functionality.

This module provides tool interfaces for extracting and processing document content
using NVIDIA's nv_ingest framework.
"""

import json
from pathlib import Path
from typing import Any

from world_understanding.functions.knowledge.extract_document_content import (
    extract_document_content as extract_document_content_func,
)
from world_understanding.functions.knowledge.extract_document_content import (
    split_document_content_by_type as split_document_content_by_type_func,
)
from world_understanding.tools.base import ToolInput, ToolOutput, register_tool


# Input/Output Models for extract_document_content
class ExtractDocumentContentInput(ToolInput):
    """Input for document content extraction tool."""

    source: str | list[str]  # Path(s) to documents or directory
    output_dir: str | None = None  # Directory to save the extracted content as JSON
    save_content_only: bool = (
        True  # If True, only save content; if False, save full metadata
    )
    batch_size: int = 32  # Batch size for processing files
    max_retries: int = 3  # Maximum number of retries for processing files


class ExtractDocumentContentOutput(ToolOutput):
    """Output for document content extraction tool."""

    extracted_content: dict[
        str, list[dict[str, Any]]
    ]  # file_path -> list of extracted content
    document_count: int  # Total number of documents processed
    content_types: dict[str, int]  # Count of each content type extracted
    extraction_errors: list[str] = []  # Any errors during extraction


# Input/Output Models for split_document_content
class SplitDocumentContentInput(ToolInput):
    """Input for splitting document content by type."""

    input_file_path: str  # Path to JSON file with extracted content
    output_dir: str  # Directory to save split content files


class SplitDocumentContentOutput(ToolOutput):
    """Output for splitting document content by type."""

    created_files: dict[str, list[str]]  # doc_name -> list of created file paths
    total_files_created: int  # Total number of files created
    content_type_distribution: dict[str, int]  # Distribution of content types
    processing_errors: list[str] = []  # Any errors during processing


# Tool: Extract Document Content
@register_tool(
    name="extract_document_content",
    version="0.1.0",
    tags=["knowledge", "document", "extraction", "nv-ingest", "cpu"],
    description="Extract content from various document formats using NVIDIA nv_ingest",
    input_model=ExtractDocumentContentInput,
    output_model=ExtractDocumentContentOutput,
)
def extract_document_content_tool(
    inputs: ExtractDocumentContentInput,
) -> ExtractDocumentContentOutput:
    """
    Extract content from documents using NVIDIA's nv_ingest framework.

    This tool processes documents to extract text, tables, charts, images,
    and other content types. It supports batch processing and various
    document formats including PDF, DOCX, TXT, and more.

    Args:
        inputs: ExtractDocumentContentInput containing:
            - source: Path(s) to documents or directory to scan
            - save_content_only: Whether to save only content or full metadata
            - output_dir: Optional directory to save extracted content as JSON file
            - batch_size: Batch size for processing files
            - max_retries: Maximum number of retries for processing files

    Returns:
        ExtractDocumentContentOutput containing:
            - extracted_content: Dictionary mapping file paths to extracted content
            - document_count: Number of documents processed
            - content_types: Count of each content type found
            - extraction_errors: Any errors encountered

    Note:
        Requires NVIDIA_API_KEY environment variable to be set.
    """
    extraction_errors: list[str] = []
    content_types_count: dict[str, int] = {}
    extracted_content_result: dict[str, list[dict[str, Any]]] = {}
    document_count_result = 0

    try:
        # Convert source to appropriate format
        source = inputs.source
        if isinstance(source, str):
            # Single path string
            source_for_func = source
        else:
            # List of paths
            source_for_func = source

        # Call the core function
        extracted_content = extract_document_content_func(
            source=source_for_func,
            save_content_only=inputs.save_content_only,
            batch_size=inputs.batch_size,
            max_retries=inputs.max_retries,
        )

        if inputs.output_dir:
            # Create save directory if it doesn't exist
            save_dir = Path(inputs.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save the extracted content to JSON (including base64 images)
            with open(save_dir / "extracted_content.json", "w", encoding="utf-8") as f:
                json.dump(extracted_content, f, indent=2)

        # Count content types
        for _file_path, contents in extracted_content.items():
            for content_item in contents:
                doc_type = content_item.get("document_type", "unknown")
                content_types_count[doc_type] = content_types_count.get(doc_type, 0) + 1

        # Populate output
        extracted_content_result = extracted_content
        document_count_result = len(extracted_content)
        content_types_count = content_types_count
        extraction_errors = extraction_errors

    except FileNotFoundError as e:
        extraction_errors.append(f"File not found: {str(e)}")

    except PermissionError as e:
        extraction_errors.append(f"Permission denied: {str(e)}")

    except ValueError as e:
        extraction_errors.append(f"Invalid input: {str(e)}")

    except RuntimeError as e:
        extraction_errors.append(f"Processing failed: {str(e)}")

    except Exception as e:
        extraction_errors.append(f"Unexpected error: {str(e)}")

    # Create output with all required fields
    output = ExtractDocumentContentOutput(
        extracted_content=extracted_content_result,
        document_count=document_count_result,
        content_types=content_types_count,
        extraction_errors=extraction_errors,
    )
    return output


# Tool: Split Document Content by Type
@register_tool(
    name="split_document_content",
    version="0.1.0",
    tags=["knowledge", "document", "splitting", "cpu"],
    description="Split extracted document content by type for multimodal processing",
    input_model=SplitDocumentContentInput,
    output_model=SplitDocumentContentOutput,
)
def split_document_content_tool(
    inputs: SplitDocumentContentInput,
) -> SplitDocumentContentOutput:
    """
    Split extracted document content by type for multimodal vector store.

    This tool takes the JSON output from extract_document_content and splits
    it into separate files based on content type (text, image, structured data).
    This is useful for preparing data for multimodal vector stores.

    Args:
        inputs: SplitDocumentContentInput containing:
            - input_file_path: Path to JSON file with extracted content
            - output_dir: Directory to save split content files

    Returns:
        SplitDocumentContentOutput containing:
            - created_files: Dictionary mapping document names to created file paths
            - total_files_created: Total number of files created
            - content_type_distribution: Distribution of content types
            - processing_errors: Any errors encountered

    Note:
        The input file should be a JSON file created by extract_document_content
        with save_content_only=True.
    """
    processing_errors: list[str] = []
    content_type_distribution: dict[str, int] = {}
    created_files_result: dict[str, list[str]] = {}
    total_files_result = 0

    try:
        # Call the core function
        created_files = split_document_content_by_type_func(
            input_file_path=inputs.input_file_path, output_dir=inputs.output_dir
        )

        # Calculate statistics
        total_files = 0
        for _doc_name, file_paths in created_files.items():
            total_files += len(file_paths)

            # Count content types from file extensions/names
            for file_path in file_paths:
                path = Path(file_path)
                # Extract content type from filename pattern: name_0000_type.ext
                parts = path.stem.split("_")
                if len(parts) >= 3:
                    content_type = parts[-1]
                    content_type_distribution[content_type] = (
                        content_type_distribution.get(content_type, 0) + 1
                    )

        # Convert Path objects to strings in created_files
        created_files_str = {
            doc_name: [str(p) for p in paths]
            for doc_name, paths in created_files.items()
        }

        # Populate output
        created_files_result = created_files_str
        total_files_result = total_files
        content_type_distribution = content_type_distribution
        processing_errors = processing_errors

    except FileNotFoundError as e:
        processing_errors.append(f"File not found: {str(e)}")

    except ValueError as e:
        processing_errors.append(f"Invalid input: {str(e)}")

    except Exception as e:
        processing_errors.append(f"Unexpected error: {str(e)}")

    # Create output with all required fields
    output = SplitDocumentContentOutput(
        created_files=created_files_result,
        total_files_created=total_files_result,
        content_type_distribution=content_type_distribution,
        processing_errors=processing_errors,
    )
    return output
