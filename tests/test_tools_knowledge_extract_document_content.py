# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for document content extraction tools.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from world_understanding.tools.knowledge.extract_document_content import (
    ExtractDocumentContentInput,
    ExtractDocumentContentOutput,
    SplitDocumentContentInput,
    SplitDocumentContentOutput,
    extract_document_content_tool,
    split_document_content_tool,
)


class TestExtractDocumentContentTool:
    """Test suite for extract_document_content_tool."""

    def test_input_model_validation(self):
        """Test that input model validates correctly."""
        # Valid single path input
        input1 = ExtractDocumentContentInput(
            source="/path/to/document.pdf", save_content_only=True
        )
        assert input1.source == "/path/to/document.pdf"
        assert input1.save_content_only is True

        # Valid list of paths input
        input2 = ExtractDocumentContentInput(
            source=["/path/to/doc1.pdf", "/path/to/doc2.docx"], save_content_only=False
        )
        assert len(input2.source) == 2
        assert input2.save_content_only is False

    def test_output_model_structure(self):
        """Test that output model has correct structure."""
        output = ExtractDocumentContentOutput(
            extracted_content={
                "/path/to/doc.pdf": [
                    {"document_type": "text", "content": "Sample text"},
                    {"document_type": "image", "content": "base64_image_data"},
                ]
            },
            document_count=1,
            content_types={"text": 1, "image": 1},
            extraction_errors=[],
        )

        assert output.document_count == 1
        assert "text" in output.content_types
        assert "image" in output.content_types
        assert len(output.extraction_errors) == 0

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.extract_document_content_func"
    )
    def test_successful_extraction_single_file(self, mock_extract_func):
        """Test successful extraction from a single file."""
        # Mock the core function
        mock_extract_func.return_value = {
            "/path/to/doc.pdf": [
                {"document_type": "text", "content": "Extracted text content"},
                {"document_type": "structured", "content": "Table data"},
            ]
        }

        # Create input
        inputs = ExtractDocumentContentInput(
            source="/path/to/doc.pdf", save_content_only=True
        )

        # Call tool
        output = extract_document_content_tool(inputs)

        # Verify results
        assert output.document_count == 1
        assert "/path/to/doc.pdf" in output.extracted_content
        assert len(output.extracted_content["/path/to/doc.pdf"]) == 2
        assert output.content_types["text"] == 1
        assert output.content_types["structured"] == 1
        assert len(output.extraction_errors) == 0

        # Verify function was called correctly
        mock_extract_func.assert_called_once_with(
            source="/path/to/doc.pdf",
            save_content_only=True,
            batch_size=32,
            max_retries=3,
        )

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.extract_document_content_func"
    )
    def test_successful_extraction_multiple_files(self, mock_extract_func):
        """Test successful extraction from multiple files."""
        # Mock the core function
        mock_extract_func.return_value = {
            "/path/to/doc1.pdf": [{"document_type": "text", "content": "Content 1"}],
            "/path/to/doc2.docx": [
                {"document_type": "text", "content": "Content 2"},
                {"document_type": "image", "content": "Image data"},
            ],
        }

        # Create input
        inputs = ExtractDocumentContentInput(
            source=["/path/to/doc1.pdf", "/path/to/doc2.docx"], save_content_only=False
        )

        # Call tool
        output = extract_document_content_tool(inputs)

        # Verify results
        assert output.document_count == 2
        assert "/path/to/doc1.pdf" in output.extracted_content
        assert "/path/to/doc2.docx" in output.extracted_content
        assert output.content_types["text"] == 2
        assert output.content_types["image"] == 1
        assert len(output.extraction_errors) == 0

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.extract_document_content_func"
    )
    def test_file_not_found_error(self, mock_extract_func):
        """Test handling of FileNotFoundError."""
        # Mock the core function to raise FileNotFoundError
        mock_extract_func.side_effect = FileNotFoundError("Document not found")

        # Create input
        inputs = ExtractDocumentContentInput(source="/nonexistent/doc.pdf")

        # Call tool
        output = extract_document_content_tool(inputs)

        # Verify error handling
        assert output.document_count == 0
        assert len(output.extracted_content) == 0
        assert len(output.extraction_errors) == 1
        assert "File not found" in output.extraction_errors[0]

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.extract_document_content_func"
    )
    def test_permission_error(self, mock_extract_func):
        """Test handling of PermissionError."""
        # Mock the core function to raise PermissionError
        mock_extract_func.side_effect = PermissionError("Access denied")

        # Create input
        inputs = ExtractDocumentContentInput(source="/protected/doc.pdf")

        # Call tool
        output = extract_document_content_tool(inputs)

        # Verify error handling
        assert output.document_count == 0
        assert len(output.extraction_errors) == 1
        assert "Permission denied" in output.extraction_errors[0]

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.extract_document_content_func"
    )
    def test_runtime_error(self, mock_extract_func):
        """Test handling of RuntimeError."""
        # Mock the core function to raise RuntimeError
        mock_extract_func.side_effect = RuntimeError("Processing failed")

        # Create input
        inputs = ExtractDocumentContentInput(source="/path/to/doc.pdf")

        # Call tool
        output = extract_document_content_tool(inputs)

        # Verify error handling
        assert output.document_count == 0
        assert len(output.extraction_errors) == 1
        assert "Processing failed" in output.extraction_errors[0]

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.extract_document_content_func"
    )
    def test_output_dir_saves_json(self, mock_extract_func):
        """Test saving extracted content to JSON file when output_dir is provided."""
        import base64
        import json
        import tempfile
        from pathlib import Path

        # Mock the core function with mixed content including base64 images
        mock_extract_func.return_value = {
            "/path/to/doc.pdf": [
                {"document_type": "text", "content": "Some text content"},
                {
                    "document_type": "image",
                    "content": base64.b64encode(b"fake_image_data").decode(),
                },
                {"document_type": "structured", "content": "Table data"},
                {
                    "document_type": "image",
                    "content": base64.b64encode(b"another_image").decode(),
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input with output_dir
            inputs = ExtractDocumentContentInput(
                source="/path/to/doc.pdf", save_content_only=True, output_dir=temp_dir
            )

            # Call tool
            output = extract_document_content_tool(inputs)

            # Verify output
            assert output.document_count == 1
            assert output.content_types["text"] == 1
            assert output.content_types["image"] == 2
            assert output.content_types["structured"] == 1

            # Check that JSON file was created
            save_dir = Path(temp_dir)
            json_file = save_dir / "extracted_content.json"
            assert json_file.exists()

            # Verify JSON content includes base64 images as-is
            with open(json_file) as f:
                saved_content = json.load(f)

            assert len(saved_content["/path/to/doc.pdf"]) == 4

            # Text content should be unchanged
            assert (
                saved_content["/path/to/doc.pdf"][0]["content"] == "Some text content"
            )

            # Image content should still be base64
            assert (
                saved_content["/path/to/doc.pdf"][1]["content"]
                == base64.b64encode(b"fake_image_data").decode()
            )
            assert (
                saved_content["/path/to/doc.pdf"][3]["content"]
                == base64.b64encode(b"another_image").decode()
            )

            # Structured content should be unchanged
            assert saved_content["/path/to/doc.pdf"][2]["content"] == "Table data"


class TestSplitDocumentContentTool:
    """Test suite for split_document_content_tool."""

    def test_input_model_validation(self):
        """Test that input model validates correctly."""
        inputs = SplitDocumentContentInput(
            input_file_path="/path/to/extracted.json", output_dir="/path/to/output"
        )

        assert inputs.input_file_path == "/path/to/extracted.json"
        assert inputs.output_dir == "/path/to/output"

    def test_output_model_structure(self):
        """Test that output model has correct structure."""
        output = SplitDocumentContentOutput(
            created_files={
                "document1": ["/output/document1/doc1_0000_text.txt"],
                "document2": [
                    "/output/document2/doc2_0000_text.txt",
                    "/output/document2/doc2_0001_image.png",
                ],
            },
            total_files_created=3,
            content_type_distribution={"text": 2, "image": 1},
            processing_errors=[],
        )

        assert output.total_files_created == 3
        assert "text" in output.content_type_distribution
        assert output.content_type_distribution["text"] == 2

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.split_document_content_by_type_func"
    )
    def test_successful_split(self, mock_split_func):
        """Test successful splitting of document content."""
        # Mock the core function
        mock_split_func.return_value = {
            "document1": [
                Path("/output/document1/document1_0000_text.txt"),
                Path("/output/document1/document1_0001_structured.txt"),
            ],
            "document2": [
                Path("/output/document2/document2_0000_text.txt"),
                Path("/output/document2/document2_0001_image.png"),
            ],
        }

        # Create input
        inputs = SplitDocumentContentInput(
            input_file_path="/path/to/extracted.json", output_dir="/path/to/output"
        )

        # Call tool
        output = split_document_content_tool(inputs)

        # Verify results
        assert output.total_files_created == 4
        assert "document1" in output.created_files
        assert "document2" in output.created_files
        assert len(output.created_files["document1"]) == 2
        assert len(output.created_files["document2"]) == 2

        # Check content type distribution
        assert output.content_type_distribution["text"] == 2
        assert output.content_type_distribution["structured"] == 1
        assert output.content_type_distribution["image"] == 1

        # Verify no errors
        assert len(output.processing_errors) == 0

        # Verify function was called correctly
        mock_split_func.assert_called_once_with(
            input_file_path="/path/to/extracted.json", output_dir="/path/to/output"
        )

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.split_document_content_by_type_func"
    )
    def test_file_not_found_error_split(self, mock_split_func):
        """Test handling of FileNotFoundError in split tool."""
        # Mock the core function to raise FileNotFoundError
        mock_split_func.side_effect = FileNotFoundError("Input file not found")

        # Create input
        inputs = SplitDocumentContentInput(
            input_file_path="/nonexistent/file.json", output_dir="/path/to/output"
        )

        # Call tool
        output = split_document_content_tool(inputs)

        # Verify error handling
        assert output.total_files_created == 0
        assert len(output.created_files) == 0
        assert len(output.processing_errors) == 1
        assert "File not found" in output.processing_errors[0]

    @patch(
        "world_understanding.tools.knowledge.extract_document_content.split_document_content_by_type_func"
    )
    def test_value_error_split(self, mock_split_func):
        """Test handling of ValueError in split tool."""
        # Mock the core function to raise ValueError
        mock_split_func.side_effect = ValueError("Invalid JSON format")

        # Create input
        inputs = SplitDocumentContentInput(
            input_file_path="/path/to/invalid.json", output_dir="/path/to/output"
        )

        # Call tool
        output = split_document_content_tool(inputs)

        # Verify error handling
        assert output.total_files_created == 0
        assert len(output.processing_errors) == 1
        assert "Invalid input" in output.processing_errors[0]

    def test_integration_with_temp_files(self):
        """Test integration with temporary files (without actual nv_ingest)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock extracted content JSON file
            extracted_data = {
                "/path/to/doc.pdf": [
                    {"document_type": "text", "content": "Sample text content"},
                    {
                        "document_type": "image",
                        "content": "aW1hZ2VfZGF0YQ==",
                    },  # base64 "image_data"
                ]
            }

            input_file = Path(temp_dir) / "extracted.json"
            with open(input_file, "w") as f:
                json.dump(extracted_data, f)

            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()

            # Create input
            inputs = SplitDocumentContentInput(
                input_file_path=str(input_file), output_dir=str(output_dir)
            )

            # Call tool (this will call the actual function)
            with patch(
                "world_understanding.tools.knowledge.extract_document_content.split_document_content_by_type_func"
            ) as mock_split:
                mock_split.return_value = {
                    "doc": [
                        Path(output_dir) / "doc" / "doc_0000_text.txt",
                        Path(output_dir) / "doc" / "doc_0001_image.png",
                    ]
                }

                output = split_document_content_tool(inputs)

                # Verify results
                assert output.total_files_created == 2
                assert "doc" in output.created_files
                assert len(output.created_files["doc"]) == 2


def test_tool_registration():
    """Test that tools are properly registered."""
    from world_understanding.tools.base import get_tool_registry

    registry = get_tool_registry()

    # Check that our tools are registered
    assert "extract_document_content" in registry
    assert "split_document_content" in registry

    # Verify tool specifications
    extract_tool = registry["extract_document_content"]
    assert extract_tool.spec.version == "0.1.0"
    assert extract_tool.spec.input_model == ExtractDocumentContentInput
    assert extract_tool.spec.output_model == ExtractDocumentContentOutput

    split_tool = registry["split_document_content"]
    assert split_tool.spec.version == "0.1.0"
    assert split_tool.spec.input_model == SplitDocumentContentInput
    assert split_tool.spec.output_model == SplitDocumentContentOutput
