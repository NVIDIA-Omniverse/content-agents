# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Build Dataset APIs."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.build_dataset import (
    BuildDatasetPdfVectorstoreInput,
    BuildDatasetPdfVectorstoreOutput,
    BuildDatasetPrepareDatasetInput,
    BuildDatasetPrepareDatasetOutput,
    BuildDatasetUsdInput,
    BuildDatasetUsdOutput,
    build_dataset_pdf_vectorstore,
    build_dataset_prepare_dataset,
    build_dataset_usd,
)

# ============================================================================
# USD Dataset Building Tests
# ============================================================================


class TestBuildDatasetUsdInput:
    """Tests for BuildDatasetUsdInput validation."""

    def test_usd_input_valid(self, tmp_path):
        """Test creating valid BuildDatasetUsdInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = BuildDatasetUsdInput(
            config=config_file,
            extract_metadata=True,
            verbose=True,
        )

        assert params.config == config_file
        assert params.extract_metadata is True

    def test_usd_input_missing_config(self, tmp_path):
        """Test BuildDatasetUsdInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            BuildDatasetUsdInput(config=config_file)


class TestBuildDatasetUsd:
    """Tests for build_dataset_usd function."""

    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_single_file(self, mock_create_workflow, tmp_path):
        """Test USD dataset building for single file."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("usd_path: /path/to/model.usd")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "num_prims": 50,
                "num_images": 150,
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetUsdInput(config=config_file)
        result = build_dataset_usd(params)

        # Verify
        assert result.success is True
        assert result.dataset_path == tmp_path / "dataset.jsonl"
        assert result.num_prims == 50
        assert result.num_images == 150

    @patch("material_agent.batch_processor.process_usd_batch", new_callable=AsyncMock)
    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_batch(self, mock_create_workflow, mock_batch, tmp_path):
        """Test USD dataset building in batch mode."""
        # Setup
        config_file = tmp_path / "config.yaml"
        usd_dir = tmp_path / "usd_files"
        usd_dir.mkdir()
        config_file.write_text(f"usd_dir: {usd_dir}")

        # Mock workflow
        mock_workflow = Mock()
        mock_create_workflow.return_value = mock_workflow

        # Mock batch processor (async)
        mock_batch.return_value = {
            "results": {
                "model1.usd": {
                    "status": "success",
                    "num_prims": 30,
                    "num_images": 90,
                    "output_dir": str(tmp_path / "output/model1"),
                },
                "model2.usd": {
                    "status": "success",
                    "num_prims": 40,
                    "num_images": 120,
                    "output_dir": str(tmp_path / "output/model2"),
                },
            },
            "num_files_processed": 2,
            "num_files_failed": 0,
        }

        # Execute
        params = BuildDatasetUsdInput(config=config_file)
        result = build_dataset_usd(params)

        # Verify
        assert result.success is True
        assert result.batch_results is not None
        assert len(result.batch_results) == 2

    @patch("material_agent.workflows.create_usd_data_preparation_workflow_from_config")
    def test_build_usd_exception(self, mock_create_workflow, tmp_path):
        """Test USD dataset building when exception occurs."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("usd_path: /path/to/model.usd")

        # Mock workflow that raises exception
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError("Build failed"))
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetUsdInput(config=config_file)
        result = build_dataset_usd(params)

        # Verify
        assert result.success is False
        assert "Build failed" in result.error


# ============================================================================
# PDF VectorStore Building Tests
# ============================================================================


class TestBuildDatasetPdfVectorstoreInput:
    """Tests for BuildDatasetPdfVectorstoreInput validation."""

    def test_pdf_input_valid(self, tmp_path):
        """Test creating valid BuildDatasetPdfVectorstoreInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = BuildDatasetPdfVectorstoreInput(
            config=config_file,
            verbose=True,
        )

        assert params.config == config_file
        assert params.verbose is True


class TestBuildDatasetPdfVectorstore:
    """Tests for build_dataset_pdf_vectorstore function."""

    @patch(
        "material_agent.workflows.factory.create_pdf_vectorstore_workflow_from_config"
    )
    def test_build_pdf_vectorstore_success(self, mock_create_workflow, tmp_path):
        """Test PDF vectorstore building success."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "workflow_completed": True,
                "vectorstore_result": {
                    "save_path": str(tmp_path / "vectorstore"),
                    "num_documents_indexed": 100,
                    "num_texts": 80,
                    "num_images": 20,
                    "embedding_dimension": 768,
                },
                "extraction_result": {"document_count": 10},
                "split_result": {"total_files_created": 100},
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetPdfVectorstoreInput(config=config_file)
        result = build_dataset_pdf_vectorstore(params)

        # Verify
        assert result.success is True
        assert result.vectorstore_path == tmp_path / "vectorstore"
        assert result.num_documents_indexed == 100
        assert result.num_texts == 80
        assert result.num_images == 20

    @patch(
        "material_agent.workflows.factory.create_pdf_vectorstore_workflow_from_config"
    )
    def test_build_pdf_vectorstore_failed(self, mock_create_workflow, tmp_path):
        """Test PDF vectorstore building when workflow fails."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that fails
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "workflow_completed": False,
                "error": "Failed to build vectorstore",
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetPdfVectorstoreInput(config=config_file)
        result = build_dataset_pdf_vectorstore(params)

        # Verify
        assert result.success is False
        assert "Failed to build vectorstore" in result.error


# ============================================================================
# Prepare Dataset Tests
# ============================================================================


class TestBuildDatasetPrepareDatasetInput:
    """Tests for BuildDatasetPrepareDatasetInput validation."""

    def test_prepare_input_valid(self, tmp_path):
        """Test creating valid BuildDatasetPrepareDatasetInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = BuildDatasetPrepareDatasetInput(
            config=config_file,
            verbose=True,
        )

        assert params.config == config_file
        assert params.verbose is True


class TestBuildDatasetPrepareDataset:
    """Tests for build_dataset_prepare_dataset function."""

    @patch(
        "material_agent.workflows.factory.create_prepare_dataset_workflow_from_config"
    )
    def test_prepare_dataset_success(self, mock_create_workflow, tmp_path):
        """Test dataset preparation success."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "dataset_entries": [
                    {"id": "entry1", "specification": "spec1"},
                    {"id": "entry2", "specification": "spec2"},
                ],
                "failed_models": ["model3"],
                "dataset_jsonl_path": str(tmp_path / "dataset.jsonl"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetPrepareDatasetInput(config=config_file)
        result = build_dataset_prepare_dataset(params)

        # Verify
        assert result.success is True
        assert len(result.dataset_entries) == 2
        assert result.failed_models == ["model3"]
        assert result.dataset_jsonl_path == tmp_path / "dataset.jsonl"

    @patch(
        "material_agent.workflows.factory.create_prepare_dataset_workflow_from_config"
    )
    def test_prepare_dataset_not_complete(self, mock_create_workflow, tmp_path):
        """Test dataset preparation when workflow doesn't complete."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that doesn't complete
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value={"dataset_entries": None})
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BuildDatasetPrepareDatasetInput(config=config_file)
        result = build_dataset_prepare_dataset(params)

        # Verify
        assert result.success is False
        assert "did not complete" in result.error.lower()


# ============================================================================
# Output Tests
# ============================================================================


class TestBuildDatasetOutputs:
    """Tests for build dataset output dataclasses."""

    def test_usd_output_success(self, tmp_path):
        """Test creating successful BuildDatasetUsdOutput."""
        output = BuildDatasetUsdOutput(
            success=True,
            dataset_path=tmp_path / "dataset.jsonl",
            num_prims=100,
            num_images=300,
        )

        assert output.success is True
        assert output.dataset_path == tmp_path / "dataset.jsonl"
        assert output.num_prims == 100

    def test_pdf_output_success(self, tmp_path):
        """Test creating successful BuildDatasetPdfVectorstoreOutput."""
        output = BuildDatasetPdfVectorstoreOutput(
            success=True,
            vectorstore_path=tmp_path / "vectorstore",
            num_documents_indexed=50,
            num_texts=40,
            num_images=10,
        )

        assert output.success is True
        assert output.vectorstore_path == tmp_path / "vectorstore"
        assert output.num_documents_indexed == 50

    def test_prepare_output_success(self, tmp_path):
        """Test creating successful BuildDatasetPrepareDatasetOutput."""
        output = BuildDatasetPrepareDatasetOutput(
            success=True,
            dataset_jsonl_path=tmp_path / "dataset.jsonl",
            dataset_entries=[{"id": "1"}, {"id": "2"}],
            failed_models=["model3"],
        )

        assert output.success is True
        assert len(output.dataset_entries) == 2
        assert output.failed_models == ["model3"]
