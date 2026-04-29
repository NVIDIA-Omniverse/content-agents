# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent workflow factory."""

import pytest
from world_understanding.agentic.workflows import Workflow

from material_agent.workflows.factory import (
    create_benchmark_workflow_from_config,
    create_evaluation_workflow_from_config,
    create_identify_asset_workflow_from_config,
    create_pdf_vectorstore_workflow_from_config,
    create_prediction_workflow_from_config,
)


class TestWorkflowFactory:
    """Tests for workflow factory functions."""

    def test_create_prediction_workflow_from_config(self):
        """Test creating config-driven prediction workflow."""
        workflow = create_prediction_workflow_from_config()

        assert isinstance(workflow, Workflow)
        assert workflow.name == "Config-Driven Prediction"
        assert (
            workflow.description
            == "Config-driven prediction workflow without evaluation"
        )
        assert len(workflow.tasks) == 6  # Config, Provision, Load, Infer, Report, Save

        # Verify task order
        task_names = [task.name for task in workflow.tasks]
        expected_tasks = [
            "PredictConfigLoading",
            "ModelProvisioning",
            "DatasetLoading",
            "VLMInference",
            "GeneratePredictionReport",
            "SavePredictions",
        ]
        assert task_names == expected_tasks

    def test_create_evaluation_workflow_from_config(self):
        """Test creating config-driven evaluation workflow."""
        workflow = create_evaluation_workflow_from_config()

        assert isinstance(workflow, Workflow)
        assert workflow.name == "Config-Driven Evaluation"
        assert len(workflow.tasks) == 4  # Config, Provision, Evaluate, Report

        # Verify task order
        task_names = [task.name for task in workflow.tasks]
        expected_tasks = [
            "EvaluateConfigLoading",
            "ModelProvisioning",
            "Evaluation",
            "GenerateEvaluationReport",
        ]
        assert task_names == expected_tasks

    def test_create_benchmark_workflow_from_config(self):
        """Test creating config-driven benchmark workflow."""
        workflow = create_benchmark_workflow_from_config()

        assert isinstance(workflow, Workflow)
        assert workflow.name == "Config-Driven Benchmark"
        assert workflow.description == "Flat configuration-driven benchmark workflow"
        assert (
            len(workflow.tasks) == 8
        )  # Config, Provision, Load, Infer, Report1, Save, Eval, Report2

        # Verify task order
        task_names = [task.name for task in workflow.tasks]
        expected_tasks = [
            "BenchmarkConfigLoading",
            "ModelProvisioning",
            "DatasetLoading",
            "VLMInference",
            "GeneratePredictionReport",
            "SavePredictions",
            "Evaluation",
            "GenerateEvaluationReport",
        ]
        assert task_names == expected_tasks

    def test_create_pdf_vectorstore_workflow_from_config(self):
        """Test creating PDF to vectorstore workflow."""
        workflow = create_pdf_vectorstore_workflow_from_config()

        assert isinstance(workflow, Workflow)
        assert workflow.name == "Config-Driven PDF to VectorStore"
        assert (
            workflow.description
            == "Convert PDF documents to a searchable multimodal vector store"
        )
        assert len(workflow.tasks) == 4  # Config, Extract, Split, Build

        # Verify task types - first task is PDFVectorstoreConfigTask
        from material_agent.tasks import PDFVectorstoreConfigTask

        assert isinstance(workflow.tasks[0], PDFVectorstoreConfigTask)

        # Verify ToolTasks have correct tool names
        from world_understanding.agentic.tasks import ToolTask

        assert isinstance(workflow.tasks[1], ToolTask)
        assert workflow.tasks[1].tool_name == "extract_document_content"
        assert workflow.tasks[1].name == "Extract PDF Content"

        assert isinstance(workflow.tasks[2], ToolTask)
        assert workflow.tasks[2].tool_name == "split_document_content"
        assert workflow.tasks[2].name == "Split Content by Type"

        assert isinstance(workflow.tasks[3], ToolTask)
        assert workflow.tasks[3].tool_name == "build_multimodal_vector_store"
        assert workflow.tasks[3].name == "Build Vector Store"

    def test_create_identify_asset_workflow_from_config(self):
        """Test creating asset identification workflow."""
        workflow = create_identify_asset_workflow_from_config()

        assert isinstance(workflow, Workflow)
        assert workflow.name == "Asset Identification"
        assert (
            workflow.description
            == "Identify asset type and description from preview images"
        )
        assert len(workflow.tasks) == 1

        # Verify task type
        from world_understanding.agentic.usd_tasks import IdentifyAssetTask

        assert isinstance(workflow.tasks[0], IdentifyAssetTask)

    def test_workflow_object_stores(self):
        """Test that workflows use appropriate object stores."""
        from world_understanding.utils.object_store import (
            InMemoryObjectStore,
            TempDirObjectStore,
        )

        # Prediction uses TempDirObjectStore for image handling
        prediction_workflow = create_prediction_workflow_from_config()
        assert isinstance(prediction_workflow.object_store, TempDirObjectStore)

        # Evaluation uses InMemoryObjectStore (smaller data)
        evaluation_workflow = create_evaluation_workflow_from_config()
        assert isinstance(evaluation_workflow.object_store, InMemoryObjectStore)

        # Benchmark uses TempDirObjectStore for image handling
        benchmark_workflow = create_benchmark_workflow_from_config()
        assert isinstance(benchmark_workflow.object_store, TempDirObjectStore)

        # PDF workflow uses TempDirObjectStore for large files
        pdf_workflow = create_pdf_vectorstore_workflow_from_config()
        assert isinstance(pdf_workflow.object_store, TempDirObjectStore)
