# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Dataset APIs for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from material_agent.api.types import APIResult

logger = logging.getLogger(__name__)


# ============================================================================
# USD Dataset Building API
# ============================================================================


@dataclass
class BuildDatasetUsdInput:
    """Input parameters for USD dataset building API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        source_override: Optional path to USD file or directory (overrides config)
        output_dir_override: Optional output directory (overrides config)
        extract_metadata: Extract prim metadata
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    source_override: Path | None = None  # Can be file or directory
    output_dir_override: Path | None = None
    extract_metadata: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        # Handle config as either Path or dict
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not self.config.exists():
                raise FileNotFoundError(f"Config file not found: {self.config}")

        if self.source_override:
            self.source_override = Path(self.source_override)

        if self.output_dir_override:
            self.output_dir_override = Path(self.output_dir_override)


@dataclass
class BuildDatasetUsdOutput(APIResult):
    """Output results from USD dataset building API."""

    dataset_path: Path | None = None
    num_prims: int = 0
    num_images: int = 0
    batch_results: dict[str, dict[str, Any]] | None = None  # For batch processing
    raw_result: dict[str, Any] | None = None


async def abuild_dataset_usd(params: BuildDatasetUsdInput) -> BuildDatasetUsdOutput:
    """Build a dataset from USD file(s) by rendering views of each prim.

    This command will intelligently handle both single file and batch processing:
    - If config has 'usd_path': processes a single USD file
    - If config has 'usd_dir': processes all USD files in that directory

    For batch processing, subdirectories will be created for each USD file.

    Args:
        params: USD dataset building input parameters

    Returns:
        BuildDatasetUsdOutput with results or error information
    """
    import yaml

    logger.info("Starting USD dataset building via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info(f"Configuration file: {params.config}")

    try:
        # Load config - either from file or use provided dict
        if isinstance(params.config, dict):
            config_data = params.config
        else:
            with open(params.config, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

        # Determine if source override points to a directory or file
        is_batch_mode = False
        if params.source_override:
            if params.source_override.is_dir():
                is_batch_mode = True
        elif "usd_dir" in config_data:
            is_batch_mode = True
        elif "usd_path" not in config_data:
            raise ValueError(
                "Configuration must contain either 'usd_path' (for single file) "
                "or 'usd_dir' (for batch processing)"
            )

        if is_batch_mode:
            return await _build_dataset_usd_batch(params, config_data)
        else:
            return await _build_dataset_usd_single(params)

    except Exception as e:
        logger.error(f"Error building USD dataset: {str(e)}", exc_info=True)
        return BuildDatasetUsdOutput(
            success=False,
            error=str(e),
        )


async def _build_dataset_usd_single(
    params: BuildDatasetUsdInput,
) -> BuildDatasetUsdOutput:
    """Build dataset from a single USD file."""
    from material_agent.workflows import (
        create_usd_data_preparation_workflow_from_config,
    )

    logger.info("Processing single USD file")

    workflow = create_usd_data_preparation_workflow_from_config()

    initial_context: dict[str, Any] = {}

    # Add config as either path or dict
    if isinstance(params.config, dict):
        initial_context["config_dict"] = params.config
    else:
        initial_context["config_path"] = params.config

    if params.source_override:
        initial_context["source_override"] = params.source_override
        logger.info(f"Using USD source override: {params.source_override}")

    if params.output_dir_override:
        initial_context["output_dir_override"] = params.output_dir_override
        logger.info(f"Using output directory override: {params.output_dir_override}")

    if params.extract_metadata:
        initial_context["extract_prim_metadata"] = params.extract_metadata
        logger.info("Metadata extraction enabled")

    # Run workflow
    logger.info("Executing dataset build workflow")
    result = await workflow.arun(initial_context)

    return BuildDatasetUsdOutput(
        success=True,
        dataset_path=(
            Path(result["dataset_path"]) if result.get("dataset_path") else None
        ),
        num_prims=result.get("num_prims", 0),
        num_images=result.get("num_images", 0),
        raw_result=result,
    )


async def _build_dataset_usd_batch(
    params: BuildDatasetUsdInput, config_data: dict[str, Any]
) -> BuildDatasetUsdOutput:
    """Build datasets from multiple USD files in a directory."""
    from material_agent.batch_processor import process_usd_batch
    from material_agent.workflows import (
        create_usd_data_preparation_workflow_from_config,
    )

    logger.info("Detected batch processing mode")

    # Get USD directory
    if params.source_override and params.source_override.is_dir():
        usd_dir = params.source_override
        logger.info(f"Using USD directory override: {usd_dir}")
    elif "usd_dir" in config_data:
        # For file-based config, resolve relative to config file
        # For dict-based config, use as-is (must be absolute or relative to cwd)
        if isinstance(params.config, Path):
            config_dir = params.config.parent
            usd_dir = config_dir / Path(config_data["usd_dir"])
            usd_dir = usd_dir.resolve()
        else:
            usd_dir = Path(config_data["usd_dir"])
        logger.info(f"Using usd_dir from config: {usd_dir}")
    else:
        raise ValueError("Batch mode requires usd_dir in config or --source directory")

    # Get output directory
    if params.output_dir_override:
        batch_output_dir = params.output_dir_override
    elif "output_dir" in config_data:
        # For file-based config, resolve relative to config file
        # For dict-based config, use as-is (must be absolute or relative to cwd)
        if isinstance(params.config, Path):
            config_dir = params.config.parent
            batch_output_dir = config_dir / Path(config_data["output_dir"])
            batch_output_dir = batch_output_dir.resolve()
        else:
            batch_output_dir = Path(config_data["output_dir"])
    else:
        batch_output_dir = Path("output")

    # Check if USD directory exists
    if not usd_dir.exists():
        raise FileNotFoundError(f"USD directory not found: {usd_dir}")

    # Create workflow once
    workflow = create_usd_data_preparation_workflow_from_config()

    # Prepare base context
    base_context: dict[str, Any] = {}

    # Add config as either path or dict
    if isinstance(params.config, dict):
        base_context["config_dict"] = params.config
    else:
        base_context["config_path"] = params.config

    if params.extract_metadata:
        base_context["extract_prim_metadata"] = params.extract_metadata

    # Run batch processor
    batch_result = await process_usd_batch(
        usd_dir=usd_dir,
        batch_output_dir=batch_output_dir,
        workflow_runner=lambda ctx: workflow.arun(ctx),
        base_context=base_context,
    )

    results = batch_result["results"]
    successful_builds = batch_result["num_files_processed"]
    failed_builds = batch_result["num_files_failed"]

    logger.info(
        f"Batch processing complete: {successful_builds} successful, "
        f"{failed_builds} failed"
    )

    return BuildDatasetUsdOutput(
        success=failed_builds == 0,
        batch_results=results,
        raw_result=batch_result,
    )


# ============================================================================
# PDF VectorStore Building API
# ============================================================================


@dataclass
class BuildDatasetPdfVectorstoreInput:
    """Input parameters for PDF vectorstore building API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        source_override: Optional path to PDF file or directory (overrides config)
        output_dir_override: Optional output directory (overrides config)
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    source_override: Path | None = None  # PDF file or directory
    output_dir_override: Path | None = None
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        # Handle config as either Path or dict
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not self.config.exists():
                raise FileNotFoundError(f"Config file not found: {self.config}")

        if self.source_override:
            self.source_override = Path(self.source_override)

        if self.output_dir_override:
            self.output_dir_override = Path(self.output_dir_override)


@dataclass
class BuildDatasetPdfVectorstoreOutput(APIResult):
    """Output results from PDF vectorstore building API."""

    vectorstore_path: Path | None = None
    num_documents_indexed: int = 0
    num_texts: int = 0
    num_images: int = 0
    embedding_dimension: int = 0
    extraction_result: dict[str, Any] | None = None
    split_result: dict[str, Any] | None = None
    raw_result: dict[str, Any] | None = None


async def abuild_dataset_pdf_vectorstore(
    params: BuildDatasetPdfVectorstoreInput,
) -> BuildDatasetPdfVectorstoreOutput:
    """Build a multimodal vector store from PDF documents.

    This command processes PDF files to extract content (text, images, tables),
    splits them by type, and creates a searchable vector store.

    Args:
        params: PDF vectorstore building input parameters

    Returns:
        BuildDatasetPdfVectorstoreOutput with results or error information
    """
    logger.info("Starting PDF vectorstore building via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info(f"Configuration file: {params.config}")

    if params.source_override:
        logger.info(f"Source override: {params.source_override}")
    if params.output_dir_override:
        logger.info(f"Output directory override: {params.output_dir_override}")

    try:
        # Import workflow factory
        from material_agent.workflows.factory import (
            create_pdf_vectorstore_workflow_from_config,
        )

        # Prepare initial context with config and overrides
        initial_context: dict[str, Any] = {
            "source_override": (
                str(params.source_override) if params.source_override else None
            ),
            "output_dir_override": (
                str(params.output_dir_override) if params.output_dir_override else None
            ),
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
        else:
            initial_context["config_path"] = str(params.config)

        # Create workflow
        workflow = create_pdf_vectorstore_workflow_from_config()

        # Run the workflow
        logger.info("Processing PDFs and building vector store...")
        result = await workflow.arun(initial_context=initial_context)

        # Check if workflow completed successfully
        if result.get("workflow_completed"):
            logger.info("PDF vectorstore workflow completed successfully")

            vectorstore_result = result.get("vectorstore_result", {})

            return BuildDatasetPdfVectorstoreOutput(
                success=True,
                vectorstore_path=(
                    Path(vectorstore_result["save_path"])
                    if vectorstore_result.get("save_path")
                    else None
                ),
                num_documents_indexed=vectorstore_result.get(
                    "num_documents_indexed", 0
                ),
                num_texts=vectorstore_result.get("num_texts", 0),
                num_images=vectorstore_result.get("num_images", 0),
                embedding_dimension=vectorstore_result.get("embedding_dimension", 0),
                extraction_result=result.get("extraction_result"),
                split_result=result.get("split_result"),
                raw_result=result,
            )
        else:
            error_msg = result.get("error", "PDF vectorstore workflow failed")
            logger.error(error_msg)
            return BuildDatasetPdfVectorstoreOutput(
                success=False,
                error=error_msg,
            )

    except Exception as e:
        logger.error(f"Error building PDF vectorstore: {str(e)}", exc_info=True)
        return BuildDatasetPdfVectorstoreOutput(
            success=False,
            error=str(e),
        )


# ============================================================================
# Prepare Dataset API
# ============================================================================


@dataclass
class BuildDatasetPrepareDatasetInput:
    """Input parameters for prepare dataset API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        vector_store_override: Optional path to vector store (overrides config)
        dataset_override: Optional path to dataset directory (overrides config)
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    vector_store_override: Path | None = None
    dataset_override: Path | None = None
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        # Handle config as either Path or dict
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not self.config.exists():
                raise FileNotFoundError(f"Config file not found: {self.config}")

        if self.vector_store_override:
            self.vector_store_override = Path(self.vector_store_override)

        if self.dataset_override:
            self.dataset_override = Path(self.dataset_override)


@dataclass
class BuildDatasetPrepareDatasetOutput(APIResult):
    """Output results from prepare dataset API."""

    dataset_jsonl_path: Path | None = None
    dataset_entries: list[dict[str, Any]] = field(default_factory=list)
    failed_models: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


async def abuild_dataset_prepare_dataset(
    params: BuildDatasetPrepareDatasetInput,
) -> BuildDatasetPrepareDatasetOutput:
    """Prepare dataset with CMF specifications for benchmark or prediction.

    This command prepares datasets by extracting CMF specifications
    for model numbers using the spec_rag functionality. Can prepare either
    benchmark datasets (with ground truth) or prediction datasets (without
    ground truth).

    Args:
        params: Prepare dataset input parameters

    Returns:
        BuildDatasetPrepareDatasetOutput with results or error information
    """
    logger.info("Starting prepare dataset via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info(f"Configuration file: {params.config}")

    if params.vector_store_override:
        logger.info(f"Vector store override: {params.vector_store_override}")
    if params.dataset_override:
        logger.info(f"Dataset override: {params.dataset_override}")

    try:
        # Import workflow factory
        from material_agent.workflows.factory import (
            create_prepare_dataset_workflow_from_config,
        )

        # Create config-driven workflow
        logger.info("Creating prepare dataset workflow...")
        workflow = create_prepare_dataset_workflow_from_config()

        # Prepare initial context with config and overrides
        initial_context: dict[str, Any] = {
            "vector_store_override": (
                str(params.vector_store_override)
                if params.vector_store_override
                else None
            ),
            "dataset_override": (
                str(params.dataset_override) if params.dataset_override else None
            ),
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
        else:
            initial_context["config_path"] = str(params.config)

        # Run the workflow
        logger.info("Running prepare dataset workflow...")
        result = await workflow.arun(initial_context=initial_context)

        # Check if workflow completed successfully
        if result.get("dataset_entries") is not None:
            dataset_entries = result.get("dataset_entries", [])
            failed_models = result.get("failed_models", [])
            dataset_jsonl_path = result.get("dataset_jsonl_path")

            logger.info(
                f"Dataset preparation completed: {len(dataset_entries)} entries, "
                f"{len(failed_models)} failed"
            )

            return BuildDatasetPrepareDatasetOutput(
                success=True,
                dataset_jsonl_path=(
                    Path(dataset_jsonl_path) if dataset_jsonl_path else None
                ),
                dataset_entries=dataset_entries,
                failed_models=failed_models,
                raw_result=result,
            )
        else:
            error_msg = "Prepare dataset workflow did not complete successfully"
            logger.error(error_msg)
            return BuildDatasetPrepareDatasetOutput(
                success=False,
                error=error_msg,
            )

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}", exc_info=True)
        return BuildDatasetPrepareDatasetOutput(
            success=False,
            error=str(e),
        )


def build_dataset_usd(params: BuildDatasetUsdInput) -> BuildDatasetUsdOutput:
    """Build dataset from USD files synchronously."""
    return asyncio.run(abuild_dataset_usd(params))


def build_dataset_pdf_vectorstore(
    params: BuildDatasetPdfVectorstoreInput,
) -> BuildDatasetPdfVectorstoreOutput:
    """Build PDF vector store synchronously."""
    return asyncio.run(abuild_dataset_pdf_vectorstore(params))


def build_dataset_prepare_dataset(
    params: BuildDatasetPrepareDatasetInput,
) -> BuildDatasetPrepareDatasetOutput:
    """Prepare dataset synchronously."""
    return asyncio.run(abuild_dataset_prepare_dataset(params))
