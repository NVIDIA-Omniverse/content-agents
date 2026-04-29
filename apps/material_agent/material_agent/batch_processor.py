# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch processing utilities for material agent workflows."""

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def process_usd_batch(
    usd_dir: Path,
    batch_output_dir: Path,
    workflow_runner: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    base_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process multiple USD files in batch mode asynchronously.

    This utility handles the common pattern of:
    1. Finding all USD files in a directory
    2. Running a workflow for each file asynchronously
    3. Tracking success/failure
    4. Aggregating results

    Args:
        usd_dir: Directory containing USD files
        batch_output_dir: Base output directory for results
        workflow_runner: Async callable that takes context dict and returns result dict
        base_context: Optional base context to merge with per-file context

    Returns:
        Dictionary with batch processing results:
        - output_dir: Base output directory
        - num_files_processed: Number of successfully processed files
        - num_files_failed: Number of failed files
        - total_files: Total number of files found
        - results: Dictionary mapping filename to result details

    Raises:
        RuntimeError: If no USD files found or if all files fail to process
    """
    if not usd_dir.exists():
        raise RuntimeError(f"USD directory not found: {usd_dir}")

    # Find all USD files recursively
    usd_files = (
        list(usd_dir.rglob("*.usd"))
        + list(usd_dir.rglob("*.usda"))
        + list(usd_dir.rglob("*.usdc"))
    )

    if not usd_files:
        raise RuntimeError(f"No USD files found in directory: {usd_dir}")

    logger.info(f"Found {len(usd_files)} USD files to process")
    logger.info(f"  USD directory: {usd_dir}")
    logger.info(f"  Output directory: {batch_output_dir}")

    # Create base output directory
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each USD file
    successful = 0
    failed = 0
    results = {}
    base_context = base_context or {}

    for usd_file in usd_files:
        usd_name = usd_file.stem
        dataset_output_dir = batch_output_dir / usd_name

        logger.info(f"  Processing {usd_file.name} -> {dataset_output_dir}")

        try:
            # Prepare context for this specific file
            file_context = dict(base_context)  # Copy base context
            file_context["source_override"] = usd_file
            file_context["output_dir_override"] = dataset_output_dir

            # Run workflow for this file
            result = await workflow_runner(file_context)

            # Check result
            if not result or "error" in result:
                logger.warning(f"  ✗ Failed to process {usd_file.name}")
                results[usd_name] = {
                    "status": "failed",
                    "usd_file": str(usd_file),
                    "output_dir": str(dataset_output_dir),
                    "error": (
                        result.get("error", "Unknown error")
                        if result
                        else "Empty result"
                    ),
                }
                failed += 1
            else:
                logger.info(f"  ✓ Successfully processed {usd_file.name}")
                results[usd_name] = {
                    "status": "success",
                    "usd_file": str(usd_file),
                    "output_dir": str(dataset_output_dir),
                    "dataset_path": result.get("dataset_path", "N/A"),
                    "num_prims": result.get("num_prims", 0),
                    "num_images": result.get("num_images", 0),
                }
                successful += 1

        except Exception as e:
            logger.error(f"  ✗ Error processing {usd_file.name}: {e}")
            results[usd_name] = {
                "status": "failed",
                "usd_file": str(usd_file),
                "output_dir": str(dataset_output_dir),
                "error": str(e),
            }
            failed += 1

    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

    if failed > 0 and successful == 0:
        raise RuntimeError(f"All {failed} USD files failed to process")

    # Return aggregated results
    return {
        "output_dir": str(batch_output_dir),
        "num_files_processed": successful,
        "num_files_failed": failed,
        "total_files": len(usd_files),
        "results": results,
    }
