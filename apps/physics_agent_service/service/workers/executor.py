# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution using Physics Agent Python async API.

Calls arun_pipeline directly - no wrappers or thread pools needed!
"""

import logging
from datetime import UTC, datetime

from physics_agent.api import PipelineInput, arun_pipeline

from ..events.listener import FastAPIEventListener
from ..runtime import get_event_bus
from ..runtime.events import ProgressEvent, StepState

logger = logging.getLogger(__name__)


async def execute_pipeline_async(
    session_id: str,
    config_dict: dict,
    session_manager,
    only_steps: list[str] | None = None,
) -> None:
    """Execute pipeline workflow using Physics Agent Python async API."""
    logger.info(f"Pipeline execution started for {session_id[:8]}...")

    session_dir = session_manager.get_session_dir(session_id)

    listener = FastAPIEventListener(
        session_id,
        session_dir,
    )

    result = await arun_pipeline(
        PipelineInput(
            config=config_dict,
            event_listener=listener,
            only_steps=only_steps or [],
            verbose=False,
        )
    )

    if not result.success:
        raise RuntimeError(f"Pipeline failed: {result.error}")

    logger.info(f"Pipeline completed_steps: {result.completed_steps}")
    if result.step_results:
        for step, data in result.step_results.items():
            logger.info(f"  {step}: {data}")

    stats = _extract_stats_from_result(result, session_dir)
    logger.info(f"Pipeline stats for {session_id[:8]}: {stats}")

    metadata = await session_manager.get_session_metadata(session_id)
    duration_seconds = 0
    if metadata and metadata.get("created_at"):
        created_at = datetime.fromisoformat(metadata["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    await session_manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": stats,
            "duration_seconds": duration_seconds,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )

    # Sync key artifacts to store (uploads to S3 if configured).
    # Only sync the result files — skip rendered images (can be thousands of PNGs)
    # which are too large to upload reliably and not needed cross-instance.
    synced = 0
    for prefix in (
        "cache/predictions/",
        "cache/dataset/dataset.jsonl",
        "cache/physics/",
    ):
        try:
            n = await session_manager.sync_to_store(session_id, prefix=prefix)
            synced += n
        except Exception as e:
            logger.warning(
                f"Failed to sync {prefix} to store for {session_id[:8]}: {e}"
            )
    if synced > 0:
        logger.info(f"Synced {synced} artifact file(s) to store for {session_id[:8]}")

    # Signal SSE clients that artifacts are now in the store and the pipeline is fully done.
    # This fires AFTER update_session + sync_to_store so clients get "done" only when
    # status and artifacts are already available in S3.
    # Guard: only emit if this instance built up a snapshot (i.e., was the executing instance).
    # Avoids creating a stale empty snapshot on cross-instance calls or in tests.
    event_bus = get_event_bus()
    if event_bus.get_snapshot(session_id) is not None:
        await event_bus.emit(
            ProgressEvent(
                session_id=session_id,
                step="pipeline",
                state=StepState.COMPLETED,
                percent=100,
                message="Pipeline artifacts synced and ready",
                extra={"pipeline_ready": True},
            )
        )

    logger.info(f"Pipeline execution completed for {session_id[:8]}")


def _extract_stats_from_result(result, session_dir=None) -> dict:
    """Extract statistics from pipeline result."""
    stats = {
        "prims_processed": 0,
        "images_generated": 0,
        "predictions_made": 0,
    }

    step_results = result.step_results or {}

    if "predict" in step_results:
        stats["predictions_made"] = step_results["predict"].get("predictions_count", 0)

    raw_result = result.raw_result or {}

    if "build_dataset_usd_result" in raw_result:
        usd_result = raw_result["build_dataset_usd_result"]
        stats["prims_processed"] = usd_result.get("num_prims", 0)
        stats["images_generated"] = usd_result.get("num_images", 0)

    if (
        stats["prims_processed"] == 0
        and "build_dataset_prepare_dataset_result" in raw_result
    ):
        prepare_result = raw_result["build_dataset_prepare_dataset_result"]
        stats["prims_processed"] = prepare_result.get("num_entries", 0)

    if stats["prims_processed"] == 0:
        dataset_info = raw_result.get("dataset_info", {})
        stats["prims_processed"] = dataset_info.get("num_entries", 0)

    if session_dir and (
        stats["prims_processed"] == 0 or stats["predictions_made"] == 0
    ):
        stats = _count_stats_from_files(session_dir, stats)

    return stats


def _count_stats_from_files(session_dir, stats: dict) -> dict:
    """Count stats from actual files in session directory."""
    from pathlib import Path

    session_path = Path(session_dir)

    if stats["prims_processed"] == 0:
        dataset_file = session_path / "cache" / "dataset" / "dataset.jsonl"
        if dataset_file.exists():
            try:
                with open(dataset_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["prims_processed"] = len(lines)
            except Exception as e:
                logger.warning(f"Failed to count dataset entries: {e}")

    if stats["images_generated"] == 0:
        dataset_dir = session_path / "cache" / "dataset"
        if dataset_dir.exists():
            try:
                image_count = len(list(dataset_dir.glob("**/*.png")))
                stats["images_generated"] = image_count
            except Exception as e:
                logger.warning(f"Failed to count images: {e}")

    if stats["predictions_made"] == 0:
        predictions_file = session_path / "cache" / "predictions" / "predictions.jsonl"
        if predictions_file.exists():
            try:
                with open(predictions_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["predictions_made"] = len(lines)
            except Exception as e:
                logger.warning(f"Failed to count predictions: {e}")

    return stats
