# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution using MAA Python async API.

Calls arun_pipeline directly - no wrappers or thread pools needed!
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from material_agent.api import PipelineInput, arun_pipeline
from world_understanding.telemetry import get_current_span, traced
from world_understanding.telemetry.attributes import MAAttributes

from ..events.listener import FastAPIEventListener
from ..events.telemetry_listener import TelemetryEventListener
from ..runtime.bus import get_event_bus
from ..runtime.events import ProgressEvent, StepState
from ..session.manager import SessionManager

logger = logging.getLogger(__name__)


@traced("maa.pipeline.execution")
async def execute_pipeline_async(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: SessionManager,
    user_email: str = "",
) -> None:
    """Execute pipeline workflow using MAA Python async API.

    Handles cancellation and errors by updating session metadata so
    sessions never get stuck in "running" state.

    Args:
        session_id: Session identifier
        config_dict: Complete unified pipeline config dict (built in router)
        session_manager: SessionManager instance
        user_email: User email address for telemetry
    """
    try:
        await _execute_pipeline_inner(
            session_id, config_dict, session_manager, user_email
        )
    except asyncio.CancelledError:
        logger.info(f"Pipeline cancelled for {session_id[:8]}")
        await session_manager.update_session(
            session_id,
            {
                "status": "cancelled",
                "cancelled_at": datetime.now(UTC).isoformat(),
            },
        )
        raise  # Re-raise so JobRegistry cleanup runs
    except Exception as e:
        logger.error(f"Pipeline failed for {session_id[:8]}: {e}")
        await session_manager.update_session(
            session_id,
            {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now(UTC).isoformat(),
            },
        )
        raise


async def _execute_pipeline_inner(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: SessionManager,
    user_email: str = "",
) -> None:
    """Inner pipeline execution logic.

    Args:
        session_id: Session identifier
        config_dict: Complete unified pipeline config dict (built in router)
        session_manager: SessionManager instance
        user_email: User email address for telemetry
    """
    logger.info(f"Pipeline execution started for {session_id[:8]}...")

    # Get session directory for thumbnail creation
    session_dir = session_manager.get_session_dir(session_id)

    # Extract session-specific material icons from config
    # Materials are stored under config["materials"]["entries"]
    session_material_icons: dict[str, str] = {}
    materials_entries = config_dict.get("materials", {}).get("entries", [])
    for entry in materials_entries:
        if "name" in entry and "icon" in entry:
            session_material_icons[entry["name"]] = entry["icon"]

    logger.info(
        f"Loaded {len(session_material_icons)} material icons from config "
        f"(materials_entries count: {len(materials_entries)})"
    )
    if session_material_icons:
        # Log first 3 material icon mappings for debugging
        sample = list(session_material_icons.items())[:3]
        logger.info(f"Sample material icons: {sample}")

    # Create event listener with telemetry wrapper for per-step timing
    inner_listener = FastAPIEventListener(
        session_id,
        session_dir,
        session_material_icons=session_material_icons,
    )
    telemetry_listener = TelemetryEventListener(inner_listener)

    # Read asset metadata from session for telemetry
    metadata = await session_manager.get_session_metadata(session_id)
    asset_info = metadata.get("asset", {}) if metadata else {}

    vlm_model = (
        config_dict.get("steps", {}).get("predict", {}).get("vlm", {}).get("model", "")
    )

    # Set pipeline attributes on the root span from @traced decorator
    span = get_current_span()
    if span:
        span.set_attribute(MAAttributes.PIPELINE_SESSION_ID, session_id)
        # User email is sent only to our self-hosted Langfuse instance for
        # internal per-user pipeline tracing. No external data collection.
        span.set_attribute(MAAttributes.PIPELINE_USER_EMAIL, user_email)
        # Langfuse-specific: maps to native user/session fields for dashboard filtering
        span.set_attribute(MAAttributes.LANGFUSE_USER_ID, user_email)
        span.set_attribute(MAAttributes.LANGFUSE_SESSION_ID, session_id)
        if vlm_model:
            span.set_attribute(MAAttributes.PIPELINE_VLM_MODEL, vlm_model)
            span.set_attribute(MAAttributes.LANGFUSE_META_VLM_MODEL, vlm_model)
        if asset_info:
            if asset_info.get("filename"):
                span.set_attribute(MAAttributes.ASSET_FILENAME, asset_info["filename"])
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILENAME,
                    asset_info["filename"],
                )
            if asset_info.get("file_size_bytes"):
                span.set_attribute(
                    MAAttributes.ASSET_FILE_SIZE_BYTES,
                    asset_info["file_size_bytes"],
                )
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILE_SIZE,
                    asset_info["file_size_bytes"],
                )
            if asset_info.get("file_extension"):
                span.set_attribute(
                    MAAttributes.ASSET_FILE_EXTENSION,
                    asset_info["file_extension"],
                )
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILE_EXT,
                    asset_info["file_extension"],
                )

    # Call async API directly - no wrapper or thread pool needed!
    result = await arun_pipeline(
        PipelineInput(
            config=config_dict,
            event_listener=telemetry_listener,
            verbose=False,
        )
    )

    if not result.success:
        # Emit step spans before raising so they appear as children
        _emit_step_spans(
            session_id=session_id,
            step_timings=telemetry_listener.get_step_timings(),
        )
        # Set failure status on root span
        if span:
            span.set_attribute(MAAttributes.PIPELINE_STATUS, "failed")
        raise RuntimeError(f"Pipeline failed: {result.error}")

    # Extract stats from step results and save to session metadata
    # Debug: log available data
    logger.info(f"Pipeline completed_steps: {result.completed_steps}")
    logger.info(
        f"Pipeline step_results keys: {list(result.step_results.keys()) if result.step_results else 'None'}"
    )
    if result.step_results:
        for step, data in result.step_results.items():
            logger.info(f"  {step}: {data}")
    if result.raw_result:
        logger.info(f"Pipeline raw_result keys: {list(result.raw_result.keys())}")

    stats = _extract_stats_from_result(result, session_dir)
    logger.info(f"Pipeline stats for {session_id[:8]}: {stats}")

    # Calculate duration
    duration_seconds = 0
    if metadata and metadata.get("created_at"):
        created_at = datetime.fromisoformat(metadata["created_at"])
        duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    # Build step_timings dict for persistence
    step_timings_dict = {}
    for t in telemetry_listener.get_step_timings():
        duration_ns = t["completed_at_ns"] - t["started_at_ns"]
        step_timings_dict[t["name"]] = duration_ns / 1_000_000_000

    # Save results to session metadata (include status to ensure atomicity
    # with results - prevents race where EventBus sets status="completed"
    # before stats are persisted)
    await session_manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": stats,
            "duration_seconds": duration_seconds,
            "step_timings": step_timings_dict,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )

    # Force the in-memory progress snapshot to terminal state after metadata is
    # persisted. Otherwise /status can prefer a stale running EventBus snapshot
    # over the completed session metadata.
    event_bus = get_event_bus()
    if event_bus.get_snapshot(session_id) is not None:
        try:
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step="pipeline",
                    state=StepState.COMPLETED,
                    percent=100,
                    message="Pipeline completed successfully",
                    extra={"pipeline_completed": True, **stats},
                )
            )
        except Exception:
            logger.exception(
                "Failed to emit completion event for session %s; "
                "metadata remains completed",
                session_id,
            )

    # Set completion attributes on root span
    if span:
        span.set_attribute(MAAttributes.PIPELINE_STATUS, "completed")
        span.set_attribute(MAAttributes.PIPELINE_DURATION_SECONDS, duration_seconds)
        span.set_attribute(
            MAAttributes.PIPELINE_PRIM_COUNT,
            stats.get("original_prim_count", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_PRIMS_PROCESSED,
            stats.get("prims_processed", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_IMAGES_GENERATED,
            stats.get("images_generated", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_PREDICTIONS_MADE,
            stats.get("predictions_made", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_MATERIALS_APPLIED,
            stats.get("materials_applied", 0),
        )

    # Emit per-step child spans from TelemetryEventListener
    _emit_step_spans(
        session_id=session_id,
        step_timings=telemetry_listener.get_step_timings(),
    )

    logger.info(f"Pipeline execution completed for {session_id[:8]}")


def _emit_step_spans(
    session_id: str,
    step_timings: list[dict] | None = None,
) -> None:
    """Emit OTel child spans for each pipeline step.

    These appear as children of the current active span (the root
    ``maa.pipeline.execution`` span created by the ``@traced`` decorator
    on ``execute_pipeline_async``).
    """
    if not step_timings:
        return

    try:
        from world_understanding.telemetry import get_tracer

        tracer = get_tracer(__name__)
        for timing in step_timings:
            step_name = timing["name"]
            step_status = timing["status"]
            duration_ns = timing["completed_at_ns"] - timing["started_at_ns"]
            duration_secs = duration_ns / 1_000_000_000

            with tracer.start_as_current_span(
                f"maa.pipeline.step.{step_name}"
            ) as step_span:
                step_span.set_attribute(MAAttributes.PIPELINE_STEP_NAME, step_name)
                step_span.set_attribute(MAAttributes.PIPELINE_STEP_STATUS, step_status)
                step_span.set_attribute(
                    MAAttributes.PIPELINE_STEP_DURATION_SECONDS, duration_secs
                )
                step_span.set_attribute(MAAttributes.PIPELINE_SESSION_ID, session_id)
                if timing.get("error"):
                    step_span.set_attribute(
                        MAAttributes.PIPELINE_STEP_ERROR, timing["error"]
                    )

    except Exception as e:
        logger.warning(f"Failed to emit step telemetry: {e}")


def _extract_stats_from_result(result, session_dir=None) -> dict:
    """Extract statistics from pipeline result.

    Args:
        result: PipelineOutput from arun_pipeline
        session_dir: Optional session directory for file-based fallback

    Returns:
        Dictionary with extracted stats
    """
    stats = {
        "original_prim_count": 0,
        "prims_processed": 0,
        "images_generated": 0,
        "predictions_made": 0,
        "materials_applied": 0,
    }

    # Use step_results for basic info
    step_results = result.step_results or {}

    # Extract predictions_count from predict step (this field IS extracted)
    if "predict" in step_results:
        stats["predictions_made"] = step_results["predict"].get("predictions_count", 0)
    elif "benchmark" in step_results:
        stats["predictions_made"] = step_results["benchmark"].get(
            "predictions_count", 0
        )

    # Extract materials_applied count from apply step
    if "apply" in step_results:
        materials_applied = step_results["apply"].get("materials_applied", {})
        # materials_applied is a dict mapping material names to prim paths
        stats["materials_applied"] = (
            len(materials_applied) if isinstance(materials_applied, dict) else 0
        )

    # For num_prims and num_images, we need to look at raw_result
    # which contains the full workflow context
    raw_result = result.raw_result or {}

    # First, try to get the ORIGINAL prim count (before optimization)
    # This is stored in pipeline_results by the optimize_usd step
    original_prim_count = 0

    # Check pipeline_results (where step outputs are stored)
    pipeline_results = raw_result.get("pipeline_results", {})
    logger.debug(f"pipeline_results keys: {list(pipeline_results.keys())}")

    if "optimize_usd" in pipeline_results:
        optimize_outputs = pipeline_results["optimize_usd"]
        logger.debug(f"optimize_usd outputs: {optimize_outputs}")
        original_prim_count = optimize_outputs.get("original_prim_count", 0)
        logger.debug(
            f"original_prim_count from optimize_outputs: {original_prim_count}"
        )
        # Also check optimization_metadata within the step outputs
        if original_prim_count == 0 and "optimization_metadata" in optimize_outputs:
            opt_metadata = optimize_outputs["optimization_metadata"]
            if opt_metadata:
                original_prim_count = opt_metadata.get("original_prim_count", 0)
                logger.debug(
                    f"original_prim_count from optimization_metadata: {original_prim_count}"
                )

    # Fallback: check optimization_metadata at top level
    if original_prim_count == 0 and "optimization_metadata" in raw_result:
        opt_metadata = raw_result["optimization_metadata"]
        original_prim_count = opt_metadata.get("original_prim_count", 0)
        logger.debug(
            f"original_prim_count from top-level optimization_metadata: {original_prim_count}"
        )

    # Fallback: check context directly
    if original_prim_count == 0:
        original_prim_count = raw_result.get("original_prim_count", 0)
        logger.debug(
            f"original_prim_count from raw_result directly: {original_prim_count}"
        )

    logger.info(f"Final original_prim_count: {original_prim_count}")

    # Store original prim count (before optimization)
    stats["original_prim_count"] = original_prim_count

    # Try to get stats from pipeline_results (where step outputs are stored)
    # Check build_dataset_usd step outputs
    if "build_dataset_usd" in pipeline_results:
        usd_result = pipeline_results["build_dataset_usd"]
        logger.debug(f"build_dataset_usd outputs: {usd_result}")
        stats["prims_processed"] = usd_result.get("num_prims", 0)
        stats["images_generated"] = usd_result.get("num_images", 0)
        logger.debug(
            f"From build_dataset_usd: prims={stats['prims_processed']}, images={stats['images_generated']}"
        )

    # Legacy fallback: check for build_dataset_usd_result directly in raw_result
    if stats["prims_processed"] == 0 and "build_dataset_usd_result" in raw_result:
        usd_result = raw_result["build_dataset_usd_result"]
        stats["prims_processed"] = usd_result.get("num_prims", 0)
        stats["images_generated"] = usd_result.get("num_images", 0)

    # Check build_dataset_prepare_dataset step outputs
    if (
        stats["prims_processed"] == 0
        and "build_dataset_prepare_dataset" in pipeline_results
    ):
        prepare_result = pipeline_results["build_dataset_prepare_dataset"]
        logger.debug(f"build_dataset_prepare_dataset outputs: {prepare_result}")
        stats["prims_processed"] = prepare_result.get("num_entries", 0)

    # Legacy fallback: check for build_dataset_prepare_dataset_result
    if (
        stats["prims_processed"] == 0
        and "build_dataset_prepare_dataset_result" in raw_result
    ):
        prepare_result = raw_result["build_dataset_prepare_dataset_result"]
        stats["prims_processed"] = prepare_result.get("num_entries", 0)

    # If still no prims count, try to get from dataset loading
    if stats["prims_processed"] == 0:
        # Count from completed_steps or dataset
        dataset_info = raw_result.get("dataset_info", {})
        stats["prims_processed"] = dataset_info.get("num_entries", 0)

    # Fallback: count from actual files in session directory
    if session_dir and (
        stats["prims_processed"] == 0
        or stats["images_generated"] == 0
        or stats["predictions_made"] == 0
    ):
        stats = _count_stats_from_files(session_dir, stats)

    # If optimize_usd didn't run, original_prim_count equals prims_processed
    # (no optimization means original == processed)
    if stats["original_prim_count"] == 0 and stats["prims_processed"] > 0:
        stats["original_prim_count"] = stats["prims_processed"]
        logger.debug(
            f"Set original_prim_count to prims_processed (no optimization): {stats['original_prim_count']}"
        )

    return stats


def _count_stats_from_files(session_dir, stats: dict) -> dict:
    """Count stats from actual files in session directory.

    Args:
        session_dir: Path to session directory
        stats: Current stats dict to update

    Returns:
        Updated stats dict
    """
    from pathlib import Path

    session_path = Path(session_dir)

    # Count entries from dataset.jsonl
    if stats["prims_processed"] == 0:
        dataset_file = session_path / "cache" / "dataset" / "dataset.jsonl"
        if dataset_file.exists():
            try:
                with open(dataset_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["prims_processed"] = len(lines)
                    logger.info(
                        f"Counted {stats['prims_processed']} entries from dataset.jsonl"
                    )
            except Exception as e:
                logger.warning(f"Failed to count dataset entries: {e}")

    # Count images from dataset directory
    if stats["images_generated"] == 0:
        dataset_dir = session_path / "cache" / "dataset"
        if dataset_dir.exists():
            try:
                image_count = len(list(dataset_dir.glob("**/*.png")))
                stats["images_generated"] = image_count
                logger.info(f"Counted {image_count} images in dataset directory")
            except Exception as e:
                logger.warning(f"Failed to count images: {e}")

    # Count predictions from predictions.jsonl
    if stats["predictions_made"] == 0:
        predictions_file = session_path / "cache" / "predictions" / "predictions.jsonl"
        if predictions_file.exists():
            try:
                with open(predictions_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["predictions_made"] = len(lines)
                    logger.info(
                        f"Counted {stats['predictions_made']} predictions from predictions.jsonl"
                    )
            except Exception as e:
                logger.warning(f"Failed to count predictions: {e}")

    return stats
