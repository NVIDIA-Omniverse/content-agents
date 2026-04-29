# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifacts API endpoints - Downloads and reports."""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from ..session.manager import SessionManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/artifacts", tags=["artifacts"])

# Global session manager (initialized by main app)
session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global session_manager
    session_manager = manager


async def _generate_report_on_demand(
    session_dir: Path, predictions_path: Path, dataset_path: Path
) -> None:
    """Generate prediction HTML report on-demand."""
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    dataset = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    import sys

    service_dir = Path(__file__).parent.parent.parent
    apps_dir = service_dir.parent
    repo_root = apps_dir.parent
    for path in [str(apps_dir), str(repo_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    from physics_agent.tasks.reporting import GeneratePredictionReportTask

    task = GeneratePredictionReportTask()

    report_context = {
        "predictions": predictions,
        "failed_predictions": [],
        "dataset": dataset,
        "output_dir": str(predictions_path.parent),
        "dataset_path": str(dataset_path),
    }

    await asyncio.to_thread(task.run, report_context, None)


async def _serve_artifact(
    manager: SessionManager,
    session_id: str,
    artifact_type: str,
    media_type: str,
    filename: str,
) -> FileResponse | StreamingResponse:
    """Serve an artifact from local disk or store (S3)."""
    # Try local path first (fast path for the executing instance)
    local_path = await manager.get_artifact_path(session_id, artifact_type)
    if local_path:
        return FileResponse(local_path, media_type=media_type, filename=filename)

    # Fall back to store (S3 — works cross-instance)
    stream = await manager.get_artifact_stream(session_id, artifact_type)
    if stream:
        return StreamingResponse(
            stream,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            background=BackgroundTask(stream.close),
        )

    raise HTTPException(
        status_code=404, detail=f"{artifact_type.capitalize()} not available"
    )


@router.get("/{session_id}/predictions")
async def download_predictions(session_id: str):
    """Download predictions JSONL file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "predictions",
        "application/x-ndjson",
        "predictions.jsonl",
    )


@router.get("/{session_id}/report")
async def view_prediction_report(session_id: str):
    """View prediction HTML report in browser.

    Generates the report on-demand if it doesn't exist yet.
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    report_path = session_dir / "cache" / "predictions" / "report.html"

    if not report_path.exists():
        logger.info(f"Report not found for {session_id[:8]}, generating on-demand...")

        predictions_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
        dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"

        # Pull from store (S3) if files are missing locally (cross-instance case)
        if not predictions_path.exists() or not dataset_path.exists():
            pulled = await manager.sync_from_store(session_id, prefix="cache/")
            if pulled > 0:
                logger.info(
                    f"Pulled {pulled} artifact(s) from store for report generation"
                )

        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail="Predictions not available yet")

        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not available")

        try:
            await _generate_report_on_demand(
                session_dir, predictions_path, dataset_path
            )
            logger.info(f"Report generated on-demand for {session_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to generate report for {session_id[:8]}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Report generation failed: {str(e)}"
            )

    return FileResponse(report_path, media_type="text/html")


@router.get("/{session_id}/dataset")
async def download_dataset(session_id: str):
    """Download dataset JSONL file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "dataset",
        "application/x-ndjson",
        "dataset.jsonl",
    )


@router.get("/{session_id}/output-usd")
async def download_output_usd(session_id: str) -> Response:
    # Annotated as the starlette Response base class rather than the
    # concrete `FileResponse | StreamingResponse` union — FastAPI treats
    # endpoint-level union return annotations as a pydantic response_model
    # and fails at route registration ("Invalid args for response field").
    # Response covers both subclasses for MyPy while letting FastAPI pass
    # the result through untouched.
    """Download the simulation-ready USD written by the apply_physics step.

    Returned only when the pipeline has completed with apply_physics enabled
    (the service default). The file is the input USD flattened and augmented
    with UsdPhysics schemas (RigidBodyAPI, CollisionAPI, MassAPI, MaterialAPI)
    on each predicted prim, plus a PhysicsScene. Consumable by PhysX / Isaac.
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "output_usd",
        "application/octet-stream",
        "scene_physics.usda",
    )
