# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifacts API endpoints - Downloads and reports."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response

from ..session.manager import SessionManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/artifacts", tags=["artifacts"])

# Content type mapping for common file extensions
CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".usd": "application/octet-stream",
    ".usda": "text/plain",
    ".usdc": "application/octet-stream",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".html": "text/html",
    ".pdf": "application/pdf",
}

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


async def _try_serve_file_with_fallback(
    manager: SessionManager,
    session_id: str,
    key: str,
    local_path: Path,
    media_type: str | None = None,
    filename: str | None = None,
) -> Response | FileResponse | RedirectResponse | None:
    """Serve a file with fallback from presigned URL → store → local.

    Args:
        manager: Session manager
        session_id: Session identifier
        key: Store key for the file
        local_path: Local filesystem path
        media_type: MIME type (auto-detected if None)
        filename: Download filename (for Content-Disposition)

    Returns:
        Response object (redirect, streaming, or file response), or ``None`` if
        the artifact is not available anywhere.
    """
    # Auto-detect media type from extension if not provided
    if media_type is None:
        suffix = local_path.suffix.lower()
        media_type = CONTENT_TYPES.get(suffix, "application/octet-stream")

    # 1. Try presigned URL (redirect)
    url = await manager.make_public_url(session_id, key)
    if url:
        return RedirectResponse(url, status_code=302)

    # 2. Try reading from store (streaming response)
    data = await manager.read_from_store(session_id, key)
    if data is not None:
        headers = {}
        if filename:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(content=data, media_type=media_type, headers=headers)

    # 3. Fallback to local file
    if local_path.exists():
        return FileResponse(
            local_path,
            media_type=media_type,
            filename=filename,
        )

    return None


async def _serve_file_with_fallback(
    manager: SessionManager,
    session_id: str,
    key: str,
    local_path: Path,
    media_type: str | None = None,
    filename: str | None = None,
) -> Response | FileResponse | RedirectResponse:
    """Serve a file with fallback from presigned URL → store → local.

    Raises:
        HTTPException: If file not found anywhere.
    """
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        key,
        local_path,
        media_type=media_type,
        filename=filename,
    )
    if response is not None:
        return response

    raise HTTPException(status_code=404, detail="Artifact not found")


def _scene_render_candidates(
    metadata: dict[str, Any] | None,
    session_dir: Path,
) -> list[tuple[str, Path, str]]:
    """Return store keys/local paths for large-scene render fallbacks."""
    candidates: list[tuple[str, Path, str]] = []
    scene_metadata = metadata.get("scene", {}) if metadata else {}
    rendered_images = (
        scene_metadata.get("rendered_images", [])
        if isinstance(scene_metadata, dict)
        else []
    )
    if isinstance(rendered_images, list):
        for image in rendered_images:
            if not isinstance(image, str) or not image:
                continue
            image_path = Path(image)
            filename = image_path.name
            if not filename:
                continue
            local_path = image_path if image_path.is_absolute() else session_dir / image
            candidates.append((f"output/{filename}", local_path, filename))
            mirrored_path = session_dir / "output" / filename
            if mirrored_path != local_path:
                candidates.append((f"output/{filename}", mirrored_path, filename))

    for image_path in sorted((session_dir / "output").glob("composed_scene_*.png")):
        candidates.append((f"output/{image_path.name}", image_path, image_path.name))

    unique: list[tuple[str, Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, local_path, filename in candidates:
        marker = (key, str(local_path))
        if marker in seen:
            continue
        seen.add(marker)
        unique.append((key, local_path, filename))
    return unique


async def _generate_report_on_demand(
    session_dir: Path, predictions_path: Path, dataset_path: Path
) -> None:
    """Generate prediction HTML report on-demand.

    This is called only when the /report endpoint is accessed, preventing
    blocking operations during the predict step.

    Args:
        session_dir: Session directory
        predictions_path: Path to predictions.jsonl
        dataset_path: Path to dataset.jsonl
    """

    # Load predictions
    predictions = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    # Load dataset
    dataset = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    # Import lazily because report generation pulls in material-agent runtime deps.
    from material_agent.tasks.reporting import GeneratePredictionReportTask

    task = GeneratePredictionReportTask()

    # Prepare context
    report_context = {
        "predictions": predictions,
        "failed_predictions": [],
        "dataset": dataset,
        "output_dir": str(predictions_path.parent),
        "dataset_path": str(dataset_path),
    }

    # Run report generation in thread pool (blocks this coroutine but not event loop)
    await asyncio.to_thread(task.run, report_context, None)


@router.get("/{session_id}/output")
async def download_output_usd(session_id: str):
    """Download flattened output USD file with applied materials.

    Returns the flattened USD file that was sent to rendering, not the layered version.
    This ensures the downloaded file matches what was actually rendered.

    Args:
        session_id: Session identifier

    Returns:
        Flattened USD file as download
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)

    # Try flattened version first
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        "output/scene_with_materials_flat.usd",
        session_dir / "output" / "scene_with_materials_flat.usd",
        filename="scene_with_materials_flat.usd",
    )
    if response:
        return response

    # Large-scene rendering writes this sibling flat file before mirroring.
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        "output/composed_scene_flat.usd",
        session_dir / "output" / "composed_scene_flat.usd",
        filename="scene_with_materials_flat.usd",
    )
    if response:
        return response

    # Fallback to non-flattened version
    logger.warning(
        f"Flattened USD not found for {session_id[:8]}, trying non-flattened version"
    )
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        "output/scene_with_materials.usd",
        session_dir / "output" / "scene_with_materials.usd",
        filename="scene_with_materials.usd",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404,
        detail="Output USD not available. Pipeline may not be completed.",
    )


@router.api_route("/{session_id}/final-render", methods=["GET", "HEAD"])
async def download_final_render(session_id: str):
    """Download final render image (output USD with materials applied).

    Args:
        session_id: Session identifier

    Returns:
        Final render PNG image
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    metadata = await manager.get_session_metadata(session_id)
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        "output/scene_with_materials.png",
        session_dir / "output" / "scene_with_materials.png",
        media_type="image/png",
    )
    if response:
        return response

    if metadata and metadata.get("pipeline_type") == "large_scene":
        for key, local_path, filename in _scene_render_candidates(
            metadata,
            session_dir,
        ):
            response = await _try_serve_file_with_fallback(
                manager,
                session_id,
                key,
                local_path,
                media_type="image/png",
                filename=filename,
            )
            if response:
                return response

    raise HTTPException(
        status_code=404,
        detail="Final render not available. Pipeline may not have completed the render step.",
    )


@router.get("/{session_id}/predictions")
async def download_predictions(session_id: str):
    """Download predictions JSONL file.

    Args:
        session_id: Session identifier

    Returns:
        Predictions JSONL file
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/predictions/predictions.jsonl",
        session_dir / "cache" / "predictions" / "predictions.jsonl",
        media_type="application/x-ndjson",
        filename="predictions.jsonl",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Predictions not available")


@router.get("/{session_id}/scene-manifest")
async def download_scene_manifest(session_id: str):
    """Download the large-scene manifest JSON file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        "scene/manifest.json",
        session_dir / "scene" / "manifest.json",
        media_type="application/json",
        filename="manifest.json",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Scene manifest not available")


@router.get("/{session_id}/scene-validation-report")
async def download_scene_validation_report(session_id: str):
    """Download the large-scene validation report JSON file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        "scene/validation_report.json",
        session_dir / "scene" / "validation_report.json",
        media_type="application/json",
        filename="validation_report.json",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404,
        detail="Scene validation report not available",
    )


@router.get("/{session_id}/scene-predictions")
async def download_scene_predictions(session_id: str):
    """Download collated large-scene per-asset predictions JSONL."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        "scene/predictions.jsonl",
        session_dir / "scene" / "predictions.jsonl",
        media_type="application/x-ndjson",
        filename="scene_predictions.jsonl",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Scene predictions not available")


@router.get("/{session_id}/cluster-map")
async def download_cluster_map(session_id: str):
    """Download the prim clustering map JSONL file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/clusters/cluster_map.jsonl",
        session_dir / "cache" / "clusters" / "cluster_map.jsonl",
        media_type="application/x-ndjson",
        filename="cluster_map.jsonl",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Cluster map not available")


@router.get("/{session_id}/cluster-report")
async def view_cluster_report(session_id: str):
    """View the prim clustering HTML report."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/clusters/cluster_report.html",
        session_dir / "cache" / "clusters" / "cluster_report.html",
        media_type="text/html",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Cluster report not available")


@router.get("/{session_id}/cluster-summary")
async def download_cluster_summary(session_id: str):
    """Download the lightweight prim clustering summary JSON file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/clusters/cluster_summary.json",
        session_dir / "cache" / "clusters" / "cluster_summary.json",
        media_type="application/json",
        filename="cluster_summary.json",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Cluster summary not available")


@router.get("/{session_id}/cluster-representatives")
async def download_cluster_representatives(session_id: str):
    """Download the representative-only dataset used for clustered prediction."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/clusters/dataset_representatives.jsonl",
        session_dir / "cache" / "clusters" / "dataset_representatives.jsonl",
        media_type="application/x-ndjson",
        filename="dataset_representatives.jsonl",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404,
        detail="Cluster representatives dataset not available",
    )


@router.get("/{session_id}/optimization-report")
async def view_optimization_report(session_id: str):
    """View optimization JSON report in browser.

    Args:
        session_id: Session identifier

    Returns:
        Optimization report JSON
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = "cache/optimized/optimized_input.metadata.json"
    session_dir = manager.get_session_dir(session_id)
    report_path = session_dir / "cache" / "optimized" / "optimized_input.metadata.json"

    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        key,
        report_path,
        media_type="application/json",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404,
        detail="Optimization report is not available. Pipeline may not have completed the optimization step.",
    )


@router.get("/{session_id}/report")
async def view_prediction_report(session_id: str):
    """View prediction HTML report in browser.

    Generates the report on-demand if it doesn't exist yet.
    This prevents blocking the predict step with heavy HTML generation.

    Args:
        session_id: Session identifier

    Returns:
        Prediction report HTML served for viewing (not download)
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = "cache/predictions/prediction_report.html"
    session_dir = manager.get_session_dir(session_id)
    report_path = session_dir / "cache" / "predictions" / "prediction_report.html"

    # Try serving from presigned URL or store first
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        key,
        report_path,
        media_type="text/html",
    )
    if response:
        return response

    # Report doesn't exist anywhere - try to generate on-demand
    logger.info(f"Report not found for {session_id[:8]}, generating on-demand...")

    # Check if predictions exist
    predictions_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"

    if not predictions_path.exists():
        await manager.sync_from_store(session_id, prefix="cache/predictions/")
    if not dataset_path.exists():
        await manager.sync_from_store(session_id, prefix="cache/dataset/")

    if not predictions_path.exists():
        raise HTTPException(status_code=404, detail="Predictions not available yet")

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not available")

    # Generate report in background thread to avoid blocking
    try:
        await _generate_report_on_demand(session_dir, predictions_path, dataset_path)
        logger.info(f"✓ Report generated on-demand for {session_id[:8]}")
    except Exception as e:
        logger.error(f"Failed to generate report for {session_id[:8]}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {str(e)}"
        )

    # make sure the report is synced to the store
    await manager.sync_session_to_store(session_id)

    # Serve HTML for viewing (not as download)
    return FileResponse(report_path, media_type="text/html")
