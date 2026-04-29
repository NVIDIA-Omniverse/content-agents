# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifacts API endpoints - Downloads for texture pipeline outputs."""

import io
import logging
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

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


def _zip_directory(directory: Path, prefix: str = "", pattern: str = "*") -> io.BytesIO:
    """Create a ZIP archive of matching files in a directory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(directory.glob(pattern)):
            if file_path.is_file():
                arcname = f"{prefix}/{file_path.name}" if prefix else file_path.name
                zf.write(file_path, arcname)
    buffer.seek(0)
    return buffer


@router.get("/{session_id}/materials")
async def download_materials(session_id: str):
    """Download discovered materials JSON file."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    materials_path = manager.get_artifact_path(session_id, "materials")
    if not materials_path or not materials_path.exists():
        raise HTTPException(status_code=404, detail="Materials data not available")

    return FileResponse(
        materials_path,
        media_type="application/json",
        filename="materials.json",
    )


@router.get("/{session_id}/textures")
async def download_textures_zip(session_id: str):
    """Download all blended textures as a ZIP archive."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    textures_dir = manager.get_artifact_dir(session_id, "textures")
    if not textures_dir or not textures_dir.exists():
        raise HTTPException(status_code=404, detail="Textures not available")

    png_files = list(textures_dir.glob("*.png"))
    if not png_files:
        raise HTTPException(status_code=404, detail="No texture files found")

    buffer = _zip_directory(textures_dir, prefix="textures", pattern="*.png")

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=textures_{session_id[:8]}.zip"
        },
    )


@router.get("/{session_id}/textures/{filename}")
async def download_single_texture(session_id: str, filename: str):
    """Download a single texture file."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    textures_dir = manager.get_artifact_dir(session_id, "textures")
    if not textures_dir:
        raise HTTPException(status_code=404, detail="Textures not available")

    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = textures_dir / filename

    if not file_path.resolve().is_relative_to(textures_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Texture file not found: {filename}"
        )

    return FileResponse(file_path, media_type="image/png")


@router.get("/{session_id}/output")
async def download_output(session_id: str):
    """Download the textured output as a self-contained USDZ archive.

    The USDZ bundles the USD file with all texture images into a single
    download — no separate texture download needed.
    """
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    usdz_path = manager.get_artifact_path(session_id, "output_usdz")
    if not usdz_path or not usdz_path.exists():
        raise HTTPException(status_code=404, detail="Output USDZ not available")

    return FileResponse(
        usdz_path,
        media_type="model/vnd.usdz+zip",
        filename="textured_output.usdz",
    )


@router.get("/{session_id}/renders")
async def download_renders_zip(session_id: str):
    """Download all rendered images as a ZIP archive."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    renders_dir = manager.get_artifact_dir(session_id, "renders")
    if not renders_dir or not renders_dir.exists():
        raise HTTPException(status_code=404, detail="Renders not available")

    png_files = list(renders_dir.glob("*.png"))
    if not png_files:
        raise HTTPException(status_code=404, detail="No render files found")

    buffer = _zip_directory(renders_dir, prefix="renders", pattern="*.png")

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=renders_{session_id[:8]}.zip"
        },
    )


@router.get("/{session_id}/renders/{filename}")
async def download_single_render(session_id: str, filename: str):
    """Download a single rendered image."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    renders_dir = manager.get_artifact_dir(session_id, "renders")
    if not renders_dir:
        raise HTTPException(status_code=404, detail="Renders not available")

    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = renders_dir / filename

    if not file_path.resolve().is_relative_to(renders_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Render file not found: {filename}"
        )

    return FileResponse(file_path, media_type="image/png")


@router.get("/{session_id}/preview/{filename}")
async def download_preview(session_id: str, filename: str):
    """Download a preview/thumbnail image."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    preview_dir = session_dir / "preview"

    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = preview_dir / filename

    if not file_path.resolve().is_relative_to(preview_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Preview file not found: {filename}"
        )

    return FileResponse(file_path, media_type="image/png")
