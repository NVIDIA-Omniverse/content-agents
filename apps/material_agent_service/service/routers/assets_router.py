# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assets API endpoints - Images and previews."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response

from ..models.responses import PreviewImage, PreviewList
from ..session.manager import SessionManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/assets", tags=["assets"])

# Content type mapping for common file extensions
CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
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


def _get_generated_reference_entry(
    metadata: dict | None, reference_id: str | None = None
) -> dict | None:
    if not metadata:
        return None

    generated_refs = metadata.get("generated_reference_images", [])
    if reference_id is None:
        return generated_refs[-1] if generated_refs else None

    for ref in generated_refs:
        if ref.get("id") == reference_id:
            return ref
    return None


async def _serve_file_with_fallback(
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
        Response object (redirect, streaming, or file response), or None if not found
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


@router.api_route("/{session_id}/input-render", methods=["GET", "HEAD"])
async def get_input_render(session_id: str):
    """Get the input USD render (before material assignment).

    This preview is generated automatically after upload to show the original scene.

    Args:
        session_id: Session identifier

    Returns:
        PNG image of the input scene

    Raises:
        404: If session not found or render not yet complete
        503: If render is still in progress
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = "input/input_render.png"
    session_dir = manager.get_session_dir(session_id)
    input_render_path = session_dir / key

    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        input_render_path,
        media_type="image/png",
    )
    if response:
        return response

    metadata = await manager.get_session_metadata(session_id)
    if metadata and metadata.get("preview_render_status") == "failed":
        detail = metadata.get("preview_render_error") or "Input render failed"
        raise HTTPException(status_code=424, detail=detail)

    # Check if it's still rendering
    temp_config = session_dir / ".input_render_config.yaml"
    if temp_config.exists() or (
        metadata and metadata.get("preview_render_status") == "rendering"
    ):
        raise HTTPException(status_code=503, detail="Input render still in progress")

    raise HTTPException(status_code=404, detail="Input render not available")


@router.api_route(
    "/{session_id}/generated-ref", methods=["GET", "HEAD"], response_model=None
)
async def get_generated_ref(session_id: str):
    """Get the AI-generated reference image.

    This image is generated interactively by the user via the
    generate-reference-image endpoint before pipeline submission.

    Args:
        session_id: Session identifier

    Returns:
        PNG image of the generated reference

    Raises:
        404: If session not found or image not generated
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    generated_ref = _get_generated_reference_entry(metadata)
    key = generated_ref.get("key") if generated_ref else "input/generated_ref_0.png"
    if not isinstance(key, str):
        raise HTTPException(
            status_code=404, detail="Generated reference image not available"
        )
    session_dir = manager.get_session_dir(session_id)
    generated_ref_path = session_dir / key

    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        generated_ref_path,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404, detail="Generated reference image not available"
    )


@router.api_route(
    "/{session_id}/generated-ref/{reference_id}",
    methods=["GET", "HEAD"],
    response_model=None,
)
async def get_generated_ref_by_id(session_id: str, reference_id: str):
    """Get a generated reference image by its explicit reference ID."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    generated_ref = _get_generated_reference_entry(metadata, reference_id)
    if not generated_ref:
        raise HTTPException(status_code=404, detail="Generated reference not found")

    key = generated_ref.get("key")
    if not isinstance(key, str):
        raise HTTPException(status_code=404, detail="Generated reference not found")

    session_dir = manager.get_session_dir(session_id)
    generated_ref_path = session_dir / key

    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        generated_ref_path,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404, detail="Generated reference image not available"
    )


@router.api_route("/{session_id}/preview/{image_name}", methods=["GET", "HEAD"])
async def get_preview_image(session_id: str, image_name: str):
    """Get a preview image (thumbnail) from the rendering process.

    Thumbnails are 128×128 resized versions stored in cache/preview/.

    Args:
        session_id: Session identifier
        image_name: Preview image filename

    Returns:
        PNG image
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)

    # Try cache/preview/ first (new event-driven path)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        f"cache/preview/{image_name}",
        session_dir / "cache" / "preview" / image_name,
        media_type="image/png",
    )
    if response:
        return response

    # Fallback to preview/ (old path for backward compatibility)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        f"preview/{image_name}",
        session_dir / "preview" / image_name,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404, detail=f"Preview image not found: {image_name}"
    )


@router.get("/{session_id}/previews", response_model=PreviewList)
async def list_preview_images(session_id: str) -> PreviewList:
    """List all available preview images.

    Args:
        session_id: Session identifier

    Returns:
        List of preview images with URLs
    """
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    preview_images = metadata.get("preview_images", [])

    previews = [
        PreviewImage(
            name=img,
            url=f"/assets/{session_id}/preview/{img}",
            prim_path=None,  # Could extract from filename if needed
            created_at=metadata["updated_at"],  # Approximate
        )
        for img in preview_images
    ]

    return PreviewList(session_id=session_id, previews=previews, total=len(previews))


@router.get("/{session_id}/reference/{image_name}")
async def get_reference_image(session_id: str, image_name: str):
    """Get a reference image uploaded for this session.

    Reference images are stored in input/reference_images/ and used by the VLM
    to understand the target appearance/materials of the asset.

    Args:
        session_id: Session identifier
        image_name: Reference image filename (e.g., reference_1.png)

    Returns:
        Image file (PNG or JPG)
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = f"input/reference_images/{image_name}"
    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        session_dir / key,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Reference image not found")


@router.get("/{session_id}/references")
async def list_reference_images(session_id: str):
    """List all reference images uploaded for this session.

    Args:
        session_id: Session identifier

    Returns:
        JSON list of reference image filenames with URLs
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    ref_dir = manager.get_session_dir(session_id) / "input" / "reference_images"

    references = []
    if ref_dir.exists():
        # Get all image files
        for img_path in sorted(ref_dir.glob("reference_*.*")):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                references.append(
                    {
                        "name": img_path.name,
                        "url": f"/assets/{session_id}/reference/{img_path.name}",
                    }
                )

    return {
        "session_id": session_id,
        "references": references,
        "total": len(references),
    }


@router.get("/{session_id}/reference-pdf/{pdf_name}")
async def get_reference_pdf(session_id: str, pdf_name: str):
    """Get a reference PDF uploaded for this session.

    Reference PDFs are stored in input/reference_pdfs/ and converted to images
    during the prepare_dataset step for VLM processing.

    Args:
        session_id: Session identifier
        pdf_name: Reference PDF filename (e.g., reference_0000.pdf)

    Returns:
        PDF file
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = f"input/reference_pdfs/{pdf_name}"
    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        session_dir / key,
        media_type="application/pdf",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Reference PDF not found")


@router.get("/{session_id}/reference-pdfs")
async def list_reference_pdfs(session_id: str):
    """List all reference PDFs uploaded for this session.

    Args:
        session_id: Session identifier

    Returns:
        JSON list of reference PDF filenames with URLs
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    pdf_dir = manager.get_session_dir(session_id) / "input" / "reference_pdfs"

    pdfs = []
    if pdf_dir.exists():
        # Get all PDF files
        for pdf_path in sorted(pdf_dir.glob("reference_*.pdf")):
            pdfs.append(
                {
                    "name": pdf_path.name,
                    "url": f"/assets/{session_id}/reference-pdf/{pdf_path.name}",
                }
            )

    return {
        "session_id": session_id,
        "pdfs": pdfs,
        "total": len(pdfs),
    }


@router.get("/{session_id}/pdf-pages")
async def list_rendered_pdf_pages(session_id: str):
    """List rendered PDF page images for this session.

    PDF pages are converted to images during the prepare_dataset step
    and stored in cache/dataset/pdf_0/, pdf_1/, etc.

    Args:
        session_id: Session identifier

    Returns:
        JSON list of rendered page images grouped by PDF index
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    dataset_dir = manager.get_session_dir(session_id) / "cache" / "dataset"
    pages: list[dict] = []

    if dataset_dir.exists():
        for pdf_dir in sorted(dataset_dir.glob("pdf_*")):
            if not pdf_dir.is_dir():
                continue
            for img_path in sorted(pdf_dir.glob("*.png")):
                pages.append(
                    {
                        "name": img_path.name,
                        "pdf_index": pdf_dir.name,
                        "url": f"/assets/{session_id}/pdf-page/{pdf_dir.name}/{img_path.name}",
                    }
                )

    return {
        "session_id": session_id,
        "pages": pages,
        "total": len(pages),
    }


@router.get("/{session_id}/pdf-page/{pdf_index}/{page_name}")
async def get_rendered_pdf_page(session_id: str, pdf_index: str, page_name: str):
    """Get a single rendered PDF page image.

    Args:
        session_id: Session identifier
        pdf_index: PDF directory name (e.g., pdf_0)
        page_name: Image filename (e.g., spec_page_001.png)

    Returns:
        PNG image file
    """
    import re

    if not re.match(r"^pdf_\d+$", pdf_index):
        raise HTTPException(status_code=400, detail="Invalid pdf_index format")
    if ".." in page_name or "/" in page_name:
        raise HTTPException(status_code=400, detail="Invalid page_name")

    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = f"cache/dataset/{pdf_index}/{page_name}"
    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        session_dir / key,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="PDF page image not found")
