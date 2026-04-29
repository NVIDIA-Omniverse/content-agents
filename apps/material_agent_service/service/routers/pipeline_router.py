# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API endpoints - Core workflow operations."""

import asyncio
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# Import API defaults (replaces service config defaults)
from material_agent.api.defaults import (
    DEFAULT_CAMERA_DIRECTIONS,
    DEFAULT_USD_PRIM_WARNING_THRESHOLD,
)
from sse_starlette import EventSourceResponse
from world_understanding.utils.usd.stage import get_stage_info_from_path

from ..config import config
from ..models.requests import RegenerateRequest
from ..models.responses import (
    PipelineError,
    PipelineResults,
    PipelineStatus,
    SessionCreated,
)
from ..runtime import get_event_bus, get_job_registry
from ..runtime.events import ProgressEvent, StepState
from ..session.manager import SessionManager
from ..workers.executor import execute_pipeline_async

logger = logging.getLogger(__name__)

_GENERATED_REFERENCE_STATUS_READY = "ready"

# Create router
router = APIRouter(prefix="/pipeline", tags=["pipeline"])


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
    metadata: dict | None, reference_id: str
) -> dict | None:
    if not metadata:
        return None
    for ref in metadata.get("generated_reference_images", []):
        if ref.get("id") == reference_id:
            return ref
    return None


def _session_accepts_generated_reference(metadata: dict | None) -> bool:
    return bool(
        metadata
        and metadata.get("status", "pending") == _GENERATED_REFERENCE_STATUS_READY
    )


async def _ensure_input_render_local(
    manager: SessionManager, session_id: str, session_dir: Path
) -> Path | None:
    """Return the local input preview, hydrating it from the store if needed."""
    input_render_key = "input/input_render.png"
    input_render = session_dir / input_render_key
    if input_render.exists():
        return input_render

    await manager.sync_from_store(session_id, prefix=input_render_key)
    if input_render.exists():
        return input_render

    data = await manager.read_from_store(session_id, input_render_key)
    if data is None:
        return None

    input_render.parent.mkdir(parents=True, exist_ok=True)
    input_render.write_bytes(data)
    return input_render


# Streaming upload helper to avoid loading large files into memory
def _validate_materials_yaml_content(
    materials_data: object,
    base_dir: Path,
) -> tuple[str, list[dict]]:
    """Validate parsed materials.yaml content and resolve library path.

    This is a shared helper used by both the ZIP extraction flow and the legacy
    YAML-only fallback during regeneration.

    Validates:
    - materials_data is a dict
    - materials_data["materials"] exists and is a dict
    - library_path is a non-empty string
    - entries is a non-empty list of dicts
    - Resolved library_path stays within base_dir (path traversal protection)
    - Library file exists on disk

    Args:
        materials_data: Parsed YAML data (from yaml.safe_load)
        base_dir: Directory containing materials.yaml (for resolving library_path)

    Returns:
        Tuple of (absolute_library_path, entries_list)

    Raises:
        HTTPException if validation fails
    """
    # Enforce YAML shape - must be a dict with 'materials' dict
    if not isinstance(materials_data, dict):
        error_msg = f"materials.yaml must be a YAML dictionary, got {type(materials_data).__name__}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    materials_section = materials_data.get("materials")
    if not isinstance(materials_section, dict):
        error_msg = (
            f"materials.yaml must have a 'materials' dictionary at top level. "
            f"Found top-level keys: {list(materials_data.keys())}, "
            f"materials type: {type(materials_section).__name__ if materials_section else 'None'}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    library_path_relative = materials_section.get("library_path")
    entries = materials_section.get("entries", [])

    logger.info(
        f"Parsed materials.yaml: library_path={library_path_relative}, "
        f"entries_count={len(entries) if entries else 0}"
    )

    # Validate library_path is a non-empty string
    if not library_path_relative or not isinstance(library_path_relative, str):
        error_msg = (
            "materials.yaml must specify materials.library_path as a non-empty string. "
            f"Found top-level keys: {list(materials_data.keys())}, "
            f"materials section keys: {list(materials_section.keys())}, "
            f"library_path type: {type(library_path_relative).__name__}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    if not isinstance(entries, list) or not entries:
        error_msg = (
            "materials.yaml must contain a non-empty list in materials.entries. "
            f"Found type={type(entries).__name__}, "
            f"len={len(entries) if hasattr(entries, '__len__') else 'n/a'}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Ensure each entry is a mapping (dict-like)
    if not all(isinstance(e, dict) for e in entries):
        types = {type(e).__name__ for e in entries}
        error_msg = (
            "materials.entries must be a list of objects (YAML mappings). "
            f"Got element types: {sorted(types)}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Resolve and validate USD library file exists (relative to base_dir)
    # Validate library_path doesn't escape base_dir (defense in depth)
    library_path = (base_dir / library_path_relative).resolve()
    base_dir_resolved = base_dir.resolve()

    # Ensure resolved path is within base_dir
    try:
        library_path.relative_to(base_dir_resolved)
    except ValueError:
        error_msg = (
            f"library_path escapes base directory: '{library_path_relative}' "
            f"(resolved to: {library_path}, base: {base_dir_resolved})"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    logger.info(f"Looking for USD library at: {library_path}")

    if not library_path.exists():
        # List available USD files for helpful error message
        available_usd = [
            f.name
            for f in base_dir.iterdir()
            if f.is_file() and f.suffix in (".usd", ".usda", ".usdc")
        ]
        available_msg = (
            f" Available: {available_usd}" if available_usd else " No USD files found"
        )
        error_msg = (
            f"USD library file not found: '{library_path_relative}' "
            f"(resolved to: {library_path}).{available_msg} "
            f"Base directory: {base_dir}. "
            f"Ensure library_path in materials.yaml matches the actual file name."
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    logger.info(
        f"Validated materials.yaml: {len(entries)} materials, library: {library_path.name}"
    )

    return str(library_path), entries


async def _stream_copy(
    upload: UploadFile, dest: Path, chunk_size: int = 2 * 1024 * 1024
) -> int:
    """Stream upload file to disk in chunks to avoid memory spikes.

    Args:
        upload: FastAPI UploadFile to stream
        dest: Destination path on disk
        chunk_size: Chunk size in bytes (default 2MB)

    Returns:
        Total bytes written
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    with dest.open("wb") as f:
        while True:
            data = await upload.read(chunk_size)
            if not data:
                break
            f.write(data)
            total_bytes += len(data)

    return total_bytes


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory.

    Looks for scene.* with any valid USD extension (.usd, .usda, .usdc, .usdz).

    Args:
        session_dir: Session directory path

    Returns:
        Path to the input USD file, or None if not found
    """
    input_dir = session_dir / "input"
    for ext in [".usd", ".usda", ".usdc", ".usdz"]:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


def _extract_and_validate_materials_zip(
    zip_path: Path,
    extract_dir: Path,
) -> tuple[str, list[dict]]:
    """Extract materials zip and validate contents.

    The zip must contain:
    - materials.yaml: Material definitions in service format (materials.entries)
    - USD library file: Referenced by library_path in materials.yaml

    Icons (thumbs/) are optional.

    Expected zip structure (created via `zip -r my.zip custom_materials/`):
        my.zip
        └── custom_materials/
            ├── materials.yaml
            └── materials_libs.usda

    Also supports flat structure (materials.yaml at zip root).

    Args:
        zip_path: Path to the uploaded zip file
        extract_dir: Directory to extract contents to

    Returns:
        Tuple of (materials_library_path, materials_entries)

    Raises:
        HTTPException if validation fails
    """
    # Extract zip
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400,
            detail="Invalid ZIP file. Please upload a valid ZIP archive.",
        )

    # Check materials.yaml exists - first at root, then in subdirectory
    materials_yaml_path = extract_dir / "materials.yaml"
    base_dir = extract_dir

    if not materials_yaml_path.exists():
        # Look for materials.yaml in a subdirectory (e.g., zip -r x.zip custom_materials/)
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        found = False
        for subdir in subdirs:
            candidate = subdir / "materials.yaml"
            if candidate.exists():
                materials_yaml_path = candidate
                base_dir = subdir
                found = True
                logger.info(f"Found materials.yaml in subdirectory: {subdir.name}/")
                break

        if not found:
            error_msg = (
                f"materials.zip must contain materials.yaml (at root or in a subdirectory). "
                f"Searched in: {extract_dir} and subdirectories: {[d.name for d in subdirs]}"
            )
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg,
            )

    # Parse materials.yaml
    try:
        with open(materials_yaml_path, encoding="utf-8") as f:
            materials_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        error_msg = f"Invalid materials.yaml: {e}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=400,
            detail=error_msg,
        ) from e

    # Validate YAML content and resolve library path using shared helper
    library_path, entries = _validate_materials_yaml_content(materials_data, base_dir)

    logger.info(
        f"Validated materials zip: {len(entries)} materials, "
        f"library: {Path(library_path).name}"
    )

    return library_path, entries


async def _render_input_preview(
    session_id: str,
    session_dir: Path,
    original_usd_path: Path | None = None,
) -> None:
    """Render preview of input USD (before material assignment).

    This runs in the background after upload to show users what their scene looks like.
    Creates a single rendered view stored as input/input_render.png.

    Uses the shared ``RenderScenePreviewTask`` via the material agent's
    ``create_render_preview_workflow_from_config`` factory.

    Args:
        session_id: Session identifier
        session_dir: Session directory
        original_usd_path: Original file path on disk (desktop mode). When
            provided, the renderer opens from the original location so that
            relative payload/sublayer references resolve correctly.
    """

    manager = get_session_manager()
    await manager.update_session(
        session_id,
        {"preview_render_status": "rendering", "preview_render_error": None},
    )

    try:
        logger.info(
            f"Rendering input preview for {session_id[:8]}... "
            f"(original_usd_path={original_usd_path})"
        )

        # Find input USD file (supports .usd, .usda, .usdc, .usdz)
        input_usd = _find_input_usd(session_dir)
        if not input_usd:
            message = f"No input USD found for session {session_id[:8]}"
            logger.warning(message)
            await manager.update_session(
                session_id,
                {"preview_render_status": "failed", "preview_render_error": message},
            )
            return
        output_path = session_dir / "input" / "input_render.png"

        # For desktop mode: use the original file path so that relative
        # payload/sublayer references (e.g. @./Payload/Contents.usda@)
        # resolve against the original directory on disk.
        if original_usd_path and original_usd_path.is_file():
            input_usd = original_usd_path
            logger.info(f"Using original USD path for render: {original_usd_path}")
        logger.info(f"Resolved input_usd for render: {input_usd}")

        # Create config for the render_preview workflow
        preview_config = {
            "usd_path": str(input_usd),
            "output_dir": str(session_dir / "input"),
            "backend": "remote",
            "image_width": 512,
            "image_height": 512,
            "cameras": ["+x+y+z"],
            "camera_margin": 1.0,
            "background_color": [1.0, 1.0, 1.0],
            "should_reset_materials": False,
            "use_lights": True,
            "flatten_before_render": False,
        }

        # Create temp config
        temp_config_path = session_dir / ".input_render_config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(preview_config, f)

        # Import and run render-preview workflow
        from material_agent.workflows import create_render_preview_workflow_from_config

        workflow = create_render_preview_workflow_from_config()

        # Run in thread pool (sync workflow)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, workflow.run, {"config_path": str(temp_config_path)}
        )

        # Get rendered image from result
        rendered_images = result.get("rendered_preview_paths", [])
        if rendered_images and Path(rendered_images[0]).exists():
            # Rename to standard name
            import shutil

            shutil.move(rendered_images[0], output_path)
            logger.info(f"✓ Input preview rendered: {output_path.name}")
            await manager.update_session(
                session_id,
                {"preview_render_status": "ready", "preview_render_error": None},
            )
            # Mirror to external store if configured
            try:
                await manager.put_file_to_store(
                    session_id,
                    "input/input_render.png",
                    str(output_path),
                    content_type="image/png",
                )
            except Exception as e:
                logger.warning(f"Failed to mirror input_render.png to store: {e}")
        else:
            message = "Input preview render failed - no output generated"
            logger.warning(message)
            await manager.update_session(
                session_id,
                {"preview_render_status": "failed", "preview_render_error": message},
            )

    except Exception as e:
        message = f"Failed to render input preview for {session_id[:8]}: {e}"
        logger.error(message)
        await manager.update_session(
            session_id,
            {"preview_render_status": "failed", "preview_render_error": message},
        )
        # Don't fail the pipeline - this is just a nice-to-have preview
    finally:
        # Always remove the temp config marker so the assets endpoint
        # stops returning 503 ("still in progress").
        temp_marker = session_dir / ".input_render_config.yaml"
        temp_marker.unlink(missing_ok=True)


@router.post("/{session_id}/generate-reference-image")
async def generate_reference_image(
    session_id: str,
    prompt: str = Form(..., description="Text prompt describing the desired look"),
) -> dict:
    """Generate a photorealistic reference image from the input preview + prompt.

    This endpoint is called interactively after the preview render is ready.
    The user provides a text prompt describing desired materials/look, and
    the system generates a reference image using an image-generation model.

    The generated image is saved to the session and returned as an explicit
    reference_id. The full pipeline uses it only when that ID is submitted.

    Args:
        session_id: Session identifier (from upload-usd)
        prompt: Text description of desired look

    Returns:
        JSON with status and image URL
    """

    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    if not _session_accepts_generated_reference(metadata):
        raise HTTPException(
            status_code=409,
            detail="Generated references can only be created before the pipeline is queued.",
        )

    if not config.image_gen_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Image generation backend '{config.image_gen_backend}' is not "
                "configured. Check MA_IMAGE_GEN_* and the required API key."
            ),
        )

    session_dir = manager.get_session_dir(session_id)

    # Check that the preview render exists, hydrating local cache if needed.
    input_render = await _ensure_input_render_local(manager, session_id, session_dir)
    if input_render is None:
        raise HTTPException(
            status_code=400,
            detail="Input preview not yet available. Wait for preview rendering to complete.",
        )

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    reference_id = uuid.uuid4().hex
    output_dir = session_dir / "input" / "generated_references" / reference_id
    output_key = f"input/generated_references/{reference_id}/generated_ref_0.png"

    try:
        logger.info(
            f"Generating reference image for {session_id[:8]}: {prompt[:80]}..."
        )

        image_gen_config = {"backend": config.image_gen_backend}
        if config.image_gen_model:
            image_gen_config["model"] = config.image_gen_model
        if config.image_gen_base_url:
            image_gen_config["base_url"] = config.image_gen_base_url

        # Build config for the generate_reference_image workflow
        gen_ref_config = {
            "rendered_preview_paths": [str(input_render)],
            "image_gen": image_gen_config,
            "prompt": prompt.strip(),
            "output_dir": str(output_dir),
            "num_images": 1,
        }

        # Write temp config
        temp_config_path = session_dir / ".gen_ref_config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(gen_ref_config, f)

        # Import and run workflow
        from material_agent.workflows import (
            create_generate_reference_image_workflow_from_config,
        )

        workflow = create_generate_reference_image_workflow_from_config()

        # Run in thread pool (sync workflow, may take ~20-30s)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, workflow.run, {"config_path": str(temp_config_path)}
        )

        # Check result
        generated_paths = result.get("generated_reference_image_paths", [])
        if generated_paths and Path(generated_paths[0]).exists():
            latest_metadata = await manager.get_session_metadata(session_id)
            if not _session_accepts_generated_reference(latest_metadata):
                shutil.rmtree(output_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Generated reference was discarded because the pipeline "
                        "has already been queued."
                    ),
                )

            logger.info(f"✓ Reference image generated for {session_id[:8]}")

            # Mirror to external store if configured
            try:
                await manager.put_file_to_store(
                    session_id,
                    output_key,
                    generated_paths[0],
                    content_type="image/png",
                )
            except Exception as e:
                logger.warning(f"Failed to mirror generated ref to store: {e}")

            image_url = f"/assets/{session_id}/generated-ref/{reference_id}"
            await manager.add_generated_reference_image(
                session_id,
                {
                    "id": reference_id,
                    "key": output_key,
                    "path": generated_paths[0],
                    "prompt": prompt.strip(),
                    "image_url": image_url,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )

            return {
                "status": "ok",
                "reference_id": reference_id,
                "image_url": image_url,
            }
        else:
            raise RuntimeError("No image generated")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to generate reference image for %s: %s",
            session_id[:8],
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate reference image. Check server logs for details.",
        )


@router.delete("/{session_id}/generated-reference-image/{reference_id}")
async def delete_generated_reference_image(session_id: str, reference_id: str) -> dict:
    """Delete a generated-reference image from the session metadata."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    if not _session_accepts_generated_reference(metadata):
        raise HTTPException(
            status_code=409,
            detail="Generated references can only be deleted before the pipeline is queued.",
        )

    removed = await manager.remove_generated_reference_image(session_id, reference_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Generated reference not found")

    key = removed.get("key")
    if isinstance(key, str):
        local_path = manager.get_session_dir(session_id) / key
        local_path.unlink(missing_ok=True)
        parent = local_path.parent
        try:
            parent.rmdir()
        except OSError:
            pass

    return {"status": "deleted", "reference_id": reference_id}


@router.post("/upload-usd", response_model=SessionCreated, status_code=201)
async def upload_usd_immediate(
    usd_file: UploadFile = File(..., description="USD file to upload and preview"),
) -> SessionCreated:
    """Upload USD file immediately and trigger input preview render.

    This endpoint is called immediately when user selects a file (before pipeline configuration).
    It creates a session, saves the USD, and triggers a background preview render.

    Args:
        usd_file: USD file to upload

    Returns:
        Session creation response with session_id
    """
    manager = get_session_manager()

    # Generate unique session ID
    session_id = str(uuid.uuid4())

    # Validate file extension
    if usd_file.filename:
        ext = Path(usd_file.filename).suffix.lower()
        if ext not in config.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {ext}. Allowed: {config.allowed_extensions}",
            )

    # Create session directory structure
    session_dir = await manager.create_session(
        session_id,
        config={"status": "uploading", "filename": usd_file.filename},
    )

    # Save uploaded USD file using streaming, preserving original extension
    original_ext = (
        Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
    )
    usd_path = session_dir / "input" / f"scene{original_ext}"
    try:
        total_bytes = await _stream_copy(usd_file, usd_path)
        size_mb = total_bytes / (1024 * 1024)

        if size_mb > config.max_upload_size_mb:
            usd_path.unlink(missing_ok=True)
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
            )

        logger.info(
            f"USD uploaded for session {session_id[:8]}: {size_mb:.2f}MB ({original_ext})"
        )

        # Store asset metadata in session for telemetry
        original_filename = usd_file.filename or f"scene{original_ext}"
        await manager.update_session(
            session_id,
            {
                "asset": {
                    "filename": original_filename,
                    "file_size_bytes": total_bytes,
                    "file_extension": original_ext,
                }
            },
        )

        # Mirror uploaded USD to external store if configured
        try:
            await manager.put_file_to_store(
                session_id,
                f"input/scene{original_ext}",
                str(usd_path),
                content_type="application/octet-stream",
            )
        except Exception as e:
            logger.warning(f"Failed to mirror USD to store: {e}")

        # Trigger background input preview render IMMEDIATELY
        await manager.update_session(
            session_id,
            {"status": "ready", "preview_render_status": "rendering"},
        )
        asyncio.create_task(_render_input_preview(session_id, session_dir))
        logger.info(f"✓ Input preview render triggered for {session_id[:8]}...")

        return SessionCreated(
            session_id=session_id,
            status="ready",
            message="USD uploaded, preview rendering in background",
            estimated_duration_minutes=0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload USD: {e}")
        await manager.delete_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to upload USD: {e}")


@router.post("/open-usd", response_model=SessionCreated, status_code=201)
async def open_usd_local(
    file_path: str = Body(
        ..., embed=True, description="Absolute path to a local USD file"
    ),
) -> SessionCreated:
    """Open a local USD file by path (desktop mode).

    Instead of uploading bytes, the server reads the file directly from the
    local filesystem.  Validation, session creation, and preview rendering
    match the ``upload-usd`` endpoint exactly.

    Args:
        file_path: Absolute path to a USD file on the local machine.

    Returns:
        Session creation response with session_id.
    """
    manager = get_session_manager()

    src = Path(file_path)

    # --- validate -----------------------------------------------------------
    if not src.is_absolute():
        raise HTTPException(
            status_code=400, detail="file_path must be an absolute path"
        )

    if not src.is_file():
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    ext = src.suffix.lower()
    if ext not in config.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {ext}. Allowed: {config.allowed_extensions}",
        )

    size_bytes = src.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > config.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
        )

    # --- session setup -------------------------------------------------------
    session_id = str(uuid.uuid4())
    session_dir = await manager.create_session(
        session_id,
        config={"status": "uploading", "filename": src.name},
    )

    dest = session_dir / "input" / f"scene{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Copy the entire source directory so that relative payload/sublayer
    # references (e.g. @./Payload/Contents.usda@) are available for the
    # full pipeline (optimize_usd, build_dataset, etc.).
    max_dir_bytes = config.max_upload_size_mb * 1024 * 1024 * 5
    total_dir_size = sum(f.stat().st_size for f in src.parent.rglob("*") if f.is_file())
    if total_dir_size > max_dir_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Source directory too large: {total_dir_size / (1024 * 1024):.1f}MB. "
                f"Max: {max_dir_bytes / (1024 * 1024):.0f}MB"
            ),
        )
    shutil.copytree(str(src.parent), str(dest.parent), dirs_exist_ok=True)

    # Rename the main USD to the canonical scene{ext} name that the
    # rest of the pipeline expects (_find_input_usd looks for "scene.*").
    copied_src = dest.parent / src.name
    if copied_src.resolve() != dest.resolve() and copied_src.exists():
        copied_src.rename(dest)

    logger.info(f"USD opened for session {session_id[:8]}: {size_mb:.2f}MB ({ext})")

    # Store asset metadata (mirrors upload-usd)
    await manager.update_session(
        session_id,
        {
            "asset": {
                "filename": src.name,
                "file_size_bytes": size_bytes,
                "file_extension": ext,
            }
        },
    )

    # Mirror to external store if configured
    try:
        await manager.put_file_to_store(
            session_id,
            f"input/scene{ext}",
            str(dest),
            content_type="application/octet-stream",
        )
    except Exception as e:
        logger.warning(f"Failed to mirror USD to store: {e}")

    # Trigger background input preview render — pass the original path so
    # that payload/sublayer references resolve against the source directory.
    await manager.update_session(
        session_id,
        {"status": "ready", "preview_render_status": "rendering"},
    )
    asyncio.create_task(
        _render_input_preview(session_id, session_dir, original_usd_path=src)
    )
    logger.info(f"Input preview render triggered for {session_id[:8]}...")

    return SessionCreated(
        session_id=session_id,
        status="ready",
        message="USD opened, preview rendering in background",
        estimated_duration_minutes=0,
    )


@router.post("", response_model=SessionCreated, status_code=202)
async def create_pipeline(
    usd_file: UploadFile = File(
        None, description="USD file to process (optional if ``session_id`` provided)"
    ),
    session_id: str = Form(
        None, description="Existing session ID (from ``/upload-usd`` endpoint)"
    ),
    user_email: str = Form(
        ..., description="User email address for usage tracking and telemetry"
    ),
    reference_images: list[UploadFile] = File(
        default=[],
        description="Reference images to help VLM understand the object (optional)",
    ),
    reference_pdfs: list[UploadFile] = File(
        default=[],
        description="Reference PDFs to convert to images for VLM (optional)",
    ),
    materials_zip: UploadFile | None = File(
        None,
        description="ZIP file containing custom materials (materials.yaml + USD library)",
    ),
    reference_descriptions: str = Form(
        default="",
        description='JSON array of descriptions for each reference image (e.g., \'["view 1", "view 2"]\') (optional)',
    ),
    generated_reference_id: str = Form(
        default="",
        description="Generated reference ID returned by generate-reference-image (optional)",
    ),
    user_prompt: str = Form(
        default="",
        description="Custom user prompt for VLM (optional)",
    ),
    camera_views: str = Form(
        default="+x+y+z,-x-y-z",
        description="Comma-separated camera views for rendering (default: ``+x+y+z,-x-y-z``)",
    ),
    steps: str = Form(
        default="",
        description="Comma-separated steps to run (optional, default: all steps)",
    ),
    optimize_usd: str = Form(
        default="true",
        description="Enable USD optimization step (true/false, default: true)",
    ),
    enable_deinstance: str = Form(
        default="true",
        description="Enable deinstance operation when optimize_usd is true (true/false, default: true)",
    ),
    enable_split: str = Form(
        default="true",
        description="Enable split meshes operation when optimize_usd is true (true/false, default: true)",
    ),
    enable_deduplicate: str = Form(
        default="true",
        description="Enable deduplicate operation when optimize_usd is true (true/false, default: true)",
    ),
    skip_instances: str = Form(
        default="true",
        description="Skip instance prims during dataset building (true/false, default: true)",
    ),
    skip_prototypes: str = Form(
        default="false",
        description="Skip prototype prims during dataset building (true/false, default: false)",
    ),
    skip_existing_materials: str = Form(
        default="false",
        description="Skip prims with existing material bindings (true/false, default: false)",
    ),
    pdf_descriptions: str = Form(
        default="",
        description='JSON array of descriptions for each reference PDF (e.g., \'["spec sheet", "manual"]\') (optional)',
    ),
    pdf_first_page: int | None = Form(
        default=None,
        description="First page to convert from PDFs (1-indexed, optional)",
    ),
    pdf_last_page: int | None = Form(
        default=None,
        description="Last page to convert from PDFs (1-indexed, optional)",
    ),
    vlm_model: str | None = Form(
        default=None,
        description="VLM model to use for prediction (optional, uses server default if not specified)",
    ),
    vlm_max_workers: int | None = Form(
        default=None,
        description="Maximum parallel VLM workers for prediction (optional, default: 64)",
    ),
    material_library: str = Form(
        default="default",
        description="Material library ID to use (default: 'default'). Ignored when materials_zip is provided.",
    ),
    layer_only: str = Form(
        default="false",
        description=(
            "Output only a material binding layer instead of a full USD "
            "(true/false, default: false). When true, the output USD "
            "contains only material definitions and bindings as 'over' "
            "opinions, preserving the original scene structure."
        ),
    ),
) -> SessionCreated:
    """Create and execute a material assignment pipeline.

    Two modes:
    1. New session: Provide usd_file, creates new session and uploads USD
    2. Existing session: Provide session_id (from /upload-usd), skips USD upload
    """
    manager = get_session_manager()

    # Parse camera views (use API default if not provided)
    camera_view_list = [v.strip() for v in camera_views.split(",") if v.strip()]
    if not camera_view_list:
        camera_view_list = DEFAULT_CAMERA_DIRECTIONS

    # Parse steps
    steps_list = None
    if steps:
        steps_list = [s.strip() for s in steps.split(",") if s.strip()]

    # Use default user prompt if not provided
    user_prompt_text = user_prompt.strip() if user_prompt else None

    # Two execution paths:
    if session_id:
        # Path 1: Use existing session (USD already uploaded via /upload-usd)
        logger.info(f"Using existing session {session_id[:8]}...")

        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        session_dir = manager.get_session_dir(session_id)

        # Update session config with pipeline parameters
        await manager.update_session(
            session_id,
            {
                "camera_views": camera_view_list,
                "user_prompt": user_prompt_text,
                "has_reference_images": len(reference_images) > 0,
                "num_reference_images": len(reference_images),
                "has_reference_pdfs": len(reference_pdfs) > 0,
                "num_reference_pdfs": len(reference_pdfs),
                "optimize_usd": optimize_usd.lower() == "true",
                "vlm_model": vlm_model,
                "steps": steps_list,
                "generated_reference_id": generated_reference_id or None,
            },
        )

    else:
        # Path 2: New session (legacy flow - upload USD now)
        if not usd_file:
            raise HTTPException(
                status_code=400, detail="Either usd_file or session_id must be provided"
            )

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Validate file extension
        if usd_file.filename:
            ext = Path(usd_file.filename).suffix.lower()
            if ext not in config.allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {ext}. Allowed: {config.allowed_extensions}",
                )

        # Create session directory structure
        session_dir = await manager.create_session(
            session_id,
            config={
                "camera_views": camera_view_list,
                "user_prompt": user_prompt_text,
                "has_reference_images": len(reference_images) > 0,
                "num_reference_images": len(reference_images),
                "has_reference_pdfs": len(reference_pdfs) > 0,
                "num_reference_pdfs": len(reference_pdfs),
                "optimize_usd": optimize_usd.lower() == "true",
                "vlm_model": vlm_model,
                "steps": steps_list,
                "generated_reference_id": generated_reference_id or None,
            },
        )

        # Save uploaded USD file using streaming, preserving original extension
        original_ext = (
            Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
        )
        usd_path = session_dir / "input" / f"scene{original_ext}"
        try:
            # Stream file to disk in chunks (2MB at a time)
            total_bytes = await _stream_copy(usd_file, usd_path)

            # Check file size after streaming
            size_mb = total_bytes / (1024 * 1024)
            if size_mb > config.max_upload_size_mb:
                # Remove the file if it exceeds limit
                usd_path.unlink(missing_ok=True)
                await manager.delete_session(session_id)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
                )

            logger.info(
                f"Saved USD file for session {session_id}: {size_mb:.2f}MB ({original_ext})"
            )

            # Store asset metadata in session for telemetry
            original_filename = usd_file.filename or f"scene{original_ext}"
            await manager.update_session(
                session_id,
                {
                    "asset": {
                        "filename": original_filename,
                        "file_size_bytes": total_bytes,
                        "file_extension": original_ext,
                    }
                },
            )

            # Mirror uploaded USD to external store if configured
            try:
                await manager.put_file_to_store(
                    session_id,
                    f"input/scene{original_ext}",
                    str(usd_path),
                    content_type="application/octet-stream",
                )
            except Exception as e:
                logger.warning(f"Failed to mirror USD to store: {e}")

            # Trigger background render of input USD (preview before material assignment)
            # This runs in parallel while user configures other settings
            asyncio.create_task(_render_input_preview(session_id, session_dir))
            logger.info(f"Triggered input preview render for {session_id[:8]}...")

        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except Exception as e:
            logger.error(f"Failed to save USD file: {e}")
            await manager.delete_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to save USD file: {e}")

    # Store user_email at the top level of session metadata
    await manager.update_session(session_id, {"user_email": user_email})

    # Validate input USD exists (both new + existing session flows)
    # Supports .usd, .usda, .usdc, .usdz extensions
    input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path:
        # May be on a different instance — pull input/ from store and retry
        pulled = await manager.sync_from_store(session_id, prefix="input/")
        if pulled > 0:
            logger.info(
                f"Pulled {pulled} input file(s) from store for session {session_id[:8]}"
            )
        input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path:
        raise HTTPException(
            status_code=400,
            detail="Input USD not found for session",
        )

    # Parse reference image descriptions if provided
    ref_descriptions = []
    if reference_descriptions:
        try:
            ref_descriptions = json.loads(reference_descriptions)
            if not isinstance(ref_descriptions, list):
                ref_descriptions = []
        except json.JSONDecodeError:
            logger.warning("Invalid reference_descriptions JSON, ignoring")

    # Parse PDF descriptions if provided
    pdf_desc_list = []
    if pdf_descriptions:
        try:
            pdf_desc_list = json.loads(pdf_descriptions)
            if not isinstance(pdf_desc_list, list):
                pdf_desc_list = []
        except json.JSONDecodeError:
            logger.warning("Invalid pdf_descriptions JSON, ignoring")

    # Save reference images if provided using streaming
    ref_image_paths = []
    if reference_images:
        reference_dir = session_dir / "input" / "reference_images"
        reference_dir.mkdir(parents=True, exist_ok=True)

        for i, ref_image in enumerate(reference_images):
            try:
                # Stream reference image to disk
                ref_ext = (
                    Path(ref_image.filename).suffix if ref_image.filename else ".png"
                )
                ref_path = reference_dir / f"reference_{i:04d}{ref_ext}"

                await _stream_copy(ref_image, ref_path)
                ref_image_paths.append(str(ref_path))

                logger.info(f"Saved reference image {i + 1}/{len(reference_images)}")

                # Mirror to external store if configured
                try:
                    ct = "image/png" if str(ref_ext).lower() == ".png" else "image/jpeg"
                    await manager.put_file_to_store(
                        session_id,
                        f"input/reference_images/reference_{i:04d}{ref_ext}",
                        str(ref_path),
                        content_type=ct,
                    )
                except Exception as e:
                    logger.warning(f"Failed to mirror reference image to store: {e}")

            except Exception as e:
                logger.warning(f"Failed to save reference image {i}: {e}")
                # Continue with other images

        # Save descriptions metadata if provided
        if ref_descriptions:
            ref_metadata = reference_dir / "descriptions.json"
            with open(ref_metadata, "w") as f:
                json.dump(ref_descriptions, f)
            logger.info(f"Saved {len(ref_descriptions)} reference image descriptions")

    # Save reference PDFs if provided using streaming
    ref_pdf_paths = []
    if reference_pdfs:
        pdf_dir = session_dir / "input" / "reference_pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        for i, ref_pdf in enumerate(reference_pdfs):
            try:
                # Validate PDF extension
                pdf_ext = (
                    Path(ref_pdf.filename).suffix.lower()
                    if ref_pdf.filename
                    else ".pdf"
                )
                if pdf_ext != ".pdf":
                    logger.warning(
                        f"Skipping non-PDF file: {ref_pdf.filename} (extension: {pdf_ext})"
                    )
                    continue

                # Stream PDF to disk
                pdf_path = pdf_dir / f"reference_{i:04d}.pdf"
                await _stream_copy(ref_pdf, pdf_path)
                ref_pdf_paths.append(str(pdf_path))

                logger.info(f"Saved reference PDF {i + 1}/{len(reference_pdfs)}")

                # Mirror to external store if configured
                try:
                    await manager.put_file_to_store(
                        session_id,
                        f"input/reference_pdfs/reference_{i:04d}.pdf",
                        str(pdf_path),
                        content_type="application/pdf",
                    )
                except Exception as e:
                    logger.warning(f"Failed to mirror reference PDF to store: {e}")

            except Exception as e:
                logger.warning(f"Failed to save reference PDF {i}: {e}")
                # Continue with other PDFs

    # Resolve materials: custom zip > selected library > default library
    has_custom_materials = False

    # Start with selected library (or default)
    selected_lib = config.get_library(material_library)
    if selected_lib:
        session_materials_library = selected_lib.library_path
        session_materials_entries = selected_lib.entries
    else:
        # Fall back to default library
        session_materials_library = config.materials_library_path
        session_materials_entries = config.materials
        if material_library != config.default_library_id:
            logger.warning(
                f"Unknown material library '{material_library}', "
                f"falling back to default"
            )

    if materials_zip and materials_zip.filename:
        logger.info(f"Processing custom materials zip: {materials_zip.filename}")

        # Create materials directory in session
        materials_dir = session_dir / "materials"
        materials_dir.mkdir(parents=True, exist_ok=True)

        # Save zip file with size check
        zip_path = materials_dir / "materials.zip"
        try:
            total_bytes = await _stream_copy(materials_zip, zip_path)
            size_mb = total_bytes / (1024 * 1024)

            # Apply same size limit as USD files
            if size_mb > config.max_upload_size_mb:
                zip_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Materials ZIP too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
                )

            logger.info(f"Saved materials zip: {zip_path} ({size_mb:.2f}MB)")

            # Extract and validate
            try:
                session_materials_library, session_materials_entries = (
                    _extract_and_validate_materials_zip(zip_path, materials_dir)
                )
                has_custom_materials = True
            except HTTPException as e:
                logger.error(
                    f"Failed to validate materials zip: {e.detail}. "
                    f"Zip path: {zip_path}, Extract dir: {materials_dir}"
                )
                raise

            # Update session metadata
            await manager.update_session(
                session_id,
                {
                    "has_custom_materials": True,
                    "custom_materials_count": len(session_materials_entries),
                },
            )

            logger.info(
                f"Using custom materials: {len(session_materials_entries)} entries, "
                f"library: {session_materials_library}"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to process materials zip: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process materials zip: {e}",
            )

    # Build complete MAA API config dict here at entry point
    from material_agent.api import build_unified_pipeline_config

    # Determine steps
    pipeline_steps = steps_list or [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "apply",
        "render",
    ]

    # Add optimize_usd step if enabled (prepend to run first)
    optimize_usd_enabled = optimize_usd.lower() == "true"
    if optimize_usd_enabled and "optimize_usd" not in pipeline_steps:
        pipeline_steps = ["optimize_usd"] + pipeline_steps
        logger.info("USD optimization step enabled")

    # Warn early if USD is very large (many prims) so UI can communicate latency
    threshold = DEFAULT_USD_PRIM_WARNING_THRESHOLD
    stage_info = await asyncio.to_thread(get_stage_info_from_path, input_usd_path)
    prim_count = stage_info.get("prim_count") if stage_info else None
    if prim_count is not None and prim_count > threshold:
        warn_step = (
            "build_dataset_usd"
            if "build_dataset_usd" in pipeline_steps
            else (pipeline_steps[0] if pipeline_steps else "pipeline")
        )
        warn_msg = (
            f"WARNING: Input USD contains {prim_count} prims (>{threshold}). "
            "Processing may be slow."
        )
        logger.warning("[%s] %s", session_id[:8], warn_msg)
        await get_event_bus().emit(
            ProgressEvent(
                session_id=session_id,
                step=warn_step,
                state=StepState.RUNNING,
                percent=0,
                message=warn_msg,
                extra={"prim_count": prim_count, "prim_warning_threshold": threshold},
            )
        )

    # Determine VLM model to use (user-selected or server default)
    selected_vlm_model = vlm_model if vlm_model else config.vlm_model

    # Parse VLM model string to extract backend if specified
    # Only "nim" prefix indicates a different backend; other provider prefixes
    # are model names for whichever backend the deployment configured.
    # e.g., "nim/nvidia/cosmos-reason2-8b" -> backend="nim", model="nvidia/cosmos-reason2-8b"
    # e.g., "gcp/google/gemini-3.1-pro-preview" ->
    #   backend=default, model="gcp/google/gemini-3.1-pro-preview"
    selected_vlm_backend = config.vlm_backend
    if selected_vlm_model and selected_vlm_model.startswith("nim/"):
        selected_vlm_backend = "nim"
        selected_vlm_model = selected_vlm_model[4:]  # Remove "nim/" prefix

    # Build base config (use session-specific materials if custom zip was provided)
    pipeline_config = build_unified_pipeline_config(
        project_name=session_id,
        session_id=session_id,
        input_usd_path=str(input_usd_path),
        output_usd_path=str(session_dir / "output" / "scene_with_materials.usd"),
        materials_library_path=session_materials_library,
        materials_entries=session_materials_entries,
        vlm_backend=selected_vlm_backend,
        vlm_model=selected_vlm_model,
        llm_backend=config.llm_backend,
        llm_model=config.llm_model,
        user_prompt=user_prompt_text,
        enabled_steps=pipeline_steps,
        working_dir=str(session_dir / "cache"),
    )

    # Override max_workers for predict step if specified
    if vlm_max_workers is not None and "predict" in pipeline_config.get("steps", {}):
        pipeline_config["steps"]["predict"]["max_workers"] = vlm_max_workers

    # Configure optimize_usd step if enabled
    if optimize_usd_enabled:
        # Validate at least one operation is enabled
        enable_deinstance_bool = enable_deinstance.lower() == "true"
        enable_split_bool = enable_split.lower() == "true"
        enable_deduplicate_bool = enable_deduplicate.lower() == "true"

        if not any(
            [enable_deinstance_bool, enable_split_bool, enable_deduplicate_bool]
        ):
            raise HTTPException(
                status_code=400,
                detail="At least one optimization operation must be enabled when optimize_usd is true. "
                "Please select Deinstance, Split Meshes, or Deduplicate Geometry.",
            )

        optimization_config = {
            "scene_optimizer_settings": {
                "enable_deinstance": enable_deinstance_bool,
                "enable_split_meshes": enable_split_bool,
                "enable_deduplicate": enable_deduplicate_bool,
                # Use defaults for other settings
                "generate_report": True,
                "capture_stats": True,
                "verbose": False,
                "wait_for_assets": False,
                "stage_timeout": 180.0,
                "output_format": "usdc",
                "extract_geom_subset_indices": True,
            },
            # Flatten prototypes before optimization:
            # - Converts abstract prototypes (over/class) to def
            # - Inlines all referenced geometry
            # - Removes prototype prims
            "flatten_prototypes": True,
        }

        # Add to optimize_usd step config
        if "optimize_usd" not in pipeline_config["steps"]:
            pipeline_config["steps"]["optimize_usd"] = {}
        pipeline_config["steps"]["optimize_usd"]["optimization_config"] = (
            optimization_config
        )

        logger.info(
            f"Optimization config: deinstance={enable_deinstance}, "
            f"split={enable_split}, deduplicate={enable_deduplicate}"
        )

    # Parse skip_instances, skip_prototypes, and skip_existing_materials flags
    skip_instances_bool = skip_instances.lower() == "true"
    skip_prototypes_bool = skip_prototypes.lower() == "true"
    skip_existing_materials_bool = skip_existing_materials.lower() == "true"

    # Force skip_instances=true, skip_prototypes=false when optimize_usd is enabled
    # This allows processing of prototype prims after they are converted from abstract to def
    if optimize_usd_enabled:
        skip_instances_bool = True
        skip_prototypes_bool = False
        logger.info(
            "optimize_usd enabled: forcing skip_instances=true, skip_prototypes=false, flatten_prototypes=true"
        )

    # Log VLM model selection
    if vlm_model:
        logger.info(f"Using user-selected VLM model: {selected_vlm_model}")

    # Log materials source for debugging
    if has_custom_materials:
        logger.info("Pipeline using CUSTOM materials from uploaded zip")
    elif selected_lib and selected_lib.id != config.default_library_id:
        logger.info(
            f"Pipeline using library '{selected_lib.id}' "
            f"({len(session_materials_entries)} materials)"
        )
    else:
        logger.info("Pipeline using SERVER DEFAULT materials")

    # Add reference images to input config
    if ref_image_paths:
        pipeline_config["input"]["reference_images"] = ref_image_paths

    # Explicitly inject a generated reference image when the caller selected one.
    if generated_reference_id:
        metadata = await manager.get_session_metadata(session_id)
        generated_ref = _get_generated_reference_entry(metadata, generated_reference_id)
        if not generated_ref:
            raise HTTPException(
                status_code=400,
                detail=f"Generated reference not found: {generated_reference_id}",
            )

        generated_key = generated_ref.get("key")
        if not isinstance(generated_key, str) or not generated_key:
            raise HTTPException(
                status_code=400,
                detail=f"Generated reference is missing a file key: {generated_reference_id}",
            )

        generated_ref_path = session_dir / generated_key
        if not generated_ref_path.exists():
            await manager.sync_from_store(session_id, prefix=generated_key)

        if not generated_ref_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Generated reference file is not available: {generated_reference_id}",
            )

        existing_refs = pipeline_config["input"].get("reference_images", [])
        pipeline_config["input"]["reference_images"] = existing_refs + [
            str(generated_ref_path)
        ]
        logger.info(
            "Injected selected generated reference image into pipeline config: %s",
            generated_reference_id,
        )

    # Add reference PDFs to input config with conversion settings
    if ref_pdf_paths:
        pipeline_config["input"]["reference_pdfs"] = ref_pdf_paths

        # Add PDF conversion settings (dpi=150, format=png are defaults)
        if "build_dataset_prepare_dataset" not in pipeline_config.get("steps", {}):
            pipeline_config["steps"]["build_dataset_prepare_dataset"] = {}

        pdf_conversion_config = {
            "dpi": 150,  # Default DPI
            "format": "png",  # Default format
        }
        if pdf_first_page is not None:
            pdf_conversion_config["first_page"] = pdf_first_page
        if pdf_last_page is not None:
            pdf_conversion_config["last_page"] = pdf_last_page

        pipeline_config["steps"]["build_dataset_prepare_dataset"]["pdf_conversion"] = (
            pdf_conversion_config
        )
        logger.info(
            f"Configured PDF conversion: {len(ref_pdf_paths)} PDFs, "
            f"pages {pdf_first_page or 'all'}-{pdf_last_page or 'all'}"
        )

    # Configure rendering for build_dataset_usd
    if "build_dataset_usd" in pipeline_config.get("steps", {}):
        # Use dict format for per-mode rendering configuration
        pipeline_config["steps"]["build_dataset_usd"]["renderer"].update(
            {
                "rendering_modes": {
                    "prim_only": {
                        "margin": 1.2,
                        "cameras": camera_view_list,
                        "camera_focus_mode": "prim",
                    },
                    "composition": {
                        "margin": 6.0,
                        "cameras": ["+x", "+y", "+z"],
                        "camera_focus_mode": "stage",
                        "skip_occluded_images": False,
                    },
                },
                "num_views": len(camera_view_list),
            }
        )

        # Configure prim_filters for skip_instances and skip_prototypes
        if "prim_filters" not in pipeline_config["steps"]["build_dataset_usd"]:
            pipeline_config["steps"]["build_dataset_usd"]["prim_filters"] = {}
        pipeline_config["steps"]["build_dataset_usd"]["prim_filters"].update(
            {
                "skip_instances": skip_instances_bool,
                "skip_prototypes": skip_prototypes_bool,
            }
        )

        # Set batch_size for async NVCF rendering (validated: 64 optimal for 128 instances)
        if "batch_size" not in pipeline_config["steps"]["build_dataset_usd"]:
            pipeline_config["steps"]["build_dataset_usd"]["batch_size"] = 64

        # Configure skip_existing_materials (at step level)
        pipeline_config["steps"]["build_dataset_usd"]["skip_existing_materials"] = (
            skip_existing_materials_bool
        )

    # Configure prepare_dataset with image prompts (dynamic based on uploaded images)
    if "build_dataset_prepare_dataset" in pipeline_config.get("steps", {}):
        # Build reference image prompts from descriptions or use defaults
        ref_prompts = []
        if ref_descriptions and len(ref_descriptions) == len(ref_image_paths):
            # Use user-provided descriptions
            ref_prompts = [
                f"This is a reference image: {desc}" for desc in ref_descriptions
            ]
        elif len(ref_image_paths) > 0:
            # Generate default prompts
            ref_prompts = [
                f"This is reference image {i + 1} of the asset you will match this look exactly"
                for i in range(len(ref_image_paths))
            ]

        vlm_image_prompts = {
            "reference_images": ref_prompts,
            "composition": "This is an orthographic view of the object with the part of interest highlighted with an orange outline.",
            "prim_only": "This is a rendered part of interest only without highlighting.",
        }

        # Add prompts for reference PDFs if any were uploaded
        if ref_pdf_paths:
            if pdf_desc_list and len(pdf_desc_list) == len(ref_pdf_paths):
                # Use user-provided descriptions
                vlm_image_prompts["reference_pdfs"] = [
                    (
                        f"This is a reference PDF: {desc}"
                        if desc
                        else "This is a reference PDF page of the asset. You will match this look exactly"
                    )
                    for desc in pdf_desc_list
                ]
            else:
                # Use default prompt
                vlm_image_prompts["reference_pdfs"] = (
                    "This is a reference PDF page of the asset. "
                    "You will match this look exactly"
                )

        pipeline_config["steps"]["build_dataset_prepare_dataset"]["prompts"].update(
            {"vlm_image_prompts": vlm_image_prompts}
        )

    # Configure predict step with VLM settings
    if "predict" in pipeline_config.get("steps", {}):
        vlm_config = {
            "backend": selected_vlm_backend,
            "model": selected_vlm_model,
            "temperature": config.vlm_temperature,
            "llmgateway": config.llmgateway_config,
        }

        # Use local NIM endpoint if configured (set by Helm when vlmNim is enabled)
        if selected_vlm_backend == "nim":
            nim_base_url = os.environ.get("MA_VLM_NIM_BASE_URL")
            if nim_base_url:
                vlm_config["base_url"] = nim_base_url

        # Special configuration for Cosmos Reason 2 (NIM backend)
        if selected_vlm_model and "cosmos-reason2" in selected_vlm_model:
            vlm_config.update(
                {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 16384,
                    "reasoning_budget": 16384,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            )
            # Cosmos Reason 2 uses <thinking> tag instead of <reasoning>
            prep_step = pipeline_config.get("steps", {}).get(
                "build_dataset_prepare_dataset", {}
            )
            if prep_step:
                prompts = prep_step.setdefault("prompts", {})
                from material_agent.tasks.prepare_dataset import (
                    _VLM_SYSTEM_PROMPT_TEMPLATE,
                )

                base_prompt = prompts.get("vlm_system", _VLM_SYSTEM_PROMPT_TEMPLATE)
                prompts["vlm_system"] = base_prompt.replace(
                    "<reasoning>", "<thinking>"
                ).replace("</reasoning>", "</thinking>")
            logger.info(f"Using Cosmos Reason 2 via NIM backend: {selected_vlm_model}")

        pipeline_config["steps"]["predict"]["vlm"] = vlm_config

        # Route LLM (structured-output parsing) through the same local NIM
        # sidecar when MA_VLM_NIM_BASE_URL is set. This gives an air-gapped
        # deployment with zero external inference endpoints: one NIM serves
        # both VLM predict and LLM parsing via the same OpenAI-compatible API.
        nim_base_url = os.environ.get("MA_VLM_NIM_BASE_URL")
        if nim_base_url:
            existing_llm = pipeline_config["steps"]["predict"].get("llm", {}) or {}
            llm_config = {
                **existing_llm,
                "backend": "nim",
                "model": selected_vlm_model,
                "base_url": nim_base_url,
            }
            pipeline_config["steps"]["predict"]["llm"] = llm_config
            logger.info(
                f"MA_VLM_NIM_BASE_URL set — routing LLM through local NIM: "
                f"{selected_vlm_model} @ {nim_base_url}"
            )

        # Configure report image compression to reduce file size
        pipeline_config["steps"]["predict"]["report"] = {
            "image_max_size": 256,  # Downscale images to max 256x256 pixels
            "image_format": "jpeg",  # Use JPEG instead of PNG (smaller)
            "image_quality": 75,  # JPEG quality (1-100)
        }

    # Configure apply step
    layer_only_bool = layer_only.lower() == "true"
    if "apply" not in pipeline_config.get("steps", {}):
        pipeline_config["steps"]["apply"] = {}
    pipeline_config["steps"]["apply"]["layer_only"] = layer_only_bool
    if layer_only_bool:
        pipeline_config["steps"]["apply"]["flatten_output"] = False
        logger.info("Layer-only mode: output will contain only material bindings")

    if "render" in pipeline_config.get("steps", {}):
        pipeline_config["steps"]["render"]["image_size"] = [512, 512]

    # Register and start pipeline execution with JobRegistry
    await manager.update_session(session_id, {"status": "pending"})
    job_registry = get_job_registry()
    await job_registry.register(
        session_id,
        execute_pipeline_async(
            session_id=session_id,
            config_dict=pipeline_config,
            session_manager=manager,
            user_email=user_email,
        ),
    )

    logger.info(f"Pipeline registered and queued for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message="Pipeline queued for execution",
        estimated_duration_minutes=15,  # Rough estimate
    )


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(session_id: str) -> PipelineStatus:
    """Get pipeline execution status with detailed progress.

    Reads from in-memory event bus state for fast, real-time accuracy.
    Falls back to disk-based SessionManager only for completed/stopped sessions.

    Args:
        session_id: Session identifier

    Returns:
        Detailed status including current step progress and preview images
    """
    from datetime import datetime

    event_bus = get_event_bus()
    manager = get_session_manager()

    # Try in-memory state first (active sessions)
    snapshot = event_bus.get_snapshot(session_id)

    if snapshot:
        # Active session - read from in-memory state (fast path, <1ms)
        metadata = snapshot

        # Get preview images from session manager (disk-based, but light)
        session_meta = await manager.get_session_metadata(session_id)
        preview_images = session_meta.get("preview_images", []) if session_meta else []

    else:
        # Session not in event bus - check disk for completed/old sessions
        metadata = await manager.get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")

        preview_images = metadata.get("preview_images", [])

    # Build preview image URLs (using new assets router path)
    preview_urls = [f"/assets/{session_id}/preview/{img}" for img in preview_images]

    # Calculate elapsed time dynamically
    created_at = datetime.fromisoformat(metadata["created_at"])
    elapsed_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    # Determine if can cancel (only if running)
    can_cancel = metadata.get("status") in ["pending", "running"]

    return PipelineStatus(
        session_id=session_id,
        status=metadata["status"],
        current_step=metadata.get("current_step"),
        completed_steps=metadata.get("completed_steps", []),
        overall_progress=metadata.get("overall_progress", {}),
        preview_images=preview_urls,
        can_cancel=can_cancel,
        elapsed_seconds=elapsed_seconds,
        created_at=metadata["created_at"],
        updated_at=metadata["updated_at"],
    )


@router.get("/{session_id}/results", response_model=PipelineResults | PipelineError)
async def get_pipeline_results(session_id: str):
    """Get pipeline execution results (only available when completed).

    Args:
        session_id: Session identifier

    Returns:
        Results if completed, error if failed, or 202 if still running
    """
    manager = get_session_manager()

    # make sure the session is synced to the store
    await manager.sync_session_to_store(session_id)

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    status = metadata["status"]

    if status == "completed":
        # Wait for stats if the executor hasn't saved them yet.
        # This handles the race window between when the EventBus sets
        # status="completed" and when the executor persists results.
        results = metadata.get("results") or {}
        stats_ready = any(v for v in results.values() if v)
        if not stats_ready:
            for _attempt in range(6):
                await asyncio.sleep(0.5)
                metadata = await manager.get_session_metadata(session_id)
                if not metadata:
                    raise HTTPException(status_code=404, detail="Session not found")
                results = metadata.get("results") or {}
                if any(v for v in results.values() if v):
                    break

        return PipelineResults(
            session_id=session_id,
            status=status,
            stats=metadata.get("results", {}),
            timings=metadata.get("timings_breakdown"),
            download_urls={
                "output_usd": f"/artifacts/{session_id}/output",
                "predictions": f"/artifacts/{session_id}/predictions",
                "report": f"/artifacts/{session_id}/report",
            },
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
        )

    elif status == "failed":
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=metadata.get("error", "Unknown error"),
            failed_step=metadata.get("failed_step", "unknown"),
            completed_steps=[s["name"] for s in metadata.get("completed_steps", [])],
            partial_results=metadata.get("partial_results"),
        )

    else:
        # Still running, pending, or cancelling
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline still {status}. Check status endpoint for progress.",
        )


@router.post("/{session_id}/cancel")
async def cancel_pipeline(session_id: str):
    """Cancel a running pipeline.

    Uses JobRegistry to cancel the asyncio.Task directly for immediate,
    deterministic cancellation (no file markers needed).

    Args:
        session_id: Session identifier

    Returns:
        Cancellation acknowledgment
    """
    job_registry = get_job_registry()
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    if metadata["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel pipeline with status: {metadata['status']}",
        )

    # Write cancel signal to store (visible to all instances)
    await manager.request_cancellation(session_id)

    # Also try local cancellation (fast path if this is the executing instance)
    if job_registry.is_running(session_id):
        await job_registry.cancel(session_id)

    return {
        "session_id": session_id,
        "status": "cancelling",
        "message": "Pipeline cancellation requested",
    }


@router.get("/{session_id}/events")
async def stream_progress_events(session_id: str):
    """Stream real-time progress events via Server-Sent Events (SSE).

    This endpoint provides live updates as the pipeline executes. The web UI
    can subscribe to this stream to show real-time progress without polling.

    Args:
        session_id: Session identifier

    Returns:
        SSE event stream with progress updates

    Example client (JavaScript):
        const eventSource = new EventSource(`/pipeline/${sessionId}/events`);
        eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            console.log(`Step: ${data.step}, Progress: ${data.percent}%`);
        });
    """
    event_bus = get_event_bus()

    # Verify session exists (either in EventBus or SessionManager)
    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is None:
        # Check if it exists in session manager but hasn't started yet
        manager = get_session_manager()
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        # If the pipeline is actively running but has no event bus snapshot,
        # it must be executing on a different instance. Return 503 so the
        # client falls back to polling. Don't 503 for "pending" — the
        # pipeline may just be waiting for the executor on this instance.
        metadata = await manager.get_session_metadata(session_id)
        status = (metadata or {}).get("status", "unknown")
        if status == "running":
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pipeline is running on a different instance; use polling instead"
                ),
            )

    async def event_generator():
        """Generate SSE events from the session's event queue."""
        queue = event_bus.get_queue(session_id)

        try:
            while True:
                # Wait for next event (with timeout to allow connection checks)
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Serialize event as JSON
                    event_data = event.model_dump_json()

                    # Yield SSE-formatted message
                    yield {
                        "event": "progress",
                        "data": event_data,
                    }

                    # Stop streaming only when OVERALL pipeline completes or fails
                    # Don't stop when individual steps complete (e.g., render at 50%)
                    should_close = False

                    if event.state in ["failed", "cancelled"]:
                        # Always close on error/cancel
                        should_close = True
                    elif event.state == "completed" and event.overall_percent >= 100:
                        # Only close when overall pipeline is 100% done
                        should_close = True

                    if should_close:
                        # Send final event then close stream
                        yield {
                            "event": "done",
                            "data": f'{{"session_id": "{session_id}", "final_state": "{event.state}"}}',
                        }
                        break

                except TimeoutError:
                    # Send keepalive ping
                    yield {"event": "ping", "data": "keepalive"}

        except asyncio.CancelledError:
            logger.debug(f"SSE stream cancelled for {session_id[:8]}...")
            raise

    return EventSourceResponse(event_generator(), ping=15)


@router.post("/{session_id}/regenerate", response_model=SessionCreated, status_code=202)
async def regenerate_pipeline(
    session_id: str,
    request: RegenerateRequest,
) -> SessionCreated:
    """Regenerate specific pipeline steps from cached data.

    Useful for re-running apply step with different settings without re-rendering.

    Args:
        session_id: Session identifier
        request: Regeneration request with steps and overrides

    Returns:
        Session status (same session_id)
    """
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    # Cannot regenerate if still running
    if metadata["status"] in ["pending", "running", "cancelling"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot regenerate while pipeline is {metadata['status']}",
        )

    # Update config with overrides
    original_config = metadata.get("config", {}).copy()

    # Override user prompt in metadata if provided (None means "no override")
    if request.user_prompt is not None:
        original_config["user_prompt"] = request.user_prompt

    # Get session directory and build complete config for regeneration
    from material_agent.api import build_unified_pipeline_config
    from material_agent.api.defaults import (
        DEFAULT_VLM_BACKEND,
        DEFAULT_VLM_LLMGATEWAY_CONFIG,
        DEFAULT_VLM_MODEL,
        DEFAULT_VLM_TEMPERATURE,
    )

    session_dir = manager.get_session_dir(session_id)
    camera_view_list = original_config.get("camera_views", DEFAULT_CAMERA_DIRECTIONS)
    steps_to_run = [s.value for s in request.steps]

    # Check if session has custom materials from previous run
    session_materials_library = config.materials_library_path
    session_materials_entries = config.materials

    materials_dir = session_dir / "materials"
    materials_zip_path = materials_dir / "materials.zip"

    # Prefer reusing validated ZIP if it exists (handles subdirectory layouts)
    if materials_zip_path.exists():
        logger.info(f"Regeneration: reloading materials from {materials_zip_path}")
        try:
            # Reuse the same validation function that handles subdirectories
            session_materials_library, session_materials_entries = (
                _extract_and_validate_materials_zip(materials_zip_path, materials_dir)
            )
            logger.info(
                f"Regeneration using custom materials: {len(session_materials_entries)} entries"
            )
        except HTTPException as e:
            # Expected validation errors - log and fall back to defaults
            logger.warning(
                f"Failed to validate custom materials for regeneration: {e.detail}. "
                f"Falling back to server defaults."
            )
        except Exception as e:
            # Unexpected errors - log and fall back to defaults
            logger.warning(
                f"Unexpected error loading custom materials for regeneration: {e}. "
                f"Falling back to server defaults."
            )
    elif (materials_dir / "materials.yaml").exists():
        # Fallback: try direct materials.yaml if ZIP was deleted (legacy)
        materials_yaml_path = materials_dir / "materials.yaml"
        logger.info(
            f"Regeneration: loading materials from legacy YAML {materials_yaml_path}"
        )
        try:
            with open(materials_yaml_path, encoding="utf-8") as f:
                materials_data = yaml.safe_load(f)

            # Use shared validation helper (same checks as ZIP flow)
            session_materials_library, session_materials_entries = (
                _validate_materials_yaml_content(materials_data, materials_dir)
            )
            logger.info(
                f"Regeneration using custom materials (legacy): "
                f"{len(session_materials_entries)} entries"
            )
        except yaml.YAMLError as e:
            logger.warning(
                f"Invalid YAML in materials.yaml for regeneration: {e}. "
                f"Falling back to server defaults."
            )
        except HTTPException as e:
            # Expected validation errors from shared helper - fall back to defaults
            logger.warning(
                f"Failed to validate custom materials (legacy) for regeneration: {e.detail}. "
                f"Falling back to server defaults."
            )
        except Exception as e:
            # Unexpected errors - log and fall back to defaults
            logger.warning(
                f"Unexpected error loading custom materials from YAML for regeneration: {e}. "
                f"Falling back to server defaults."
            )

    # Build config for regeneration (same as create_pipeline)
    # Supports .usd, .usda, .usdc, .usdz extensions
    input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path:
        raise HTTPException(
            status_code=400,
            detail="Input USD not found for session",
        )

    pipeline_config = build_unified_pipeline_config(
        project_name=session_id,
        session_id=session_id,
        input_usd_path=str(input_usd_path),
        output_usd_path=str(session_dir / "output" / "scene_with_materials.usd"),
        materials_library_path=session_materials_library,
        materials_entries=session_materials_entries,
        vlm_backend=DEFAULT_VLM_BACKEND,
        vlm_model=DEFAULT_VLM_MODEL,
        user_prompt=request.user_prompt,
        enabled_steps=steps_to_run,
        working_dir=str(session_dir / "cache"),
    )

    # Add reference images if they exist
    ref_images_dir = session_dir / "input" / "reference_images"
    if ref_images_dir.exists():
        ref_files = sorted(ref_images_dir.glob("reference_*"))
        if ref_files:
            pipeline_config["input"]["reference_images"] = [str(f) for f in ref_files]

    # Add reference PDFs if they exist
    ref_pdfs_dir = session_dir / "input" / "reference_pdfs"
    if ref_pdfs_dir.exists():
        pdf_files = sorted(ref_pdfs_dir.glob("reference_*.pdf"))
        if pdf_files:
            pipeline_config["input"]["reference_pdfs"] = [str(f) for f in pdf_files]

            # Add default PDF conversion settings for regeneration
            if "build_dataset_prepare_dataset" not in pipeline_config.get("steps", {}):
                pipeline_config["steps"]["build_dataset_prepare_dataset"] = {}

            pipeline_config["steps"]["build_dataset_prepare_dataset"][
                "pdf_conversion"
            ] = {
                "dpi": 150,
                "format": "png",
            }

    # Configure rendering for build_dataset_usd (same as create_pipeline)
    if "build_dataset_usd" in pipeline_config.get("steps", {}):
        # Use dict format for per-mode rendering configuration
        pipeline_config["steps"]["build_dataset_usd"]["renderer"].update(
            {
                "rendering_modes": {
                    "prim_only": {
                        "margin": 1.2,
                        "cameras": camera_view_list,
                        "camera_focus_mode": "prim",
                    },
                    "composition": {
                        "margin": 6.0,
                        "cameras": ["+x", "+y", "+z"],
                        "camera_focus_mode": "stage",
                        "skip_occluded_images": False,
                    },
                },
                "num_views": len(camera_view_list),
            }
        )

        # Set batch_size for async NVCF rendering (validated: 64 optimal for 128 instances)
        if "batch_size" not in pipeline_config["steps"]["build_dataset_usd"]:
            pipeline_config["steps"]["build_dataset_usd"]["batch_size"] = 64

    if "predict" in pipeline_config.get("steps", {}):
        pipeline_config["steps"]["predict"]["vlm"] = {
            "backend": DEFAULT_VLM_BACKEND,
            "model": DEFAULT_VLM_MODEL,
            "temperature": DEFAULT_VLM_TEMPERATURE,
            "llmgateway": DEFAULT_VLM_LLMGATEWAY_CONFIG,
        }
        # Configure report image compression to reduce file size
        pipeline_config["steps"]["predict"]["report"] = {
            "image_max_size": 256,  # Downscale images to max 256x256 pixels
            "image_format": "jpeg",  # Use JPEG instead of PNG (smaller)
            "image_quality": 75,  # JPEG quality (1-100)
        }

    # Configure apply step for layer_only mode
    if request.layer_only:
        if "apply" not in pipeline_config.get("steps", {}):
            pipeline_config["steps"]["apply"] = {}
        pipeline_config["steps"]["apply"]["layer_only"] = True
        pipeline_config["steps"]["apply"]["flatten_output"] = False
        logger.info("Regeneration: layer-only mode enabled")

    if "render" in pipeline_config.get("steps", {}):
        pipeline_config["steps"]["render"]["image_size"] = [512, 512]

    # Reset session status for regeneration
    await manager.update_session(
        session_id,
        {
            "status": "pending",
            "current_step": None,
            "config": original_config,
            "can_cancel": True,
        },
    )

    # Read user_email from session metadata for telemetry
    user_email = metadata.get("user_email", "")

    # Register and start regeneration with JobRegistry
    job_registry = get_job_registry()
    await job_registry.register(
        session_id,
        execute_pipeline_async(
            session_id=session_id,
            config_dict=pipeline_config,
            session_manager=manager,
            user_email=user_email,
        ),
    )

    logger.info(f"Pipeline regeneration registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message=f"Regenerating steps: {', '.join(s.value for s in request.steps)}",
    )


@router.get("/{session_id}/event-log")
async def get_event_log(session_id: str):
    """Get the persisted event log for a session.

    This allows replaying the full event history for completed sessions.

    Args:
        session_id: Session identifier

    Returns:
        List of event objects
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    log_file = manager.get_session_dir(session_id) / "event_log.jsonl"

    if not log_file.exists():
        return {"events": []}

    # Load events from log file
    events = []
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        return {"events": events, "total": len(events)}

    except Exception as e:
        logger.error(f"Failed to load event log for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load event log: {e}")


@router.get("/sessions/{session_id}/materials/icon/{material_name:path}")
async def get_session_material_icon(session_id: str, material_name: str):
    """Serve material icon from session's custom materials.

    This endpoint serves icons for custom materials uploaded via ZIP files.
    Icons are stored in the session's materials directory.

    Args:
        session_id: Session identifier
        material_name: Material name (URL-encoded)

    Returns:
        PNG image file
    """
    from urllib.parse import unquote, unquote_plus

    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Decode URL-encoded name
    decoded_name = unquote_plus(unquote(material_name))

    logger.info(
        f"[SESSION_ICON] Request: session={session_id[:8]}, material='{decoded_name}'"
    )

    # Get session materials directory
    session_dir = manager.get_session_dir(session_id)
    materials_dir = session_dir / "materials"

    if not materials_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Session has no custom materials",
        )

    # Find materials.yaml - check root and subdirectories
    yaml_path = materials_dir / "materials.yaml"
    base_dir = materials_dir

    if not yaml_path.exists():
        # Look for materials.yaml in a subdirectory (zip structure)
        for subdir in materials_dir.iterdir():
            if subdir.is_dir() and (subdir / "materials.yaml").exists():
                yaml_path = subdir / "materials.yaml"
                base_dir = subdir
                break

    if not yaml_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Session materials.yaml not found",
        )

    # Load materials.yaml and find icon path for material
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        entries = data.get("materials", {}).get("entries", [])
        icon_rel_path = None

        for entry in entries:
            if entry.get("name") == decoded_name:
                icon_rel_path = entry.get("icon")
                break

        if not icon_rel_path:
            logger.warning(
                f"[SESSION_ICON] Material not found: '{decoded_name}' in {yaml_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Icon not found for material: {decoded_name}",
            )

        icon_path = base_dir / icon_rel_path

        if not icon_path.exists():
            logger.warning(f"[SESSION_ICON] Icon file not found: {icon_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Icon file not found: {icon_rel_path}",
            )

        # Security: ensure path is inside materials directory
        try:
            icon_path.resolve().relative_to(base_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")

        logger.info(f"[SESSION_ICON] Serving: {icon_path}")
        return FileResponse(icon_path, media_type="image/png")

    except yaml.YAMLError as e:
        logger.error(f"[SESSION_ICON] Failed to parse materials.yaml: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse materials config: {e}",
        )
