# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API endpoints - Core workflow operations."""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sse_starlette import EventSourceResponse
from world_understanding.utils.s3_utils import download_file_from_s3

from ..config import config
from ..models.requests import RegenerateRequest
from ..models.responses import (
    PipelineError,
    PipelineResults,
    PipelineStatus,
    SessionCreated,
)
from ..runtime import ProgressEvent, StepState, get_event_bus, get_job_registry
from ..session.manager import SessionManager
from ..workers.executor import execute_pipeline_async

logger = logging.getLogger(__name__)

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


async def _stream_copy(
    upload: UploadFile,
    dest: Path,
    chunk_size: int = 2 * 1024 * 1024,
    max_bytes: int = 0,
) -> int:
    """Stream upload file to disk in chunks to avoid memory spikes.

    Args:
        upload: FastAPI upload file.
        dest: Destination path.
        chunk_size: Read chunk size in bytes.
        max_bytes: Maximum allowed bytes (0 = unlimited).

    Raises:
        HTTPException: If the file exceeds max_bytes during streaming.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    with dest.open("wb") as f:
        while True:
            data = await upload.read(chunk_size)
            if not data:
                break
            total_bytes += len(data)
            if max_bytes and total_bytes > max_bytes:
                dest.unlink(missing_ok=True)
                size_mb = total_bytes / (1024 * 1024)
                limit_mb = max_bytes / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: >{size_mb:.1f}MB. Max: {limit_mb:.0f}MB",
                )
            f.write(data)

    return total_bytes


_VALID_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}


def _download_s3_to_session(s3_uri: str, session_dir: Path) -> Path:
    """Download a USD file from S3 into session_dir/input/."""
    if not s3_uri.startswith("s3://") or s3_uri.count("/") < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid S3 URI format: {s3_uri}. "
            "Expected s3://bucket/path/to/file.ext",
        )

    s3_filename = s3_uri.rstrip("/").rsplit("/", 1)[-1]
    if not s3_filename:
        raise HTTPException(
            status_code=400,
            detail=f"S3 URI must include an object key: {s3_uri}",
        )
    ext = Path(s3_filename).suffix.lower()
    if ext not in _VALID_USD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid USD file type in S3 URI: {ext}. "
            f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
        )

    local_path = session_dir / "input" / f"scene{ext}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_file_from_s3(s3_uri, local_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"S3 object not found: {s3_uri}")
    except PermissionError:
        raise HTTPException(
            status_code=403, detail=f"Access denied to S3 object: {s3_uri}"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download from S3: {e}")

    size_mb = local_path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_upload_size_mb:
        local_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"S3 file too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
        )

    return local_path


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory."""
    input_dir = session_dir / "input"
    for ext in [".usd", ".usda", ".usdc", ".usdz"]:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


def build_default_pipeline_config(
    session_id: str,
    usd_path: str,
    working_dir: str,
    material_textures: dict[str, Any] | None = None,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Build a default pipeline config dict from ServiceConfig defaults.

    Args:
        session_id: Session identifier
        usd_path: Path to input USD file
        working_dir: Working directory for pipeline output
        material_textures: Per-material prompt/opacity overrides
        user_prompt: Aesthetic direction for auto-prompt generation

    Returns:
        Pipeline config dict compatible with config_to_context()
    """
    image_gen_config: dict[str, Any] = {
        "backend": config.image_gen_backend,
    }
    if config.image_gen_model:
        image_gen_config["model"] = config.image_gen_model
    if config.image_gen_base_url:
        image_gen_config["base_url"] = config.image_gen_base_url

    return {
        "project": {
            "name": session_id,
            "session_id": session_id,
            "working_dir": working_dir,
        },
        "input": {
            "usd_path": usd_path,
        },
        "texture": {
            "backend": config.texture_backend,
            "image_gen": image_gen_config,
            "size": config.texture_size,
            "workers": config.texture_workers,
        },
        "material_textures": material_textures or {},
        "auto_prompt": {
            "user_prompt": user_prompt or "",
            "default_opacity": config.blend_opacity,
            "llm": {
                "backend": config.llm_backend,
                "model": config.llm_model,
                **({"base_url": config.llm_base_url} if config.llm_base_url else {}),
            },
        },
        "variations": {"count": 1},
        "steps": {
            "prepare_uvs": {"enabled": True},
            "discover_materials": {"enabled": True},
            "generate_prompts": {"enabled": True},
            "render_previews": {"enabled": False},
            "generate_textures": {
                "enabled": True,
                "skip_existing": True,
                "max_workers": config.texture_workers,
            },
            "blend_textures": {
                "enabled": True,
                "default_opacity": config.blend_opacity,
                "output_size": config.texture_size,
            },
            "apply_textures": {"enabled": True},
            "render": {"enabled": False},
        },
    }


@router.post("/upload-usd", response_model=SessionCreated, status_code=201)
async def upload_usd_immediate(
    usd_file: UploadFile = File(
        None, description="USD file to upload (provide this OR s3_uri)"
    ),
    s3_uri: str = Form(
        None,
        description="S3 URI to a USD file (e.g. s3://bucket/path/scene.usdz)",
    ),
) -> SessionCreated:
    """Upload a USD file and create a session for later pipeline execution.

    Two input modes:
    1. **File upload**: Provide ``usd_file`` (multipart).
    2. **S3 reference**: Provide ``s3_uri`` -- the service downloads server-side.

    Use the returned session_id with ``POST /pipeline`` to start processing.
    """
    if not usd_file and not s3_uri:
        raise HTTPException(
            status_code=400,
            detail="Either usd_file or s3_uri must be provided",
        )
    if usd_file and s3_uri:
        raise HTTPException(
            status_code=400,
            detail="Provide either usd_file or s3_uri, not both",
        )

    manager = get_session_manager()
    session_id = str(uuid.uuid4())

    if s3_uri:
        session_dir = manager.create_session(session_id)
        try:
            local_path = _download_s3_to_session(s3_uri, session_dir)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
            return SessionCreated(
                session_id=session_id,
                status="ready",
                message=f"USD downloaded from S3 successfully ({size_mb:.1f}MB)",
                estimated_duration_minutes=0,
            )
        except HTTPException:
            manager.delete_session(session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to download USD from S3: {e}")
            manager.delete_session(session_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download USD from S3: {e}",
            )

    # File upload path
    if usd_file and usd_file.filename:
        ext = Path(usd_file.filename).suffix.lower()
        if ext not in _VALID_USD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid USD file type: {ext}. "
                f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
            )

    session_dir = manager.create_session(session_id)

    original_ext = (
        Path(usd_file.filename).suffix.lower()
        if usd_file and usd_file.filename
        else ".usd"
    )
    usd_path = session_dir / "input" / f"scene{original_ext}"

    max_bytes = config.max_upload_size_mb * 1024 * 1024
    try:
        total_bytes = await _stream_copy(usd_file, usd_path, max_bytes=max_bytes)
        size_mb = total_bytes / (1024 * 1024)

        logger.info(
            f"USD uploaded for session {session_id[:8]}: "
            f"{size_mb:.2f}MB ({original_ext})"
        )

        return SessionCreated(
            session_id=session_id,
            status="ready",
            message="USD uploaded successfully",
            estimated_duration_minutes=0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload USD: {e}")
        manager.delete_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to upload USD: {e}")


@router.post("", response_model=SessionCreated, status_code=202)
async def create_pipeline(
    usd_file: UploadFile = File(
        None,
        description="USD file to process (optional if session_id or s3_uri provided)",
    ),
    session_id: str = Form(
        None,
        description="Existing session ID (from /upload-usd endpoint)",
    ),
    s3_uri: str = Form(
        None,
        description="S3 URI to a USD file (e.g. s3://bucket/path/scene.usdz)",
    ),
    material_textures_json: str = Form(
        default="",
        description='Per-material texture config as JSON string, e.g. {"Steel": {"prompt": "rusted steel", "opacity": 0.85}}',
    ),
    user_prompt: str = Form(
        default="",
        description="Aesthetic direction for auto-prompt generation (e.g. 'old and weathered'). "
        "Used to auto-generate prompts for materials not covered by material_textures_json.",
    ),
) -> SessionCreated:
    """Create and execute a texture generation pipeline.

    Three input modes:
    1. **Existing session**: Provide ``session_id`` (from ``/upload-usd``).
    2. **File upload**: Provide ``usd_file``, creates new session.
    3. **S3 reference**: Provide ``s3_uri``, downloads from S3 server-side.

    Optionally provide ``material_textures_json`` to specify per-material
    texture prompts and blend opacity.
    """
    manager = get_session_manager()

    # Parse material_textures from JSON
    material_textures: dict[str, Any] | None = None
    if material_textures_json and material_textures_json.strip():
        try:
            material_textures = json.loads(material_textures_json)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in material_textures_json: {e}",
            )

    if session_id:
        # Path 1: reuse existing session
        if not manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        # Prevent concurrent re-start of a running session
        job_registry = get_job_registry()
        if job_registry.is_running(session_id):
            raise HTTPException(
                status_code=409,
                detail="Session is already running. Cancel it first or wait for completion.",
            )

        session_dir = manager.get_session_dir(session_id)

    elif s3_uri:
        # Path 2: new session with S3 download
        session_id = str(uuid.uuid4())
        session_dir = manager.create_session(session_id)

        try:
            local_path = _download_s3_to_session(s3_uri, session_dir)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
        except HTTPException:
            manager.delete_session(session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to download USD from S3: {e}")
            manager.delete_session(session_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download USD from S3: {e}",
            )

    elif usd_file:
        # Path 3: new session with USD upload
        session_id = str(uuid.uuid4())
        session_dir = manager.create_session(session_id)

        try:
            if usd_file.filename:
                ext = Path(usd_file.filename).suffix.lower()
                if ext not in _VALID_USD_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid USD file type: {ext}. "
                        f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
                    )

            original_ext = (
                Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
            )
            usd_path = session_dir / "input" / f"scene{original_ext}"
            max_bytes = config.max_upload_size_mb * 1024 * 1024
            total_bytes = await _stream_copy(usd_file, usd_path, max_bytes=max_bytes)
            size_mb = total_bytes / (1024 * 1024)

            logger.info(
                f"USD uploaded for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({original_ext})"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to save USD file: {e}")
            manager.delete_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to save USD file: {e}")

    else:
        raise HTTPException(
            status_code=400,
            detail="One of usd_file, session_id, or s3_uri must be provided",
        )

    # Find the input USD
    input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path:
        raise HTTPException(status_code=400, detail="Input USD not found for session")

    # Build pipeline config
    user_prompt_text = user_prompt.strip() if user_prompt else None
    pipeline_config = build_default_pipeline_config(
        session_id=session_id,
        usd_path=str(input_usd_path),
        working_dir=str(session_dir / "cache"),
        material_textures=material_textures,
        user_prompt=user_prompt_text,
    )

    # Save resolved config for audit / regeneration
    config_path = session_dir / "input" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(pipeline_config, f, default_flow_style=False)

    # Update session metadata
    manager.update_session(
        session_id,
        {
            "config": {
                "project_name": session_id,
                "usd_path": str(input_usd_path),
                "has_usd_upload": usd_file is not None
                and usd_file.filename is not None,
                "s3_uri": s3_uri,
                "material_textures": material_textures,
            },
        },
    )

    # Register and start pipeline execution
    job_registry = get_job_registry()
    await job_registry.register(
        session_id,
        execute_pipeline_async(
            session_id=session_id,
            config_dict=pipeline_config,
            session_manager=manager,
        ),
    )

    logger.info(f"Pipeline registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message="Pipeline queued for execution",
        estimated_duration_minutes=10,
    )


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(session_id: str) -> PipelineStatus:
    """Get pipeline execution status with detailed progress.

    Reads from in-memory event bus state for fast, real-time accuracy.
    Falls back to disk-based SessionManager for completed/stopped sessions.
    """
    from datetime import UTC, datetime

    event_bus = get_event_bus()
    manager = get_session_manager()

    # Try in-memory state first (active sessions)
    snapshot = event_bus.get_snapshot(session_id)

    if snapshot:
        metadata = snapshot
        session_meta = manager.get_session_metadata(session_id)
        preview_images = session_meta.get("preview_images", []) if session_meta else []
    else:
        metadata = manager.get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")
        preview_images = metadata.get("preview_images", [])

    preview_urls = [f"/artifacts/{session_id}/preview/{img}" for img in preview_images]

    created_at = datetime.fromisoformat(metadata["created_at"])
    elapsed_seconds = int((datetime.now(UTC) - created_at).total_seconds())
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


@router.get(
    "/{session_id}/results",
    response_model=PipelineResults | PipelineError,
)
async def get_pipeline_results(session_id: str):
    """Get pipeline execution results (only available when completed)."""
    manager = get_session_manager()

    metadata = manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    status = metadata["status"]

    if status == "completed":
        return PipelineResults(
            session_id=session_id,
            status=status,
            stats=metadata.get("results", {}),
            download_urls={
                "materials": f"/artifacts/{session_id}/materials",
                "textures": f"/artifacts/{session_id}/textures",
                "output": f"/artifacts/{session_id}/output",
                "renders": f"/artifacts/{session_id}/renders",
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
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline still {status}. Check status endpoint for progress.",
        )


@router.post("/{session_id}/cancel")
async def cancel_pipeline(session_id: str):
    """Cancel a running pipeline."""
    job_registry = get_job_registry()
    manager = get_session_manager()

    if not job_registry.is_running(session_id):
        metadata = manager.get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")

        if metadata["status"] not in ["pending", "running"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel pipeline with status: {metadata['status']}",
            )

        raise HTTPException(
            status_code=500,
            detail="Session not in job registry. Cannot cancel.",
        )

    # request_cancellation drops the `.cancel` marker (so the worker's
    # between-step is_cancelled() checkpoint sees it) and persists "cancelling"
    # to disk. The CANCELLING bus event then mirrors that into the in-memory
    # snapshot used by /status and notifies SSE subscribers. Both writers are
    # idempotent against terminal state — if the worker finished naturally in
    # the window after our is_running() guard, neither will downgrade the
    # final status.
    manager.request_cancellation(session_id)
    event_bus = get_event_bus()
    snapshot = event_bus.get_snapshot(session_id)
    current_step = (
        (snapshot or {}).get("current_step", {}).get("name", "pipeline")
        if snapshot
        else "pipeline"
    )
    await event_bus.emit(
        ProgressEvent(
            session_id=session_id,
            step=current_step,
            state=StepState.CANCELLING,
            message="Pipeline cancellation requested",
        )
    )

    # job_registry.cancel internally fires task.cancel() and waits up to 5s
    # for the worker to finish (cooperative path or asyncio cancellation).
    cancelled = await job_registry.cancel(session_id)

    if not cancelled:
        raise HTTPException(
            status_code=400,
            detail="Failed to cancel pipeline. It may have already completed.",
        )

    return {
        "session_id": session_id,
        "status": "cancelling",
        "message": "Pipeline cancellation requested",
    }


@router.get("/{session_id}/events")
async def stream_progress_events(session_id: str):
    """Stream real-time progress events via Server-Sent Events (SSE).

    Example client (JavaScript):
        const eventSource = new EventSource(`/pipeline/${sessionId}/events`);
        eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            console.log(`Step: ${data.step}, Progress: ${data.percent}%`);
        });
    """
    event_bus = get_event_bus()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is None:
        manager = get_session_manager()
        if not manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        """Generate SSE events from the session's event queue."""
        queue = event_bus.get_queue(session_id)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    event_data = event.model_dump_json()

                    yield {
                        "event": "progress",
                        "data": event_data,
                    }

                    should_close = False

                    if event.state in ["failed", "cancelled"]:
                        should_close = True
                    elif (
                        event.state == "completed"
                        and event.overall_percent
                        and event.overall_percent >= 100
                    ):
                        should_close = True

                    if should_close:
                        yield {
                            "event": "done",
                            "data": f'{{"session_id": "{session_id}", "final_state": "{event.state}"}}',
                        }
                        break

                except TimeoutError:
                    yield {"event": "ping", "data": "keepalive"}

        except asyncio.CancelledError:
            logger.debug(f"SSE stream cancelled for {session_id[:8]}...")
            raise

    return EventSourceResponse(event_generator(), ping=15)


@router.post(
    "/{session_id}/regenerate",
    response_model=SessionCreated,
    status_code=202,
)
async def regenerate_pipeline(
    session_id: str,
    request: RegenerateRequest,
) -> SessionCreated:
    """Regenerate specific pipeline steps from cached data.

    Useful for re-running texture generation with different prompts/opacity
    without re-discovering materials.
    """
    manager = get_session_manager()

    metadata = manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    if metadata["status"] in ["pending", "running", "cancelling"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot regenerate while pipeline is {metadata['status']}",
        )

    # Load the original config from session
    session_dir = manager.get_session_dir(session_id)
    config_path = session_dir / "input" / "config.yaml"

    if not config_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Original config not found for session",
        )

    with open(config_path) as f:
        pipeline_config = yaml.safe_load(f)

    # Determine which steps to re-run
    only_steps = [s.value for s in request.steps]

    # Override material_textures if provided
    if request.material_textures is not None:
        pipeline_config["material_textures"] = request.material_textures

    # Reset session status
    original_status = metadata["status"]
    manager.update_session(
        session_id,
        {
            "status": "pending",
            "current_step": None,
            "can_cancel": True,
        },
    )

    # Register and start regeneration
    job_registry = get_job_registry()
    try:
        await job_registry.register(
            session_id,
            execute_pipeline_async(
                session_id=session_id,
                config_dict=pipeline_config,
                session_manager=manager,
                only_steps=only_steps,
            ),
        )
    except Exception:
        manager.update_session(session_id, {"status": original_status})
        raise

    logger.info(f"Pipeline regeneration registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message=f"Regenerating steps: {', '.join(s.value for s in request.steps)}",
    )


@router.get("/{session_id}/event-log")
async def get_event_log(session_id: str) -> dict[str, Any]:
    """Get the persisted event log for a session."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    log_file = manager.get_session_dir(session_id) / "event_log.jsonl"

    if not log_file.exists():
        return {"events": []}

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
