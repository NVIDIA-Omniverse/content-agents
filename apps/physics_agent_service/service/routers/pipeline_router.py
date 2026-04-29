# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API endpoints - Core workflow operations."""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from physics_agent.api.defaults import build_default_pipeline_config
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
from ..runtime import get_event_bus, get_job_registry
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
    upload: UploadFile, dest: Path, chunk_size: int = 2 * 1024 * 1024
) -> int:
    """Stream upload file to disk in chunks to avoid memory spikes."""
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
            detail=f"Invalid USD file type in S3 URI: {ext}. Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
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
    """Upload a USD file and create a session for later pipeline execution."""
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
        session_dir = await manager.create_session(session_id)
        try:
            local_path = _download_s3_to_session(s3_uri, session_dir)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
            # Push input to store so other instances can find it
            try:
                await manager.sync_to_store(session_id)
            except Exception as e:
                logger.warning(f"Failed to sync to store for {session_id[:8]}: {e}")

            return SessionCreated(
                session_id=session_id,
                status="ready",
                message=f"USD downloaded from S3 successfully ({size_mb:.1f}MB)",
                estimated_duration_minutes=0,
            )
        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to download USD from S3: {e}")
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to download USD from S3: {e}"
            )

    # File upload path
    if usd_file.filename:
        ext = Path(usd_file.filename).suffix.lower()
        if ext not in _VALID_USD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid USD file type: {ext}. Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
            )

    session_dir = await manager.create_session(session_id)

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

        # Push input to store so other instances can find it
        try:
            await manager.sync_to_store(session_id)
        except Exception as e:
            logger.warning(f"Failed to sync to store for {session_id[:8]}: {e}")

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
        await manager.delete_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to upload USD: {e}")


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory."""
    input_dir = session_dir / "input"
    for ext in [".usd", ".usda", ".usdc", ".usdz"]:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


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
    user_prompt: str = Form(
        default="",
        description="Custom user prompt for VLM (optional)",
    ),
    render_backend: str = Form(
        default="",
        description="Rendering backend: 'remote' (default, HTTP render service; the bundled compose points this at the OVRTX sidecar), 'warp' (local CUDA), or 'ovrtx' (local Vulkan subprocess)",
    ),
    optimize_usd: bool = Form(
        default=False,
        description="Enable USD optimization step (default: false). "
        "When enabled, runs Scene Optimizer before rendering/prediction "
        "and restore_usd afterward to map results back to original paths.",
    ),
    enable_deinstance: bool = Form(
        default=True,
        description="Enable deinstance operation when optimize_usd is true "
        "(default: true). Required for instanced USD assets "
        "(e.g. robot arms with shared prototypes). FastAPI accepts common "
        "boolean form values such as true/false, 1/0, yes/no, and on/off.",
    ),
    enable_split: bool = Form(
        default=False,
        description="Enable split meshes operation when optimize_usd is true "
        "(default: false).",
    ),
    enable_deduplicate: bool = Form(
        default=False,
        description="Enable deduplicate operation when optimize_usd is true "
        "(default: false).",
    ),
) -> SessionCreated:
    """Create and execute a physics agent pipeline."""
    manager = get_session_manager()

    user_prompt_text = user_prompt.strip() if user_prompt else None

    if session_id:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        session_dir = manager.get_session_dir(session_id)

    elif s3_uri:
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)

        try:
            local_path = _download_s3_to_session(s3_uri, session_dir)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to download USD from S3: {e}")
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to download USD from S3: {e}"
            )

    elif usd_file:
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)

        try:
            if usd_file.filename:
                ext = Path(usd_file.filename).suffix.lower()
                if ext not in _VALID_USD_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid USD file type: {ext}. Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
                    )

            original_ext = (
                Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
            )
            usd_path = session_dir / "input" / f"scene{original_ext}"
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

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to save USD file: {e}")
            await manager.delete_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to save USD file: {e}")

    else:
        raise HTTPException(
            status_code=400,
            detail="One of usd_file, session_id, or s3_uri must be provided",
        )

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
        raise HTTPException(status_code=400, detail="Input USD not found for session")

    render_backend_text = render_backend.strip() if render_backend else None

    if optimize_usd and not any([enable_deinstance, enable_split, enable_deduplicate]):
        raise HTTPException(
            status_code=400,
            detail="At least one optimization operation must be enabled when "
            "optimize_usd is true (enable_deinstance, enable_split, or "
            "enable_deduplicate).",
        )

    pipeline_config = build_default_pipeline_config(
        session_id=session_id,
        usd_path=str(input_usd_path),
        working_dir=str(session_dir / "cache"),
        user_prompt=user_prompt_text,
        render_backend=render_backend_text,
        optimize_usd=optimize_usd,
        enable_deinstance=enable_deinstance,
        enable_split=enable_split,
        enable_deduplicate=enable_deduplicate,
    )

    config_path = session_dir / "input" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(pipeline_config, f, default_flow_style=False)

    await manager.update_session(
        session_id,
        {
            "config": {
                "project_name": pipeline_config.get("project", {}).get("name", ""),
                "usd_path": str(input_usd_path),
                "has_usd_upload": usd_file is not None
                and usd_file.filename is not None,
                "s3_uri": s3_uri,
                "user_prompt": user_prompt_text,
                "optimize_usd": optimize_usd,
                "enable_deinstance": enable_deinstance,
                "enable_split": enable_split,
                "enable_deduplicate": enable_deduplicate,
            },
        },
    )

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
        estimated_duration_minutes=15,
    )


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(session_id: str) -> PipelineStatus:
    """Get pipeline execution status with detailed progress.

    Reads from in-memory event bus state for fast, real-time accuracy.
    Falls back to store-based SessionManager for completed/cross-instance sessions.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    # Try in-memory state first (active sessions on this instance)
    snapshot = event_bus.get_snapshot(session_id)

    if snapshot:
        metadata = snapshot
        preview_images = snapshot.get("preview_images", [])
    else:
        # Fall back to store (works cross-instance)
        metadata = await manager.get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")
        preview_images = metadata.get("preview_images", [])

    preview_urls = [f"/artifacts/{session_id}/preview/{img}" for img in preview_images]

    created_at = datetime.fromisoformat(metadata["created_at"])
    now = datetime.now(UTC)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    elapsed_seconds = int((now - created_at).total_seconds())
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
    """Get pipeline execution results (only available when completed)."""
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    status = metadata["status"]

    if status == "completed":
        return PipelineResults(
            session_id=session_id,
            status=status,
            stats=metadata.get("results", {}),
            download_urls={
                "predictions": f"/artifacts/{session_id}/predictions",
                "report": f"/artifacts/{session_id}/report",
                "dataset": f"/artifacts/{session_id}/dataset",
                "output_usd": f"/artifacts/{session_id}/output-usd",
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
    """Cancel a running pipeline.

    Works cross-instance: writes a cancel signal to the store (S3)
    so the executing instance can detect it. Also tries local cancellation.
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

    Only works when connected to the instance running the pipeline.
    For cross-instance progress, use GET /pipeline/{session_id}/status (polling).
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is None:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

    terminal_states = ("completed", "failed", "cancelled")

    # If the pipeline is not running on this instance, SSE can't stream live events.
    # Return immediately so the client falls back to polling.
    if snapshot is None:
        metadata = await manager.get_session_metadata(session_id)
        final_state = (metadata or {}).get("status", "unknown")
        if final_state not in terminal_states:
            raise HTTPException(
                status_code=503,
                detail="Pipeline is running on a different instance; use polling instead",
            )

    async def event_generator():
        queue = event_bus.get_queue(session_id)

        # Check if already terminal (late connect to same instance after completion).
        if snapshot is not None and snapshot.get("status") in terminal_states:
            final_state = snapshot["status"]
            yield {
                "event": "done",
                "data": f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}',
            }
            return

        # If cross-instance and already terminal, send done and close.
        if snapshot is None:
            metadata = await manager.get_session_metadata(session_id)
            if metadata and metadata.get("status") in terminal_states:
                final_state = metadata["status"]
                yield {
                    "event": "done",
                    "data": f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}',
                }
                return

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
                    elif event.extra and event.extra.get("pipeline_ready"):
                        # Executor fired this after update_session + sync_to_store —
                        # status and artifacts are now available in S3.
                        should_close = True

                    if should_close:
                        yield {
                            "event": "done",
                            "data": f'{{"session_id": "{session_id}", "final_state": "{event.state}"}}',
                        }
                        break

                except TimeoutError:
                    # Check store on each timeout in case pipeline completed on another instance
                    metadata = await manager.get_session_metadata(session_id)
                    if metadata and metadata.get("status") in terminal_states:
                        final_state = metadata["status"]
                        yield {
                            "event": "done",
                            "data": f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}',
                        }
                        break
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
    """Regenerate specific pipeline steps from cached data."""
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    if metadata["status"] in ["pending", "running", "cancelling"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot regenerate while pipeline is {metadata['status']}",
        )

    session_dir = manager.get_session_dir(session_id)
    config_path = session_dir / "input" / "config.yaml"

    if not config_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Original config not found for session",
        )

    with open(config_path) as f:
        pipeline_config = yaml.safe_load(f)

    only_steps = [s.value for s in request.steps]

    if request.user_prompt is not None:
        steps_section = pipeline_config.get("steps", {})
        prepare_dataset = steps_section.get("build_dataset_prepare_dataset", {})
        prompts = prepare_dataset.get("prompts", {})
        prompts["user"] = request.user_prompt
        prepare_dataset["prompts"] = prompts
        steps_section["build_dataset_prepare_dataset"] = prepare_dataset
        pipeline_config["steps"] = steps_section

    await manager.update_session(
        session_id,
        {
            "status": "pending",
            "current_step": None,
            "can_cancel": True,
        },
    )

    job_registry = get_job_registry()
    await job_registry.register(
        session_id,
        execute_pipeline_async(
            session_id=session_id,
            config_dict=pipeline_config,
            session_manager=manager,
            only_steps=only_steps,
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
    """Get the persisted event log for a session."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Try store first (works cross-instance)
    try:
        events = await manager.store.get_event_log(session_id)
        return {"events": events, "total": len(events)}
    except Exception:
        logger.warning(
            f"Failed to read event log from store for {session_id[:8]}",
            exc_info=True,
        )

    # Fall back to local file
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
