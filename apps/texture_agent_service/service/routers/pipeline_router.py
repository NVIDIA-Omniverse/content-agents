# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API endpoints - Core workflow operations."""

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from filelock import Timeout
from pydantic import ValidationError
from sse_starlette import EventSourceResponse
from world_understanding.utils.s3_utils import download_file_from_s3

from ..config import config
from ..models.requests import MaterialTextures, RegenerateRequest
from ..models.responses import (
    PipelineError,
    PipelineResults,
    PipelineStatus,
    SessionCreated,
)
from ..runtime import ProgressEvent, StepState, get_event_bus, get_job_registry
from ..sanitization import sanitize_message, sanitize_payload, sanitize_step_stats
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


def _material_textures_validation_detail(
    error: ValidationError,
    root_loc: list[str],
) -> list[dict[str, Any]]:
    """Translate pydantic locations to the API field that carried the JSON."""
    detail: list[dict[str, Any]] = []
    for item in error.errors():
        translated = dict(item)
        raw_loc = translated.get("loc", ())
        loc = list(raw_loc) if isinstance(raw_loc, list | tuple) else [raw_loc]
        if loc and loc[0] == "root":
            loc = loc[1:]
        translated["loc"] = [*root_loc, *loc]
        ctx = translated.get("ctx")
        if isinstance(ctx, dict):
            translated["ctx"] = {key: str(value) for key, value in ctx.items()}
        detail.append(translated)
    return detail


def _validate_material_textures(
    decoded: Any,
    root_loc: list[str],
) -> dict[str, Any]:
    """Validate material override payloads before accepting a pipeline job."""
    try:
        return MaterialTextures(root=decoded).as_config()
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=_material_textures_validation_detail(e, root_loc),
        )


async def _reserve_worker_slot(manager: SessionManager, session_id: str) -> Any:
    """Reserve the cross-process worker lock before acknowledging a job."""
    try:
        worker_lock = await asyncio.to_thread(
            manager.acquire_worker_lock, session_id, 0
        )
    except Timeout:
        raise HTTPException(
            status_code=409,
            detail=(
                "Session is still draining worker writes. Wait for the worker "
                "to stop before starting it."
            ),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    if await asyncio.to_thread(manager.is_worker_stalled, session_id):
        manager.release_worker_lock(worker_lock, session_id)
        raise HTTPException(
            status_code=409,
            detail=(
                "Session is still draining worker writes. Wait for the worker "
                "to stop before starting it."
            ),
        )

    return worker_lock


def _release_worker_slot_callback(
    manager: SessionManager,
    session_id: str,
    worker_lock: Any,
) -> Callable[[], None]:
    """Build a typed registry callback that releases an accepted-job lock."""

    def _release() -> None:
        manager.release_worker_lock(worker_lock, session_id)

    return _release


def _cancel_never_started_callback(
    manager: SessionManager,
    session_id: str,
) -> Callable[[], None]:
    """Mark a queued job cancelled if it never reaches the executor body."""

    def _cancel() -> None:
        try:
            manager.update_session(
                session_id,
                {
                    "status": "cancelled",
                    "can_cancel": False,
                },
            )
        except FileNotFoundError:
            return
        except Exception:
            logger.exception(
                "Failed to persist pre-start cancellation for %s", session_id[:8]
            )
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        emit_task = loop.create_task(
            get_event_bus().emit(
                ProgressEvent(
                    session_id=session_id,
                    step="pipeline",
                    state=StepState.CANCELLED,
                    message="Pipeline cancelled before startup",
                )
            )
        )

        def _log_emit_failure(task: asyncio.Task) -> None:
            try:
                task.result()
            except Exception:
                logger.exception(
                    "Failed to emit pre-start cancellation for %s", session_id[:8]
                )

        emit_task.add_done_callback(_log_emit_failure)

    return _cancel


_RUN_SCOPED_METADATA_FIELDS = (
    "error",
    "failed_step",
    "failed_step_stats",
    "failed_at",
    "partial_results",
)


def _reset_session_for_new_run(
    manager: SessionManager,
    session_id: str,
    *,
    fresh: bool,
) -> dict[str, Any]:
    """Reset run-scoped state on an existing session before a new run.

    Must be called with the cross-process worker lock held so a peer
    cancel cannot drop a fresh `.cancel` between the clear and the
    coroutine starting. Resets the four state surfaces that can leak
    from a prior run into a new one:

    - the durable `.cancel` marker (executor's between-step checkpoint),
    - the EventBus in-memory snapshot read by `/status`,
    - the EventBus per-session SSE queue read by `/events`,
    - run-scoped session metadata fields surfaced by `/sessions/{id}`.

    `fresh=True` (existing-session reuse via `POST /pipeline`) also
    clears `completed_steps`, `overall_progress`, `preview_images`, and
    `current_step` because the new run starts from scratch. `fresh=False`
    (regenerate) keeps those because a regenerate is incremental on top
    of the already-completed steps.

    Returns a snapshot of the prior values for every metadata field the
    reset overwrote. Pass it to `_restore_session_after_reset_failure`
    in an `except` block so a subsequent failure (e.g., `register()` or
    a config-write race) does not leave the session permanently in
    `pending` with all prior diagnostics wiped.
    """
    metadata = manager.get_session_metadata(session_id) or {}
    snapshot_keys: tuple[str, ...] = (
        "status",
        "current_step",
        "can_cancel",
    ) + _RUN_SCOPED_METADATA_FIELDS
    if fresh:
        snapshot_keys = snapshot_keys + (
            "completed_steps",
            "preview_images",
            "overall_progress",
        )
    snapshot = {key: metadata.get(key) for key in snapshot_keys}

    manager.clear_cancellation(session_id)
    get_event_bus().clear_session_state(session_id)

    metadata_reset: dict[str, Any] = {
        "status": "pending",
        "current_step": None,
        "can_cancel": True,
    }
    for field in _RUN_SCOPED_METADATA_FIELDS:
        metadata_reset[field] = None
    if fresh:
        metadata_reset["completed_steps"] = []
        metadata_reset["preview_images"] = []
        metadata_reset["overall_progress"] = {
            "current_step": 0,
            "total_steps": 8,
            "percent": 0,
            "estimated_remaining_seconds": None,
        }
    manager.update_session(session_id, metadata_reset)
    return snapshot


def _restore_session_after_reset_failure(
    manager: SessionManager,
    session_id: str,
    snapshot: dict[str, Any],
) -> None:
    """Re-apply the prior metadata snapshot after a post-reset failure.

    Called from `except` blocks when a step after `_reset_session_for_new_run`
    raises (validation, config-write race, register failure, etc.). Without
    this, the session would be permanently stuck in `pending` with prior
    diagnostics wiped and no executor coroutine ever scheduled.

    The bus snapshot and `.cancel` marker are deliberately not restored:
    the bus snapshot rebuilds lazily from disk on next read, and the
    `.cancel` marker reflected a pre-existing cancellation that the
    caller already chose to abandon by accepting the retry.
    """
    if not snapshot:
        return
    try:
        manager.update_session(session_id, snapshot)
    except Exception:
        logger.exception(
            "Failed to restore session metadata for %s after reset rollback",
            session_id,
        )


def _uses_per_prim_overrides(material_textures: dict[str, Any] | None) -> bool:
    """Return whether material overrides request per-prim texture units."""
    if not material_textures:
        return False
    return any(
        isinstance(override, dict) and bool(override.get("per_prim"))
        for override in material_textures.values()
    )


def _sync_texture_mode_for_overrides(
    pipeline_config: dict[str, Any],
    material_textures: dict[str, Any] | None,
) -> None:
    """Promote configs with per-prim overrides without downgrading stored mode."""
    if _uses_per_prim_overrides(material_textures):
        pipeline_config.setdefault("texture", {})["mode"] = "per_prim"


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

    pipeline_config = {
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
    _sync_texture_mode_for_overrides(pipeline_config, material_textures)
    return pipeline_config


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
        description=(
            "Per-material texture config JSON. Shape: "
            '{"Material": {"prompt": "rusted steel", "opacity": 0.85, '
            '"per_prim": {"/World/Prim": {"prompt": "scratches", "opacity": 0.65}}}}. '
            "Material prompt is required and non-empty, opacity is optional "
            "and bounded to 0.0-1.0, unknown fields are rejected, and any "
            "per_prim entry runs the request in per-prim texture mode."
        ),
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
    texture prompts, blend opacity, and per-prim overrides.
    """
    manager = get_session_manager()

    # Parse material_textures from JSON. Use 422 with a structured detail
    # list matching FastAPI's request-validation format and validate the
    # decoded wire shape before accepting the request.
    material_textures: dict[str, Any] | None = None
    if material_textures_json and material_textures_json.strip():
        try:
            decoded = json.loads(material_textures_json)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "json_invalid",
                        "loc": ["form", "material_textures_json"],
                        "msg": f"JSON decode error: {e}",
                    }
                ],
            )
        if not isinstance(decoded, dict):
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "dict_type",
                        "loc": ["form", "material_textures_json"],
                        "msg": (
                            "Input should be a JSON object mapping material "
                            "names to override dicts"
                        ),
                    }
                ],
            )
        bad_keys = [k for k, v in decoded.items() if not isinstance(v, dict)]
        if bad_keys:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "dict_type",
                        "loc": ["form", "material_textures_json", k],
                        "msg": (
                            "Per-material override must be an object with "
                            "prompt/opacity fields"
                        ),
                    }
                    for k in bad_keys
                ],
            )
        material_textures = _validate_material_textures(
            decoded, ["form", "material_textures_json"]
        )

    worker_lock: Any | None = None
    reused_existing_session = False

    if session_id:
        # Path 1: reuse existing session
        if not manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        reused_existing_session = True

        # Prevent concurrent re-start of a running session. Reserve the
        # cross-process lock before reading or mutating session files so DELETE
        # and peer POST requests cannot interleave with config/metadata writes.
        job_registry = get_job_registry()
        if job_registry.is_running(session_id):
            raise HTTPException(
                status_code=409,
                detail="Session is already running. Cancel it first or wait for completion.",
            )
        worker_lock = await _reserve_worker_slot(manager, session_id)

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

    reset_snapshot: dict[str, Any] = {}
    try:
        if worker_lock is None:
            worker_lock = await _reserve_worker_slot(manager, session_id)

        # Find the input USD
        input_usd_path = _find_input_usd(session_dir)
        if not input_usd_path:
            raise HTTPException(
                status_code=400, detail="Input USD not found for session"
            )

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

        # Update session metadata. Only fields that are safe for the public
        # ``/sessions`` and ``/sessions/{id}`` responses are persisted here:
        # the absolute ``usd_path`` is intentionally omitted because it would
        # leak the container's internal storage layout.
        input_extension = input_usd_path.suffix.lower()
        original_filename = (
            usd_file.filename if (usd_file is not None and usd_file.filename) else None
        )
        manager.update_session(
            session_id,
            {
                "config": {
                    "project_name": session_id,
                    "input_extension": input_extension,
                    "original_filename": original_filename,
                    "has_usd_upload": usd_file is not None
                    and usd_file.filename is not None,
                    "s3_uri": s3_uri,
                    "material_textures": material_textures,
                },
            },
        )

        # Reset run-scoped state on reused sessions only after every step
        # that could fail above has succeeded. The executor reads `.cancel`
        # and metadata when it starts; `/status` reads the bus snapshot.
        # Resetting earlier (then failing in validation/config write) would
        # leave the session permanently `pending` with prior diagnostics
        # wiped — see `_restore_session_after_reset_failure` for the
        # post-register rollback path.
        if reused_existing_session:
            reset_snapshot = _reset_session_for_new_run(manager, session_id, fresh=True)

        # Register and start pipeline execution
        job_registry = get_job_registry()
        await job_registry.register(
            session_id,
            execute_pipeline_async(
                session_id=session_id,
                config_dict=pipeline_config,
                session_manager=manager,
                acquire_worker_lock=False,
            ),
            on_never_started=_cancel_never_started_callback(
                manager,
                session_id,
            ),
            on_finished=_release_worker_slot_callback(
                manager,
                session_id,
                worker_lock,
            ),
        )
    except Exception:
        if worker_lock is not None:
            manager.release_worker_lock(worker_lock, session_id)
        if reset_snapshot:
            _restore_session_after_reset_failure(manager, session_id, reset_snapshot)
        raise

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

    Uses the same merged disk+bus view as ``/sessions/{sid}`` so the two
    endpoints agree on every observable field for the same session, even
    when the executor's outer exception handler persists a terminal disk
    status without emitting a corresponding bus event.
    """
    # Imported here to keep the cross-router dependency local rather than
    # introducing it at module load time.
    from .sessions_router import _build_session_view

    view = _build_session_view(session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Session not found")

    preview_images = view.get("preview_images", [])
    preview_urls = [f"/artifacts/{session_id}/preview/{img}" for img in preview_images]

    # Sanitize at read time too -- session.json files written before the
    # write-time scrubbing fix landed may still hold raw NVCF URLs or
    # absolute session paths in the failure diagnostics.
    storage_root = config.session_storage_path
    completed_steps = sanitize_payload(view.get("completed_steps", []), storage_root)
    if not isinstance(completed_steps, list):
        completed_steps = []

    return PipelineStatus(
        session_id=session_id,
        status=view["status"],
        current_step=view.get("current_step"),
        completed_steps=completed_steps,
        overall_progress=view.get("overall_progress", {}),
        preview_images=preview_urls,
        can_cancel=view["can_cancel"],
        elapsed_seconds=view["elapsed_seconds"],
        created_at=view["created_at"],
        updated_at=view["updated_at"],
        failed_step=view.get("failed_step"),
        failed_step_stats=sanitize_step_stats(
            view.get("failed_step_stats"), storage_root
        ),
    )


@router.get(
    "/{session_id}/results",
    response_model=PipelineResults | PipelineError,
)
async def get_pipeline_results(session_id: str):
    """Get pipeline execution results (only available when completed).

    Reads from the same merged disk+bus view as ``/sessions/{sid}`` and
    ``/pipeline/{sid}/status``: when the bus has reached a terminal status
    but ``_persist_status`` hasn't yet awaited its disk write, a disk-only
    read here would briefly return 202 ("still running") while the other
    two endpoints already report "completed".
    """
    from .sessions_router import _build_session_view

    view = _build_session_view(session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Session not found")

    status = view["status"]
    storage_root = config.session_storage_path

    if status == "completed":
        # Sanitize ``results`` for legacy session.json files written before
        # the executor's write-time scrubber landed. A completed run with
        # partial failures (threshold not hit) carries ``errors`` records
        # whose ``message`` field can still hold an NVCF function-
        # invocation URL or absolute session path.
        sanitized_stats = sanitize_step_stats(view.get("results", {}), storage_root)
        return PipelineResults(
            session_id=session_id,
            status=status,
            stats=sanitized_stats or {},
            download_urls={
                "materials": f"/artifacts/{session_id}/materials",
                "textures": f"/artifacts/{session_id}/textures",
                "output": f"/artifacts/{session_id}/output",
                "renders": f"/artifacts/{session_id}/renders",
            },
            duration_seconds=view.get("duration_seconds", 0),
            completed_at=view.get("completed_at", ""),
        )

    elif status == "failed":
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=sanitize_message(
                view.get("error", "Unknown error"), storage_root
            ),
            failed_step=view.get("failed_step", "unknown"),
            completed_steps=[s["name"] for s in view.get("completed_steps", [])],
            partial_results=sanitize_step_stats(
                view.get("partial_results"), storage_root
            ),
            failed_step_stats=sanitize_step_stats(
                view.get("failed_step_stats"), storage_root
            ),
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

    metadata = manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    if metadata["status"] not in ["pending", "running", "cancelling"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel pipeline with status: {metadata['status']}",
        )

    # request_cancellation drops the `.cancel` marker (so the worker's
    # between-step is_cancelled() checkpoint sees it) and persists "cancelling"
    # to disk. The CANCELLING bus event then mirrors that into the in-memory
    # snapshot used by /status and notifies SSE subscribers. Both writers are
    # idempotent against terminal state — if the worker finished naturally in
    # the window after our is_running() guard, neither will downgrade the
    # final status. In a multi-process deployment, this disk marker is the
    # only shared cancellation signal; JobRegistry only knows about local
    # asyncio tasks.
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

    if job_registry.is_running(session_id):
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
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Register the per-session queue here in the route handler rather than
    # lazily inside the generator. EventSourceResponse runs the generator
    # body only once SSE iteration starts, which opens a window between the
    # session_exists() check above and the first queue.get(). A DELETE
    # landing in that window would leave cleanup_session() with no queue to
    # notify, then the generator would call get_queue() and silently
    # setdefault() a fresh queue for an already-deleted session. Resolving
    # the queue eagerly closes that race -- cleanup_session() will see the
    # queue, push the terminal sentinel, and the generator will pick it up
    # immediately on its first iteration.
    queue = event_bus.get_queue(session_id)

    async def event_generator():
        """Generate SSE events from the session's event queue."""
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
                    # Defense-in-depth against the cleanup_session sentinel
                    # being missed (e.g. cleanup ran on a different queue
                    # object than this generator holds, or the session was
                    # deleted before any subscriber attached). Bound the
                    # post-delete idle-stream lifetime to one keepalive
                    # interval rather than indefinitely.
                    if not manager.session_exists(session_id):
                        yield {
                            "event": "done",
                            "data": (
                                f'{{"session_id": "{session_id}", '
                                f'"final_state": "deleted"}}'
                            ),
                        }
                        break
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
    worker_lock = await _reserve_worker_slot(manager, session_id)
    reset_snapshot: dict[str, Any] = {}

    try:
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

        steps_cfg = pipeline_config.get("steps", {}) or {}
        disabled_requested = [
            s for s in only_steps if not steps_cfg.get(s, {}).get("enabled", True)
        ]
        if disabled_requested:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Requested step(s) are disabled in this deploy: "
                    f"{', '.join(disabled_requested)}. The default Docker "
                    "Compose deploy does not configure a rendering backend; "
                    "render_previews and render are disabled. Either deploy "
                    "with a rendering backend configured or omit these steps."
                ),
            )

        # Override material_textures if provided
        material_textures_config = request.material_textures_config()
        if material_textures_config is not None:
            pipeline_config["material_textures"] = material_textures_config
            _sync_texture_mode_for_overrides(pipeline_config, material_textures_config)

        # Regenerate is incremental — keep completed_steps / progress —
        # but every other run-scoped state surface must be reset so the
        # executor and `/status` cannot see prior-run remnants. Reset is
        # deferred until after every step that could fail above has
        # succeeded; the snapshot drives rollback if `register()` raises.
        reset_snapshot = _reset_session_for_new_run(manager, session_id, fresh=False)

        job_registry = get_job_registry()
        await job_registry.register(
            session_id,
            execute_pipeline_async(
                session_id=session_id,
                config_dict=pipeline_config,
                session_manager=manager,
                only_steps=only_steps,
                acquire_worker_lock=False,
            ),
            on_never_started=_cancel_never_started_callback(
                manager,
                session_id,
            ),
            on_finished=_release_worker_slot_callback(
                manager,
                session_id,
                worker_lock,
            ),
        )
    except Exception:
        manager.release_worker_lock(worker_lock, session_id)
        if reset_snapshot:
            _restore_session_after_reset_failure(manager, session_id, reset_snapshot)
        raise

    logger.info(f"Pipeline regeneration registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message=f"Regenerating steps: {', '.join(s.value for s in request.steps)}",
    )


@router.get("/{session_id}/event-log")
async def get_event_log(session_id: str) -> dict[str, Any]:
    """Get the persisted event log for a session with sanitized diagnostics."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    log_file = manager.get_session_dir(session_id) / "event_log.jsonl"

    if not log_file.exists():
        return {"events": []}

    storage_root = config.session_storage_path
    events = []
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    if isinstance(event, dict):
                        if isinstance(event.get("message"), str):
                            event["message"] = sanitize_message(
                                event["message"], storage_root
                            )
                        extra = event.get("extra")
                        if isinstance(extra, dict):
                            event["extra"] = sanitize_step_stats(extra, storage_root)
                    events.append(event)

        return {"events": events, "total": len(events)}

    except Exception as e:
        logger.error(f"Failed to load event log for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load event log: {e}")
