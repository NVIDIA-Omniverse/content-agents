# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sessions API endpoints - Session CRUD operations."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import config as service_config
from ..models.responses import (
    SessionConfigSummary,
    SessionDetail,
    SessionListResponse,
    SessionSummary,
)
from ..runtime import get_event_bus, get_job_registry
from ..sanitization import sanitize_message, sanitize_payload, sanitize_step_stats
from ..session.manager import SessionManager
from .common import JSON_RESPONSE

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/sessions", tags=["sessions"])

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


# Live progress fields that the EventBus tracks per-session. The on-disk
# session.json is initialized with frozen defaults for these and only the
# top-level "status" gets re-persisted on terminal transitions, so a read
# served straight off disk reports current_step=None / completed_steps=[]
# / overall_progress.percent=0 even while the pipeline is mid-flight.
_BUS_OVERLAY_FIELDS = (
    "status",
    "current_step",
    "completed_steps",
    "overall_progress",
    "step_timings",
    "updated_at",
    "completed_at",
    "cancelled_at",
    "cancelling_at",
    "failed_at",
    "failed_step",
    "failed_step_stats",
    "error",
)

# Statuses that, once on disk, must not be downgraded by an older bus
# snapshot. The executor's outer exception handler can persist
# status="failed" directly via SessionManager.update_session() without
# emitting a FAILED event to the bus, so the bus snapshot may still
# carry a stale RUNNING/COMPLETED status while disk has the truthful
# terminal state. Without this guard, _build_session_view() would
# overlay the stale bus status and report running for an actually-
# failed session.
_DISK_TERMINAL_STATUSES = frozenset({"failed", "cancelled", "completed"})


def _build_session_view(
    session_id: str,
    *,
    check_exists: bool = True,
) -> dict[str, Any] | None:
    """Return the merged session view: disk metadata + live event-bus state.

    /sessions/{sid} and /sessions list must agree with /pipeline/{sid}/status
    on every observable field for the same session. Disk metadata alone is
    insufficient because per-step progress, timings, and terminal markers
    live in the EventBus snapshot until the worker explicitly persists them.
    Returns None if the session no longer exists on disk.
    """
    manager = get_session_manager()
    if check_exists and not manager.session_exists(session_id):
        return None

    metadata = manager.get_session_metadata(session_id)
    if metadata is None:
        return None

    disk_status = metadata.get("status")
    snapshot = get_event_bus().get_snapshot(session_id)
    if snapshot:
        for key in _BUS_OVERLAY_FIELDS:
            if key == "status" and disk_status in _DISK_TERMINAL_STATUSES:
                # Disk holds a terminal status set by a path that bypassed
                # the bus (e.g. the executor's outer exception handler).
                # Do not let an older non-terminal bus snapshot mask it.
                continue
            if key in snapshot:
                metadata[key] = snapshot[key]

    metadata["can_cancel"] = metadata.get("status") in ("pending", "running")

    created_at_str = metadata.get("created_at")
    if created_at_str:
        try:
            created_at = datetime.fromisoformat(created_at_str)
            metadata["elapsed_seconds"] = int(
                (datetime.now(UTC) - created_at).total_seconds()
            )
        except ValueError:
            pass

    return metadata


def _build_session_summary_list(manager: SessionManager) -> list[SessionSummary]:
    summaries: list[SessionSummary] = []
    list_metadata = getattr(manager, "list_session_metadata", None)
    if callable(list_metadata):
        metadata_rows = list_metadata()
    else:
        metadata_rows = [
            manager.get_session_metadata(session_id)
            for session_id in manager.list_sessions()
        ]

    for row in metadata_rows:
        if not row:
            continue
        view = dict(row)
        session_id = view.get("session_id")
        if not isinstance(session_id, str):
            continue

        disk_status = view.get("status")
        snapshot = get_event_bus().get_snapshot(session_id)
        if snapshot:
            for key in _BUS_OVERLAY_FIELDS:
                if key == "status" and disk_status in _DISK_TERMINAL_STATUSES:
                    continue
                if key in snapshot:
                    view[key] = snapshot[key]

        created_at_str = view.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
                view["elapsed_seconds"] = int(
                    (datetime.now(UTC) - created_at).total_seconds()
                )
            except ValueError:
                pass

        summaries.append(
            SessionSummary(
                session_id=session_id,
                status=view.get("status", "unknown"),
                created_at=view.get("created_at"),
                updated_at=view.get("updated_at"),
                elapsed_seconds=view.get("elapsed_seconds", 0),
                config=_build_config_summary(view.get("config")),
            )
        )

    summaries.sort(key=lambda s: s.created_at or "", reverse=True)
    return summaries


def _build_config_summary(raw_config: dict[str, Any] | None) -> SessionConfigSummary:
    """Build the public ``config`` summary from stored metadata."""
    raw = raw_config or {}
    return SessionConfigSummary(
        project_name=raw.get("project_name"),
        original_filename=raw.get("original_filename"),
        input_extension=raw.get("input_extension"),
        has_usd_upload=raw.get("has_usd_upload"),
        s3_uri=raw.get("s3_uri"),
        material_textures=raw.get("material_textures"),
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """List all sessions with sanitized metadata."""
    manager = get_session_manager()
    summaries = await asyncio.to_thread(_build_session_summary_list, manager)

    return SessionListResponse(sessions=summaries, total=len(summaries))


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str) -> SessionDetail:
    """Get detailed session information with sanitized error fields."""
    view = await asyncio.to_thread(_build_session_view, session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Session not found")

    storage_root = service_config.session_storage_path
    error = view.get("error")
    sanitized_error = sanitize_message(error, storage_root) if error else None
    completed_steps = sanitize_payload(view.get("completed_steps", []), storage_root)
    if not isinstance(completed_steps, list):
        completed_steps = []

    return SessionDetail(
        session_id=view.get("session_id", session_id),
        status=view.get("status", "unknown"),
        created_at=view.get("created_at"),
        updated_at=view.get("updated_at"),
        elapsed_seconds=view.get("elapsed_seconds", 0),
        ttl_expires_at=view.get("ttl_expires_at"),
        config=_build_config_summary(view.get("config")),
        current_step=view.get("current_step"),
        completed_steps=completed_steps,
        overall_progress=view.get("overall_progress"),
        preview_images=view.get("preview_images", []),
        can_cancel=view.get("can_cancel", False),
        error=sanitized_error,
        failed_step=view.get("failed_step"),
        failed_step_stats=sanitize_step_stats(
            view.get("failed_step_stats"), storage_root
        ),
        partial_results=sanitize_step_stats(view.get("partial_results"), storage_root),
        results=sanitize_step_stats(view.get("results"), storage_root),
        duration_seconds=view.get("duration_seconds"),
        completed_at=view.get("completed_at"),
    )


@router.delete(
    "/{session_id}",
    status_code=204,
    responses={404: JSON_RESPONSE, 409: JSON_RESPONSE, 500: JSON_RESPONSE},
)
async def delete_session(session_id: str):
    """Delete a session and all its artifacts."""
    manager = get_session_manager()

    if not await asyncio.to_thread(manager.session_exists, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    job_registry = get_job_registry()
    if job_registry.is_running(session_id):
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot delete an active session. Cancel it and wait for the worker "
                "to stop before deleting."
            ),
        )

    if await asyncio.to_thread(manager.is_worker_active, session_id):
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot delete an active session. A worker is still writing "
                "artifacts for this session."
            ),
        )

    event_bus = get_event_bus()

    # Retry deletion with backoff
    max_retries = 3
    for attempt in range(max_retries):
        success = await asyncio.to_thread(manager.delete_session, session_id)
        if success:
            await event_bus.cleanup_session(session_id)
            return None

        if await asyncio.to_thread(manager.is_worker_active, session_id):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot delete an active session. A worker is still writing "
                    "artifacts for this session."
                ),
            )

        if not await asyncio.to_thread(manager.session_exists, session_id):
            await event_bus.cleanup_session(session_id)
            return None

        if attempt < max_retries - 1:
            await asyncio.sleep(0.05 * (attempt + 1))

    raise HTTPException(status_code=500, detail="Failed to delete session")
