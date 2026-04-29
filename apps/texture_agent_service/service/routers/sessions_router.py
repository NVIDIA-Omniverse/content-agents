# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sessions API endpoints - Session CRUD operations."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from ..runtime.registry import get_job_registry
from ..session.manager import SessionManager

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


@router.get("")
async def list_sessions():
    """List all sessions with metadata."""
    manager = get_session_manager()

    sessions = []
    session_dirs = (
        manager.storage_path.iterdir() if manager.storage_path.exists() else []
    )

    for session_dir in session_dirs:
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        metadata = manager.get_session_metadata(session_id)

        if metadata:
            sessions.append(
                {
                    "session_id": session_id,
                    "status": metadata.get("status", "unknown"),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at"),
                    "elapsed_seconds": metadata.get("elapsed_seconds", 0),
                    "config": metadata.get("config", {}),
                }
            )

    # Sort by created_at (newest first)
    sessions.sort(key=lambda x: x["created_at"] or "", reverse=True)

    return {"sessions": sessions, "total": len(sessions)}


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get detailed session information."""
    manager = get_session_manager()

    metadata = manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    return metadata


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a session and all its artifacts."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Cancel any running job first
    job_registry = get_job_registry()
    if job_registry.is_running(session_id):
        await job_registry.cancel(session_id)

    # Retry deletion with backoff
    max_retries = 3
    for attempt in range(max_retries):
        success = await asyncio.to_thread(manager.delete_session, session_id)
        if success:
            return None

        if not await asyncio.to_thread(manager.session_exists, session_id):
            return None

        if attempt < max_retries - 1:
            await asyncio.sleep(0.05 * (attempt + 1))

    raise HTTPException(status_code=500, detail="Failed to delete session")
