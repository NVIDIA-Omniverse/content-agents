# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for physics agent pipeline executions.

Delegates all persistence to a pluggable SessionStore (local or S3).
All public methods are async.
"""

import asyncio
import logging
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any

from ..runtime.progress import (
    STEP_COMPLETION_PERCENT,
    STEP_DISPLAY_NAMES,
    STEP_NUMBER,
    STEP_WEIGHTS,
    TOTAL_VISIBLE_STEPS,
)
from ..storage import LocalSessionStore, SessionStore
from ..storage.base import METADATA_KEY

logger = logging.getLogger(__name__)

# Session IDs are server-generated UUID4 strings but are also accepted back from
# URL path parameters (e.g. GET /sessions/{id}/...), so they must be validated
# before reaching any code that builds a filesystem path or storage key from
# them. The pattern is intentionally case-insensitive to tolerate normal UUID
# casing variance; it still rejects `../`, `/`, empty, and non-hex inputs.
_SESSION_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class InvalidSessionIdError(ValueError):
    """Raised when a session_id fails format validation.

    Subclasses ValueError so existing `except ValueError` / `pytest.raises(ValueError)`
    keeps working, but lets FastAPI's exception handler target just this class
    instead of swallowing every ValueError in the app.
    """


def _validate_session_id(session_id: str) -> str:
    """Validate that session_id has UUID shape; reject otherwise."""
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise InvalidSessionIdError(f"Invalid session_id: {session_id!r}")
    return session_id


class SessionManager:
    """Manages pipeline sessions and their artifacts.

    Wraps a SessionStore for persistence and keeps a local directory
    for pipeline working data (GPU rendering needs fast local I/O).
    """

    def __init__(
        self,
        storage_path: Path | str,
        ttl_hours: int = 24,
        store: SessionStore | None = None,
    ):
        self.storage_path = Path(storage_path)
        self.ttl_hours = ttl_hours
        self.store = store or LocalSessionStore(root_dir=str(self.storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock for safe read-modify-write."""
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def create_session(
        self, session_id: str, config: dict[str, Any] | None = None
    ) -> Path:
        """Create a new session with local dirs and store entry."""
        session_id = _validate_session_id(session_id)
        session_dir = self.storage_path / session_id

        # Create local directory structure (pipeline needs fast local I/O)
        (session_dir / "input").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "dataset").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "predictions").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "physics").mkdir(parents=True, exist_ok=True)
        (session_dir / "preview").mkdir(parents=True, exist_ok=True)

        # Initialize store entry
        await self.store.init_session(session_id)

        metadata = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": TOTAL_VISIBLE_STEPS,
                "percent": 0,
                "estimated_remaining_seconds": None,
            },
            "preview_images": [],
            "can_cancel": True,
            "elapsed_seconds": 0,
            "config": config or {},
            "ttl_expires_at": (
                datetime.now(UTC) + timedelta(hours=self.ttl_hours)
            ).isoformat(),
        }

        await self.store.put_json(session_id, METADATA_KEY, metadata)
        logger.info(f"Created session: {session_id}")
        return session_dir

    def get_session_dir(self, session_id: str) -> Path:
        """Get path to local session directory."""
        return self.storage_path / _validate_session_id(session_id)

    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists in the store."""
        session_id = _validate_session_id(session_id)
        return await self.store.exists(session_id, METADATA_KEY)

    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata from store."""
        session_id = _validate_session_id(session_id)
        return await self.store.get_json(session_id, METADATA_KEY)

    async def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update session metadata (read-modify-write with lock)."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            metadata = await self.store.get_json(session_id, METADATA_KEY)
            if not metadata:
                logger.warning(f"Cannot update non-existent session: {session_id}")
                return

            metadata.update(updates)
            metadata["updated_at"] = datetime.now(UTC).isoformat()

            created_at = datetime.fromisoformat(metadata["created_at"])
            now = datetime.now(UTC)
            # Handle naive datetimes from older sessions
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            metadata["elapsed_seconds"] = int((now - created_at).total_seconds())

            await self.store.put_json(session_id, METADATA_KEY, metadata)

    async def update_step_progress(
        self,
        session_id: str,
        step_name: str,
        progress: dict[str, Any],
    ) -> None:
        """Update progress for current step."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            metadata = await self.store.get_json(session_id, METADATA_KEY)
            if not metadata:
                return

            step_info = {
                "display": STEP_DISPLAY_NAMES.get(step_name, step_name),
                "step_num": STEP_NUMBER.get(step_name, 0),
            }

            current_step_info = metadata.get("current_step")
            if current_step_info and current_step_info.get("name") == step_name:
                started_at = datetime.fromisoformat(current_step_info["started_at"])
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=UTC)
                elapsed = int((datetime.now(UTC) - started_at).total_seconds())
                current_step_info["progress"] = progress
                current_step_info["elapsed_seconds"] = elapsed
            else:
                current_step_info = {
                    "name": step_name,
                    "display_name": step_info["display"],
                    "started_at": datetime.now(UTC).isoformat(),
                    "progress": progress,
                    "elapsed_seconds": 0,
                }

            metadata["current_step"] = current_step_info

            step_num = step_info["step_num"]
            if step_num > 0:
                step_progress_percent = progress.get("percent", 0)

                # Map the in-flight step percent through the shared weighted
                # range so the store-backed /status fallback matches the
                # EventBus (predict at 100% step-progress → 90% overall, not
                # 100% as the old raw passthrough produced).
                weights = STEP_WEIGHTS.get(step_name)
                if weights is not None:
                    start, end = weights
                    overall_percent = start + int(
                        (end - start) * step_progress_percent / 100
                    )
                else:
                    overall_percent = step_progress_percent

                metadata["overall_progress"]["current_step"] = step_num
                metadata["overall_progress"]["percent"] = min(100, overall_percent)

            metadata["updated_at"] = datetime.now(UTC).isoformat()
            await self.store.put_json(session_id, METADATA_KEY, metadata)

    async def mark_step_completed(
        self,
        session_id: str,
        step_name: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Mark a step as completed."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            metadata = await self.store.get_json(session_id, METADATA_KEY)
            if not metadata:
                return

            current_step_info = metadata.get("current_step")
            if current_step_info and current_step_info["name"] == step_name:
                started_at = datetime.fromisoformat(current_step_info["started_at"])
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=UTC)
                completed_at = datetime.now(UTC)
                duration = int((completed_at - started_at).total_seconds())

                completed_step = {
                    "name": step_name,
                    "display_name": current_step_info["display_name"],
                    "started_at": current_step_info["started_at"],
                    "completed_at": completed_at.isoformat(),
                    "duration_seconds": duration,
                    "stats": stats or {},
                }

                if "completed_steps" not in metadata:
                    metadata["completed_steps"] = []
                metadata["completed_steps"].append(completed_step)

                if "timings" not in metadata:
                    metadata["timings"] = {}
                metadata["timings"][step_name] = duration

                metadata["current_step"] = None

                # Keep current_step monotonic and clamp via STEP_NUMBER so the
                # optional optimize_usd step doesn't push the counter past
                # total_steps (optimize_usd collapses onto slot 1 alongside
                # identify_asset).
                metadata["overall_progress"]["current_step"] = max(
                    metadata["overall_progress"].get("current_step", 0),
                    STEP_NUMBER.get(step_name, len(metadata["completed_steps"])),
                )

                # Name-based percent lookup keeps this path in sync with the
                # EventBus regardless of which subset of steps actually ran.
                current_percent = metadata["overall_progress"].get("percent", 0)
                snapped = STEP_COMPLETION_PERCENT.get(step_name)
                if snapped is not None:
                    metadata["overall_progress"]["percent"] = max(
                        current_percent, snapped
                    )

                metadata["updated_at"] = datetime.now(UTC).isoformat()
                await self.store.put_json(session_id, METADATA_KEY, metadata)

    async def add_preview_image(self, session_id: str, image_name: str) -> None:
        """Add a preview image to the session."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            metadata = await self.store.get_json(session_id, METADATA_KEY)
            if not metadata:
                return

            if "preview_images" not in metadata:
                metadata["preview_images"] = []

            if image_name not in metadata["preview_images"]:
                metadata["preview_images"].append(image_name)
                await self.store.put_json(session_id, METADATA_KEY, metadata)

    async def update_preview_images(
        self, session_id: str, image_names: list[str]
    ) -> None:
        """Update the list of preview images."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            metadata = await self.store.get_json(session_id, METADATA_KEY)
            if not metadata:
                return

            metadata["preview_images"] = image_names
            await self.store.put_json(session_id, METADATA_KEY, metadata)

    async def is_cancelled(self, session_id: str) -> bool:
        """Check if session has been cancelled (works cross-instance via store)."""
        session_id = _validate_session_id(session_id)
        return await self.store.exists(session_id, ".cancel")

    async def request_cancellation(self, session_id: str) -> None:
        """Request cancellation — visible to all instances via store."""
        session_id = _validate_session_id(session_id)
        if not await self.session_exists(session_id):
            logger.warning(f"Cannot cancel non-existent session: {session_id}")
            return

        await self.store.put_bytes(session_id, ".cancel", b"")
        await self.update_session(session_id, {"status": "cancelling"})
        logger.info(f"Cancellation requested for session: {session_id}")

    async def get_artifact_path(
        self, session_id: str, artifact_type: str
    ) -> Path | None:
        """Get path to a local session artifact."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)

        artifact_map = {
            "predictions": session_dir / "cache" / "predictions" / "predictions.jsonl",
            "dataset": session_dir / "cache" / "dataset" / "dataset.jsonl",
            # apply_physics writes <stem>_physics.usda into cache/physics/.
            # The uploaded USD is normalized to scene.<ext>, so the stem is
            # always "scene" → cache/physics/scene_physics.usda.
            "output_usd": session_dir / "cache" / "physics" / "scene_physics.usda",
        }

        path = artifact_map.get(artifact_type)
        if path and path.exists():
            return path

        return None

    async def get_artifact_stream(
        self, session_id: str, artifact_type: str
    ) -> IO[bytes] | None:
        """Get artifact as a byte stream from store (works for S3)."""
        session_id = _validate_session_id(session_id)
        key_map = {
            "predictions": "cache/predictions/predictions.jsonl",
            "dataset": "cache/dataset/dataset.jsonl",
            "output_usd": "cache/physics/scene_physics.usda",
        }
        key = key_map.get(artifact_type)
        if not key:
            return None

        if not await self.store.exists(session_id, key):
            return None

        return await self.store.open_read(session_id, key)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from store and local disk."""
        session_id = _validate_session_id(session_id)
        try:
            await self.store.delete_session(session_id)
        except Exception as e:
            logger.error(f"Failed to delete session {session_id} from store: {e}")
            return False

        # Also clean up local directory (with retry for transient failures)
        session_dir = self.get_session_dir(session_id)
        if session_dir.exists():
            for attempt in range(3):
                try:
                    shutil.rmtree(session_dir)
                    break
                except OSError as e:
                    if attempt == 2:
                        logger.warning(f"Failed to delete local session dir: {e}")
                    else:
                        await asyncio.sleep(0.5 * (attempt + 1))

        # Clean up lock
        self._locks.pop(session_id, None)

        logger.info(f"Deleted session: {session_id}")
        return True

    async def list_sessions(self) -> list[str]:
        """List all session IDs from the store."""
        return await self.store.list_sessions()

    async def sync_to_store(self, session_id: str, prefix: str = "") -> int:
        """Sync local session files to the store (uploads to S3 if configured)."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        if not session_dir.exists():
            return 0
        return await self.store.sync_from_local(
            session_id, str(session_dir), prefix=prefix
        )

    async def sync_from_store(self, session_id: str, prefix: str = "") -> int:
        """Pull files from the store to local session directory (downloads from S3 if configured)."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return await self.store.sync_to_local(
            session_id, str(session_dir), prefix=prefix
        )

    async def cleanup_expired_sessions(self) -> int:
        """Remove sessions past their TTL."""
        cleaned = 0
        now = datetime.now(UTC)

        session_ids = await self.list_sessions()
        for session_id in session_ids:
            metadata = await self.get_session_metadata(session_id)
            if not metadata:
                continue

            expires_at_str = metadata.get("ttl_expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
                if now > expires_at:
                    logger.info(f"Cleaning up expired session: {session_id}")
                    if await self.delete_session(session_id):
                        cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired sessions")

        return cleaned
