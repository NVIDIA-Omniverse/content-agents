# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for pipeline executions."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..storage.base import METADATA_KEY, SessionStore
from ..storage.local_store import LocalSessionStore

logger = logging.getLogger(__name__)

# Cancel signal key
CANCEL_KEY = ".cancel"

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

    All storage operations are delegated to the configured SessionStore backend.
    If no store is provided, defaults to LocalSessionStore.
    All methods are async to support non-blocking I/O.
    """

    def __init__(
        self,
        storage_path: Path | str,
        ttl_hours: int = 24,
        store: SessionStore | None = None,
    ):
        """Initialize session manager.

        Args:
            storage_path: Base directory for local session storage (used as
                          default root for LocalSessionStore if no store provided)
            ttl_hours: Time-to-live for sessions in hours
            store: Storage backend (defaults to LocalSessionStore at storage_path)
        """
        self.storage_path = Path(storage_path)
        self.ttl_hours = ttl_hours
        self._update_locks: dict[str, asyncio.Lock] = {}

        # Default to LocalSessionStore if no store provided
        if store is None:
            self._store = LocalSessionStore(str(self.storage_path))
        else:
            self._store = store

        # Ensure storage directory exists (for local store compatibility)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @property
    def store(self) -> SessionStore:
        """Get the configured storage backend."""
        return self._store

    # ---------- Session Lifecycle ----------

    async def create_session(
        self, session_id: str, config: dict[str, Any] | None = None
    ) -> Path:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            config: Optional configuration dict
        """
        session_id = _validate_session_id(session_id)
        # Initialize session in store
        await self._store.init_session(session_id)

        # Create local directory structure (for backward compat with file-based ops)
        session_dir = self.storage_path / session_id

        # Create directory structure
        (session_dir / "input").mkdir(parents=True, exist_ok=True)
        (session_dir / "materials").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "dataset").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "predictions").mkdir(parents=True, exist_ok=True)
        (session_dir / "preview").mkdir(parents=True, exist_ok=True)
        (session_dir / "output").mkdir(parents=True, exist_ok=True)

        # Initialize session metadata
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": 3,
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

        # Save metadata via store
        await self._store.put_json(session_id, METADATA_KEY, metadata)

        logger.info(f"Created session: {session_id}")
        return session_dir

    def get_session_dir(self, session_id: str) -> Path:
        """Get path to local session directory.

        Args:
            session_id: Session identifier

        Returns:
            Path to session directory
        """
        return self.storage_path / _validate_session_id(session_id)

    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists
        """
        session_id = _validate_session_id(session_id)
        return await self._store.exists(session_id, METADATA_KEY)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its artifacts.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        session_id = _validate_session_id(session_id)
        try:
            await self._store.delete_session(session_id)
            self._update_locks.pop(session_id, None)
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    # ---------- Metadata Operations ----------

    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata dict or None if not found
        """
        session_id = _validate_session_id(session_id)
        return await self._store.get_json(session_id, METADATA_KEY)

    async def get_session_metadata_batch(
        self, session_ids: list[str]
    ) -> list[dict[str, Any] | None]:
        """Get metadata for multiple sessions in a single batch.

        Uses the store's batch method to reuse a single connection.

        Args:
            session_ids: List of session identifiers

        Returns:
            List of metadata dicts (or None), matching input order
        """
        session_ids = [_validate_session_id(sid) for sid in session_ids]
        return await self._store.get_json_batch(session_ids, METADATA_KEY)

    def _get_update_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock for update_session.

        Prevents concurrent read-modify-write races when multiple callers
        (e.g., EventBus and executor) update the same session concurrently.
        """
        return self._update_locks.setdefault(session_id, asyncio.Lock())

    async def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update session metadata.

        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update
        """
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            metadata = await self.get_session_metadata(session_id)
            if not metadata:
                logger.warning(f"Cannot update non-existent session: {session_id}")
                return

            # Update fields
            metadata.update(updates)
            metadata["updated_at"] = datetime.now(UTC).isoformat()

            # Recalculate elapsed time
            created_at_str = metadata["created_at"].replace("Z", "")
            created_at = datetime.fromisoformat(created_at_str)
            metadata["elapsed_seconds"] = int(
                (datetime.now(UTC) - created_at).total_seconds()
            )

            # Save updated metadata
            await self._store.put_json(session_id, METADATA_KEY, metadata)

            # sync session to store
            await self.sync_session_to_store(session_id)

    async def update_step_progress(
        self,
        session_id: str,
        step_name: str,
        progress: dict[str, Any],
    ) -> None:
        """Update progress for current step.

        Args:
            session_id: Session identifier
            step_name: Name of current step
            progress: Progress dict with current, total, percent, message
        """
        session_id = _validate_session_id(session_id)
        metadata = await self.get_session_metadata(session_id)
        if not metadata:
            return

        # Map step names to display names and step numbers
        step_info_map = {
            "build_dataset_usd": {"display": "Rendering USD Scene", "step_num": 1},
            "predict": {"display": "Running VLM Predictions", "step_num": 2},
            "apply": {"display": "Applying Materials", "step_num": 3},
        }

        step_info = step_info_map.get(step_name, {"display": step_name, "step_num": 0})

        # Calculate elapsed time for current step
        current_step_info = metadata.get("current_step")
        if current_step_info and current_step_info.get("name") == step_name:
            # Update existing step
            started_at = datetime.fromisoformat(current_step_info["started_at"])
            elapsed = int((datetime.now(UTC) - started_at).total_seconds())
            current_step_info["progress"] = progress
            current_step_info["elapsed_seconds"] = elapsed
        else:
            # New step started
            current_step_info = {
                "name": step_name,
                "display_name": step_info["display"],
                "started_at": datetime.now(UTC).isoformat(),
                "progress": progress,
                "elapsed_seconds": 0,
            }

        metadata["current_step"] = current_step_info

        # Update overall progress based on current step
        step_num = step_info["step_num"]
        if step_num > 0:
            step_progress_percent = progress.get("percent", 0)
            # Progress percent is already pre-scaled to overall progress
            overall_percent = step_progress_percent

            metadata["overall_progress"]["current_step"] = step_num
            metadata["overall_progress"]["percent"] = min(100, overall_percent)

        await self._store.put_json(session_id, METADATA_KEY, metadata)

    async def mark_step_completed(
        self,
        session_id: str,
        step_name: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Mark a step as completed.

        Args:
            session_id: Session identifier
            step_name: Name of completed step
            stats: Optional statistics from step execution
        """
        session_id = _validate_session_id(session_id)
        metadata = await self.get_session_metadata(session_id)
        if not metadata:
            return

        current_step_info = metadata.get("current_step")
        if current_step_info and current_step_info["name"] == step_name:
            # Calculate duration
            started_at = datetime.fromisoformat(current_step_info["started_at"])
            completed_at = datetime.now(UTC)
            duration = int((completed_at - started_at).total_seconds())

            # Create completed step record
            completed_step = {
                "name": step_name,
                "display_name": current_step_info["display_name"],
                "started_at": current_step_info["started_at"],
                "completed_at": completed_at.isoformat(),
                "duration_seconds": duration,
                "stats": stats or {},
            }

            # Add to completed steps
            if "completed_steps" not in metadata:
                metadata["completed_steps"] = []
            metadata["completed_steps"].append(completed_step)

            # Track timing for this step
            if "timings" not in metadata:
                metadata["timings"] = {}
            metadata["timings"][step_name] = duration

            # Clear current step
            metadata["current_step"] = None

            # Update overall progress
            completed_count = len(metadata["completed_steps"])
            metadata["overall_progress"]["current_step"] = completed_count

            # Use cumulative percentages based on unequal allocation
            # Rendering: 0-50%, Prediction: 50-90%, Apply: 90-100%
            cumulative_percents = [50, 90, 100]
            if completed_count <= len(cumulative_percents):
                metadata["overall_progress"]["percent"] = cumulative_percents[
                    completed_count - 1
                ]
            else:
                metadata["overall_progress"]["percent"] = 100

            await self._store.put_json(session_id, METADATA_KEY, metadata)

    async def add_preview_image(self, session_id: str, image_name: str) -> None:
        """Add a preview image to the session.

        Args:
            session_id: Session identifier
            image_name: Name of preview image file
        """
        session_id = _validate_session_id(session_id)
        metadata = await self.get_session_metadata(session_id)
        if not metadata:
            return

        if "preview_images" not in metadata:
            metadata["preview_images"] = []

        # Add if not already in list
        if image_name not in metadata["preview_images"]:
            metadata["preview_images"].append(image_name)
            await self._store.put_json(session_id, METADATA_KEY, metadata)

    async def update_preview_images(
        self, session_id: str, image_names: list[str]
    ) -> None:
        """Update the list of preview images.

        Args:
            session_id: Session identifier
            image_names: List of preview image filenames
        """
        session_id = _validate_session_id(session_id)
        metadata = await self.get_session_metadata(session_id)
        if not metadata:
            return

        metadata["preview_images"] = image_names
        await self._store.put_json(session_id, METADATA_KEY, metadata)

    async def add_generated_reference_image(
        self, session_id: str, entry: dict[str, Any]
    ) -> None:
        """Append a generated-reference image metadata record."""
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            metadata = await self.get_session_metadata(session_id)
            if not metadata:
                return

            generated_refs = list(metadata.get("generated_reference_images", []))
            generated_refs.append(entry)
            metadata["generated_reference_images"] = generated_refs
            metadata["updated_at"] = datetime.now(UTC).isoformat()
            await self._store.put_json(session_id, METADATA_KEY, metadata)
            await self.sync_session_to_store(session_id)

    async def remove_generated_reference_image(
        self, session_id: str, reference_id: str
    ) -> dict[str, Any] | None:
        """Remove a generated-reference metadata record by ID."""
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            metadata = await self.get_session_metadata(session_id)
            if not metadata:
                return None

            generated_refs = list(metadata.get("generated_reference_images", []))
            kept_refs = [ref for ref in generated_refs if ref.get("id") != reference_id]
            if len(kept_refs) == len(generated_refs):
                return None

            removed_ref = next(
                ref for ref in generated_refs if ref.get("id") == reference_id
            )
            metadata["generated_reference_images"] = kept_refs
            metadata["updated_at"] = datetime.now(UTC).isoformat()
            await self._store.put_json(session_id, METADATA_KEY, metadata)
            await self.sync_session_to_store(session_id)
            return removed_ref

    # ---------- Cancellation ----------

    async def is_cancelled(self, session_id: str) -> bool:
        """Check if session has been cancelled.

        Args:
            session_id: Session identifier

        Returns:
            True if cancellation signal exists
        """
        session_id = _validate_session_id(session_id)
        return await self._store.exists(session_id, CANCEL_KEY)

    async def request_cancellation(self, session_id: str) -> None:
        """Request cancellation of a running pipeline.

        Args:
            session_id: Session identifier
        """
        session_id = _validate_session_id(session_id)
        if not await self.session_exists(session_id):
            logger.warning(f"Cannot cancel non-existent session: {session_id}")
            return

        # Create cancellation signal
        await self._store.put_bytes(session_id, CANCEL_KEY, b"")

        # Update status
        await self.update_session(session_id, {"status": "cancelling"})

        logger.info(f"Cancellation requested for session: {session_id}")

    # ---------- Artifact Operations ----------

    async def get_artifact_path(
        self, session_id: str, artifact_type: str
    ) -> Path | None:
        """Get path to a session artifact (local filesystem).

        Args:
            session_id: Session identifier
            artifact_type: Type of artifact (output_usd, predictions, etc.)

        Returns:
            Path to artifact or None if not found
        """
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)

        artifact_map = {
            "output_usd": session_dir / "output" / "scene_with_materials.usd",
            "predictions": session_dir / "cache" / "predictions" / "predictions.jsonl",
            "dataset": session_dir / "cache" / "dataset" / "dataset.jsonl",
        }

        path = artifact_map.get(artifact_type)
        if path and path.exists():
            return path

        return None

    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None:
        """Generate a presigned/public URL for an artifact if store supports it.

        Args:
            session_id: Session identifier
            key: Artifact key (e.g., "input/input_render.png")
            expires_seconds: URL expiration time

        Returns:
            Presigned URL string or None if not supported
        """
        session_id = _validate_session_id(session_id)
        return await self._store.make_public_url(session_id, key, expires_seconds)

    async def put_file_to_store(
        self,
        session_id: str,
        key: str,
        file_path: str,
        content_type: str | None = None,
    ) -> None:
        """Copy a file to the store.

        Args:
            session_id: Session identifier
            key: Artifact key (e.g., "input/scene.usd")
            file_path: Local file path to copy
            content_type: Optional MIME type
        """
        session_id = _validate_session_id(session_id)
        await self._store.put_file(session_id, key, file_path, content_type)

    async def put_bytes_to_store(
        self,
        session_id: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        """Write bytes to the store.

        Args:
            session_id: Session identifier
            key: Artifact key
            data: Bytes to write
            content_type: Optional MIME type
        """
        session_id = _validate_session_id(session_id)
        await self._store.put_bytes(session_id, key, data, content_type)

    async def exists_in_store(self, session_id: str, key: str) -> bool:
        """Check if a file exists in the store.

        Args:
            session_id: Session identifier
            key: Artifact key (e.g., "input/input_render.png")

        Returns:
            True if file exists in store
        """
        session_id = _validate_session_id(session_id)
        return await self._store.exists(session_id, key)

    async def read_from_store(self, session_id: str, key: str) -> bytes | None:
        """Read file content from the store.

        Args:
            session_id: Session identifier
            key: Artifact key (e.g., "input/input_render.png")

        Returns:
            File content as bytes, or None if not found
        """
        session_id = _validate_session_id(session_id)
        try:
            if not await self._store.exists(session_id, key):
                return None
            stream = await self._store.open_read(session_id, key)
            return stream.read()
        except Exception as e:
            logger.warning(f"Failed to read {key} from store: {e}")
            return None

    # ---------- Sync ----------

    async def sync_session_to_store(self, session_id: str, prefix: str = "") -> int:
        """Sync all local session files to the remote store.

        For local storage, this is a no-op.
        For S3 storage, uploads all local files to S3.

        Args:
            session_id: Session identifier
            prefix: Optional prefix to filter files (e.g., "output/")

        Returns:
            Number of files synced
        """
        session_id = _validate_session_id(session_id)
        local_dir = str(self.get_session_dir(session_id))
        count = await self._store.sync_from_local(session_id, local_dir, prefix)
        if count > 0:
            logger.info(f"Synced {count} files to store for session {session_id[:8]}")
        return count

    async def sync_from_store(self, session_id: str, prefix: str = "") -> int:
        """Pull files from the store to local session directory.

        For local storage, this is a no-op.
        For S3 storage, downloads files from S3 to local disk.

        Args:
            session_id: Session identifier
            prefix: Optional prefix to filter files (e.g., "input/")

        Returns:
            Number of files downloaded
        """
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        count = await self._store.sync_to_local(
            session_id, str(session_dir), prefix=prefix
        )
        if count > 0:
            logger.info(f"Pulled {count} files from store for session {session_id[:8]}")
        return count

    # ---------- Session Listing & Cleanup ----------

    async def list_sessions(self) -> list[str]:
        """List all session IDs.

        Delegates to the configured storage backend to list sessions.
        For S3 storage, sessions are listed from the remote bucket.
        For local storage, sessions are listed from the local directory.

        Returns:
            List of session IDs
        """
        return await self._store.list_sessions()

    async def cleanup_expired_sessions(self) -> int:
        """Remove sessions past their TTL.

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0
        now = datetime.now(UTC)

        for session_id in await self.list_sessions():
            metadata = await self.get_session_metadata(session_id)

            if not metadata:
                continue

            if self.store.kind == "local" and self.ttl_hours > 0:
                expires_at_str = metadata.get("ttl_expires_at")
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str.replace("Z", ""))
                    if now > expires_at:
                        logger.info(f"Cleaning up expired session: {session_id}")
                        if await self.delete_session(session_id):
                            cleaned += 1
            else:
                logger.info(
                    "Skipping cleanup of session: %s (not local or TTL not enabled)",
                    session_id,
                )

        return cleaned

    async def cleanup_stale_local_cache(self, max_age_hours: float = 24.0) -> int:
        """Clean up stale local session cache.

        For S3 storage, syncs old sessions to remote and removes local files
        to free up disk space. Sessions not updated for longer than max_age_hours
        are considered stale.

        For local storage, this is a no-op since files are already in their
        final location.

        Args:
            max_age_hours: Maximum age in hours before cleanup (default: 24)

        Returns:
            Number of sessions cleaned up
        """
        return await self._store.cleanup_stale_local_sessions(
            str(self.storage_path), max_age_hours
        )
