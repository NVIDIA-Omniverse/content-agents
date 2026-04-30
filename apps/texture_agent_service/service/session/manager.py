# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for texture agent pipeline executions."""

import json
import logging
import os
import re
import shutil
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_session_id(session_id: str) -> str:
    """Validate a session id before it is used in filesystem paths."""
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid session_id")
    if session_id in {".", ".."}:
        raise ValueError("Invalid session_id")
    if "/" in session_id or "\\" in session_id:
        raise ValueError("Invalid session_id")
    return session_id


class SessionManager:
    """Manages pipeline sessions and their artifacts."""

    def __init__(self, storage_path: Path | str, ttl_hours: int = 24):
        """Initialize session manager.

        Args:
            storage_path: Base directory for session storage
            ttl_hours: Time-to-live for sessions in hours
        """
        self.storage_path = Path(storage_path)
        self.ttl_hours = ttl_hours
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        """Return a validated path guaranteed to remain under storage_path."""
        validate_session_id(session_id)
        storage_root = self.storage_path.resolve()
        session_dir = (storage_root / session_id).resolve()
        if not session_dir.is_relative_to(storage_root):
            raise ValueError("Invalid session_id")
        return session_dir

    def _require_session_dir(self, session_id: str) -> Path:
        """Return an existing session dir without creating lock parents."""
        session_dir = self._session_dir(session_id)
        metadata_path = session_dir / "session.json"
        if not session_dir.is_dir() or not metadata_path.is_file():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return session_dir

    @contextmanager
    def _session_lock(self, session_id: str):
        """Acquire an exclusive file lock for a session's metadata.

        Raises filelock.Timeout if the lock cannot be acquired within 10s.
        """
        lock_path = self._require_session_dir(session_id) / "session.json.lock"
        lock = FileLock(lock_path, timeout=10)
        try:
            with lock:
                yield
        except Timeout:
            logger.warning(f"Lock timeout for session {session_id}")
            raise

    @contextmanager
    def worker_lock(self, session_id: str, timeout: float = 10):
        """Acquire a cross-process lock while a worker may write artifacts."""
        lock = self.acquire_worker_lock(session_id, timeout=timeout)
        try:
            yield lock
        finally:
            self.release_worker_lock(lock, session_id)

    def acquire_worker_lock(self, session_id: str, timeout: float = 10) -> FileLock:
        """Acquire the cross-process worker lock and return its handle.

        Routes use this as an accepted-job reservation before returning 202;
        the registry releases it when the queued/running job exits. This keeps
        DELETE/TTL cleanup serialized with jobs even before the executor starts.
        """
        lock_path = self._require_session_dir(session_id) / ".worker.lock"
        lock = FileLock(lock_path, timeout=timeout, thread_local=False)
        try:
            lock.acquire()
            return lock
        except Timeout:
            logger.warning(f"Worker lock timeout for session {session_id}")
            raise

    def release_worker_lock(self, lock: FileLock, session_id: str) -> None:
        """Release a worker lock handle acquired by acquire_worker_lock."""
        try:
            lock.release()
        except Exception:
            logger.exception("Failed to release worker lock for %s", session_id)

    def _worker_stalled_path(self, session_id: str) -> Path:
        """Return marker path for a worker thread that outlived cancellation."""
        return self._session_dir(session_id) / ".worker.stalled"

    @staticmethod
    def _current_boot_id() -> str | None:
        try:
            return (
                Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="utf-8")
                .strip()
            )
        except OSError:
            return None

    @staticmethod
    def _process_start_ticks(pid: int) -> str | None:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        parts = stat.rsplit(") ", 1)
        if len(parts) != 2:
            return None
        fields = parts[1].split()
        if len(fields) < 20:
            return None
        return fields[19]

    @classmethod
    def _current_stalled_owner(cls) -> dict[str, Any]:
        pid = os.getpid()
        return {
            "pid": pid,
            "boot_id": cls._current_boot_id(),
            "process_start_ticks": cls._process_start_ticks(pid),
        }

    @classmethod
    def _stalled_owner_is_live(cls, marker: dict[str, Any]) -> bool:
        pid = marker.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False

        marker_boot_id = marker.get("boot_id")
        current_boot_id = cls._current_boot_id()
        if (
            isinstance(marker_boot_id, str)
            and current_boot_id is not None
            and marker_boot_id != current_boot_id
        ):
            return False

        marker_start = marker.get("process_start_ticks")
        current_start = cls._process_start_ticks(pid)
        if current_start is None:
            return False
        if isinstance(marker_start, str) and marker_start != current_start:
            return False

        return True

    def mark_worker_stalled(self, session_id: str, reason: str) -> None:
        """Mark that a background worker may still be writing artifacts."""
        marker_path = self._worker_stalled_path(session_id)
        tmp_path = marker_path.with_name(f"{marker_path.name}.{os.getpid()}.tmp")
        marker = {
            "reason": reason,
            "created_at": datetime.now(UTC).isoformat(),
            **self._current_stalled_owner(),
        }
        try:
            tmp_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
            os.replace(tmp_path, marker_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def clear_worker_stalled(self, session_id: str) -> None:
        """Clear the stalled-worker marker after the background thread exits."""
        try:
            self._worker_stalled_path(session_id).unlink(missing_ok=True)
        except ValueError:
            return

    def is_worker_stalled(self, session_id: str) -> bool:
        """Check whether cancellation left a background worker still draining."""
        try:
            marker_path = self._worker_stalled_path(session_id)
        except ValueError:
            return False
        if not marker_path.exists():
            return False

        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Treating unreadable stalled-worker marker as active for %s: %s",
                session_id,
                e,
            )
            return True

        if isinstance(marker, dict) and self._stalled_owner_is_live(marker):
            return True

        marker_path.unlink(missing_ok=True)
        logger.info("Cleared stale stalled-worker marker for %s", session_id)
        return False

    def is_worker_active(self, session_id: str) -> bool:
        """Check whether another worker currently holds the session write lock."""
        if not self.session_exists(session_id):
            return False

        if self.is_worker_stalled(session_id):
            return True

        lock_path = self._session_dir(session_id) / ".worker.lock"
        lock = FileLock(lock_path, timeout=0)
        try:
            with lock:
                return False
        except Timeout:
            return True

    def create_session(
        self, session_id: str, config: dict[str, Any] | None = None
    ) -> Path:
        """Create a new session directory structure.

        Args:
            session_id: Unique session identifier
            config: Optional configuration dict

        Returns:
            Path to session directory
        """
        session_dir = self._session_dir(session_id)

        # Create directory structure for texture pipeline
        (session_dir / "input").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "prepared").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "discovery").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "previews").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "generated").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "textures").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "output").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "renders").mkdir(parents=True, exist_ok=True)
        (session_dir / "preview").mkdir(parents=True, exist_ok=True)

        metadata = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": 8,
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

        self._save_metadata(session_id, metadata)
        logger.info(f"Created session: {session_id}")
        return session_dir

    def get_session_dir(self, session_id: str) -> Path:
        """Get path to session directory."""
        return self._session_dir(session_id)

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        try:
            session_dir = self._session_dir(session_id)
        except ValueError:
            return False
        return session_dir.is_dir() and (session_dir / "session.json").is_file()

    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata with retry logic."""
        if not self.session_exists(session_id):
            return None

        metadata_path = self._session_dir(session_id) / "session.json"
        if not metadata_path.exists():
            return None

        for attempt in range(3):
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                if attempt < 2:
                    time.sleep(0.05)
                    continue
                logger.warning(f"Failed to read metadata after 3 attempts: {e}")
                return None

    def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update session metadata."""
        try:
            with self._session_lock(session_id):
                metadata = self.get_session_metadata(session_id)
                if not metadata:
                    logger.warning(f"Cannot update non-existent session: {session_id}")
                    return

                metadata.update(updates)
                metadata["updated_at"] = datetime.now(UTC).isoformat()

                created_at = datetime.fromisoformat(metadata["created_at"])
                metadata["elapsed_seconds"] = int(
                    (datetime.now(UTC) - created_at).total_seconds()
                )

                self._save_metadata(session_id, metadata)
        except FileNotFoundError:
            logger.warning(f"Cannot update non-existent session: {session_id}")

    def update_step_progress(
        self,
        session_id: str,
        step_name: str,
        progress: dict[str, Any],
    ) -> None:
        """Update progress for current step."""
        with self._session_lock(session_id):
            self._update_step_progress_locked(session_id, step_name, progress)

    def _update_step_progress_locked(
        self,
        session_id: str,
        step_name: str,
        progress: dict[str, Any],
    ) -> None:
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            return

        step_info_map = {
            "prepare_uvs": {"display": "Preparing UV Coordinates", "step_num": 1},
            "discover_materials": {
                "display": "Discovering Materials",
                "step_num": 2,
            },
            "generate_prompts": {
                "display": "Generating Texture Prompts",
                "step_num": 3,
            },
            "render_previews": {
                "display": "Rendering Material Previews",
                "step_num": 4,
            },
            "generate_textures": {
                "display": "Generating PBR Textures",
                "step_num": 5,
            },
            "blend_textures": {"display": "Blending Textures", "step_num": 6},
            "apply_textures": {
                "display": "Applying Textures to USD",
                "step_num": 7,
            },
            "render": {"display": "Rendering Final Output", "step_num": 8},
        }

        step_info = step_info_map.get(step_name, {"display": step_name, "step_num": 0})

        current_step_info = metadata.get("current_step")
        if current_step_info and current_step_info.get("name") == step_name:
            started_at = datetime.fromisoformat(current_step_info["started_at"])
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
            metadata["overall_progress"]["current_step"] = step_num

        self._save_metadata(session_id, metadata)

    def mark_step_completed(
        self,
        session_id: str,
        step_name: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Mark a step as completed."""
        with self._session_lock(session_id):
            self._mark_step_completed_locked(session_id, step_name, stats)

    def _mark_step_completed_locked(
        self,
        session_id: str,
        step_name: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            return

        current_step_info = metadata.get("current_step")
        if current_step_info and current_step_info["name"] == step_name:
            started_at = datetime.fromisoformat(current_step_info["started_at"])
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

            completed_count = len(metadata["completed_steps"])
            metadata["overall_progress"]["current_step"] = completed_count

            # Cumulative progress per step completion
            cumulative_percents = {
                "prepare_uvs": 3,
                "discover_materials": 5,
                "generate_prompts": 10,
                "render_previews": 20,
                "generate_textures": 75,
                "blend_textures": 85,
                "apply_textures": 95,
                "render": 100,
            }
            metadata["overall_progress"]["percent"] = cumulative_percents.get(
                step_name, metadata["overall_progress"]["percent"]
            )

            self._save_metadata(session_id, metadata)

    def add_preview_image(self, session_id: str, image_name: str) -> None:
        """Add a preview image to the session."""
        with self._session_lock(session_id):
            metadata = self.get_session_metadata(session_id)
            if not metadata:
                return

            if "preview_images" not in metadata:
                metadata["preview_images"] = []

            if image_name not in metadata["preview_images"]:
                metadata["preview_images"].append(image_name)
                self._save_metadata(session_id, metadata)

    def update_preview_images(self, session_id: str, image_names: list[str]) -> None:
        """Update the list of preview images."""
        with self._session_lock(session_id):
            metadata = self.get_session_metadata(session_id)
            if not metadata:
                return

            metadata["preview_images"] = image_names
            self._save_metadata(session_id, metadata)

    def is_cancelled(self, session_id: str) -> bool:
        """Check if session has been cancelled."""
        cancel_file = self._session_dir(session_id) / ".cancel"
        return cancel_file.exists()

    def clear_cancellation(self, session_id: str) -> None:
        """Remove the durable `.cancel` marker for a session.

        Callers must hold the cross-process worker lock (see
        ``acquire_worker_lock``) before clearing the marker so a concurrent
        ``request_cancellation`` cannot drop a fresh marker between the
        clear and the new run starting.

        Idempotent: missing marker is a no-op. A missing session directory
        is also tolerated so callers do not have to special-case it after
        delete races; the durable cancellation state simply does not exist.
        """
        try:
            cancel_file = self._session_dir(session_id) / ".cancel"
        except ValueError:
            return
        try:
            cancel_file.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning(
                f"Failed to clear cancellation marker for {session_id}: {exc}"
            )

    def request_cancellation(self, session_id: str) -> None:
        """Request cancellation of a running pipeline.

        Idempotent against terminal states: if the session has already
        landed in completed/failed/cancelled (e.g. it finished naturally
        between the cancel route's is_running check and this call), the
        marker is still dropped for any worker still observing it but the
        terminal status is preserved.

        The read-check-write is done atomically under _session_lock so a
        concurrent worker-side update_session(... "completed") cannot
        interleave between the terminal check and the cancelling write.
        """
        if not self.session_exists(session_id):
            logger.warning(f"Cannot cancel non-existent session: {session_id}")
            return

        cancel_file = self._session_dir(session_id) / ".cancel"
        cancel_file.touch()

        with self._session_lock(session_id):
            metadata = self.get_session_metadata(session_id)
            if not metadata:
                logger.warning(f"Cannot cancel non-existent session: {session_id}")
                return

            current_status = metadata.get("status")
            if current_status in ("completed", "failed", "cancelled"):
                logger.info(
                    f"Cancellation requested but session {session_id} already in "
                    f"terminal state: {current_status}"
                )
                return

            metadata["status"] = "cancelling"
            metadata["updated_at"] = datetime.now(UTC).isoformat()
            created_at = datetime.fromisoformat(metadata["created_at"])
            metadata["elapsed_seconds"] = int(
                (datetime.now(UTC) - created_at).total_seconds()
            )
            self._save_metadata(session_id, metadata)

        logger.info(f"Cancellation requested for session: {session_id}")

    def get_artifact_path(self, session_id: str, artifact_type: str) -> Path | None:
        """Get path to a session artifact."""
        session_dir = self.get_session_dir(session_id)

        artifact_map = {
            "materials": session_dir / "cache" / "discovery" / "materials.json",
            "output_usd": session_dir / "cache" / "output" / "textured_output.usd",
            "output_usdz": session_dir / "cache" / "output" / "textured_output.usdz",
        }

        path = artifact_map.get(artifact_type)
        if path and path.exists():
            return path

        return None

    def get_artifact_dir(self, session_id: str, artifact_type: str) -> Path | None:
        """Get path to a session artifact directory."""
        session_dir = self.get_session_dir(session_id)

        dir_map = {
            "textures": session_dir / "cache" / "textures",
            "renders": session_dir / "cache" / "renders",
            "generated": session_dir / "cache" / "generated",
            "previews": session_dir / "cache" / "previews",
        }

        path = dir_map.get(artifact_type)
        if path and path.exists():
            return path

        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its artifacts."""
        try:
            session_dir = self.get_session_dir(session_id)
        except ValueError:
            logger.warning(f"Invalid session id for delete: {session_id}")
            return False

        if not session_dir.exists():
            logger.warning(f"Session not found: {session_id}")
            return False
        if not (session_dir / "session.json").is_file():
            logger.warning(f"Session metadata not found: {session_id}")
            return False

        try:
            with self.worker_lock(session_id, timeout=0):
                if self.is_worker_stalled(session_id):
                    logger.warning(
                        f"Cannot delete stalled worker session: {session_id}"
                    )
                    return False
                with self._session_lock(session_id):
                    shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {session_id}")
            return True
        except Timeout:
            logger.warning(f"Could not acquire lock to delete session {session_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def cleanup_expired_sessions(self) -> list[str]:
        """Remove sessions past their TTL.

        Returns the list of session IDs whose on-disk directories were
        removed. Caller is responsible for releasing any in-memory state
        (event-bus snapshot/queue) keyed by these IDs -- without that
        follow-up, periodic TTL cleanup would slowly leak stale
        per-session bus state in long-running deployments.
        """
        cleaned: list[str] = []
        now = datetime.now(UTC)

        for session_dir in self.storage_path.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            if not (session_dir / "session.json").is_file():
                continue

            try:
                with self.worker_lock(session_id, timeout=0):
                    if self.is_worker_stalled(session_id):
                        logger.debug(f"Skipping session {session_id} (worker stalled)")
                        continue
                    with self._session_lock(session_id):
                        metadata = self.get_session_metadata(session_id)
                        if not metadata:
                            continue

                        if self.is_worker_stalled(session_id):
                            logger.debug(
                                f"Skipping session {session_id} (worker stalled)"
                            )
                            continue
                        expires_at_str = metadata.get("ttl_expires_at")
                        if expires_at_str:
                            expires_at = datetime.fromisoformat(expires_at_str)
                            if now > expires_at:
                                logger.info(
                                    f"Cleaning up expired session: {session_id}"
                                )
                                shutil.rmtree(session_dir)
                                cleaned.append(session_id)
            except Timeout:
                logger.debug(f"Skipping session {session_id} (lock busy)")
                continue
            except Exception as e:
                logger.warning(f"Error cleaning session {session_id}: {e}")
                continue

        if cleaned:
            logger.info(f"Cleaned up {len(cleaned)} expired sessions")

        return cleaned

    def _save_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Save session metadata to disk atomically."""
        metadata_path = self._session_dir(session_id) / "session.json"
        tmp_path = metadata_path.with_suffix(".json.tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        os.replace(tmp_path, metadata_path)
