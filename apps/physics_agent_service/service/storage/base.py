# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import BinaryIO, Protocol

# Standard key for session metadata JSON
METADATA_KEY = "session.json"


class SessionStore(Protocol):
    @property
    def kind(self) -> str: ...

    # Lifecycle
    async def init_session(self, session_id: str) -> None: ...
    async def delete_session(self, session_id: str) -> None: ...
    async def list_sessions(self, use_cache: bool = True) -> list[str]:
        """List all session IDs in the store.

        Args:
            use_cache: If True, may return cached results for performance.
                       Set to False to force a refresh. (Only affects S3 store)

        Returns:
            List of session IDs
        """
        ...

    def invalidate_sessions_cache(self) -> None:
        """Invalidate the sessions cache.

        For S3 store, clears the cached session list so the next
        list_sessions() call fetches fresh data. For local store, this is a no-op.
        """
        ...

    # Artifacts (images, usd, report, predictions, etc.)
    async def put_bytes(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> None: ...
    async def put_file(
        self, session_id: str, key: str, file_path: str, content_type: str | None = None
    ) -> None: ...
    async def open_read(self, session_id: str, key: str) -> BinaryIO: ...
    async def exists(self, session_id: str, key: str) -> bool: ...
    async def list_keys(self, session_id: str, prefix: str = "") -> list[str]: ...

    # Metadata/Status/Events
    async def put_json(self, session_id: str, key: str, obj: dict) -> None: ...
    async def get_json(self, session_id: str, key: str) -> dict | None: ...
    async def append_event(self, session_id: str, event: dict) -> None: ...
    async def get_event_log(self, session_id: str) -> list[dict]: ...

    # Public access (for images/files) — may return presigned URL or None if proxy-only
    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None: ...

    # Sync between local and remote storage
    async def sync_to_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """Sync files from remote storage to local session directory.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter keys (e.g., "input/")

        Returns:
            Number of files downloaded
        """
        ...

    async def sync_from_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """Sync files from local session directory to remote storage.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter files (e.g., "output/")

        Returns:
            Number of files synced
        """
        ...

    # Local cache cleanup (for remote stores like S3)
    async def cleanup_stale_local_sessions(
        self, local_storage_path: str, max_age_hours: float = 24.0
    ) -> int:
        """Clean up stale local session directories.

        For remote stores (S3), syncs sessions to remote and removes local cache
        if the session hasn't been updated for longer than max_age_hours.

        For local stores, this is a no-op since files are already in their
        final location.

        Args:
            local_storage_path: Root path where local sessions are stored
            max_age_hours: Maximum age in hours before a session is considered stale

        Returns:
            Number of sessions cleaned up
        """
        ...
