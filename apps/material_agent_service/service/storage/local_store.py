# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from .base import SessionStore

logger = logging.getLogger(__name__)


class LocalSessionStore(SessionStore):
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)

    @property
    def kind(self) -> str:
        return "local"

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    async def init_session(self, session_id: str) -> None:
        self._session_dir(session_id).mkdir(parents=True, exist_ok=True)

    async def delete_session(self, session_id: str) -> None:
        base = self._session_dir(session_id)
        if not base.exists():
            return
        for attempt in range(3):
            try:
                shutil.rmtree(base)
                return
            except OSError as e:
                if attempt == 2:
                    raise
                logger.warning(f"Retry {attempt + 1}/3 deleting {session_id[:8]}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))

    async def list_sessions(self, use_cache: bool = True) -> list[str]:
        """List all session IDs in the local store.

        Args:
            use_cache: Ignored for local storage (filesystem is fast,
                       no caching needed)

        Returns:
            List of session IDs (directory names that contain session.json)
        """
        sessions: list[str] = []
        if not self.root.exists():
            return sessions

        for session_dir in self.root.iterdir():
            if session_dir.is_dir():
                # Check if session.json exists to confirm it's a valid session
                if (session_dir / "session.json").exists():
                    sessions.append(session_dir.name)

        return sessions

    def invalidate_sessions_cache(self) -> None:
        """No-op for local storage - no caching needed."""
        pass

    async def put_bytes(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        path = self._session_dir(session_id) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def put_file(
        self, session_id: str, key: str, file_path: str, content_type: str | None = None
    ) -> None:
        path = self._session_dir(session_id) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, path)

    async def open_read(self, session_id: str, key: str):
        return open(self._session_dir(session_id) / key, "rb")  # noqa: SIM115

    async def exists(self, session_id: str, key: str) -> bool:
        return (self._session_dir(session_id) / key).exists()

    async def list_keys(self, session_id: str, prefix: str = "") -> list[str]:
        base = self._session_dir(session_id)
        if not base.exists():
            return []
        keys: list[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(base))
                if rel.startswith(prefix):
                    keys.append(rel)
        return keys

    async def put_json(self, session_id: str, key: str, obj: dict) -> None:
        await self.put_bytes(
            session_id, key, json.dumps(obj).encode("utf-8"), "application/json"
        )

    async def get_json(self, session_id: str, key: str) -> dict | None:
        path = self._session_dir(session_id) / key
        if not path.exists():
            return None
        return json.loads(path.read_text())

    async def get_json_batch(
        self, session_ids: list[str], key: str
    ) -> list[dict | None]:
        return [await self.get_json(sid, key) for sid in session_ids]

    async def append_event(self, session_id: str, event: dict) -> None:
        path = self._session_dir(session_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    async def get_event_log(self, session_id: str) -> list[dict]:
        path = self._session_dir(session_id) / "events.jsonl"
        if not path.exists():
            return []
        text = path.read_text()
        return [json.loads(line) for line in text.splitlines()]

    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None:
        return None

    async def sync_from_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """No-op for local storage - files are already local.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter files

        Returns:
            0 (no files synced, already local)
        """
        return 0

    async def sync_to_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """Copy files from store to local dir (no-op if they are the same path).

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter files (e.g., "input/")

        Returns:
            Number of files copied
        """
        store_dir = self._session_dir(session_id)
        local_dir = Path(local_session_dir)
        if store_dir == local_dir:
            return 0
        count = 0
        for file_path in store_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(store_dir))
            if prefix and not rel_path.startswith(prefix):
                continue
            dest = local_dir / rel_path
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(file_path), str(dest))
                count += 1
        return count

    async def cleanup_stale_local_sessions(
        self, local_storage_path: str, max_age_hours: float = 24.0
    ) -> int:
        """No-op for local storage - no remote sync needed.

        Args:
            local_storage_path: Root path where local sessions are stored
            max_age_hours: Maximum age in hours (ignored)

        Returns:
            0 (no cleanup needed for local storage)
        """
        return 0
