# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from .base import EVENT_LOG_KEY, METADATA_KEY, SessionStore

logger = logging.getLogger(__name__)


class LocalSessionStore(SessionStore):
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)

    @property
    def kind(self) -> str:
        return "local"

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def _safe_key_path(self, session_id: str, key: str) -> Path:
        base = self._session_dir(session_id).resolve()
        path = (base / key).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"Invalid key outside session directory: {key}")
        return path

    def init_session(self, session_id: str) -> None:
        self._session_dir(session_id).mkdir(parents=True, exist_ok=True)

    def delete_session(self, session_id: str) -> None:
        base = self._session_dir(session_id)
        if base.exists():
            shutil.rmtree(base)

    def delete_key(self, session_id: str, key: str) -> None:
        self._safe_key_path(session_id, key).unlink(missing_ok=True)

    def list_sessions(self, use_cache: bool = True) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            session_dir.name
            for session_dir in self.root.iterdir()
            if session_dir.is_dir() and (session_dir / "session.json").is_file()
        )

    def list_session_metadata(self, use_cache: bool = True) -> list[dict]:
        sessions: list[dict] = []
        for session_id in self.list_sessions(use_cache=use_cache):
            metadata = self.get_json(session_id, METADATA_KEY)
            if metadata is not None:
                sessions.append(metadata)
        return sessions

    def update_session_index(self, session_id: str, metadata: dict) -> None:
        return None

    def invalidate_sessions_cache(self) -> None:
        return None

    def put_bytes(
        self,
        session_id: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        path = self._safe_key_path(session_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_bytes(data)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def put_file(
        self,
        session_id: str,
        key: str,
        file_path: str,
        content_type: str | None = None,
    ) -> None:
        path = self._safe_key_path(session_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, path)

    def open_read(self, session_id: str, key: str) -> BinaryIO:
        return open(self._safe_key_path(session_id, key), "rb")  # noqa: SIM115

    def exists(self, session_id: str, key: str) -> bool:
        return self._safe_key_path(session_id, key).exists()

    @staticmethod
    def _prefix_matches(rel_path: str, prefix: str) -> bool:
        clean_prefix = prefix.lstrip("/")
        if not clean_prefix:
            return True
        if clean_prefix.endswith("/"):
            return rel_path.startswith(clean_prefix)
        return rel_path == clean_prefix or rel_path.startswith(f"{clean_prefix}/")

    def list_keys(self, session_id: str, prefix: str = "") -> list[str]:
        base = self._session_dir(session_id)
        if not base.exists():
            return []

        keys: list[str] = []
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel_path = path.relative_to(base).as_posix()
            if prefix and not self._prefix_matches(rel_path, prefix):
                continue
            keys.append(rel_path)
        return sorted(keys)

    def put_json(self, session_id: str, key: str, obj: dict) -> None:
        self.put_bytes(
            session_id,
            key,
            json.dumps(obj, indent=2).encode("utf-8"),
            "application/json",
        )

    def put_json_if_absent(self, session_id: str, key: str, obj: dict) -> bool:
        path = self._safe_key_path(session_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "xb") as f:
                f.write(json.dumps(obj, indent=2).encode("utf-8"))
            return True
        except FileExistsError:
            return False

    def get_json(self, session_id: str, key: str) -> dict | None:
        path = self._safe_key_path(session_id, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete_json_if_match(
        self,
        session_id: str,
        key: str,
        predicate: Callable[[dict], bool],
    ) -> bool:
        current = self.get_json(session_id, key)
        if current is None or not predicate(dict(current)):
            return False
        self.delete_key(session_id, key)
        return True

    def update_json(
        self,
        session_id: str,
        key: str,
        updater: Callable[[dict], dict | None],
    ) -> dict | None:
        current = self.get_json(session_id, key)
        if current is None:
            return None
        updated = updater(dict(current))
        if updated is None:
            return current
        self.put_json(session_id, key, updated)
        return updated

    def append_event(self, session_id: str, event: dict) -> None:
        path = self._safe_key_path(session_id, EVENT_LOG_KEY)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def get_event_log(self, session_id: str) -> list[dict]:
        path = self._safe_key_path(session_id, EVENT_LOG_KEY)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def make_public_url(
        self,
        session_id: str,
        key: str,
        expires_seconds: int = 3600,
    ) -> str | None:
        return None

    def sync_to_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str = "",
    ) -> int:
        store_dir = self._session_dir(session_id)
        local_dir = Path(local_session_dir)
        if store_dir.resolve() == local_dir.resolve():
            return 0
        if not store_dir.exists():
            return 0

        count = 0
        for file_path in store_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(store_dir).as_posix()
            if prefix and not self._prefix_matches(rel_path, prefix):
                continue
            dest = local_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            count += 1
        return count

    def sync_from_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str = "",
    ) -> int:
        local_dir = Path(local_session_dir)
        store_dir = self._session_dir(session_id)
        if local_dir.resolve() == store_dir.resolve():
            return 0
        if not local_dir.exists():
            return 0

        count = 0
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(local_dir).as_posix()
            if prefix and not self._prefix_matches(rel_path, prefix):
                continue
            dest = store_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            count += 1
        return count
