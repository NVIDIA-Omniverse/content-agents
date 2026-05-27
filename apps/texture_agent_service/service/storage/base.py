# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from typing import BinaryIO, Protocol

METADATA_KEY = "session.json"
CANCEL_KEY = ".cancel"
EVENT_LOG_KEY = "event_log.jsonl"
WORKER_RESERVATION_KEY = ".worker.reservation"


class SessionStore(Protocol):
    @property
    def kind(self) -> str: ...

    def init_session(self, session_id: str) -> None: ...
    def delete_session(self, session_id: str) -> None: ...
    def delete_key(self, session_id: str, key: str) -> None: ...
    def list_sessions(self, use_cache: bool = True) -> list[str]: ...
    def list_session_metadata(self, use_cache: bool = True) -> list[dict]: ...
    def update_session_index(self, session_id: str, metadata: dict) -> None: ...
    def invalidate_sessions_cache(self) -> None: ...

    def put_bytes(
        self,
        session_id: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None: ...
    def put_file(
        self,
        session_id: str,
        key: str,
        file_path: str,
        content_type: str | None = None,
    ) -> None: ...
    def open_read(self, session_id: str, key: str) -> BinaryIO: ...
    def exists(self, session_id: str, key: str) -> bool: ...
    def list_keys(self, session_id: str, prefix: str = "") -> list[str]: ...

    def put_json(self, session_id: str, key: str, obj: dict) -> None: ...
    def put_json_if_absent(self, session_id: str, key: str, obj: dict) -> bool: ...
    def get_json(self, session_id: str, key: str) -> dict | None: ...
    def delete_json_if_match(
        self,
        session_id: str,
        key: str,
        predicate: Callable[[dict], bool],
    ) -> bool: ...
    def update_json(
        self,
        session_id: str,
        key: str,
        updater: Callable[[dict], dict | None],
    ) -> dict | None: ...

    def append_event(self, session_id: str, event: dict) -> None: ...
    def get_event_log(self, session_id: str) -> list[dict]: ...

    def make_public_url(
        self,
        session_id: str,
        key: str,
        expires_seconds: int = 3600,
    ) -> str | None: ...

    def sync_to_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str = "",
    ) -> int: ...
    def sync_from_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str = "",
    ) -> int: ...
