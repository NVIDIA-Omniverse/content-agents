# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...service.routers import pipeline_router, sessions_router
from ...service.runtime import ProgressEvent, StepState
from ...service.runtime.bus import get_event_bus, init_event_bus
from ...service.session.manager import SessionManager


def _build_session_app(tmp_path: Path) -> tuple[TestClient, SessionManager]:
    manager = SessionManager(tmp_path)
    init_event_bus(manager)
    pipeline_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    app.include_router(sessions_router.router)
    return TestClient(app), manager


def test_delete_missing_session_returns_json_error(tmp_path: Path) -> None:
    client, _ = _build_session_app(tmp_path)

    response = client.delete("/sessions/missing-session")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == "Session not found"


def test_invalid_session_id_routes_return_not_found(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)

    requests = [
        ("GET", "/sessions/%2E%2E"),
        ("DELETE", "/sessions/%2E%2E"),
        ("GET", "/pipeline/%2E%2E/status"),
        ("GET", "/pipeline/%2E%2E/results"),
        ("POST", "/pipeline/%2E%2E/cancel"),
        ("GET", "/pipeline/%2E%2E/event-log"),
    ]

    for method, path in requests:
        response = client.request(method, path)
        assert response.status_code == 404

    assert manager.storage_path.exists()


def test_delete_session_openapi_documents_json_errors(tmp_path: Path) -> None:
    client, _ = _build_session_app(tmp_path)
    responses = client.app.openapi()["paths"]["/sessions/{session_id}"]["delete"][
        "responses"
    ]

    for status_code in ("404", "409", "500"):
        content = responses[status_code]["content"]

        assert list(content) == ["application/json"]
        assert content["application/json"]["schema"] == {}


def test_delete_session_clears_runtime_status_snapshot(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "completed-session"
    manager.create_session(sid)
    asyncio.run(
        get_event_bus().emit(
            ProgressEvent(
                session_id=sid,
                step="render",
                state=StepState.RUNNING,
                percent=50,
            )
        )
    )

    assert client.get(f"/pipeline/{sid}/status").status_code == 200

    response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 204
    assert get_event_bus().get_snapshot(sid) is None
    assert client.get(f"/pipeline/{sid}/status").status_code == 404


def test_delete_running_session_returns_conflict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "running-session"
    manager.create_session(sid)

    class RunningJobRegistry:
        def is_running(self, session_id: str) -> bool:
            return session_id == sid

    monkeypatch.setattr(
        sessions_router,
        "get_job_registry",
        lambda: RunningJobRegistry(),
    )

    response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 409
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == (
        "Cannot delete an active session. Cancel it and wait for the worker "
        "to stop before deleting."
    )
    assert manager.session_exists(sid) is True


def test_delete_worker_locked_session_returns_conflict(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "worker-locked-session"
    manager.create_session(sid)

    with manager.worker_lock(sid):
        response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 409
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == (
        "Cannot delete an active session. A worker is still writing artifacts "
        "for this session."
    )
    assert manager.session_exists(sid) is True


def test_delete_corrupt_stalled_marker_returns_conflict(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "corrupt-stalled-session"
    manager.create_session(sid)
    marker_path = manager.get_session_dir(sid) / ".worker.stalled"
    marker_path.write_text("{", encoding="utf-8")

    response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 409
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == (
        "Cannot delete an active session. A worker is still writing artifacts "
        "for this session."
    )
    assert marker_path.exists() is True
    assert manager.session_exists(sid) is True


def test_delete_stale_cancelling_session_succeeds(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "cancelling-session"
    manager.create_session(sid)
    asyncio.run(
        get_event_bus().emit(
            ProgressEvent(
                session_id=sid,
                step="render",
                state=StepState.RUNNING,
                percent=50,
            )
        )
    )
    manager.update_session(sid, {"status": "cancelling"})

    response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 204
    assert manager.session_exists(sid) is False
    assert get_event_bus().get_snapshot(sid) is None
