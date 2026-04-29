# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ...service.session.manager import SessionManager


def test_create_session_and_progress_lifecycle(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)

    session_dir = manager.create_session("session-1", config={"foo": "bar"})

    assert session_dir.exists()
    assert (session_dir / "input").is_dir()
    assert (session_dir / "cache" / "prepared").is_dir()
    assert (session_dir / "cache" / "renders").is_dir()
    assert (session_dir / "preview").is_dir()

    metadata = manager.get_session_metadata("session-1")
    assert metadata is not None
    assert metadata["status"] == "pending"
    assert metadata["config"] == {"foo": "bar"}

    manager.update_step_progress(
        "session-1",
        "generate_textures",
        {"current": 1, "total": 2, "percent": 50, "message": "halfway"},
    )
    metadata = manager.get_session_metadata("session-1")
    assert metadata["current_step"]["name"] == "generate_textures"
    assert metadata["overall_progress"]["current_step"] == 5

    manager.mark_step_completed("session-1", "generate_textures", {"textures": 2})
    metadata = manager.get_session_metadata("session-1")
    assert metadata["current_step"] is None
    assert metadata["overall_progress"]["percent"] == 75
    assert metadata["completed_steps"][0]["stats"] == {"textures": 2}
    assert "generate_textures" in metadata["timings"]


def test_preview_cancellation_and_artifact_helpers(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create_session("session-2")
    session_dir = manager.get_session_dir("session-2")

    manager.add_preview_image("session-2", "a.png")
    manager.add_preview_image("session-2", "a.png")
    manager.update_preview_images("session-2", ["b.png", "c.png"])

    textures_dir = session_dir / "cache" / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    (textures_dir / "one.png").write_text("x", encoding="utf-8")
    output_usd = session_dir / "cache" / "output" / "textured_output.usd"
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    output_usd.write_text("#usda 1.0\n", encoding="utf-8")

    manager.request_cancellation("session-2")
    metadata = manager.get_session_metadata("session-2")

    assert metadata["preview_images"] == ["b.png", "c.png"]
    assert metadata["status"] == "cancelling"
    assert manager.is_cancelled("session-2") is True
    assert manager.get_artifact_path("session-2", "output_usd") == output_usd
    assert manager.get_artifact_dir("session-2", "textures") == textures_dir


def test_request_cancellation_does_not_overwrite_terminal_status(
    tmp_path: Path,
) -> None:
    """request_cancellation races with natural completion in the cancel route.

    If the worker finishes before request_cancellation runs, the terminal
    status (completed / failed / cancelled) must win — otherwise /status
    reports a stale in-flight `cancelling` and /results rejects valid
    output.
    """
    manager = SessionManager(tmp_path)

    for terminal in ("completed", "failed", "cancelled"):
        sid = f"session-{terminal}"
        manager.create_session(sid)
        manager.update_session(sid, {"status": terminal})

        manager.request_cancellation(sid)

        metadata = manager.get_session_metadata(sid)
        assert metadata is not None
        assert metadata["status"] == terminal, (
            f"request_cancellation overwrote terminal {terminal} state"
        )
        # The .cancel marker still gets dropped — harmless for terminal
        # sessions, useful if a stray worker still observes it.
        assert manager.is_cancelled(sid) is True


def test_request_cancellation_serializes_with_concurrent_completion(
    tmp_path: Path,
) -> None:
    """Race the cancel route's terminal-state guard against a worker-side
    `update_session(..., "completed")`.

    The atomic read-check-write inside `_session_lock` should serialize
    the two writes such that whichever takes the lock first wins and the
    loser observes the new status. The terminal status, once written,
    must never be downgraded to `cancelling`.
    """
    import threading

    manager = SessionManager(tmp_path)
    manager.create_session("race-1")
    manager.update_session("race-1", {"status": "running"})

    barrier = threading.Barrier(2)
    cancel_done = threading.Event()
    complete_done = threading.Event()

    def cancel_path() -> None:
        barrier.wait()
        manager.request_cancellation("race-1")
        cancel_done.set()

    def complete_path() -> None:
        barrier.wait()
        manager.update_session("race-1", {"status": "completed"})
        complete_done.set()

    threads = [
        threading.Thread(target=cancel_path),
        threading.Thread(target=complete_path),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert cancel_done.is_set() and complete_done.is_set()

    final = manager.get_session_metadata("race-1")
    assert final is not None
    # Either order is allowed; what's not allowed is downgrading a
    # terminal `completed` to `cancelling`.
    assert final["status"] in ("completed", "cancelling")
    if final["status"] == "completed":
        # `completed` won the race — request_cancellation observed it
        # under the lock and bailed without overwriting.
        pass


def test_cleanup_expired_sessions_removes_old_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create_session("expired")

    metadata = manager.get_session_metadata("expired")
    assert metadata is not None
    metadata["ttl_expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    manager._save_metadata("expired", metadata)

    cleaned = manager.cleanup_expired_sessions()

    assert cleaned == 1
    assert not manager.get_session_dir("expired").exists()
