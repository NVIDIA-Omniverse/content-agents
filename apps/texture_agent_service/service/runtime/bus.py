# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event bus for pipeline progress events."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from .events import ProgressEvent, StepState

logger = logging.getLogger(__name__)


class EventBus:
    """Central event bus for pipeline progress.

    Manages:
    - Per-session event queues for SSE streaming
    - Canonical in-memory state snapshot for /status API
    - Event application logic to update state
    """

    def __init__(self, session_manager: Any = None):
        """Initialize event bus.

        Args:
            session_manager: Shared SessionManager instance for persistence.
                If None, persistence methods become no-ops.
        """
        self._queues: dict[str, asyncio.Queue[ProgressEvent]] = {}
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._session_manager = session_manager

    def get_queue(self, session_id: str) -> asyncio.Queue[ProgressEvent]:
        """Get or create event queue for a session."""
        return self._queues.setdefault(session_id, asyncio.Queue())

    def get_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Get current in-memory state snapshot for a session."""
        return self._state.get(session_id)

    async def emit(self, event: ProgressEvent) -> None:
        """Emit an event: update state and queue for subscribers."""
        pending_persists: list[tuple[str, str]] = []

        async with self._lock:
            self._apply_event_to_state(event, pending_persists)

            state = self._state.get(event.session_id)
            if state:
                event.overall_percent = state.get("overall_progress", {}).get(
                    "percent", 0
                )
                logger.info(
                    f"[EventBus] {event.session_id[:8]}... {event.step}: "
                    f"step={event.percent or 0}% → overall={event.overall_percent}% (state={event.state.value})"
                )
            else:
                event.overall_percent = 0

            queue = self.get_queue(event.session_id)
            await queue.put(event)

        # Persist status changes and event log outside the lock
        for session_id, status in pending_persists:
            await self._persist_status(session_id, status)
        await self._save_event_to_log(event)

    def _apply_event_to_state(
        self, event: ProgressEvent, pending_persists: list[tuple[str, str]]
    ) -> None:
        """Apply event to update canonical in-memory state."""
        session_id = event.session_id

        if session_id not in self._state:
            self._state[session_id] = {
                "session_id": session_id,
                "status": "pending",
                "created_at": event.timestamp,
                "updated_at": event.timestamp,
                "current_step": None,
                "completed_steps": [],
                "overall_progress": {
                    "current_step": 0,
                    "total_steps": 8,
                    "percent": 0,
                },
                "step_timings": {},
            }

        state = self._state[session_id]
        state["updated_at"] = event.timestamp

        if event.state == StepState.RUNNING:
            if (
                state.get("current_step") is None
                or state["current_step"].get("name") != event.step
            ):
                state["current_step"] = {
                    "name": event.step,
                    "display_name": self._get_display_name(event.step),
                    "started_at": event.timestamp,
                    "progress": {
                        "current": event.current or 0,
                        "total": event.total or 1,
                        "percent": event.percent or 0,
                        "message": event.message or "",
                    },
                    "elapsed_seconds": 0,
                }
                if state["status"] == "pending":
                    pending_persists.append((state["session_id"], "running"))
                state["status"] = "running"
            else:
                state["current_step"]["progress"] = {
                    "current": event.current or 0,
                    "total": event.total or 1,
                    "percent": event.percent or 0,
                    "message": event.message or "",
                }
                started_at = datetime.fromisoformat(state["current_step"]["started_at"])
                now = datetime.fromisoformat(event.timestamp)
                state["current_step"]["elapsed_seconds"] = int(
                    (now - started_at).total_seconds()
                )

            self._update_overall_progress(state, event.step, event.percent or 0)

        elif event.state == StepState.COMPLETED:
            if (
                state.get("current_step")
                and state["current_step"]["name"] == event.step
            ):
                started_at = datetime.fromisoformat(state["current_step"]["started_at"])
                now = datetime.fromisoformat(event.timestamp)
                duration = int((now - started_at).total_seconds())

                completed_step = {
                    "name": event.step,
                    "display_name": state["current_step"]["display_name"],
                    "started_at": state["current_step"]["started_at"],
                    "completed_at": event.timestamp,
                    "duration_seconds": duration,
                    "stats": event.extra or {},
                }
                state["completed_steps"].append(completed_step)
                state["step_timings"][event.step] = duration
                state["current_step"] = None

                self._update_overall_progress_on_completion(
                    state, event.step, pending_persists
                )

            elif event.extra and event.extra.get("pipeline_completed"):
                state["overall_progress"]["percent"] = 100
                state["status"] = "completed"
                state["completed_at"] = datetime.now(UTC).isoformat()
                state["current_step"] = None
                pending_persists.append((state["session_id"], "completed"))

        elif event.state == StepState.FAILED:
            state["status"] = "failed"
            state["error"] = event.message or "Unknown error"
            state["failed_step"] = event.step
            state["failed_at"] = event.timestamp
            pending_persists.append((state["session_id"], "failed"))

        elif event.state == StepState.CANCELLED:
            state["status"] = "cancelled"
            state["cancelled_at"] = event.timestamp
            pending_persists.append((state["session_id"], "cancelled"))

        elif event.state == StepState.CANCELLING:
            # Don't downgrade a terminal state. If the worker finished or
            # failed between the cancel route's is_running check and this
            # event reaching the bus, the terminal status wins.
            if state.get("status") in ("completed", "failed", "cancelled"):
                return
            state["status"] = "cancelling"
            state["cancelling_at"] = event.timestamp
            pending_persists.append((state["session_id"], "cancelling"))

    def _get_display_name(self, step: str) -> str:
        """Get human-readable display name for step."""
        display_map = {
            "prepare_uvs": "Preparing UV Coordinates",
            "discover_materials": "Discovering Materials",
            "generate_prompts": "Generating Texture Prompts",
            "render_previews": "Rendering Material Previews",
            "generate_textures": "Generating PBR Textures",
            "blend_textures": "Blending Textures",
            "apply_textures": "Applying Textures to USD",
            "render": "Rendering Final Output",
        }
        return display_map.get(step, step)

    def _update_overall_progress(
        self, state: dict, step: str, step_percent: int
    ) -> None:
        """Update overall progress based on current step progress.

        Uses weighted allocation across 8 texture pipeline steps:
        - prepare_uvs: 0-3%
        - discover_materials: 3-5%
        - generate_prompts: 5-10%
        - render_previews: 10-20%
        - generate_textures: 20-75% (dominant cost)
        - blend_textures: 75-85%
        - apply_textures: 85-95%
        - render: 95-100%
        """
        step_weights = {
            "prepare_uvs": (0, 3),
            "discover_materials": (3, 5),
            "generate_prompts": (5, 10),
            "render_previews": (10, 20),
            "generate_textures": (20, 75),
            "blend_textures": (75, 85),
            "apply_textures": (85, 95),
            "render": (95, 100),
        }

        if step in step_weights:
            start, end = step_weights[step]
            overall = start + int((end - start) * step_percent / 100)
            state["overall_progress"]["percent"] = min(100, overall)

    def _update_overall_progress_on_completion(
        self,
        state: dict,
        step: str,
        pending_persists: list[tuple[str, str]],
    ) -> None:
        """Update overall progress when a step completes."""
        completion_percent = {
            "prepare_uvs": 3,
            "discover_materials": 5,
            "generate_prompts": 10,
            "render_previews": 20,
            "generate_textures": 75,
            "blend_textures": 85,
            "apply_textures": 95,
            "render": 100,
        }

        if step in completion_percent:
            state["overall_progress"]["percent"] = completion_percent[step]

        completed_count = len(state["completed_steps"])
        state["overall_progress"]["current_step"] = completed_count

        if state["overall_progress"]["percent"] >= 100:
            state["status"] = "completed"
            state["completed_at"] = datetime.now(UTC).isoformat()
            pending_persists.append((state["session_id"], "completed"))

    async def _persist_status(self, session_id: str, status: str) -> None:
        """Persist session status to SessionManager on disk."""
        if self._session_manager is None:
            return
        try:
            manager = self._session_manager
            if manager.session_exists(session_id):
                await asyncio.to_thread(
                    manager.update_session, session_id, {"status": status}
                )
                logger.info(f"Persisted {status} status for session {session_id}")

        except Exception as e:
            logger.warning(f"Failed to persist {status} status: {e}")

    async def _save_event_to_log(self, event: ProgressEvent) -> None:
        """Save event to persistent log file for replay."""
        if self._session_manager is None:
            return
        try:
            manager = self._session_manager
            if manager.session_exists(event.session_id):
                session_dir = manager.get_session_dir(event.session_id)
                log_file = session_dir / "event_log.jsonl"

                event_dict = event.model_dump()

                def _write():
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event_dict) + "\n")

                await asyncio.to_thread(_write)

        except Exception as e:
            logger.debug(f"Failed to save event to log: {e}")

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session from event bus."""
        async with self._lock:
            self._queues.pop(session_id, None)
            self._state.pop(session_id, None)


# Global singleton event bus
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def init_event_bus(session_manager: Any) -> EventBus:
    """Initialize the global event bus with a shared session manager."""
    global _event_bus
    _event_bus = EventBus(session_manager=session_manager)
    return _event_bus
