# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry Event Listener - Wraps FastAPIEventListener to capture per-step timings.

This listener delegates all event handling to the inner FastAPIEventListener
while recording start/end timestamps for each pipeline step. The executor
reads these timings after pipeline completion to emit OTel spans and persist
them in session metadata.
"""

import logging
import time
from typing import Any

from world_understanding.agentic.events import EventListener

logger = logging.getLogger(__name__)


class TelemetryEventListener:
    """Wrapper around an EventListener that captures per-step timing data.

    Forwards all logging and event calls to the inner listener unchanged.
    Tracks step start/end timestamps for telemetry emission.
    """

    def __init__(self, inner: EventListener) -> None:
        self._inner = inner
        self._step_timings: dict[str, dict[str, Any]] = {}

    # =================================================================
    # Logging Methods (forwarded to inner listener)
    # =================================================================

    def info(self, message: str, **kwargs: Any) -> None:
        self._inner.info(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._inner.debug(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._inner.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._inner.error(message, **kwargs)

    # =================================================================
    # Event Handling
    # =================================================================

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        """Handle event: record timing data then forward to inner listener."""
        # Only track step-level events for telemetry (not task-level like VLMInference)
        step_name = data.get("step_name")

        if event_type == "step.started" and step_name:
            self._step_timings[step_name] = {
                "name": step_name,
                "started_at_ns": time.time_ns(),
                "completed_at_ns": None,
                "status": "running",
                "error": None,
            }

        elif event_type == "step.completed" and step_name:
            if step_name in self._step_timings:
                self._step_timings[step_name]["completed_at_ns"] = time.time_ns()
                self._step_timings[step_name]["status"] = "completed"

        elif event_type == "step.failed" and step_name:
            if step_name in self._step_timings:
                self._step_timings[step_name]["completed_at_ns"] = time.time_ns()
                self._step_timings[step_name]["status"] = "failed"
                self._step_timings[step_name]["error"] = data.get("error") or data.get(
                    "message", ""
                )

        # Always forward to inner listener
        self._inner.event(event_type, data, **kwargs)

    # =================================================================
    # Public API
    # =================================================================

    def get_step_timings(self) -> list[dict[str, Any]]:
        """Return collected step timings.

        Returns:
            List of dicts with keys: name, started_at_ns, completed_at_ns,
            status, error. Only includes steps that have both start and end
            timestamps.
        """
        return [
            timing
            for timing in self._step_timings.values()
            if timing["started_at_ns"] is not None
            and timing["completed_at_ns"] is not None
        ]
