# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune execution wrapper — drives :func:`physics_agent.tuning.arun_tune`.

Mirrors :mod:`workers.executor` (pipeline) so the JobRegistry / EventBus /
SessionManager wiring is identical. The big difference: we install a
threading.Event that is polled on every trial — when the user POSTs
``/tune/{id}/cancel`` it sets the event and the optimizer exits cleanly
between trials.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from physics_agent.tuning import (
    BoTorchUnavailableError,
    OvPhysXUnavailableError,
    TuneInput,
    TuningCancelledError,
    arun_tune,
)
from world_understanding.agentic.events import EventListener

from ..runtime import get_event_bus
from ..runtime.events import ProgressEvent, StepState

logger = logging.getLogger(__name__)


def _finite_best_score(value: object) -> float | None:
    """Round 15 (doyubkim blocker #2): coerce ``best_score`` to a JSON-safe
    finite float or ``None`` before persisting it to session metadata.

    The runner emits ``float("inf")`` for the cancelled-before-first-trial
    path (see ``physics_agent.tuning.runner._handle_zero_trial_cancel``);
    a backend overflow during a normal trial can also stamp ``inf``. Both
    routes write through ``update_session({...,"results":{"best_score":...}})``,
    which is later returned by ``GET /tune/{id}/status``. Starlette's JSON
    encoder rejects ``inf`` / ``-inf`` / ``nan`` outright and raises
    ``ValueError: Out of range float values are not JSON compliant``,
    turning a clean cancel into a 500 at every status poll. Sanitising at
    the WRITE site means every later read — ``/status``, ``/results``,
    artifact sync, refine-loop re-entry — sees a finite-only number or
    ``None`` instead.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _tune_results_metadata(result: Any) -> dict[str, Any]:
    """Return the session ``results`` payload for completed or partial tunes."""
    return {
        "best_params": dict(getattr(result, "best_params", {}) or {}),
        "best_score": _finite_best_score(getattr(result, "best_score", None)),
        "n_trials": int(getattr(result, "n_trials", 0) or 0),
        "optimizer_used": str(getattr(result, "optimizer_used", "") or ""),
        "engine_used": str(getattr(result, "engine_used", "") or ""),
    }


def _has_partial_tune_results(result: Any) -> bool:
    """Return True when a failed tune still has useful result artifacts."""
    if int(getattr(result, "n_trials", 0) or 0) > 0:
        return True
    if getattr(result, "best_params", None):
        return True
    return bool(getattr(result, "artifacts", None))


class _TuneEventListener(EventListener):
    """Adapter from the tuning runner's events → FastAPI ProgressEvent bus.

    Intentionally minimal — tuning has a much smaller event vocabulary than
    the full physics pipeline (started / trial.completed / completed /
    failed / warning) so we don't need the multi-step bookkeeping of the
    pipeline listener.
    """

    def __init__(self, session_id: str, max_trials: int):
        self.session_id = session_id
        self.max_trials = max(max_trials, 1)
        self.bus = get_event_bus()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None
        self.best_score: float | None = None
        self.best_params: dict[str, float] | None = None
        self.n_trials = 0

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.info(f"[tune {self.session_id[:8]}] {message}", *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.debug(f"[tune {self.session_id[:8]}] {message}", *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.warning(f"[tune {self.session_id[:8]}] {message}", *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.error(f"[tune {self.session_id[:8]}] {message}", *args, **kwargs)

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        if event_type == "tune.started":
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.RUNNING,
                percent=0,
                message="Tuning started",
                extra=data,
            )
        elif event_type == "tune.trial.completed":
            self.n_trials += 1
            score = float(data.get("score", float("inf")))
            failed = bool(data.get("failed", False))
            if not failed and (self.best_score is None or score < self.best_score):
                self.best_score = score
                self.best_params = dict(data.get("params") or {})
            percent = int(min(100, 100 * self.n_trials / self.max_trials))
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.RUNNING,
                current=self.n_trials,
                total=self.max_trials,
                percent=percent,
                message=(
                    f"Trial {self.n_trials}/{self.max_trials}: "
                    f"score={score:.4g}{' (failed)' if failed else ''}"
                ),
                extra={
                    "trial_index": data.get("trial_index"),
                    "score": score,
                    "params": data.get("params"),
                    "failed": failed,
                    "best_score": self.best_score,
                    "best_params": self.best_params,
                },
            )
        elif event_type == "tune.completed":
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.COMPLETED,
                percent=100,
                message="Tuning completed",
                extra=dict(data),
            )
        elif event_type == "tune.cancelled":
            # Round 11 thread #10: the runner now emits ``tune.cancelled``
            # for cancelled runs; map directly to the CANCELLED step state
            # so SSE clients see the right terminal frame on the wire.
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.CANCELLED,
                message="Tuning cancelled",
                extra=dict(data),
            )
        elif event_type == "tune.failed":
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.FAILED,
                message=data.get("error", "Tuning failed"),
                extra=dict(data),
            )
        else:
            return

        self._emit_threadsafe(ev)

    def _emit_threadsafe(self, ev: ProgressEvent) -> None:
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.bus.emit(ev)))


async def _emit_terminal_bus_event(
    session_id: str,
    state: StepState,
    message: str,
    *,
    error: str | None = None,
) -> None:
    """Emit a terminal CANCELLED/FAILED bus frame for early-exit branches.

    Round 11 thread #3: the BoTorch/OvPhysX-unavailable, asyncio.Cancelled,
    TuningCancelledError, and generic-Exception handlers in
    ``execute_tune_async`` exit before the post-finally code that emits the
    terminal event. Same-instance ``stream_tune_events`` clients only
    short-circuit on FAILED/CANCELLED/tune_ready, so without an explicit
    emit here SSE consumers hang until the 30s timeout fallback notices the
    durable metadata flip.
    """
    bus = get_event_bus()
    if bus.get_snapshot(session_id) is None:
        return
    extra: dict[str, Any] = {}
    if error is not None:
        extra["error"] = error
    try:
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step="tune",
                state=state,
                message=message,
                extra=extra,
            )
        )
    except Exception:
        logger.warning(
            f"Failed to emit terminal {state.name} event for {session_id[:8]}",
            exc_info=True,
        )


async def _watch_for_cancel(
    session_manager: Any,
    session_id: str,
    cancel_event: threading.Event,
    poll_interval: float = 0.25,
) -> None:
    """Watch the SessionManager for an out-of-process cancel signal.

    The /tune/{id}/cancel endpoint writes a ``.cancel`` marker via
    ``request_cancellation``; that signal is visible cross-instance, but the
    optimizer running inside ``to_thread`` only checks the ``cancel_event``
    we hand it. This task polls and bridges the two.
    """
    try:
        while not cancel_event.is_set():
            try:
                if await session_manager.is_cancelled(session_id):
                    cancel_event.set()
                    return
            except Exception:  # pragma: no cover
                logger.debug("cancel watcher poll failed", exc_info=True)
            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        return


async def execute_tune_async(
    session_id: str,
    session_manager: Any,
    scenario_path: Path | None,
    physics_usd: Path,
    *,
    user_prompt: str | None = None,
    engine: str,
    optimizer: str,
    max_trials: int,
    seed: int,
    enable_judge: bool = True,
    judge_max_iterations: int = 3,
    judge_max_tokens: int | None = None,
    judge_temperature: float | None = None,
    reference_images: list[Path] | None = None,
    reference_videos: list[Path] | None = None,
    reference_descriptions: list[str] | None = None,
    reference_video_descriptions: list[str] | None = None,
) -> None:
    """Run one tuning session end-to-end and persist results."""
    logger.info(f"Tune execution started for {session_id[:8]}...")

    session_dir = session_manager.get_session_dir(session_id)
    output_dir = session_dir / "tune"
    output_dir.mkdir(parents=True, exist_ok=True)

    listener = _TuneEventListener(session_id, max_trials=max_trials)
    cancel_event = threading.Event()

    cancel_watcher = asyncio.create_task(
        _watch_for_cancel(session_manager, session_id, cancel_event)
    )

    await session_manager.update_session(session_id, {"status": "running"})

    result = None
    try:
        try:
            result = await arun_tune(
                TuneInput(
                    scenario=scenario_path,
                    user_prompt=user_prompt,
                    physics_usd=physics_usd,
                    output_dir=output_dir,
                    reference_images=reference_images,
                    reference_videos=reference_videos,
                    reference_descriptions=reference_descriptions,
                    reference_video_descriptions=reference_video_descriptions,
                    engine=engine,
                    optimizer=optimizer,
                    max_trials=max_trials,
                    seed=seed,
                    enable_judge=enable_judge,
                    judge_max_iterations=judge_max_iterations,
                    judge_max_tokens=judge_max_tokens,
                    judge_temperature=judge_temperature,
                    cancel_event=cancel_event,
                    event_listener=listener,
                )
            )
        except BoTorchUnavailableError as e:
            await session_manager.update_session(
                session_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "failed_step": "tune",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id, StepState.FAILED, str(e), error=str(e)
            )
            raise
        except OvPhysXUnavailableError as e:
            await session_manager.update_session(
                session_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "failed_step": "tune",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id, StepState.FAILED, str(e), error=str(e)
            )
            raise
        except asyncio.CancelledError:
            # Outer task cancellation (session delete, server shutdown).
            # Set the cooperative cancel signal so the worker thread's
            # optimizer can exit between trials before the asyncio side
            # tears down — without this the thread would keep running and
            # could write into a freshly-deleted session directory.
            cancel_event.set()
            await session_manager.update_session(
                session_id,
                {
                    "status": "cancelled",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id, StepState.CANCELLED, "Tune cancelled"
            )
            raise
        except TuningCancelledError:
            # Cooperative cancellation surfaced by the runner's LLM-call
            # wrapper (interpreter or judge phase), or by the optimizer
            # detecting the cancel marker. Persist as 'cancelled' rather
            # than the generic 'failed' branch below — codex round 5.
            await session_manager.update_session(
                session_id,
                {
                    "status": "cancelled",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id, StepState.CANCELLED, "Tune cancelled"
            )
            return
        except Exception as e:
            # Any unexpected failure inside the runner (USD parse, optimizer
            # bug, backend RuntimeError, …). Without this catch the exception
            # propagates into JobRegistry, whose cleanup does NOT update
            # session metadata — leaving the session stuck in 'running'.
            logger.error(
                "Tune execution failed for %s: %s", session_id[:8], e, exc_info=True
            )
            await session_manager.update_session(
                session_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "failed_step": "tune",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id, StepState.FAILED, str(e), error=str(e)
            )
            raise
    finally:
        cancel_watcher.cancel()
        try:
            await cancel_watcher
        except asyncio.CancelledError:
            pass

    metadata = await session_manager.get_session_metadata(session_id) or {}
    duration = 0
    created_at_str = metadata.get("created_at")
    if created_at_str:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        duration = int((datetime.now(UTC) - created_at).total_seconds())

    # Re-check the cancel marker before declaring completion. The /cancel
    # endpoint accepts pending/running, so a cancel can land in the brief
    # window between arun_tune returning and us writing 'completed'. Without
    # this re-check that cancel would be silently overwritten.
    late_cancel = await session_manager.is_cancelled(session_id) or result.cancelled

    if late_cancel:
        await session_manager.update_session(
            session_id,
            {
                "status": "cancelled",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": duration,
                "results": {
                    "best_params": result.best_params,
                    # Round 15 (doyubkim blocker #2): coerce to finite/None
                    # at the write site so persisted metadata is always
                    # JSON-serialisable. A pre-first-trial cancel passes
                    # ``float("inf")`` straight from the runner.
                    "best_score": _finite_best_score(result.best_score),
                    "n_trials": result.n_trials,
                    "optimizer_used": result.optimizer_used,
                    "engine_used": result.engine_used,
                },
            },
        )
        # Emit a terminal CANCELLED bus event so same-instance SSE
        # clients in ``stream_tune_events()`` close immediately. Without
        # this they would hang until the 30s timeout fallback noticed
        # the metadata change. ``stream_tune_events`` only short-circuits
        # on FAILED / CANCELLED / tune_ready terminal events; without one
        # of those a fresh-cancel returns no terminal frame on the wire.
        bus = get_event_bus()
        if bus.get_snapshot(session_id) is not None:
            try:
                # n_trials may be 0 if cancellation landed before the
                # first trial completed; clamp the percent into [0, 100].
                pct = (
                    min(100, int(100 * result.n_trials / max(max_trials, 1)))
                    if max_trials
                    else 0
                )
                await bus.emit(
                    ProgressEvent(
                        session_id=session_id,
                        step="tune",
                        state=StepState.CANCELLED,
                        percent=pct,
                        message="Tune cancelled",
                        extra={
                            # Coerce here too — the runner stamps
                            # ``float("inf")`` on the cancelled-before-first-trial
                            # path, and Pydantic's JSON serializer (which the
                            # SSE bus uses) rejects non-finite floats in
                            # strict mode just like Starlette's encoder.
                            # Round 15 follow-up.
                            "best_score": _finite_best_score(result.best_score),
                            "best_params": result.best_params,
                            "n_trials": result.n_trials,
                        },
                    )
                )
            except Exception:
                logger.warning(
                    f"Failed to emit CANCELLED event for {session_id[:8]}",
                    exc_info=True,
                )
        # Round 12 (CX P2#3): the cancelled-after-N-trials branch can
        # produce ``best_params.json`` / ``history.jsonl`` / ``report.md``
        # the user may want to download. The successful-run path syncs
        # ``tune/`` to the multi-instance store; this branch returned
        # before that sync, so the artifact GETs would 404 on a different
        # instance. Sync here too — wrap in try/except to mirror the
        # success-path behavior (best-effort, never fails the worker).
        try:
            await session_manager.sync_to_store(session_id, prefix="tune/")
        except Exception:
            logger.warning(
                f"Failed to sync cancelled tune artifacts for {session_id[:8]}",
                exc_info=True,
            )
        return

    if not result.success:
        partial_results = (
            _tune_results_metadata(result)
            if _has_partial_tune_results(result)
            else None
        )
        updates: dict[str, Any] = {
            "status": "failed",
            "error": result.error or "Tuning failed",
            "failed_step": "tune",
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": duration,
            "can_cancel": False,
        }
        if partial_results is not None:
            # A media-backed judge/evidence failure can happen after the
            # optimizer wrote useful tune artifacts. Keep the terminal status
            # failed, but persist the same discoverable metadata shape as
            # completed/cancelled runs so REST clients can fetch artifacts.
            updates["results"] = partial_results
            updates["partial_results"] = partial_results
        await session_manager.update_session(
            session_id,
            updates,
        )
        # Symmetric with the CANCELLED branch above: emit a terminal
        # FAILED bus event so same-instance SSE clients in
        # ``stream_tune_events`` close immediately. ``stream_tune_events``
        # short-circuits on FAILED / CANCELLED / ``tune_ready``; without
        # one, the wire would hang until the 30s timeout fallback
        # noticed the durable metadata flip.
        bus = get_event_bus()
        if bus.get_snapshot(session_id) is not None:
            try:
                await bus.emit(
                    ProgressEvent(
                        session_id=session_id,
                        step="tune",
                        state=StepState.FAILED,
                        message=str(result.error or "Tuning failed"),
                        extra={"error": str(result.error or "Tuning failed")},
                    )
                )
            except Exception:
                logger.warning(
                    f"Failed to emit FAILED event for {session_id[:8]}",
                    exc_info=True,
                )
        # Round 12 (CX P2#3): same as the cancelled branch above —
        # ``not result.success`` runs may still write tune artifacts that
        # downstream clients can pull, so sync to the multi-instance store
        # before returning.
        try:
            await session_manager.sync_to_store(session_id, prefix="tune/")
        except Exception:
            logger.warning(
                f"Failed to sync failed-tune artifacts for {session_id[:8]}",
                exc_info=True,
            )
        return

    await session_manager.update_session(
        session_id,
        {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": duration,
            "can_cancel": False,
            # Round 15 (doyubkim blocker #2): same finite/None coercion as
            # the cancelled branch — a backend overflow on the final trial can
            # stamp ``inf`` even on a "successful" run.
            "results": _tune_results_metadata(result),
        },
    )

    # Sync the tune/ artifact dir to the store (multi-instance / S3 path).
    try:
        await session_manager.sync_to_store(session_id, prefix="tune/")
    except Exception:
        logger.warning(
            f"Failed to sync tune artifacts to store for {session_id[:8]}",
            exc_info=True,
        )

    bus = get_event_bus()
    if bus.get_snapshot(session_id) is not None:
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step="tune",
                state=StepState.COMPLETED,
                percent=100,
                message="Tune artifacts synced and ready",
                extra={"tune_ready": True},
            )
        )
    logger.info(f"Tune execution completed for {session_id[:8]}")
