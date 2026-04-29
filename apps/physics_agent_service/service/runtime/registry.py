# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job registry for managing pipeline task lifecycle."""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class JobRegistry:
    """Registry for managing asyncio.Task lifecycle.

    Maintains strong references to pipeline tasks to prevent GC and enables
    proper cancellation and monitoring.
    """

    def __init__(self, max_concurrent: int = 3):
        """Initialize job registry.

        Args:
            max_concurrent: Maximum concurrent pipeline jobs (semaphore limit)
        """
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def register(self, session_id: str, coro: Any) -> None:
        """Register and start a pipeline job.

        The job is created immediately (so the HTTP 202 returns right away)
        but waits on the semaphore internally before executing.  This means
        concurrent requests are queued rather than blocking the HTTP handler
        or being rejected.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
        """
        task = asyncio.create_task(self._run_with_cleanup(session_id, coro))
        self._tasks[session_id] = task

        logger.info(f"Pipeline queued for {session_id[:8]}...")

    async def _run_with_cleanup(self, session_id: str, coro: Any) -> None:
        """Wait for a semaphore slot, run the pipeline, then clean up.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
        """
        acquired = False
        coro_started = False
        try:
            await self._semaphore.acquire()
            acquired = True

            async with self._lock:
                self._active_count += 1

            logger.info(
                f"Starting pipeline for {session_id[:8]}... "
                f"(active: {self._active_count}/{self._semaphore._value + self._active_count})"
            )

            coro_started = True
            await coro
        finally:
            if not coro_started and hasattr(coro, "close"):
                coro.close()

            if acquired:
                self._semaphore.release()

                async with self._lock:
                    self._active_count -= 1

            if session_id in self._tasks:
                del self._tasks[session_id]

            logger.info(
                f"Pipeline completed/cancelled for {session_id[:8]}... "
                f"(active: {self._active_count})"
            )

    async def cancel(self, session_id: str) -> bool:
        """Cancel a running pipeline job.

        Args:
            session_id: Session identifier

        Returns:
            True if job was cancelled, False if not found or already done
        """
        task = self._tasks.get(session_id)
        if task is None:
            logger.warning(f"Cannot cancel - session not found: {session_id[:8]}...")
            return False

        if task.done():
            logger.info(f"Session already completed: {session_id[:8]}...")
            return False

        task.cancel()
        logger.info(f"Cancellation requested for {session_id[:8]}...")

        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            pass

        return True

    def get_task(self, session_id: str) -> asyncio.Task | None:
        """Get task for a session."""
        return self._tasks.get(session_id)

    def is_running(self, session_id: str) -> bool:
        """Check if a session is currently running."""
        task = self._tasks.get(session_id)
        return task is not None and not task.done()

    @property
    def active_count(self) -> int:
        """Get count of currently active jobs (past the semaphore)."""
        return self._active_count

    @property
    def registered_count(self) -> int:
        """Get count of all registered jobs (active + queued)."""
        return len(self._tasks)


# Global singleton job registry
_job_registry: JobRegistry | None = None


def get_job_registry() -> JobRegistry:
    """Get the global job registry instance."""
    global _job_registry
    if _job_registry is None:
        import os

        max_concurrent = int(os.getenv("PA_MAX_ACTIVE_SESSIONS", "1"))
        _job_registry = JobRegistry(max_concurrent=max_concurrent)
    return _job_registry
