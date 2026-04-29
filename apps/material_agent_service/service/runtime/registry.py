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

    This replaces FastAPI's BackgroundTasks which doesn't provide task handles
    or real cancellation support.
    """

    def __init__(self, max_concurrent: int = 3):
        """Initialize job registry.

        Args:
            max_concurrent: Maximum concurrent pipeline jobs (semaphore limit)
        """
        # Strong references to tasks (prevents GC mid-execution)
        self._tasks: dict[str, asyncio.Task] = {}

        # Configured concurrency limit
        self._max_concurrent = max_concurrent

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Track active count explicitly (don't use _semaphore._value)
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def register(self, session_id: str, coro: Any) -> None:
        """Register and start a pipeline job.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
        """
        # Acquire semaphore for concurrency limit
        await self._semaphore.acquire()

        async with self._lock:
            self._active_count += 1

        logger.info(
            f"Starting pipeline for {session_id[:8]}... "
            f"(active: {self._active_count}/{self._semaphore._value + self._active_count})"
        )

        # Create task with proper cleanup callback
        task = asyncio.create_task(self._run_with_cleanup(session_id, coro))
        self._tasks[session_id] = task

    async def _run_with_cleanup(self, session_id: str, coro: Any) -> None:
        """Run coroutine and clean up resources.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
        """
        try:
            await coro
        finally:
            # Always release semaphore and decrement count
            self._semaphore.release()

            async with self._lock:
                self._active_count -= 1

            # Remove from registry
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

        # Cancel the task
        task.cancel()
        logger.info(f"Cancellation requested for {session_id[:8]}...")

        try:
            # Wait for cancellation to complete
            await asyncio.wait_for(task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            # Expected - task was cancelled or didn't finish in time
            pass

        return True

    def get_task(self, session_id: str) -> asyncio.Task | None:
        """Get task for a session.

        Args:
            session_id: Session identifier

        Returns:
            Task or None if not found
        """
        return self._tasks.get(session_id)

    def is_running(self, session_id: str) -> bool:
        """Check if a session is currently running.

        Args:
            session_id: Session identifier

        Returns:
            True if session has an active task
        """
        task = self._tasks.get(session_id)
        return task is not None and not task.done()

    @property
    def active_count(self) -> int:
        """Get count of currently active jobs.

        Returns:
            Number of active jobs
        """
        return self._active_count

    @property
    def max_concurrent(self) -> int:
        """Get the configured maximum concurrent jobs.

        Returns:
            Maximum concurrent pipeline jobs
        """
        return self._max_concurrent


# Global singleton job registry
_job_registry: JobRegistry | None = None


def get_job_registry() -> JobRegistry:
    """Get the global job registry instance.

    Returns:
        Global JobRegistry instance
    """
    global _job_registry
    if _job_registry is None:
        # Read max concurrent from environment
        import os

        max_concurrent = int(os.getenv("MA_MAX_ACTIVE_SESSIONS", "3"))
        _job_registry = JobRegistry(max_concurrent=max_concurrent)
    return _job_registry
