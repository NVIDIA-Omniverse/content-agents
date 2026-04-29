# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable

from ...service.runtime.registry import JobRegistry


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.01)


async def test_cancel_running_job_releases_slot_for_queued_job() -> None:
    registry = JobRegistry(max_concurrent=1)

    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    first_release = asyncio.Event()
    queued_started = asyncio.Event()

    async def first_job() -> None:
        first_started.set()
        try:
            await first_release.wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def queued_job() -> None:
        queued_started.set()

    await registry.register("first", first_job())
    await _wait_until(first_started.is_set)

    await registry.register("queued", queued_job())
    await _wait_until(lambda: registry.registered_count == 2)
    await asyncio.sleep(0.05)

    assert registry.active_count == 1
    assert registry.is_running("first")
    assert registry.is_running("queued")
    assert not queued_started.is_set()

    assert await registry.cancel("first") is True
    await _wait_until(first_cancelled.is_set)
    await _wait_until(queued_started.is_set)
    await _wait_until(lambda: registry.registered_count == 0)

    assert registry.active_count == 0
    assert not registry.is_running("first")
    assert not registry.is_running("queued")
