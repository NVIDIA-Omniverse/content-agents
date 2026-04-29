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


async def test_register_waits_for_free_slot_before_tracking_next_job() -> None:
    registry = JobRegistry(max_concurrent=1)

    first_started = asyncio.Event()
    first_release = asyncio.Event()
    first_cancelled = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()

    async def first_job() -> None:
        first_started.set()
        try:
            await first_release.wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def second_job() -> None:
        second_started.set()
        await second_release.wait()

    await registry.register("first", first_job())
    await _wait_until(first_started.is_set)

    second_register = asyncio.create_task(registry.register("second", second_job()))
    await asyncio.sleep(0.05)

    assert not second_register.done()
    assert registry.active_count == 1
    assert registry.get_task("second") is None
    assert registry.is_running("first")

    assert await registry.cancel("first") is True
    await _wait_until(first_cancelled.is_set)

    await second_register
    await _wait_until(second_started.is_set)

    assert registry.active_count == 1
    assert registry.is_running("second")
    assert registry.get_task("first") is None
    assert registry.get_task("second") is not None

    second_release.set()
    await _wait_until(lambda: registry.get_task("second") is None)

    assert registry.active_count == 0
    assert not registry.is_running("second")
