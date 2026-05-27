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


async def test_cancelling_queued_job_does_not_release_slot_early() -> None:
    registry = JobRegistry(max_concurrent=1)

    first_started = asyncio.Event()
    first_release = asyncio.Event()
    third_started = asyncio.Event()
    third_release = asyncio.Event()

    async def first_job() -> None:
        first_started.set()
        await first_release.wait()

    async def second_job() -> None:
        raise AssertionError("queued second job should never start")

    async def third_job() -> None:
        third_started.set()
        await third_release.wait()

    await registry.register("first", first_job())
    await _wait_until(first_started.is_set)

    await registry.register("second", second_job())
    await asyncio.sleep(0.05)

    assert registry.active_count == 1
    assert registry.registered_count == 2

    assert await registry.cancel("second") is True
    await asyncio.sleep(0.05)

    assert registry.active_count == 1
    assert registry.get_task("second") is None
    assert registry.is_running("first") is True

    await registry.register("third", third_job())
    await asyncio.sleep(0.05)
    assert third_started.is_set() is False

    first_release.set()
    await _wait_until(third_started.is_set)

    assert registry.active_count == 1
    third_release.set()
    await _wait_until(lambda: registry.get_task("third") is None)
    assert registry.active_count == 0


async def test_queued_job_runs_heartbeat_callback_before_slot_opens() -> None:
    registry = JobRegistry(max_concurrent=1)

    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    heartbeats = 0

    async def first_job() -> None:
        first_started.set()
        await first_release.wait()

    async def second_job() -> None:
        second_started.set()

    def queued_heartbeat() -> None:
        nonlocal heartbeats
        heartbeats += 1

    await registry.register("first", first_job())
    await _wait_until(first_started.is_set)

    await registry.register(
        "second",
        second_job(),
        on_queued_heartbeat=queued_heartbeat,
        queued_heartbeat_interval_seconds=0.01,
    )

    await _wait_until(lambda: heartbeats >= 2)
    assert second_started.is_set() is False

    first_release.set()
    await _wait_until(second_started.is_set)
    await _wait_until(lambda: registry.get_task("second") is None)
