"""Tests for async NVCF rendering response handling."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from world_understanding.functions.graphics import render_nvcf_async
from world_understanding.functions.graphics.render_nvcf import RenderingStatus

_ONE_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


@pytest.mark.asyncio
async def test_render_cameras_retries_response_exception(monkeypatch):
    calls: list[dict[str, Any]] = []

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {"status": "exception", "error": "renderer worker unavailable"}
        return {
            "status": "success",
            "images": {"0": {"/Camera": {"images": _ONE_PIXEL_PNG}}},
        }

    monkeypatch.setattr(
        render_nvcf_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    result = await render_nvcf_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
        max_retries=1,
        retry_delay=0.0,
    )

    assert len(calls) == 2
    assert result["successful_cameras"] == 1
    assert result["failed_cameras"] == 0
    assert result["results"][0]["status"] == RenderingStatus.success
    assert result["results"][0]["frame_count"] == 1


@pytest.mark.asyncio
async def test_render_cameras_does_not_retry_load_error(monkeypatch):
    calls: list[dict[str, Any]] = []

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "load_error", "error": "invalid USD"}

    monkeypatch.setattr(
        render_nvcf_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    result = await render_nvcf_async.render_cameras_from_url(
        usd_url="https://example.com/bad.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
        max_retries=3,
        retry_delay=0.0,
    )

    assert len(calls) == 1
    assert result["successful_cameras"] == 0
    assert result["failed_cameras"] == 1
    assert result["results"][0]["status"] == "load_error"
    assert "invalid USD" in result["results"][0]["error"]


@pytest.mark.asyncio
async def test_global_nvcf_render_limit_serializes_requests(monkeypatch):
    active_requests = 0
    max_active_requests = 0
    calls = 0

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        nonlocal active_requests, max_active_requests, calls
        calls += 1
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)
        await asyncio.sleep(0.01)
        active_requests -= 1
        return {
            "status": "success",
            "images": {"0": {"/Camera": {"images": _ONE_PIXEL_PNG}}},
        }

    monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "1")
    render_nvcf_async._reset_global_nvcf_render_semaphore_for_tests()
    monkeypatch.setattr(
        render_nvcf_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    try:
        results = await asyncio.gather(
            render_nvcf_async.render_cameras_from_url(
                usd_url="https://example.com/scene-a.usda",
                cameras=["/Camera"],
                api_key="test-api-key",
                base_url="https://example.com",
            ),
            render_nvcf_async.render_cameras_from_url(
                usd_url="https://example.com/scene-b.usda",
                cameras=["/Camera"],
                api_key="test-api-key",
                base_url="https://example.com",
            ),
        )
    finally:
        render_nvcf_async._reset_global_nvcf_render_semaphore_for_tests()

    assert calls == 2
    assert max_active_requests == 1
    assert [result["successful_cameras"] for result in results] == [1, 1]
