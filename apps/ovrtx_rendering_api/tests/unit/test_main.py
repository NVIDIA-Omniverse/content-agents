# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import logging
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

service_main = importlib.import_module("service.main")


class _DummyRootLogger:
    def __init__(self, handlers: list[object]) -> None:
        self.handlers = handlers
        self.level = None

    def setLevel(self, level: int) -> None:
        self.level = level


class _FakeRenderer:
    def __init__(
        self,
        *,
        initialized: bool,
        daemon_running: bool,
        recover_result: bool = True,
        recover_hook: Callable[[], None] | None = None,
    ) -> None:
        self.is_initialized = initialized
        self.daemon_running = daemon_running
        self.recover_result = recover_result
        self.recover_hook = recover_hook
        self._recover_lock = threading.RLock()
        self.recover_calls = 0
        self.render_calls = 0

    @property
    def is_ready(self) -> bool:
        return self.is_initialized and self.daemon_running

    def recover(self, *, force: bool = False) -> bool:
        with self._recover_lock:
            if not force and self.is_ready:
                return True
            self.recover_calls += 1
            if self.recover_hook is not None:
                self.recover_hook()
            self.is_initialized = self.recover_result
            self.daemon_running = self.recover_result
            return self.recover_result

    def render(self, **_kwargs):
        self.render_calls += 1
        return {"status": "success", "error": None, "images": {}}


class _FakeTask:
    def __init__(self, done_result: bool) -> None:
        self._done_result = done_result

    def done(self) -> bool:
        return self._done_result


class _FakeDispatcher:
    def __init__(self, response: dict | None = None) -> None:
        self.render_calls = 0
        self.response = response

    def health(self):
        return {
            "status": "healthy",
            "gpu_initialized": True,
            "ready_workers": 2,
            "total_workers": 2,
        }

    def render(self, payload):
        self.render_calls += 1
        if self.response is not None:
            return self.response
        return {
            "status": "success",
            "error": None,
            "images": {},
            "url": payload["url"],
        }


def _render_request():
    return service_main.RenderRequest(
        url="data:application/octet-stream;base64,AA==",
        render_settings={
            "camera_paths": ["/World/Camera"],
            "frame_range": {"start": 0, "end": 0},
            "camera_parameters": {"width": 64, "height": 64},
        },
    )


def test_dispatcher_gpu_ids_disabled_in_worker_mode(monkeypatch):
    monkeypatch.setenv("OVRTX_GPU_WORKERS", "2")
    monkeypatch.setenv("OVRTX_WORKER_MODE", "1")

    assert service_main._dispatcher_gpu_ids() == []


def test_configure_logging_uses_basic_config_without_existing_handlers(monkeypatch):
    root_logger = _DummyRootLogger([])
    captured: dict[str, object] = {}

    def fake_basic_config(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(service_main.logging, "basicConfig", fake_basic_config)

    service_main._configure_logging(root_logger=root_logger)

    assert captured["level"] == logging.INFO
    assert "handlers" in captured


def test_configure_logging_reuses_existing_root_handlers(monkeypatch):
    root_logger = _DummyRootLogger([object()])
    basic_config_calls = 0

    def fake_basic_config(**kwargs):
        nonlocal basic_config_calls
        basic_config_calls += 1

    monkeypatch.setattr(service_main.logging, "basicConfig", fake_basic_config)

    service_main._configure_logging(root_logger=root_logger)

    assert root_logger.level == logging.INFO
    assert basic_config_calls == 0


@pytest.mark.asyncio
async def test_health_reports_initializing_before_renderer_exists(monkeypatch):
    monkeypatch.setattr(service_main, "_renderer", None)
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=False))

    response = await service_main.health()

    assert response.status == "initializing"
    assert response.gpu_initialized is False
    assert response.renderer_initialized is False
    assert response.daemon_running is False


@pytest.mark.asyncio
async def test_health_uses_dispatcher_when_configured(monkeypatch):
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(service_main, "_dispatcher", dispatcher)

    response = await service_main.health()

    assert response.status == "healthy"
    assert response.ready_workers == 2


@pytest.mark.asyncio
async def test_health_reports_unhealthy_when_renderer_creation_failed(monkeypatch):
    monkeypatch.setattr(service_main, "_dispatcher", None)
    monkeypatch.setattr(service_main, "_renderer", None)
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=True))

    response = await service_main.health()

    assert response.status == "unhealthy"
    assert response.gpu_initialized is False
    assert response.renderer_initialized is False
    assert response.daemon_running is False


@pytest.mark.asyncio
async def test_health_reports_ready_when_initialized_and_daemon_running(monkeypatch):
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=True))
    monkeypatch.setattr(
        service_main,
        "_renderer",
        _FakeRenderer(initialized=True, daemon_running=True),
    )

    response = await service_main.health()

    assert response.status == "healthy"
    assert response.gpu_initialized is True
    assert response.renderer_initialized is True
    assert response.daemon_running is True


@pytest.mark.asyncio
async def test_health_reports_unhealthy_when_initialized_daemon_died(monkeypatch):
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=True))
    monkeypatch.setattr(
        service_main,
        "_renderer",
        _FakeRenderer(initialized=True, daemon_running=False),
    )

    response = await service_main.health()

    assert response.status == "unhealthy"
    assert response.gpu_initialized is False
    assert response.renderer_initialized is True
    assert response.daemon_running is False


@pytest.mark.asyncio
async def test_health_reports_unhealthy_when_warmup_failed(monkeypatch):
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=True))
    monkeypatch.setattr(
        service_main,
        "_renderer",
        _FakeRenderer(initialized=False, daemon_running=True),
    )

    response = await service_main.health()

    assert response.status == "unhealthy"
    assert response.gpu_initialized is False
    assert response.renderer_initialized is False
    assert response.daemon_running is True


@pytest.mark.asyncio
async def test_health_reports_daemon_state_while_warmup_is_running(monkeypatch):
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=False))
    monkeypatch.setattr(
        service_main,
        "_renderer",
        _FakeRenderer(initialized=False, daemon_running=True),
    )

    response = await service_main.health()

    assert response.status == "initializing"
    assert response.gpu_initialized is False
    assert response.renderer_initialized is False
    assert response.daemon_running is True


def test_render_does_not_recover_while_warmup_is_running(monkeypatch):
    monkeypatch.setattr(service_main, "_dispatcher", None)
    renderer = _FakeRenderer(
        initialized=False,
        daemon_running=True,
        recover_result=True,
    )
    monkeypatch.setattr(service_main, "_renderer", renderer)
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=False))

    response = service_main.render(_render_request())

    assert response == {
        "status": "exception",
        "error": "Renderer not initialized",
        "images": {},
    }
    assert renderer.recover_calls == 0
    assert renderer.render_calls == 0


def test_render_uses_dispatcher_when_configured(monkeypatch):
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(service_main, "_dispatcher", dispatcher)

    response = service_main.render(_render_request())

    assert response["status"] == "success"
    assert response["url"] == "data:application/octet-stream;base64,AA=="
    assert dispatcher.render_calls == 1


def test_render_preserves_dispatcher_blank_render_payload(monkeypatch):
    dispatcher = _FakeDispatcher(
        {
            "status": "blank_render",
            "error": "1/1 OVRTX render frames are blank or near-blank.",
            "images": {"0": {"Camera": {"images": "large-payload"}}},
            "warnings": ["blank frame"],
            "blank_render_frames": [{"frame": 0, "camera": "/World/Camera"}],
        }
    )
    monkeypatch.setattr(service_main, "_dispatcher", dispatcher)

    response = service_main.render(_render_request())

    assert response["status"] == "blank_render"
    assert response["images"] == {"0": {"Camera": {"images": "large-payload"}}}
    assert response["blank_render_frames"] == [{"frame": 0, "camera": "/World/Camera"}]


def test_render_attempts_recovery_when_warmup_failed(monkeypatch):
    monkeypatch.setattr(service_main, "_dispatcher", None)
    renderer = _FakeRenderer(
        initialized=False,
        daemon_running=False,
        recover_result=True,
    )
    monkeypatch.setattr(service_main, "_renderer", renderer)
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=True))

    response = service_main.render(_render_request())

    assert response["status"] == "success"
    assert renderer.recover_calls == 1
    assert renderer.render_calls == 1


def test_render_preserves_blank_render_payload(monkeypatch):
    monkeypatch.setattr(service_main, "_dispatcher", None)
    renderer = _FakeRenderer(initialized=True, daemon_running=True)

    def blank_render(**_kwargs):
        renderer.render_calls += 1
        return {
            "status": "blank_render",
            "error": "1/1 OVRTX render frames are blank or near-blank.",
            "images": {"0": {"Camera": {"images": "large-payload"}}},
            "warnings": ["blank frame"],
            "blank_render_frames": [{"frame": 0, "camera": "/World/Camera"}],
        }

    renderer.render = blank_render
    monkeypatch.setattr(service_main, "_renderer", renderer)
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=True))

    response = service_main.render(_render_request())

    assert response["status"] == "blank_render"
    assert response["images"] == {"0": {"Camera": {"images": "large-payload"}}}
    assert response["blank_render_frames"] == [{"frame": 0, "camera": "/World/Camera"}]
    assert renderer.render_calls == 1


def test_render_recovery_is_single_flight(monkeypatch):
    first_recovery_entered = threading.Event()
    release_first_recovery = threading.Event()

    def block_first_recovery() -> None:
        first_recovery_entered.set()
        assert release_first_recovery.wait(timeout=2.0)

    renderer = _FakeRenderer(
        initialized=False,
        daemon_running=False,
        recover_result=True,
        recover_hook=block_first_recovery,
    )
    monkeypatch.setattr(service_main, "_renderer", renderer)
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=True))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service_main.render, _render_request())
        assert first_recovery_entered.wait(timeout=2.0)

        second = executor.submit(service_main.render, _render_request())
        release_first_recovery.set()

        responses = [first.result(timeout=2.0), second.result(timeout=2.0)]

    assert [response["status"] for response in responses] == ["success", "success"]
    assert renderer.recover_calls == 1
    assert renderer.render_calls == 2


def test_render_rejects_when_recovery_cannot_initialize_renderer(monkeypatch):
    renderer = _FakeRenderer(
        initialized=False,
        daemon_running=False,
        recover_result=False,
    )
    monkeypatch.setattr(service_main, "_renderer", renderer)
    monkeypatch.setattr(service_main, "_warmup_task", _FakeTask(done_result=True))

    response = service_main.render(_render_request())

    assert response == {
        "status": "exception",
        "error": "Renderer not initialized",
        "images": {},
    }
    assert renderer.recover_calls == 1
    assert renderer.render_calls == 0
