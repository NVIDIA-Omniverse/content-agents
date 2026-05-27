# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from service.dispatcher import OVRTXDispatcher, parse_gpu_workers  # noqa: E402


class _FakeResponse:
    def __init__(
        self,
        payload: dict,
        *,
        status_code: int = 200,
        text: str | None = None,
    ):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json_dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            from requests import HTTPError

            raise HTTPError(f"{self.status_code} error", response=self)
        return None

    def json(self) -> dict:
        return self._payload


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload)


def test_parse_gpu_workers_accepts_count_and_explicit_ids() -> None:
    assert parse_gpu_workers(None) == []
    assert parse_gpu_workers("") == []
    assert parse_gpu_workers("2") == ["0", "1"]
    assert parse_gpu_workers("0,1,3") == ["0", "1", "3"]
    assert parse_gpu_workers("GPU-abcd") == ["GPU-abcd"]


def test_health_aggregates_ready_workers() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0", "1"])
    dispatcher._workers[0].ready = True
    dispatcher._workers[0].renderer_initialized = True
    dispatcher._workers[0].daemon_running = True
    dispatcher._workers[0].status = "healthy"
    dispatcher._workers[1].status = "initializing"

    payload = dispatcher.health()

    assert payload["status"] == "healthy"
    assert payload["gpu_initialized"] is True
    assert payload["ready_workers"] == 1
    assert payload["total_workers"] == 2
    assert payload["workers"][0]["gpu"] == "0"
    assert payload["workers"][1]["ready"] is False


def test_render_routes_to_idle_workers(monkeypatch) -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0", "1"])
    for worker in dispatcher._workers:
        worker.ready = True
        worker.status = "healthy"

    entered_first_request = threading.Event()
    release_first_request = threading.Event()
    called_urls: list[str] = []
    lock = threading.Lock()

    def fake_post(url: str, **_kwargs):
        with lock:
            called_urls.append(url)
            call_number = len(called_urls)
        if call_number == 1:
            entered_first_request.set()
            assert release_first_request.wait(timeout=2.0)
        return _FakeResponse({"status": "success", "error": None, "images": {}})

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(dispatcher.render, {"url": "data:,x"})
        assert entered_first_request.wait(timeout=2.0)
        second = executor.submit(dispatcher.render, {"url": "data:,x"})
        release_first_request.set()
        responses = [first.result(timeout=2.0), second.result(timeout=2.0)]

    assert [response["status"] for response in responses] == ["success", "success"]
    assert called_urls == [
        "http://127.0.0.1:8100/render",
        "http://127.0.0.1:8101/render",
    ]


def test_render_times_out_when_no_worker_is_ready() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], queue_timeout_seconds=0.01)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "exception"
    assert "Timed out waiting for a ready OVRTX worker" in response["error"]


def test_render_does_not_mark_worker_unhealthy_for_client_http_error(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {"detail": "bad request"},
            status_code=422,
            text='{"detail":"bad request"}',
        )

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "exception"
    assert "HTTP 422" in response["error"]
    assert worker.ready is True
    assert worker.status == "healthy"


def test_render_preserves_blank_render_worker_detail(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {
                "detail": {
                    "status": "blank_render",
                    "error": "1/1 OVRTX render frames are blank or near-blank.",
                    "warnings": ["blank frame"],
                    "blank_render_frames": [{"frame": 0}],
                }
            },
            status_code=422,
        )

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "blank_render"
    assert response["images"] == {}
    assert response["blank_render_frames"] == [{"frame": 0}]
    assert worker.ready is True
    assert worker.status == "healthy"


def test_render_treats_fastapi_validation_422_as_client_error(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "url"],
                        "msg": "Field required",
                    }
                ]
            },
            status_code=422,
        )

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "exception"
    assert "HTTP 422" in response["error"]
    assert worker.ready is True
    assert worker.status == "healthy"


def test_render_marks_worker_unhealthy_for_server_http_error(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {"detail": "server error"},
            status_code=503,
            text='{"detail":"server error"}',
        )

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "exception"
    assert worker.ready is False
    assert worker.status == "unhealthy"


def test_check_worker_updates_readiness_from_health(monkeypatch) -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])

    class _FakeProcess:
        def poll(self):
            return None

    dispatcher._workers[0].process = _FakeProcess()

    def fake_get(url: str, **_kwargs):
        assert url == "http://127.0.0.1:8100/health"
        return _FakeResponse(
            {
                "status": "healthy",
                "gpu_initialized": True,
                "renderer_initialized": True,
                "daemon_running": True,
            }
        )

    monkeypatch.setattr("service.dispatcher.requests.get", fake_get)

    dispatcher._check_worker(dispatcher._workers[0])

    assert dispatcher._workers[0].ready is True
    assert dispatcher._workers[0].renderer_initialized is True
    assert dispatcher._workers[0].daemon_running is True
    assert dispatcher._workers[0].status == "healthy"


def test_check_worker_restarts_after_cooldown_once(monkeypatch) -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=0.01)
    worker = dispatcher._workers[0]

    class _ExitedProcess:
        returncode = 9

        def poll(self):
            return 9

    worker.process = _ExitedProcess()
    starts = 0

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1
        _worker.status = "starting"

    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._check_worker(worker)
    assert starts == 0
    first_deadline = worker.next_restart_at

    dispatcher._check_worker(worker)
    assert starts == 0
    assert worker.next_restart_at == first_deadline

    import time

    time.sleep(0.02)
    dispatcher._check_worker(worker)

    assert starts == 1


def test_check_worker_restarts_live_unhealthy_worker_after_cooldown(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=0.01)
    worker = dispatcher._workers[0]

    class _LiveProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = _LiveProcess()
    worker.process = process
    starts = 0

    def fake_get(url: str, **_kwargs):
        return _FakeResponse(
            {
                "status": "unhealthy",
                "gpu_initialized": False,
                "renderer_initialized": False,
                "daemon_running": False,
                "error": "warm-up failed",
            }
        )

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1
        _worker.status = "starting"

    monkeypatch.setattr("service.dispatcher.requests.get", fake_get)
    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._check_worker(worker)
    assert starts == 0
    assert worker.unhealthy_since is not None

    import time

    time.sleep(0.02)
    dispatcher._check_worker(worker)

    assert starts == 1
    assert process.terminated is True


def test_check_worker_restarts_live_worker_after_health_exception(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=0.01)
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    class _LiveProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = _LiveProcess()
    worker.process = process
    starts = 0

    def fake_get(url: str, **_kwargs):
        raise TimeoutError("health timed out")

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1
        _worker.status = "starting"

    monkeypatch.setattr("service.dispatcher.requests.get", fake_get)
    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._check_worker(worker)
    assert starts == 0
    assert worker.ready is False
    assert worker.status == "unhealthy"
    assert worker.unhealthy_since is not None

    time.sleep(0.02)
    dispatcher._check_worker(worker)

    assert starts == 1
    assert process.terminated is True


def test_check_worker_defers_unhealthy_restart_while_render_in_flight(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=0.01)
    worker = dispatcher._workers[0]
    worker.process = object()
    worker.ready = False
    worker.status = "unhealthy"
    worker.unhealthy_since = time.monotonic() - 1.0
    worker.in_flight = 1
    starts = 0

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1

    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._restart_unhealthy_worker_if_due(worker)

    assert starts == 0
    assert worker.process is not None
    assert worker.status == "unhealthy"


def test_stop_kills_and_reaps_stuck_worker() -> None:
    import asyncio
    import subprocess

    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]

    class _StuckProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.waits = 0

        def poll(self):
            return None if not self.killed else -9

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits += 1
            if not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

    process = _StuckProcess()
    worker.process = process

    asyncio.run(dispatcher.stop())

    assert process.terminated is True
    assert process.killed is True
    assert process.waits >= 2


def test_start_cleans_up_started_workers_on_start_failure(monkeypatch) -> None:
    import asyncio

    dispatcher = OVRTXDispatcher(gpu_ids=["0", "1"])

    class _LiveProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

        def kill(self):
            raise AssertionError("terminate should drain without kill")

    process = _LiveProcess()

    def fake_start_worker(worker):
        if worker.spec.gpu_id == "0":
            worker.process = process
            return
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    with pytest.raises(RuntimeError, match="spawn failed"):
        asyncio.run(dispatcher.start())

    assert process.terminated is True
    assert process.waited is True
    assert dispatcher._monitor_task is None


def test_monitor_survives_worker_check_exception(monkeypatch):
    import asyncio

    dispatcher = OVRTXDispatcher(gpu_ids=["0"], health_interval_seconds=0.01)
    checks = 0

    def fake_check_worker(_worker):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise OSError("spawn failed")
        dispatcher._stop_event.set()

    monkeypatch.setattr(dispatcher, "_check_worker", fake_check_worker)

    asyncio.run(dispatcher._monitor_workers())

    assert checks >= 2


def test_dispatcher_rejects_worker_port_collision() -> None:
    with pytest.raises(ValueError, match="collides"):
        OVRTXDispatcher(gpu_ids=["0"], port_base=8000, parent_port=8000)


def test_dispatcher_rejects_duplicate_gpu_ids() -> None:
    with pytest.raises(ValueError, match="duplicate GPU id"):
        OVRTXDispatcher(gpu_ids=["0", "0"])
