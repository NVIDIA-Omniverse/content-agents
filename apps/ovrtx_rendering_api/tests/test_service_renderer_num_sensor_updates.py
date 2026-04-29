# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-request ``num_sensor_updates`` plumbing.

Exercises that the render-settings field is forwarded into the backend's
``num_sensor_updates`` kwarg without touching pxr or ovrtx — the backend is
replaced with a recording stub.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

# service/ is on sys.path via [tool.pytest.ini_options].pythonpath in the
# root pyproject.toml.
from service import renderer as renderer_module


class _RecordingBackend:
    """Stand-in for OvRTXRenderingBackend that records render() kwargs."""

    def __init__(self) -> None:
        self.last_num_sensor_updates: int | None | object = object()
        self.last_render_mode: str | None | object = object()
        self.last_kwargs: dict[str, Any] = {}

    def render(self, **kwargs: Any) -> dict[str, Any]:
        self.last_num_sensor_updates = kwargs.get("num_sensor_updates")
        self.last_render_mode = kwargs.get("render_mode")
        self.last_kwargs = kwargs
        # Produce a minimally valid backend result so _to_v1_response
        # doesn't need to deal with empty payloads.
        return {
            "results": [
                {
                    "camera": "/World/Camera",
                    "successful_frames": 1,
                    "failed_frames": 0,
                    "images": [{"frame": 0, "image_base64": "fake"}],
                    "sensor_files": {},
                }
            ]
        }


class _FakeStage:
    """Truthy placeholder so ``if not stage:`` does not short-circuit."""

    def __bool__(self) -> bool:
        return True


@pytest.fixture
def renderer_with_stub_backend():
    """Produce a Renderer whose backend, stage open, and fetch are stubbed."""
    import threading
    import types

    # Skip Renderer.__init__ — it constructs OvRTXRenderingBackend, which
    # requires the ovrtx venv and a GPU. The only attributes render() needs
    # are _backend, _render_lock, and _initialized.
    backend = _RecordingBackend()
    r = renderer_module.Renderer.__new__(renderer_module.Renderer)
    r._backend = backend  # type: ignore[attr-defined]
    r._initialized = True  # type: ignore[attr-defined]
    r._render_lock = threading.Lock()  # type: ignore[attr-defined]

    # Stub the fetch (no network / disk) and pxr (no USD C++ deps).
    pxr_mod = types.ModuleType("pxr")
    pxr_mod.Usd = types.SimpleNamespace(  # type: ignore[attr-defined]
        Stage=types.SimpleNamespace(Open=lambda _p: _FakeStage())
    )

    with (
        patch.object(renderer_module, "_fetch_usd", lambda _url, _path: None),
        patch.dict("sys.modules", {"pxr": pxr_mod}),
    ):
        yield r, backend


class TestNumSensorUpdatesPrecedence:
    def test_per_request_value_is_forwarded_to_backend(
        self, renderer_with_stub_backend
    ):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
            num_sensor_updates=32,
        )
        assert backend.last_num_sensor_updates == 32

    def test_unset_passes_none_so_backend_uses_instance_default(
        self, renderer_with_stub_backend
    ):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )
        assert backend.last_num_sensor_updates is None


class TestRenderSettingsValidation:
    def test_num_sensor_updates_rejects_zero(self):
        from service.models import RenderSettings

        with pytest.raises(ValueError):
            RenderSettings(num_sensor_updates=0)

    def test_num_sensor_updates_accepts_positive(self):
        from service.models import RenderSettings

        assert RenderSettings(num_sensor_updates=1).num_sensor_updates == 1
        assert RenderSettings(num_sensor_updates=100).num_sensor_updates == 100

    def test_default_is_none_so_backend_instance_default_wins(self):
        """Unspecified field defaults to None so OVRTX_NUM_SENSOR_UPDATES still governs."""
        from service.models import RenderSettings

        assert RenderSettings().num_sensor_updates is None


class TestRenderModePrecedence:
    """Mirrors the num_sensor_updates precedence tests — per-request render_mode
    must reach the backend, and None must leave the backend's instance
    default (seeded from OVRTX_RENDER_MODE env at service startup) alone.
    """

    def test_per_request_mode_is_forwarded_to_backend(self, renderer_with_stub_backend):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
            render_mode="rt2",
        )
        assert backend.last_render_mode == "rt2"

    def test_unset_mode_passes_none_to_backend(self, renderer_with_stub_backend):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )
        assert backend.last_render_mode is None

    @pytest.mark.parametrize("mode", ["rt1", "rt2", "pt"])
    def test_all_supported_modes_are_forwarded(self, renderer_with_stub_backend, mode):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
            render_mode=mode,
        )
        assert backend.last_render_mode == mode
