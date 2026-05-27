# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdLux

from world_understanding.functions.graphics.render_time_sampled_usd import (
    render_time_sampled_usd,
)

rt = importlib.import_module(
    "world_understanding.functions.graphics.render_time_sampled_usd"
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def render(
        self,
        stage: Usd.Stage,
        cameras: list[str] | None = None,
        image_width: int = 1024,
        image_height: int | None = None,
        frames: str = "0",
        **kwargs: Any,
    ) -> dict[str, Any]:
        image_height = image_height or image_width
        selected_frames = rt._parse_frames(frames)
        selected_cameras = cameras or ["/Camera"]
        self.calls.append(
            {
                "cameras": selected_cameras,
                "frames": frames,
                "image_width": image_width,
                "image_height": image_height,
                "stage_fps": stage.GetTimeCodesPerSecond(),
            }
        )

        results = []
        for camera in selected_cameras:
            images = [
                Image.new(
                    "RGBA",
                    (image_width, image_height),
                    (frame % 255, 32, 64, 255),
                )
                for frame in selected_frames
            ]
            results.append(
                {
                    "camera": camera,
                    "images": images,
                    "render_time": 0.0,
                    "frame_count": len(images),
                    "status": "success",
                    "sensors": {},
                }
            )

        return {
            "total_cameras": len(results),
            "successful_cameras": len(results),
            "failed_cameras": 0,
            "results": results,
        }


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setattr(rt, "_make_backend", lambda renderer: backend)
    return backend


def _write_time_sampled_usd(
    path: Path,
    *,
    start: float | None = None,
    end: float | None = None,
    fps: float = 24.0,
    samples: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(fps)
    if start is not None:
        stage.SetStartTimeCode(start)
    if end is not None:
        stage.SetEndTimeCode(end)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    translate = UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp()
    for frame in samples:
        translate.Set(Gf.Vec3d(float(frame), 0.0, 0.0), Usd.TimeCode(frame))

    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(50.0)
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -5.0, 2.0))

    light = UsdLux.DomeLight.Define(stage, "/DomeLight")
    light.CreateIntensityAttr(500.0)

    stage.GetRootLayer().Save()
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_explicit_frames_are_rendered_chronologically(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    usd_path = _write_time_sampled_usd(tmp_path / "time_sampled.usda")

    paths = render_time_sampled_usd(
        usd_path,
        tmp_path / "renders",
        frames="10,0,5",
        image_width=8,
        image_height=6,
        make_mp4=False,
    )

    assert fake_backend.calls[0]["frames"] == "0,5,10"
    assert [path.name for path in paths] == [
        "frame_0000.png",
        "frame_0005.png",
        "frame_0010.png",
    ]
    assert all(path.exists() for path in paths)


def test_infers_stage_range(tmp_path: Path, fake_backend: FakeBackend) -> None:
    usd_path = _write_time_sampled_usd(
        tmp_path / "range.usda", start=2.0, end=4.0, samples=(2.0, 3.0, 4.0)
    )

    paths = render_time_sampled_usd(
        usd_path,
        tmp_path / "renders",
        image_width=4,
        image_height=4,
        make_mp4=False,
    )

    assert fake_backend.calls[0]["frames"] == "2:4"
    assert [path.name for path in paths] == [
        "frame_0002.png",
        "frame_0003.png",
        "frame_0004.png",
    ]


def test_falls_back_to_authored_time_samples(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    usd_path = _write_time_sampled_usd(tmp_path / "samples.usda", samples=(3.0, 5.0))

    render_time_sampled_usd(
        usd_path,
        tmp_path / "renders",
        image_width=4,
        image_height=4,
        make_mp4=False,
    )

    assert fake_backend.calls[0]["frames"] == "3:5"


def test_multiple_cameras_are_prefixed(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    usd_path = _write_time_sampled_usd(tmp_path / "multi_camera.usda")

    paths = render_time_sampled_usd(
        usd_path,
        tmp_path / "renders",
        frames="0",
        cameras=["/World/Cam A", "/Camera"],
        image_width=4,
        image_height=4,
        make_mp4=False,
    )

    assert fake_backend.calls[0]["cameras"] == ["/World/Cam A", "/Camera"]
    assert [path.name for path in paths] == [
        "World_Cam_A__frame_0000.png",
        "Camera__frame_0000.png",
    ]


def test_fps_override_does_not_modify_input_usd(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    usd_path = _write_time_sampled_usd(tmp_path / "readonly.usda", fps=24.0)
    before = _sha256(usd_path)

    render_time_sampled_usd(
        usd_path,
        tmp_path / "renders",
        frames="0",
        fps=30,
        image_width=4,
        image_height=4,
        make_mp4=False,
    )

    assert fake_backend.calls[0]["stage_fps"] == 30.0
    assert _sha256(usd_path) == before


def test_frame_cap_warns_but_does_not_block_extra_frames(
    tmp_path: Path,
    fake_backend: FakeBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Frame-count discrepancy is advisory (warn-not-raise).

    The recorder writes ``int(fps × duration) + 1`` samples for closed-
    interval trajectories, which trips a strict cap by exactly 1. The
    cap was downgraded to a warning so legitimate recordings render and
    only accidentally huge ones surface a log line. See
    ``_validate_frame_caps`` in ``render_time_sampled_usd``.
    """
    usd_path = _write_time_sampled_usd(tmp_path / "too_many.usda", fps=2.0)

    with caplog.at_level("WARNING", logger=rt.logger.name):
        paths = render_time_sampled_usd(
            usd_path,
            tmp_path / "renders",
            frames="0:2",
            image_width=4,
            image_height=4,
            make_mp4=False,
            max_duration_seconds=1.0,
        )

    # Render proceeded on all 3 selected frames despite the cap of 2.
    assert len(fake_backend.calls) == 1
    assert fake_backend.calls[0]["frames"] == "0:2"
    assert len(paths) == 3
    assert any(
        "frame-count discrepancy" in record.message
        and "3 time samples" in record.message
        for record in caplog.records
    )


def test_frame_count_beyond_tolerance_raises(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    """Hard cap: frame count > max_frames + 1 raises ValueError.

    The frame-cap check has a 1-frame closed-interval tolerance to
    accommodate recorders that write ``int(fps × duration) + 1``
    samples. Beyond that tolerance, ``_validate_frame_caps`` must
    abort rather than silently rendering an unbounded evidence set —
    that's the contract ``max_duration_seconds`` advertises.

    Setup: ``fps=2.0`` and ``max_duration_seconds=1.0`` give
    ``max_frames = 2`` and tolerance ``max_frames + 1 = 3``.
    Requesting frames ``0:4`` (5 frames) is 2 beyond the tolerance
    and must raise.
    """
    usd_path = _write_time_sampled_usd(
        tmp_path / "way_too_many.usda",
        fps=2.0,
        start=0.0,
        end=4.0,
        samples=(0.0, 1.0, 2.0, 3.0, 4.0),
    )

    with pytest.raises(ValueError, match="exceeds cap"):
        render_time_sampled_usd(
            usd_path,
            tmp_path / "renders",
            frames="0:4",
            image_width=4,
            image_height=4,
            make_mp4=False,
            max_duration_seconds=1.0,
        )

    # Backend must NOT have been called — the cap must trip pre-render.
    assert fake_backend.calls == []


def test_nvcf_sparse_frames_are_rejected(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    """NVCF rejects sparse frame strings up front.

    The remote NVCF frame parser only accepts ``"N"`` or ``"start:end"``;
    a sparse spec like ``"0,5,10"`` would otherwise raise mid-request
    inside ``int("0,5,10")`` with an opaque message. The driver guards
    this case explicitly before constructing the backend.
    """
    usd_path = _write_time_sampled_usd(
        tmp_path / "sparse.usda",
        start=0.0,
        end=10.0,
        samples=(0.0, 5.0, 10.0),
    )

    with pytest.raises(ValueError, match="sparse frame selections"):
        render_time_sampled_usd(
            usd_path,
            tmp_path / "renders",
            renderer="nvcf",
            frames="0,5,10",
            image_width=4,
            image_height=4,
            make_mp4=False,
        )

    # Guard trips pre-backend; no backend call should land.
    assert fake_backend.calls == []


def test_non_integer_time_samples_are_rejected(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    usd_path = _write_time_sampled_usd(tmp_path / "fractional.usda", samples=(0.5,))

    with pytest.raises(ValueError, match="non-integer time code"):
        render_time_sampled_usd(
            usd_path,
            tmp_path / "renders",
            image_width=4,
            image_height=4,
            make_mp4=False,
        )

    assert fake_backend.calls == []


def test_render_time_sampled_usd_is_exported_from_graphics_package() -> None:
    from world_understanding.functions.graphics import (
        render_time_sampled_usd as exported,
    )

    assert exported is render_time_sampled_usd


def test_authored_single_frame_range_is_honored(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    # Stage has start == end == 5 explicitly authored. Per USD this is
    # a valid single-frame range that should bypass the
    # _authored_time_sample_frames scan and render exactly frame 5.
    usd_path = _write_time_sampled_usd(
        tmp_path / "single_frame.usda",
        start=5.0,
        end=5.0,
        # No animated samples — only the explicit frame-5 range.
        samples=(),
    )

    paths = render_time_sampled_usd(
        usd_path,
        tmp_path / "renders",
        image_width=8,
        image_height=8,
        make_mp4=False,
    )
    assert len(paths) == 1
    # The driver formats single-frame selections as the bare integer.
    assert fake_backend.calls[-1]["frames"] == "5"


def test_three_way_slug_collision_dedups_to_unique_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Multi-collision: third camera's normalized slug happens to equal
    # the second camera's disambiguated suffix. The dedup loop must
    # walk the suffix counter past the second camera's claim and assign
    # a unique slug to the third.
    backend = FakeBackend()
    real_render = backend.render

    def three_camera_render(stage, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["cameras"] = ["/World/Cam A", "/World/Cam_A", "/World/Cam_A_1"]
        return real_render(stage, **kwargs)

    backend.render = three_camera_render  # type: ignore[method-assign]
    monkeypatch.setattr(rt, "_make_backend", lambda renderer: backend)

    usd_path = _write_time_sampled_usd(tmp_path / "clip.usda")
    paths = render_time_sampled_usd(
        usd_path,
        tmp_path / "renders",
        frames="0:1",
        image_width=8,
        image_height=8,
        make_mp4=False,
    )
    # 3 cameras × 2 frames = 6 unique PNG paths.
    assert len(paths) == 6
    assert len(set(paths)) == 6, "multi-collision dedup left overlapping PNG paths"


def test_collision_in_camera_slugs_does_not_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two camera paths that normalize to the same slug must produce
    # disambiguated PNG/mp4 names — otherwise the second camera's PNGs
    # silently overwrite the first's. ``/World/Cam A`` and
    # ``/World/Cam_A`` both go through _slug() to the same string;
    # the dedup in _save_rendered_images must add a suffix to the
    # second occurrence so all PNG paths stay unique.
    backend = FakeBackend()

    real_render = backend.render

    def two_camera_render(stage, **kwargs):  # type: ignore[no-untyped-def]
        # Force a slug collision regardless of what the caller passed.
        kwargs["cameras"] = ["/World/Cam A", "/World/Cam_A"]
        return real_render(stage, **kwargs)

    backend.render = two_camera_render  # type: ignore[method-assign]
    monkeypatch.setattr(rt, "_make_backend", lambda renderer: backend)

    usd_path = _write_time_sampled_usd(tmp_path / "clip.usda")
    paths = render_time_sampled_usd(
        usd_path,
        tmp_path / "renders",
        frames="0:1",
        image_width=8,
        image_height=8,
        make_mp4=False,
    )

    # 2 cameras × 2 frames = 4 unique PNG paths.
    assert len(paths) == 4
    assert len(set(paths)) == 4, "camera-slug collision overwrote PNG paths"
