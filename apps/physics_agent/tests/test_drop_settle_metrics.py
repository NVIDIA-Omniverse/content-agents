# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the drop_settle metric registry.

The registry decouples ``scenario.metric`` from a single hard-coded
scalar so the refine loop can swap in metrics like
``max_bounce_height`` without touching ``evaluate()``. Tests run on
synthetic ``[(t, pose7, vel6), ...]`` trajectories — no daemon, no USD.
"""

from __future__ import annotations

import math
import pathlib

import pytest

from physics_agent.tuning.scenarios.drop_settle import (
    _METRICS,
    MetricContext,
    _infer_up_idx,
    _metric_max_bounce_height,
    _metric_settle_distance,
    _resolve_up_idx,
)
from physics_agent.tuning.types import Scenario, TunableParam


def _scenario(metric: str = "settle_distance") -> Scenario:
    return Scenario(
        name="drop_settle",
        params=(TunableParam(name="restitution", min_value=0.0, max_value=1.0),),
        target={"drop_height_m": 0.5},
        metric=metric,
    )


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[float]:
    """Pose7 [px, py, pz, qx, qy, qz, qw]."""
    return [x, y, z, 0.0, 0.0, 0.0, 1.0]


def _vel() -> list[float]:
    """Vel6 zeros (max_bounce_height does not read velocity)."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_exposes_known_metrics() -> None:
    assert "settle_distance" in _METRICS
    assert "max_bounce_height" in _METRICS
    for fn in _METRICS.values():
        assert callable(fn)


# ---------------------------------------------------------------------------
# Up-axis inference
# ---------------------------------------------------------------------------


def test_infer_up_idx_y_up() -> None:
    assert _infer_up_idx([0.0, 0.5, 0.0]) == 1


def test_infer_up_idx_z_up() -> None:
    assert _infer_up_idx([0.0, 0.0, 0.7]) == 2


def test_infer_up_idx_origin_falls_back_to_y() -> None:
    """Corner-origin assets (rest_position == origin) default to Y-up."""
    assert _infer_up_idx([0.0, 0.0, 0.0]) == 1


# ---------------------------------------------------------------------------
# _resolve_up_idx — prefers scene_info["world_up"] over inference
# ---------------------------------------------------------------------------


def test_resolve_up_idx_prefers_world_up_for_z_up_corner_origin() -> None:
    """Corner-origin Z-up asset: rest_position is the origin (would default
    to Y-up under inference), but scene_info["world_up"] = [0, 0, 1] resolves
    to Z-up so bounce metrics measure the correct axis."""
    scene_info = {"world_up": [0.0, 0.0, 1.0]}
    assert _resolve_up_idx(scene_info, [0.0, 0.0, 0.0]) == 2


def test_resolve_up_idx_prefers_world_up_y_up() -> None:
    scene_info = {"world_up": [0.0, 1.0, 0.0]}
    assert _resolve_up_idx(scene_info, [0.0, 0.0, 0.0]) == 1


def test_resolve_up_idx_falls_back_when_world_up_missing() -> None:
    """Older callers stub scene_info without world_up — fall through to
    inference from rest_position."""
    assert _resolve_up_idx({}, [0.0, 0.5, 0.0]) == 1
    assert _resolve_up_idx(None, [0.0, 0.0, 0.7]) == 2


def test_resolve_up_idx_handles_zero_world_up_as_fallback() -> None:
    """A degenerate world_up vector (all zeros) falls back to inference."""
    assert _resolve_up_idx({"world_up": [0.0, 0.0, 0.0]}, [0.0, 0.5, 0.0]) == 1


# ---------------------------------------------------------------------------
# settle_distance metric
# ---------------------------------------------------------------------------


def test_settle_distance_equals_zero_when_final_pose_at_rest() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel()),
        (0.5, _pose(y=1.5), _vel()),
        (1.0, _pose(y=0.5), _vel()),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario(),
    )
    assert _metric_settle_distance(ctx) == pytest.approx(0.0, abs=1e-6)


def test_settle_distance_grows_with_offset() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel()),
        (1.0, _pose(y=1.5), _vel()),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario(),
    )
    # final at y=1.5, rest at y=0.5 → 1.0
    assert _metric_settle_distance(ctx) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# max_bounce_height metric
# ---------------------------------------------------------------------------


def test_max_bounce_height_finds_peak_after_first_contact() -> None:
    """Drop from y=2, touch ground at y=0.5 (rest), bounce up to y=1.4."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel()),
        (0.1, _pose(y=1.5), _vel()),
        (0.2, _pose(y=0.5), _vel()),  # first contact
        (0.3, _pose(y=1.0), _vel()),  # rebound rising
        (0.4, _pose(y=1.4), _vel()),  # peak rebound
        (0.5, _pose(y=1.0), _vel()),  # falling again
        (0.6, _pose(y=0.5), _vel()),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
    )
    score = _metric_max_bounce_height(ctx)
    # Peak rebound is 1.4 — score is its negation.
    assert score == pytest.approx(-1.4, abs=1e-6)


def test_max_bounce_height_higher_rebound_yields_lower_score() -> None:
    """A higher rebound (more bouncy) must produce a lower score so
    the optimizer drives toward larger bounce heights."""
    rest = (0.0, 0.5, 0.0)
    low_bounce = [
        (0.0, _pose(y=2.0), _vel()),
        (0.1, _pose(y=0.5), _vel()),  # contact
        (0.2, _pose(y=0.7), _vel()),  # tiny bounce
    ]
    high_bounce = [
        (0.0, _pose(y=2.0), _vel()),
        (0.1, _pose(y=0.5), _vel()),  # contact
        (0.2, _pose(y=1.8), _vel()),  # big bounce
    ]
    sc = _scenario("max_bounce_height")
    low = _metric_max_bounce_height(
        MetricContext(low_bounce, rest, up_idx=1, scenario=sc)
    )
    high = _metric_max_bounce_height(
        MetricContext(high_bounce, rest, up_idx=1, scenario=sc)
    )
    assert high < low  # negative-of-bigger is smaller


def test_max_bounce_height_z_up_uses_z_axis() -> None:
    rest = (0.0, 0.0, 0.5)  # Z-up
    trajectory = [
        (0.0, _pose(z=2.0), _vel()),
        (0.1, _pose(z=0.5), _vel()),  # contact
        (0.2, _pose(z=1.6), _vel()),  # rebound on Z
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=2,
        scenario=_scenario("max_bounce_height"),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-1.6, abs=1e-6)


def test_max_bounce_height_empty_trajectory_returns_inf() -> None:
    ctx = MetricContext(
        trajectory=[],
        rest_position=(0.0, 0.5, 0.0),
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
    )
    assert math.isinf(_metric_max_bounce_height(ctx))


def test_max_bounce_height_no_contact_returns_inf() -> None:
    """Body never touches ground, so no rebound height can be measured."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        # Starts and stays above rest+slack
        (0.0, _pose(y=3.0), _vel()),
        (0.1, _pose(y=2.5), _vel()),
        (0.2, _pose(y=2.8), _vel()),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
    )
    assert math.isinf(_metric_max_bounce_height(ctx))


def test_max_bounce_height_last_sample_contact_returns_inf() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel()),
        (0.1, _pose(y=0.51), _vel()),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
    )
    assert math.isinf(_metric_max_bounce_height(ctx))


# ---------------------------------------------------------------------------
# Unknown metric guard
# ---------------------------------------------------------------------------


def test_evaluate_rejects_unsupported_metric(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """drop_settle.evaluate raises ValueError when scenario.metric isn't in
    _METRICS — silent fallback would make artifacts report a metric the run
    didn't actually optimize, masking LLM/typo misconfigurations."""
    from world_understanding.functions.physics import (
        trajectory as trajectory_mod,
    )

    from physics_agent import recording as recording_pkg
    from physics_agent.tuning import usd_patch as usd_patch_mod
    from physics_agent.tuning.scenarios import _scene_builder as scene_builder_mod
    from physics_agent.tuning.scenarios.drop_settle import evaluate

    # Stub side effects so we never touch real USD/daemon code.
    def _fake_build(_src: object, dst: object, **_kwargs: object) -> dict[str, object]:
        pathlib.Path(dst).write_bytes(b"")  # type: ignore[arg-type]
        return {
            "body_pattern": "/Body",
            "body_prim_path": "/Body",
            "rest_position": [0.0, 0.0, 0.0],
            "drop_height_m_resolved": 0.05,
            "bbox_size_m": 0.1,
            "camera_paths": [],
        }

    monkeypatch.setattr(
        usd_patch_mod,
        "patch_physics_usd",
        lambda src, dst, params: pathlib.Path(dst).write_bytes(b""),
    )
    monkeypatch.setattr(scene_builder_mod, "build_drop_settle_scene", _fake_build)
    monkeypatch.setattr(
        recording_pkg,
        "author_trajectory_usda",
        lambda scene, traj, body, out, fps, **kwargs: pathlib.Path(out).write_bytes(
            b""
        ),
    )
    monkeypatch.setattr(
        trajectory_mod, "settle_distance", lambda traj, rest_position: 0.0
    )

    class _FakeDaemon:
        def evaluate(self, **kwargs: object) -> dict[str, object]:
            return {
                "trajectory": [(0.0, [0.0] * 7, [0.0] * 6)],
                "final_pose": [0.0] * 7,
            }

    bad_scenario = _scenario(metric="not_a_real_metric")
    physics_usd = tmp_path / "physics.usda"
    physics_usd.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported drop_settle metric"):
        evaluate(
            params={},
            scenario=bad_scenario,
            physics_usd=physics_usd,
            seed=0,
            simulator=_FakeDaemon(),  # type: ignore[arg-type]
            work_dir=tmp_path / "work",
        )
