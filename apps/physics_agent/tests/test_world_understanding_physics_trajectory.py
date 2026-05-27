# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``world_understanding.functions.physics.trajectory``.

All trajectories use the (t, pose7, vel6) 3-tuple shape — the daemon
emits velocity alongside pose, and metrics read raw simulator
velocity directly instead of finite-differencing positions.
"""

from __future__ import annotations

import math

import pytest
from world_understanding.functions.physics.trajectory import (
    fell_over,
    infer_world_up,
    max_angular_speed,
    max_linear_speed,
    settle_distance,
    settle_time,
    trajectory_summary,
)


def _identity_quat() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0]  # qx, qy, qz, qw == identity


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[float]:
    return [x, y, z] + _identity_quat()


def _vel(
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    wx: float = 0.0,
    wy: float = 0.0,
    wz: float = 0.0,
) -> list[float]:
    return [vx, vy, vz, wx, wy, wz]


def _zero_vel() -> list[float]:
    return _vel()


def test_settle_distance_zero_when_final_at_rest() -> None:
    traj = [
        (0.0, _pose(0, 1, 0), _zero_vel()),
        (1.0, _pose(0, 0.5, 0), _zero_vel()),
        (2.0, _pose(0, 0.25, 0), _zero_vel()),
    ]
    d = settle_distance(traj, rest_position=[0.0, 0.25, 0.0])
    assert d == pytest.approx(0.0, abs=1e-9)


def test_settle_distance_nonzero_when_offset() -> None:
    traj = [
        (0.0, _pose(0, 1, 0), _zero_vel()),
        (1.0, _pose(0.5, 0.25, 0), _zero_vel()),
    ]
    d = settle_distance(traj, rest_position=[0.0, 0.25, 0.0])
    assert d == pytest.approx(0.5, abs=1e-9)


def test_settle_distance_inf_on_empty_trajectory() -> None:
    assert math.isinf(settle_distance([], rest_position=[0.0, 0.0, 0.0]))


def test_two_tuple_trajectory_rejected() -> None:
    """Legacy (t, pose7) shape must produce a clear error — the daemon
    now emits velocity per sample and finite-differencing is gone."""
    legacy = [(0.0, _pose(0, 0, 0)), (1.0, _pose(0, 0, 0))]
    with pytest.raises(ValueError, match=r"3-tuple"):
        max_linear_speed(legacy)  # type: ignore[arg-type]


def test_max_linear_speed_reads_raw_velocity() -> None:
    # Velocity from simulator is exactly 1 m/s along +x, even though
    # pose stays static — proves we read vel6 not pose deltas.
    traj = [(i * 0.1, _pose(0, 0, 0), _vel(vx=1.0)) for i in range(11)]
    assert max_linear_speed(traj) == pytest.approx(1.0, abs=1e-9)


def test_max_linear_speed_zero_for_static() -> None:
    traj = [(t, _pose(0, 0, 0), _zero_vel()) for t in (0.0, 0.5, 1.0)]
    assert max_linear_speed(traj) == 0.0


def test_max_angular_speed_reads_raw_value() -> None:
    # 5 rad/s around Y, recorded directly in vel6[3:6].
    traj = [(i * 0.1, _pose(0, 0, 0), _vel(wy=5.0)) for i in range(11)]
    assert max_angular_speed(traj) == pytest.approx(5.0, abs=1e-9)


def test_settle_time_returns_when_velocity_drops() -> None:
    # Velocity above threshold for the first 5 samples, zero for the
    # rest. Sample dt = 0.1s; sustain_window = 0.2s.
    traj = []
    for i in range(5):
        traj.append((i * 0.1, _pose(0, 0, 0), _vel(vx=0.5)))
    for i in range(5, 11):
        traj.append((i * 0.1, _pose(0, 0, 0), _zero_vel()))
    t = settle_time(traj, threshold_m_s=0.1, sustain_window_s=0.2)
    assert t is not None
    # Below-threshold window starts at index 5 (t = 0.5s) and lasts.
    assert 0.4 <= t <= 0.7


def test_settle_time_returns_none_when_never_settles() -> None:
    # Always above threshold.
    traj = [(i * 0.1, _pose(0, 0, 0), _vel(vx=2.0)) for i in range(11)]
    assert settle_time(traj, threshold_m_s=0.1, sustain_window_s=0.5) is None


def test_settle_time_uses_elapsed_time_not_sample_count() -> None:
    """Regression for the round-5 off-by-one: at 10 Hz a 0.2s sustain
    window must NOT settle on two below-threshold samples (only 0.1s
    has actually elapsed).
    """
    # dt = 0.1s. Index 0 is below threshold; index 1 is also below;
    # index 2 spikes above. The previous ``round(window / dt) = 2``
    # logic would accept the [0, 1] pair and report settle_time=0.0
    # even though the body was only quiet for 0.1s.
    traj = [
        (0.0, _pose(0, 0, 0), _zero_vel()),
        (0.1, _pose(0, 0, 0), _zero_vel()),
        (0.2, _pose(0, 0, 0), _vel(vx=2.0)),
        (0.3, _pose(0, 0, 0), _vel(vx=2.0)),
    ]
    # 0.2s window — not satisfied (only 0.1s of quiet, then a spike).
    assert settle_time(traj, threshold_m_s=0.1, sustain_window_s=0.2) is None


def test_settle_time_returns_anchor_time_not_window_end() -> None:
    """When the body is below threshold from t=0 onward and the
    trajectory carries it for at least the sustain window, the function
    returns the **start** of the run (anchor time), not the time at
    which the window expires.
    """
    # dt = 0.1s. All samples are quiet; trajectory is 0.5s long which
    # is comfortably > 0.2s window.
    traj = [(i * 0.1, _pose(0, 0, 0), _zero_vel()) for i in range(6)]
    # First sample (t=0.0) anchors the below-threshold run; window
    # crossed at t=0.2. Anchor wins.
    assert settle_time(traj, threshold_m_s=0.1, sustain_window_s=0.2) == 0.0


def test_fell_over_false_for_identity() -> None:
    traj = [(t, _pose(0, 1, 0), _zero_vel()) for t in (0, 0.5, 1.0)]
    assert fell_over(traj) is False


def test_fell_over_true_for_90_deg_rotation() -> None:
    # Rotate 90° around Z so the body's local +Y rotates to world +X.
    half = math.pi / 4
    quat = [0, 0, math.sin(half), math.cos(half)]
    traj = [
        (0.0, [0, 0, 0] + _identity_quat(), _zero_vel()),
        (1.0, [0, 0, 0] + quat, _zero_vel()),
    ]
    assert fell_over(traj, max_tilt_deg=80.0) is True


def test_trajectory_summary_keys_and_types() -> None:
    traj = [(t, _pose(t, 1.0, 0), _zero_vel()) for t in (0.0, 0.5, 1.0)]
    s = trajectory_summary(traj)
    for k in (
        "n_samples",
        "duration_s",
        "final_position",
        "max_linear_speed",
        "max_angular_speed",
        "settle_time_s",
        "fell_over",
    ):
        assert k in s
    assert s["n_samples"] == 3
    assert s["duration_s"] == pytest.approx(1.0)
    assert s["fell_over"] is False


# ---------------------------------------------------------------------------
# infer_world_up + trajectory_summary world_up forwarding
# ---------------------------------------------------------------------------


def test_infer_world_up_z_up_from_negative_rest() -> None:
    """drop_settle's Z-up scenes write rest_position with only the Z
    component nonzero (typically negative, since it's body-frame). The
    helper picks the largest-magnitude component and returns a unit
    vector in the positive direction — sign-independent."""
    assert infer_world_up([0.0, 0.0, -0.0989]) == (0.0, 0.0, 1.0)


def test_infer_world_up_y_up_from_positive_rest() -> None:
    assert infer_world_up([0.0, 0.5, 0.0]) == (0.0, 1.0, 0.0)


def test_infer_world_up_falls_back_to_y_up_when_all_zero() -> None:
    """Degenerate inputs (no axis carries signal) keep the legacy Y-up
    convention so callers without rest_position aren't broken."""
    assert infer_world_up([0.0, 0.0, 0.0]) == (0.0, 1.0, 0.0)
    assert infer_world_up([]) == (0.0, 1.0, 0.0)


def test_trajectory_summary_legacy_default_is_y_up() -> None:
    """No world_up kwarg => byte-identical Y-up behaviour. A body that
    starts upright in Y-up and rotates 90° around Z (its local +Y points
    along world +X) reads as fallen under the legacy default."""
    half = math.pi / 4  # 90° / 2
    quat = [0.0, 0.0, math.sin(half), math.cos(half)]
    traj = [
        (0.0, [0, 1, 0] + _identity_quat(), _zero_vel()),
        (1.0, [0, 1, 0] + quat, _zero_vel()),
    ]
    assert trajectory_summary(traj)["fell_over"] is True


def test_trajectory_summary_z_up_aware_via_world_up() -> None:
    """The bug this fix closes: on a Z-up stage, a body that tips its
    own +Z axis but leaves +Y untouched is **invisible** to the legacy
    Y-up check. The fix makes it visible when the caller passes
    world_up=(0,0,1).

    Rotation around the world Y-axis is the cleanest demonstration:
    body's local +Y stays world +Y (Y-up check sees no tilt → False),
    while body's local +Z rotates 90° onto world +X (Z-up check sees a
    full topple → True)."""
    half = math.pi / 4  # 90° / 2
    # Quaternion for 90° rotation around the world Y axis (qx,qy,qz,qw).
    quat = [0.0, math.sin(half), 0.0, math.cos(half)]
    traj = [
        (0.0, [0, 0, 1] + _identity_quat(), _zero_vel()),
        (1.0, [0, 0, 1] + quat, _zero_vel()),
    ]
    # Z-up aware — recognises the topple.
    assert trajectory_summary(traj, world_up=(0.0, 0.0, 1.0))["fell_over"] is True
    # Legacy Y-up — same data misclassified (the body's +Y stayed put;
    # only +Z tipped, which the Y-up check ignores).
    assert trajectory_summary(traj)["fell_over"] is False


def test_trajectory_summary_z_up_upright_is_not_fallen() -> None:
    """Sanity: an upright Z-up body that doesn't rotate stays upright
    under the Z-up world_up kwarg."""
    traj = [(t, _pose(0, 0, 1.0), _zero_vel()) for t in (0.0, 0.5, 1.0)]
    assert trajectory_summary(traj, world_up=(0.0, 0.0, 1.0))["fell_over"] is False
