# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic trajectory metrics for physics-tune scenarios.

Pure-python numpy on top of the daemon's
``trajectory: list[(t_s, pose7, vel6)]`` shape — no physics imports,
no provider SDKs. Reusable across the physics-tune scenarios
(drop_settle and freeform), the validate flow, and the Validation
Agent. Also accepts the same shape produced by reading back from a
``recording.usda`` via
``world_understanding.utils.usd.time_samples.read_pose_velocity_trajectory``
— callers can derive metrics either from the in-flight daemon dict
or from a persisted recording, no shape difference.

Each metric is intentionally cheap to compute (single pass over the
trajectory) so they're safe to invoke per optimizer trial.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Pose layout. Matches ovphysx ``TensorType.RIGID_BODY_POSE``:
#   pose7 = [px, py, pz, qx, qy, qz, qw]
# Keep this constant in sync with ``time_samples._POSE7_LEN``.
_POSE7_LEN = 7

# Velocity layout. Matches ovphysx ``TensorType.RIGID_BODY_VELOCITY``:
#   vel6 = [vx, vy, vz, wx, wy, wz]
_VEL6_LEN = 6


def _trajectory_to_arrays(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``[(t, pose7, vel6), ...]`` into numpy arrays.

    Returns:
        ``times``: shape ``(N,)`` float64.
        ``poses``: shape ``(N, 7)`` float64.
        ``velocities``: shape ``(N, 6)`` float64.

    Raises:
        ValueError: if any trajectory entry is the legacy 2-tuple
            ``(t, pose7)`` shape — callers must migrate to 3-tuples.
            The daemon emits 3-tuples; the recording's
            ``read_pose_velocity_trajectory`` does too.
    """
    # Materialize once: callers pass either lists (idempotent) or — in
    # principle — generators. Walking a generator twice (validation pass
    # plus the comprehensions below) would silently produce empty arrays.
    entries = list(trajectory)
    if not entries:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, _POSE7_LEN), dtype=np.float64),
            np.zeros((0, _VEL6_LEN), dtype=np.float64),
        )
    # Validate every entry's arity, not just the first — a stale 2-tuple
    # mid-trajectory would otherwise raise a cryptic numpy unpack error.
    for i, entry in enumerate(entries):
        if len(entry) != 3:
            raise ValueError(
                f"trajectory[{i}] must be a (t, pose7, vel6) 3-tuple; got "
                f"length-{len(entry)} entry. The daemon now emits velocity "
                "alongside pose; finite-differencing is no longer supported."
            )
    times = np.asarray([float(t) for t, _, _ in entries], dtype=np.float64)
    poses = np.asarray([list(pose) for _, pose, _ in entries], dtype=np.float64)
    velocities = np.asarray([list(vel) for _, _, vel in entries], dtype=np.float64)
    if poses.shape[1] != _POSE7_LEN:
        raise ValueError(
            f"trajectory poses must be length {_POSE7_LEN}, got {poses.shape[1]}"
        )
    if velocities.shape[1] != _VEL6_LEN:
        raise ValueError(
            f"trajectory velocities must be length {_VEL6_LEN}, got "
            f"{velocities.shape[1]}"
        )
    return times, poses, velocities


def settle_distance(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
    *,
    rest_position: Sequence[float],
) -> float:
    """Euclidean distance from final position to the configured rest position.

    Lower is better — drop_settle's primary optimizer objective.

    ``rest_position`` is the parent-side computed expected resting point
    (ground_y + body bbox_half_height for drop_settle). When the body
    settles cleanly with no slide / off-axis bounce the value approaches
    zero.
    """
    _, poses, _ = _trajectory_to_arrays(trajectory)
    if poses.size == 0:
        return float("inf")
    final = poses[-1, 0:3]
    rest = np.asarray(list(rest_position)[:3], dtype=np.float64)
    return float(np.linalg.norm(final - rest))


def max_linear_speed(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
) -> float:
    """Peak ``|linear velocity|`` (m/s) across the trajectory.

    Reads the simulator-supplied velocity values directly from each
    sample's ``vel6[0:3]`` — no finite-differencing of positions. This
    matches the velocity that ``recording.usda`` persists via
    ``physics:velocity`` time samples, so calling
    ``max_linear_speed(read_pose_velocity_trajectory(stage, path))``
    produces the same number as calling it on the daemon's in-flight
    dict.
    """
    _, _, velocities = _trajectory_to_arrays(trajectory)
    if velocities.shape[0] == 0:
        return 0.0
    speeds = np.linalg.norm(velocities[:, 0:3], axis=1)
    return float(speeds.max())


def max_angular_speed(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
) -> float:
    """Peak ``|angular velocity|`` (rad/s) across the trajectory.

    Reads the simulator-supplied angular velocity from each sample's
    ``vel6[3:6]`` — the actual physical value, not a quaternion-diff
    proxy. Companion to :func:`max_linear_speed`.
    """
    _, _, velocities = _trajectory_to_arrays(trajectory)
    if velocities.shape[0] == 0:
        return 0.0
    speeds = np.linalg.norm(velocities[:, 3:6], axis=1)
    return float(speeds.max())


def settle_time(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
    *,
    threshold_m_s: float = 0.05,
    sustain_window_s: float = 0.2,
) -> float | None:
    """First time at which ``|linear velocity|`` (from the simulator)
    drops below ``threshold_m_s`` and stays there for at least
    ``sustain_window_s`` seconds.

    Returns ``None`` when the body never settles within the trajectory.

    The "stays for X seconds" check is on **elapsed time** between the
    sample stamps, not on a sample count. At 10 Hz a previous
    ``round(sustain_window_s / dt_avg)`` count would accept two
    below-threshold samples (0.1s elapsed) for a 0.2s window — biasing
    the freeform "settled" component toward early settles. We now scan
    contiguous below-threshold runs and only return when
    ``times[i] - times[start] >= sustain_window_s`` so the actual time
    span matches the contract.
    """
    times, _, velocities = _trajectory_to_arrays(trajectory)
    if velocities.shape[0] < 2:
        return None
    speeds = np.linalg.norm(velocities[:, 0:3], axis=1)
    below = speeds < float(threshold_m_s)
    sustain_window = float(sustain_window_s)
    start: int | None = None
    for i, is_below in enumerate(below):
        if not bool(is_below):
            start = None
            continue
        if start is None:
            start = int(i)
        # Elapsed wall-clock time between the run's anchor and the
        # current sample. Once it crosses the window threshold the body
        # has been below the speed threshold for at least
        # ``sustain_window_s`` seconds; report the run's anchor time.
        if float(times[i] - times[start]) >= sustain_window:
            return float(times[start])
    return None


def fell_over(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
    *,
    initial_up_local: Sequence[float] = (0.0, 1.0, 0.0),
    world_up: Sequence[float] = (0.0, 1.0, 0.0),
    max_tilt_deg: float = 80.0,
) -> bool:
    """True if the body's local up-axis ever tilts more than
    ``max_tilt_deg`` away from the world up-axis during the trajectory.

    Useful as a "did the top fall over" check for freeform scenarios.
    Body's local up at time t = quaternion-rotated ``initial_up_local``.
    """
    _, poses, _ = _trajectory_to_arrays(trajectory)
    if poses.shape[0] == 0:
        return False
    initial_up = np.asarray(initial_up_local, dtype=np.float64)
    initial_up = initial_up / max(float(np.linalg.norm(initial_up)), 1e-12)
    world_up_arr = np.asarray(world_up, dtype=np.float64)
    world_up_arr = world_up_arr / max(float(np.linalg.norm(world_up_arr)), 1e-12)
    cos_threshold = float(np.cos(np.radians(max_tilt_deg)))

    for pose in poses:
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
        # Rotate ``initial_up`` by the quaternion.
        # v' = q v q^-1; for a unit quat with v=(x,y,z) treated as a pure
        # imaginary quat, the body-frame->world rotation gives:
        # v' = v + 2 q_xyz x (q_xyz x v + qw v)
        # Use the explicit matrix form for clarity.
        rx = 1 - 2 * (qy * qy + qz * qz)
        ry = 2 * (qx * qy - qz * qw)
        rz = 2 * (qx * qz + qy * qw)
        ux = 2 * (qx * qy + qz * qw)
        uy = 1 - 2 * (qx * qx + qz * qz)
        uz = 2 * (qy * qz - qx * qw)
        fx = 2 * (qx * qz - qy * qw)
        fy = 2 * (qy * qz + qx * qw)
        fz = 1 - 2 * (qx * qx + qy * qy)
        rot = np.array([[rx, ry, rz], [ux, uy, uz], [fx, fy, fz]], dtype=np.float64)
        body_up_world = rot.T @ initial_up  # column vectors → transpose
        cos_angle = float(np.dot(body_up_world, world_up_arr))
        if cos_angle < cos_threshold:
            return True
    return False


def infer_world_up(reference_position: Sequence[float]) -> tuple[float, float, float]:
    """Pick a unit world-up vector from a reference position whose
    non-zero component identifies the stage up-axis.

    drop_settle's scenario builder writes ``rest_position`` with only
    the up-axis component nonzero (the body's expected resting height).
    This helper turns that into a normalized up-vector consumers can
    feed to :func:`fell_over` / :func:`trajectory_summary` so derived
    metrics aren't checked against the wrong axis on a Z-up stage.

    Falls back to Y-up ``(0, 1, 0)`` when the reference position is
    all-zero (no axis carries signal — degenerate inputs default to
    the legacy convention).
    """
    if not reference_position:
        return (0.0, 1.0, 0.0)
    abs_components = [abs(float(v)) for v in reference_position[:3]]
    if not abs_components or max(abs_components) <= 0.0:
        return (0.0, 1.0, 0.0)
    idx = max(range(len(abs_components)), key=abs_components.__getitem__)
    out = [0.0, 0.0, 0.0]
    out[idx] = 1.0
    return (out[0], out[1], out[2])


def trajectory_summary(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
    *,
    world_up: Sequence[float] | None = None,
) -> dict[str, float | bool | int | list[float] | None]:
    """Single-pass set of cheap metrics — used by freeform's hybrid score
    and also by the judge's text prompt for context.

    Args:
        trajectory: Daemon-shaped ``[(t, pose7, vel6), ...]`` 3-tuples.
        world_up: Optional unit world-up vector forwarded to
            :func:`fell_over` (and used as ``initial_up_local`` so an
            object that starts upright is correctly identified).
            Defaults to ``None`` → legacy Y-up behaviour
            (``(0, 1, 0)``), preserving byte-identical output for
            existing callers. Z-up callers (drop_settle, freeform) must
            pass ``(0, 0, 1)`` (or an inferred vector via
            :func:`infer_world_up`) so ``fell_over`` is checked against
            the actual stage up-axis.
    """
    times, poses, _ = _trajectory_to_arrays(trajectory)
    if world_up is None:
        fell = fell_over(trajectory)
    else:
        up = tuple(float(v) for v in world_up[:3])
        fell = fell_over(trajectory, initial_up_local=up, world_up=up)
    summary: dict[str, float | bool | int | list[float] | None] = {
        "n_samples": int(poses.shape[0]),
        "duration_s": float(times[-1]) if times.size else 0.0,
        "final_position": (
            [float(v) for v in poses[-1, 0:3]] if poses.size else [0.0, 0.0, 0.0]
        ),
        "max_linear_speed": max_linear_speed(trajectory),
        "max_angular_speed": max_angular_speed(trajectory),
        "settle_time_s": settle_time(trajectory),
        "fell_over": fell,
    }
    return summary


__all__ = [
    "settle_distance",
    "max_linear_speed",
    "max_angular_speed",
    "settle_time",
    "fell_over",
    "trajectory_summary",
    "infer_world_up",
]
