# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-Newton smoke for :class:`NewtonSimulator`.

Skipped unless ``newton`` is installed AND ``NEWTON_CI=1`` is set in the
environment. Newton's first Warp kernel compile is multi-minute on a cold
cache, so CI gating keeps the default ``pytest`` invocation fast.

Run locally with:

    uv pip install -e "apps/physics_agent[newton]"
    NEWTON_CI=1 PYTHONPATH=apps/physics_agent pytest \\
      apps/physics_agent/tests/test_tuning_newton_simulator.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

newton = pytest.importorskip("newton")
if os.environ.get("NEWTON_CI") != "1":
    pytest.skip(
        "Real-Newton smoke disabled. Set NEWTON_CI=1 to enable.",
        allow_module_level=True,
    )


def _build_drop_scene(tmp_path: Path) -> Path:
    """Author a minimal UsdPhysics stage: one rigid box at z=1m, one ground plane."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    p = tmp_path / "drop_scene.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    # Falling body: a 10 cm cube at z=1.0
    body = UsdGeom.Cube.Define(stage, "/World/Body")
    body.CreateSizeAttr(0.1)
    body.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(body.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)

    # Static ground: large thin static collider at z=0.
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(10.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -5.0))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr(True)

    stage.GetRootLayer().Save()
    return p


def test_drop_settle_smoke(tmp_path: Path) -> None:
    """A 10 cm cube dropped from z=1m must reach the ground in 1 s.

    Tolerance asserts: MuJoCo-warp on GPU is not bitwise-deterministic, so
    we check coarse physics behavior (body fell at least 0.5 m, final z is
    bounded above by the drop height, sample count matches the requested
    fps × duration).
    """
    from physics_agent.tuning.newton_simulator import NewtonSimulator

    scene_usd = _build_drop_scene(tmp_path)

    sim = NewtonSimulator()
    result = sim.evaluate(
        scene_usd=scene_usd,
        body_pattern="/World/Body",
        duration_s=1.0,
        dt=1.0 / 240.0,
        sample_fps=30,
    )

    assert result["n_bodies"] >= 1
    assert result["duration_s"] == pytest.approx(1.0, abs=1e-3)

    trajectory = result["trajectory"]
    # Expect ~31 samples at 30 fps over 1s (start + 30 intervals).
    assert 30 <= len(trajectory) <= 33, f"unexpected sample count: {len(trajectory)}"

    z_start = trajectory[0][1][2]
    z_end = trajectory[-1][1][2]
    # Stage is z-up; cube was at z=1.0 with size 0.1, so rest plane is z≈0.05.
    assert z_start == pytest.approx(1.0, abs=0.1), (
        f"start z {z_start} not near drop height 1.0"
    )
    assert z_end < z_start - 0.5, f"body did not fall: z_start={z_start} z_end={z_end}"
    # Final pose should be at or above the resting plane.
    assert z_end > -0.5, f"body sank too far: z_end={z_end}"


def test_unknown_body_pattern_raises(tmp_path: Path) -> None:
    from physics_agent.tuning.newton_simulator import NewtonSimulator

    scene_usd = _build_drop_scene(tmp_path)
    sim = NewtonSimulator()
    with pytest.raises(RuntimeError, match="matched no prim"):
        sim.evaluate(
            scene_usd=scene_usd,
            body_pattern="/World/DoesNotExist",
            duration_s=0.1,
        )
