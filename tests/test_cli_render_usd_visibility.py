# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from pxr import Usd, UsdGeom
from typer.testing import CliRunner

from world_understanding.cli import (
    _apply_camera_clip_overrides,
    _hide_render_paths,
    app,
)


def test_render_usd_parses_hide_paths(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_render_remote(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("world_understanding.cli._render_remote", fake_render_remote)

    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--output",
            str(tmp_path / "render.png"),
            "--hide",
            "/World/Wall, /World/Glass",
            "--near-clip",
            "400",
            "--far-clip",
            "1300",
        ],
    )

    assert result.exit_code == 0
    assert captured["hide_paths"] == ["/World/Wall", "/World/Glass"]
    assert captured["near_clip"] == 400.0
    assert captured["far_clip"] == 1300.0


def test_hide_render_paths_hides_imageable_subtree() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/HiddenGroup")
    child_mesh = UsdGeom.Mesh.Define(stage, "/World/HiddenGroup/Mesh")
    visible_mesh = UsdGeom.Mesh.Define(stage, "/World/VisibleMesh")

    hidden_count, missing = _hide_render_paths(
        stage,
        ["/World/HiddenGroup", "/World/Missing"],
    )

    assert hidden_count == 1
    assert missing == ["/World/Missing"]
    assert (
        UsdGeom.Imageable(child_mesh.GetPrim()).ComputeVisibility()
        == UsdGeom.Tokens.invisible
    )
    assert (
        UsdGeom.Imageable(visible_mesh.GetPrim()).ComputeVisibility()
        == UsdGeom.Tokens.inherited
    )


def test_apply_camera_clip_overrides_updates_existing_camera() -> None:
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateClippingRangeAttr().Set((1.0, 1000.0))

    updated = _apply_camera_clip_overrides(
        stage,
        ["/World/Camera"],
        near_clip=25.0,
        far_clip=250.0,
    )

    assert updated == 1
    assert tuple(camera.GetClippingRangeAttr().Get()) == (25.0, 250.0)
