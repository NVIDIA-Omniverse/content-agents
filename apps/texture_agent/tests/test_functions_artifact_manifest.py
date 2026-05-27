# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("pxr")


def _write_textured_stage(output_usd: Path, texture_ref: str) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateNew(str(output_usd))
    mat = UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    mat.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(texture_ref))
    stage.GetRootLayer().Save()


def test_artifacts_manifest_schema_contract(tmp_path: Path) -> None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        validate_artifacts_manifest_schema,
        write_artifacts_manifest,
    )

    cache = tmp_path / "session" / "cache"
    prepared_dir = cache / "prepared"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir()
    textures_dir.mkdir()

    input_usd = tmp_path / "session" / "input" / "scene.usda"
    input_usd.parent.mkdir()
    _write_textured_stage(input_usd, "")

    texture = textures_dir / "steel_albedo.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(texture)
    output_usd = output_dir / "textured_output.usda"
    _write_textured_stage(output_usd, "../textures/steel_albedo.png")

    uv_report = prepared_dir / "uv_report.json"
    uv_report.write_text(
        json.dumps(
            {
                "schema_version": "texture-agent-uv-report.v1",
                "prepared_usd": str(input_usd),
                "summary": {"mesh_count": 1, "valid_count": 1},
            }
        ),
        encoding="utf-8",
    )

    context = {
        "working_dir": str(cache),
        "usd_path": str(input_usd),
        "config": {
            "input": {"usd_path": str(input_usd)},
            "project": {"name": "demo", "session_id": "sid"},
            "steps": {"render": {"enabled": False}},
        },
        "uv_preparation": {"uv_report_path": str(uv_report), "generated": 0},
        "material_textures": {"Steel": "brushed steel"},
        "prim_paths": ["/Root/Mesh"],
        "output_usd_paths": [str(output_usd)],
        "texture_config": {
            "backend": "service",
            "endpoint": "https://example.invalid",
            "size": 8,
            "custom_parameters": {"api_key": "SHOULD_NOT_APPEAR"},
        },
        "warnings": [],
        "timings": {"total": 1.2},
    }

    manifest = build_artifacts_manifest(
        context,
        status="completed",
        service_urls={"manifest": "/artifacts/sid/manifest"},
        duration_seconds=3,
    )

    assert validate_artifacts_manifest_schema(manifest) == []
    assert manifest["outputs"]["portability"]["portable"] is True
    assert manifest["prepared"]["uv_summary"] == {"mesh_count": 1, "valid_count": 1}
    assert manifest["backend"]["endpoint"] == "<configured>"
    assert manifest["backend"]["custom_parameters"]["api_key"] == "<redacted>"

    manifest_path = write_artifacts_manifest(context, payload=manifest)
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert validate_artifacts_manifest_schema(reloaded) == []


def test_portability_validation_rejects_relative_paths_outside_bundle(
    tmp_path: Path,
) -> None:
    from texture_agent.functions.artifact_manifest import (
        validate_output_texture_portability,
    )

    cache = tmp_path / "session" / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)
    outside = tmp_path / "session" / "outside"
    outside.mkdir()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(outside / "escape.png")

    output_usd = output_dir / "textured_output.usda"
    _write_textured_stage(output_usd, "../../outside/escape.png")

    portability = validate_output_texture_portability(output_usd)

    assert portability["portable"] is False
    assert portability["non_relative_texture_paths"] == ["../../outside/escape.png"]
    assert portability["diagnostics"][0]["code"] == "PACKAGE_ABSOLUTE_TEXTURE_PATH"
