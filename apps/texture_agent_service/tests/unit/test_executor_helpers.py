# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from ...service.workers.executor import (
    _MAX_ERROR_MESSAGE_CHARS,
    _MAX_ERRORS_IN_PAYLOAD,
    _MAX_RENDER_STATS_ITEMS,
    _artifact_manifest_status,
    _extract_final_stats,
    _extract_step_stats,
    _package_usdz,
    _prepare_config_and_context,
    _task_to_step_name,
    _truncate_errors,
    _write_service_artifact_manifest,
)


class PrepareUVsTask:
    pass


class _UnknownTask:
    pass


def test_task_to_step_name_maps_known_and_unknown_classes() -> None:
    assert _task_to_step_name(PrepareUVsTask()) == "prepare_uvs"
    assert _task_to_step_name(_UnknownTask()) == "_UnknownTask"


def test_prepare_config_and_context_applies_defaults_and_creates_dirs(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    config, context = _prepare_config_and_context(
        {"input": {"usd_path": "/tmp/input.usd"}},
        session_dir,
    )

    working_dir = session_dir / "cache"
    assert config["project"]["working_dir"] == str(working_dir)
    assert context["working_dir"] == str(working_dir)
    assert context["usd_path"] == "/tmp/input.usd"
    assert (working_dir / "prepared").is_dir()
    assert (working_dir / "renders").is_dir()
    assert context["render_preview_config"]["image_width"] == 512
    assert context["render_config"]["image_width"] == 1024


def test_prepare_config_and_context_applies_runtime_endpoint_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "session"
    monkeypatch.setenv("TA_IMAGE_GEN_BACKEND", "openai")
    monkeypatch.setenv("TA_IMAGE_GEN_BASE_URL", "http://image-gen-nim:8000/v1")
    monkeypatch.setenv("TA_IMAGE_GEN_MODEL", "black-forest-labs/flux.2-klein-4b")
    monkeypatch.setenv("TA_IMAGE_GEN_API_KEY", "not-used")

    config, context = _prepare_config_and_context(
        {"input": {"usd_path": "/tmp/input.usd"}},
        session_dir,
    )

    assert config["texture"]["image_gen"] == {
        "backend": "openai",
        "base_url": "http://image-gen-nim:8000/v1",
        "model": "black-forest-labs/flux.2-klein-4b",
        "api_key": "not-used",
    }
    assert context["texture_config"]["image_gen"]["api_key"] == "not-used"


def test_extract_step_stats_and_final_stats_fall_back_to_files(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "cache" / "textures").mkdir(parents=True)
    (session_dir / "cache" / "output").mkdir(parents=True)
    (session_dir / "cache" / "renders").mkdir(parents=True)
    (session_dir / "cache" / "textures" / "one.png").write_text("x", encoding="utf-8")
    (session_dir / "cache" / "textures" / "two.png").write_text("x", encoding="utf-8")
    (session_dir / "cache" / "output" / "a.usd").write_text(
        "#usda 1.0\n", encoding="utf-8"
    )
    (session_dir / "cache" / "renders" / "final.png").write_text(
        "png", encoding="utf-8"
    )

    assert _extract_step_stats(
        "discover_materials", {"discovered_materials": [1, 2]}
    ) == {"materials_found": 2}
    # generate_textures stats include the failed-count counter (0 when
    # the step succeeded fully); the structured "errors" key is omitted
    # for the empty case so happy-path payloads stay compact.
    assert _extract_step_stats(
        "generate_textures", {"generated_textures": {"a": 1}}
    ) == {"textures_generated": 1, "textures_failed": 0}

    stats = _extract_final_stats({}, session_dir)

    assert stats == {
        "materials_found": 0,
        "textures_generated": 2,
        "output_usd_count": 1,
        "renders_count": 1,
    }


def test_extract_render_stats_surfaces_bounded_diagnostics(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)
    camera_paths = [f"/Camera_{i}" for i in range(_MAX_RENDER_STATS_ITEMS + 2)]
    focus_cameras = [
        {"camera_path": path, "prim_path": f"/Root/Mesh_{i}"}
        for i, path in enumerate(camera_paths)
    ]
    render_error = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "RENDER_EMPTY_RESULT",
        "severity": "error",
        "stage": "render",
        "message": "Renderer returned no images",
    }
    render_warning = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "RENDER_NO_CAMERA",
        "severity": "warning",
        "stage": "render",
        "message": "Added fallback camera",
    }
    context = {
        "rendered_image_paths": [],
        "render_stats": {
            "render_available": False,
            "camera_paths": camera_paths,
            "focus_cameras": focus_cameras,
        },
        "render_diagnostics": [render_warning, render_error],
        "render_errors": [render_error],
    }

    step_stats = _extract_step_stats("render", context)
    final_stats = _extract_final_stats(context, session_dir)

    assert step_stats["renders_count"] == 0
    assert step_stats["render_available"] is False
    assert step_stats["camera_paths"] == camera_paths[:_MAX_RENDER_STATS_ITEMS]
    assert step_stats["focus_cameras"] == focus_cameras[:_MAX_RENDER_STATS_ITEMS]
    assert step_stats["diagnostics"] == [render_warning, render_error]
    assert step_stats["errors"] == [render_error]

    assert final_stats["render_available"] is False
    assert final_stats["render_camera_paths"] == camera_paths[:_MAX_RENDER_STATS_ITEMS]
    assert (
        final_stats["render_focus_cameras"] == focus_cameras[:_MAX_RENDER_STATS_ITEMS]
    )
    assert final_stats["diagnostics"]["render"] == [render_warning, render_error]
    assert final_stats["errors"]["render"] == [render_error]


def test_extract_render_stats_surfaces_warning_only_diagnostics(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)
    render_warning = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "RENDER_FRAME_TOO_WIDE",
        "severity": "warning",
        "stage": "render",
        "message": "Focused render framing heuristic is below threshold",
    }
    context = {
        "rendered_image_paths": [],
        "render_stats": {"render_available": False},
        "render_diagnostics": [render_warning],
        "render_errors": [],
    }

    step_stats = _extract_step_stats("render", context)
    final_stats = _extract_final_stats(context, session_dir)

    assert "errors" not in step_stats
    assert step_stats["diagnostics"] == [render_warning]
    assert "errors" not in final_stats
    assert final_stats["diagnostics"]["render"] == [render_warning]


def test_extract_step_stats_apply_textures_surfaces_mdl_overrides() -> None:
    """Per OMPE-91783: the apply_textures step must propagate MDL override
    counts into step stats and surface a `warnings` entry when SimReady-style
    pre-baked texture inputs had to be cleared, so /status and /results no
    longer silently succeed."""
    context = {
        "output_usd_paths": ["/x/output/textured_output.usd"],
        "apply_textures_stats": {
            "applied_count": 8,
            "mdl_inputs_overridden": 2,
            "mdl_inputs_cleared": [
                "/Mat/Plastic_Blue_A:opacity_texture",
                "/Mat/Plastic_Blue_A:emissive_color_texture",
            ],
        },
    }
    stats = _extract_step_stats("apply_textures", context)

    assert stats["output_usd_count"] == 1
    assert stats["mdl_inputs_overridden"] == 2
    assert stats["mdl_inputs_cleared"] == [
        "/Mat/Plastic_Blue_A:opacity_texture",
        "/Mat/Plastic_Blue_A:emissive_color_texture",
    ]
    assert len(stats["warnings"]) == 1
    warning = stats["warnings"][0]
    assert "opacity_texture" in warning
    assert "emissive_color_texture" in warning


def test_extract_step_stats_apply_textures_no_mdl_inputs_no_warnings() -> None:
    """Materials without pre-baked MDL inputs (the common OpenPBR-only case)
    must not emit a warnings entry — that field is reserved for actual
    pipeline anomalies."""
    context = {
        "output_usd_paths": ["/x/output/textured_output.usd"],
        "apply_textures_stats": {
            "applied_count": 3,
            "mdl_inputs_overridden": 0,
            "mdl_inputs_cleared": [],
            "mdl_inputs_localized": [],
        },
    }
    stats = _extract_step_stats("apply_textures", context)

    assert stats["output_usd_count"] == 1
    assert stats["mdl_inputs_overridden"] == 0
    assert "mdl_inputs_cleared" not in stats
    assert "mdl_inputs_localized" not in stats
    assert "warnings" not in stats


def test_extract_step_stats_apply_textures_localized_inputs_no_warning() -> None:
    """Localized MDL inputs (local files copied into the bundle textures dir)
    are reported as a count + list but must NOT trigger a warning — the bundle
    is self-consistent in that case."""
    context = {
        "output_usd_paths": ["/x/output/textured_output.usd"],
        "apply_textures_stats": {
            "applied_count": 4,
            "mdl_inputs_overridden": 1,
            "mdl_inputs_cleared": [],
            "mdl_inputs_localized": ["/Mat/Plastic:opacity_texture"],
        },
    }
    stats = _extract_step_stats("apply_textures", context)

    assert stats["mdl_inputs_localized"] == ["/Mat/Plastic:opacity_texture"]
    assert "mdl_inputs_cleared" not in stats
    assert "warnings" not in stats


def test_extract_final_stats_persists_apply_textures_warnings() -> None:
    """Per OMPE-91783 round-3 review: warnings emitted during apply_textures
    must survive into the final /results payload, not just the per-step
    stream. Otherwise clients polling /results after completion see a clean
    success and miss that MDL inputs were blanked."""
    session_dir = Path("/nonexistent")
    context = {
        "discovered_materials": [],
        "generated_textures": {},
        "output_usd_paths": ["/x/output/textured_output.usd"],
        "rendered_image_paths": [],
        "apply_textures_stats": {
            "applied_count": 1,
            "mdl_inputs_overridden": 2,
            "mdl_inputs_cleared": ["/Mat/X:opacity_texture"],
            "mdl_inputs_localized": ["/Mat/X:emissive_color_texture"],
        },
    }

    stats = _extract_final_stats(context, session_dir)

    assert stats["mdl_inputs_overridden"] == 2
    assert stats["mdl_inputs_cleared"] == ["/Mat/X:opacity_texture"]
    assert stats["mdl_inputs_localized"] == ["/Mat/X:emissive_color_texture"]
    assert len(stats["warnings"]) == 1
    assert "opacity_texture" in stats["warnings"][0]


def test_package_usdz_failure_surfaces_warning_in_final_stats(
    tmp_path: Path, monkeypatch
) -> None:
    """A USDZ packaging miss must be visible in /results, not logs only."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Usd, UsdUtils

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.DefinePrim("/Root", "Xform")
    stage.GetRootLayer().Save()

    monkeypatch.setattr(
        UsdUtils,
        "CreateNewUsdzPackage",
        lambda _src, _dst: False,
    )

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    stats = _extract_final_stats(context, tmp_path)

    assert usdz is None
    assert context["usdz_packaging_failed"] is True
    assert _artifact_manifest_status(context) == "partial"
    assert stats["package_status"] == "failed"
    assert stats["usdz_packaging_failed"] is True
    assert "Failed to create USDZ package" in stats["warnings"][0]
    assert "self-contained USDZ artifact was not produced" in stats["warnings"][0]
    assert stats["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"


def test_package_usdz_blocks_missing_relative_texture_refs(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    mat = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    mat.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("../textures/missing.png"))
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    stats = _extract_final_stats(context, tmp_path)

    assert usdz is None
    assert context["output_portability"]["portable"] is False
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"
    assert _artifact_manifest_status(context) == "partial"
    assert stats["package_status"] == "failed"
    assert stats["usdz_packaging_failed"] is True
    assert stats["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"


def test_write_service_artifact_manifest_sanitizes_and_updates_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ...service.workers import executor

    monkeypatch.setattr(
        executor.service_config,
        "session_storage_path",
        str(tmp_path / "session"),
    )
    cache = tmp_path / "session" / "cache"
    cache.mkdir(parents=True)
    context = {
        "working_dir": str(cache),
        "usd_path": str(tmp_path / "session" / "input" / "scene.usd"),
        "texture_config": {
            "backend": "service",
            "endpoint": "https://abc.invocation.api.nvcf.nvidia.com/v1",
            "custom_parameters": {"api_key": "SHOULD_NOT_SURFACE"},
        },
        "package_diagnostics": [
            {
                "schema_version": "texture-agent-diagnostic.v1",
                "code": "PACKAGE_MISSING_ARTIFACT",
                "severity": "error",
                "stage": "package",
                "message": f"missing {tmp_path / 'session' / 'cache' / 'textures' / 'x.png'}",
                "recommended_action": "inspect",
                "details": {},
            }
        ],
    }

    manifest = _write_service_artifact_manifest(
        context,
        status="failed",
        service_urls={"manifest": "/artifacts/sid/manifest"},
    )

    assert manifest == context["artifacts_manifest_path"]
    payload = Path(manifest).read_text(encoding="utf-8")
    assert "texture-agent-artifacts.v1" in payload
    assert "SHOULD_NOT_SURFACE" not in payload
    assert "<session>" in payload


def test_package_usdz_rewrites_string_and_token_png_paths(tmp_path: Path) -> None:
    """Codex round-8 finding: the packager only rewrote `Sdf.AssetPath`
    PNG attributes, leaving absolute cache paths in string/token-typed
    MDL texture inputs after download. Now string and token attributes
    are also rewritten to bundle-relative `../textures/<basename>` form.
    """
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    from PIL import Image

    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_albedo.png")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_normal.png")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_orm.png")

    # We use UsdPreviewSurface (not MDL) so USDZ packaging does not chase
    # an unresolvable `omniverse://...mdl` dep — the test focuses on the
    # path-rewriting behaviour, not the MDL resolution path.
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    # Pre-rewrite shapes the packager must handle: absolute Asset, absolute
    # String, absolute Token — all PNG paths under the cache textures dir.
    shader.CreateInput("diffuseColor_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(textures_dir / "Plastic_albedo.png"))
    )
    shader.CreateInput("normal_texture", Sdf.ValueTypeNames.String).Set(
        str(textures_dir / "Plastic_normal.png")
    )
    shader.CreateInput("orm_texture", Sdf.ValueTypeNames.Token).Set(
        str(textures_dir / "Plastic_orm.png")
    )
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    assert usdz is not None
    assert Path(usdz).exists()

    # Re-read the rewritten USD and confirm all three inputs were rewritten
    # to ../textures/<basename> (bundle-relative), regardless of authored
    # type.
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert (
        out_shader.GetInput("diffuseColor_texture").Get().path
        == "../textures/Plastic_albedo.png"
    )
    assert (
        out_shader.GetInput("normal_texture").Get() == "../textures/Plastic_normal.png"
    )
    assert out_shader.GetInput("orm_texture").Get() == "../textures/Plastic_orm.png"


def test_package_usdz_does_not_rewrite_unrelated_string_attrs(tmp_path: Path) -> None:
    """Codex round-11 finding: the string/token rewrite must be scoped to
    Shader `inputs:*_texture` attributes. A non-shader string attribute, or
    a shader string attribute with a different name, that happens to end in
    ``.png`` must NOT be rewritten — those have no Asset-typed dep, so a
    rewrite would create a dangling USDZ ref.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))

    # Non-Shader prim, string attribute that happens to end in .png.
    meta_prim = stage.DefinePrim("/Root/Metadata", "Scope")
    meta_prim.CreateAttribute("note", Sdf.ValueTypeNames.String).Set(
        "see /assets/library/reference.png for the source"
    )

    # Shader prim with a string input named other than `inputs:*_texture`.
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("debug_label", Sdf.ValueTypeNames.String).Set("fallback.png")
    # Shader `inputs:*_texture` string — this one *is* in scope and must
    # be rewritten.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        str(textures_dir / "Plastic_albedo.png")
    )

    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    assert usdz is not None
    rewritten_stage = Usd.Stage.Open(str(output_usd))

    note = rewritten_stage.GetPrimAtPath("/Root/Metadata").GetAttribute("note").Get()
    assert note == "see /assets/library/reference.png for the source"

    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("debug_label").Get() == "fallback.png"
    assert (
        out_shader.GetInput("diffuse_texture").Get() == "../textures/Plastic_albedo.png"
    )


def test_package_usdz_skips_string_inputs_with_missing_files(tmp_path: Path) -> None:
    """Codex round-13 finding: even after the round-12 scope narrowing
    (Shader + `inputs:*_texture`), the packager could rewrite a string
    texture input on a non-MDL shader (or any shader skipped by
    apply_textures) to a `../textures/<basename>.png` path that the
    bundle does not actually ship. Now the packager additionally
    requires the basename to exist in `cache/textures/` before
    rewriting, so unrelated/skipped string texture refs are left as
    authored. A31-1 then blocks USDZ packaging with a structured
    PACKAGE_* diagnostic because the output is not self-contained.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    # Only the in-bundle texture exists.
    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    # In-bundle reference: must be rewritten.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        str(textures_dir / "Plastic_albedo.png")
    )
    # Out-of-scope shader string `inputs:*_texture` whose target does NOT
    # live in cache/textures: the packager must NOT rewrite this, since
    # USDZ packaging would not bundle the file and the relative rewrite
    # would dangle on the customer's machine. A31-1 should surface that
    # as a package diagnostic instead of shipping a bad archive.
    shader.CreateInput("mask_texture", Sdf.ValueTypeNames.String).Set(
        "omniverse://nucleus.example/mask.png"
    )
    shader.CreateInput("emissive_texture", Sdf.ValueTypeNames.String).Set(
        "/private/path/that_does_not_exist.png"
    )
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    assert usdz is None
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_ABSOLUTE_TEXTURE_PATH"
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # In-bundle: rewritten.
    assert (
        out_shader.GetInput("diffuse_texture").Get() == "../textures/Plastic_albedo.png"
    )
    # Out-of-bundle: untouched.
    assert (
        out_shader.GetInput("mask_texture").Get()
        == "omniverse://nucleus.example/mask.png"
    )
    assert (
        out_shader.GetInput("emissive_texture").Get()
        == "/private/path/that_does_not_exist.png"
    )


def test_package_usdz_does_not_substitute_basename_collision(tmp_path: Path) -> None:
    """Codex round-15 finding: rewriting a string-typed shader input by
    basename match alone could silently substitute the wrong texture if
    the user has another local PNG whose basename happens to collide
    with a generated/localized file. The packager now resolves the
    *original* path and only rewrites when it lives under the session's
    own ``cache/textures`` directory.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    # The agent's generated file.
    Image.new("RGB", (4, 4), (200, 50, 50)).save(textures_dir / "Plastic_albedo.png")

    # An unrelated PNG that happens to share the basename, parked in a
    # totally separate directory.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    Image.new("RGB", (4, 4), (10, 200, 10)).save(elsewhere / "Plastic_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    # The user's intentional reference to elsewhere/Plastic_albedo.png.
    # Even though `cache/textures/Plastic_albedo.png` exists, the
    # packager must NOT substitute this string with `../textures/...`.
    # A31-1 should block USDZ packaging rather than ship an archive with
    # a host-local absolute path.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        str(elsewhere / "Plastic_albedo.png")
    )
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    assert usdz is None
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_ABSOLUTE_TEXTURE_PATH"
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # Untouched — the original elsewhere/ path survived.
    assert out_shader.GetInput("diffuse_texture").Get() == str(
        elsewhere / "Plastic_albedo.png"
    )


def test_package_usdz_does_not_rewrite_out_of_bundle_asset_paths(
    tmp_path: Path,
) -> None:
    """Asset-typed refs must pass the same containment gate as string refs.

    Otherwise an absolute host-local AssetPath could be rewritten by basename
    to an in-bundle texture and pass portability validation with the wrong file.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), (200, 50, 50)).save(textures_dir / "Plastic_albedo.png")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    Image.new("RGB", (4, 4), (10, 200, 10)).save(elsewhere / "Plastic_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    mat = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    mat.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(str(elsewhere / "Plastic_albedo.png")))
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)

    assert usdz is None
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_ABSOLUTE_TEXTURE_PATH"
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_mat = UsdShade.Material(rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic"))
    out_ref = out_mat.GetPrim().GetAttribute("inputs:base_color_texture_file").Get()
    assert out_ref.path == str(elsewhere / "Plastic_albedo.png")


def test_extract_final_stats_no_apply_textures_stats_no_warning(
    tmp_path: Path,
) -> None:
    """Sessions that ran without apply_textures (or where the step recorded no
    MDL anomalies) must not emit a warnings entry into /results."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    stats = _extract_final_stats({"output_usd_paths": ["/x.usd"]}, session_dir)

    assert "warnings" not in stats
    assert "mdl_inputs_overridden" not in stats
    assert "mdl_inputs_cleared" not in stats
    assert "mdl_inputs_localized" not in stats


def test_extract_final_stats_surfaces_partial_generate_failures(
    tmp_path: Path,
) -> None:
    """A run that completed below the threshold (e.g. 1 success + 3
    failures with default 1.0) must still expose the structured failures
    on the persisted final stats. Otherwise GET /result/{session_id} after
    the SSE snapshot has been GC'd looks identical to a clean run."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    stats = _extract_final_stats(
        {
            "generated_textures": {"Good": object()},
            "generate_textures_failed_count": 3,
            "generate_textures_errors": [
                {
                    "material": "BadA",
                    "type": "RuntimeError",
                    "status": 500,
                    "message": "x",
                },
            ],
        },
        session_dir,
    )

    assert stats["textures_generated"] == 1
    assert stats["textures_generated_failed"] == 3
    assert stats["textures_failed"] == 3
    assert "generate_textures" in stats["errors"]
    assert stats["errors"]["generate_textures"][0]["status"] == 500


def test_extract_final_stats_sums_gen_and_blend_failure_counts(
    tmp_path: Path,
) -> None:
    """When both gen and blend partial-fail (different units), the
    top-level ``textures_failed`` must be the SUM, not just blend's.
    Otherwise an upstream auth issue (gen 403s) is hidden the moment
    blend introduces any of its own failures, defeating the purpose of
    the field."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    stats = _extract_final_stats(
        {
            "generated_textures": {"Good1": object()},
            "generate_textures_failed_count": 2,
            "generate_textures_errors": [
                {
                    "material": "GenBadA",
                    "type": "RuntimeError",
                    "status": 403,
                    "message": "auth",
                },
                {
                    "material": "GenBadB",
                    "type": "RuntimeError",
                    "status": 403,
                    "message": "auth",
                },
            ],
            "blend_textures_failed_count": 1,
            "blend_textures_errors": [
                {
                    "material": "BlendBadA",
                    "type": "MissingAlbedo",
                    "status": None,
                    "message": "x",
                },
            ],
        },
        session_dir,
    )

    assert stats["textures_generated_failed"] == 2
    assert stats["textures_blended_failed"] == 1
    assert stats["textures_failed"] == 3
    assert set(stats["errors"]) == {"generate_textures", "blend_textures"}


def test_extract_final_stats_omits_failure_keys_when_no_errors(
    tmp_path: Path,
) -> None:
    """Happy-path runs must not gain new top-level keys -- existing
    consumers should see the same shape they always have."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    stats = _extract_final_stats(
        {
            "generated_textures": {"Good": object()},
            "generate_textures_failed_count": 0,
        },
        session_dir,
    )

    assert "textures_failed" not in stats
    assert "errors" not in stats


def test_truncate_errors_caps_list_length() -> None:
    """Per-prim mode with backend-wide outage can produce thousands of
    error records. Persisted payloads (session.json, event_log.jsonl,
    SSE) must cap them while leaving the count visible elsewhere."""
    errors = [
        {"material": f"m{i}", "type": "T", "status": 500, "message": "x"}
        for i in range(_MAX_ERRORS_IN_PAYLOAD * 4)
    ]
    out = _truncate_errors(errors)
    assert len(out) == _MAX_ERRORS_IN_PAYLOAD


def test_truncate_errors_truncates_long_messages() -> None:
    long_msg = "X" * (_MAX_ERROR_MESSAGE_CHARS * 5)
    errors = [{"material": "m", "type": "T", "status": 500, "message": long_msg}]
    out = _truncate_errors(errors)
    assert out[0]["message"].endswith("...(truncated)")
    assert len(out[0]["message"]) <= _MAX_ERROR_MESSAGE_CHARS + len("...(truncated)")


def test_truncate_errors_preserves_short_messages_unchanged() -> None:
    record = {"material": "m", "type": "HTTPError", "status": 403, "message": "x"}
    out = _truncate_errors([record])
    assert out == [record]


def test_extract_final_stats_truncates_oversized_error_lists(tmp_path: Path) -> None:
    """A 1000-prim per-prim run with an all-fail backend must NOT
    persist 1000 error records to /result. The count survives via
    ``textures_generated_failed`` / ``textures_failed``."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    errors = [
        {"material": f"m{i}", "type": "T", "status": 500, "message": "x"}
        for i in range(1000)
    ]
    stats = _extract_final_stats(
        {
            "generated_textures": {},
            "generate_textures_failed_count": 1000,
            "generate_textures_errors": errors,
        },
        session_dir,
    )

    assert stats["textures_generated_failed"] == 1000
    assert stats["textures_failed"] == 1000
    assert len(stats["errors"]["generate_textures"]) == _MAX_ERRORS_IN_PAYLOAD
