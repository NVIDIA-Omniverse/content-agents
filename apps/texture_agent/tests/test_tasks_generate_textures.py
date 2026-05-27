# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GenerateTexturesTask error-propagation behavior.

The task uses a per-unit thread pool; a previous version logged each
per-unit failure but returned an empty result map without raising,
so the pipeline reported "complete" with zero textures generated.
These tests pin the corrected behavior: total failure raises, partial
failure logs a warning and returns whatever did succeed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.functions.texture_generation import (
    GeneratedTextures,
    GenerationResult,
    JobStatus,
)
from texture_agent.tasks.generate_textures import (
    GenerateTexturesTask,
    _cached_texture_set,
)


def _unit(key: str) -> PrimTextureUnit:
    return PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(prim_path=f"/World/Looks/{key}", name=key),
        key=key,
        prompt=f"weathered {key}",
        opacity=0.85,
    )


def _texture_sibling(albedo: str, suffix: str) -> str:
    if not albedo:
        return ""
    if albedo.endswith("_albedo.png"):
        return albedo.removesuffix("_albedo.png") + f"_{suffix}.png"
    return albedo


def _ok_status(albedo: str, key: str) -> JobStatus:
    textures = GeneratedTextures(
        albedo=albedo,
        normal=_texture_sibling(albedo, "normal"),
        orm=_texture_sibling(albedo, "orm"),
    )
    return JobStatus(
        job_id=f"job-{key}",
        status="completed",
        result=GenerationResult(
            variant_asset_uri=f"/tmp/{key}.usd",
            variant_name=key,
            generated_textures=textures,
        ),
    )


def _make_real_texture_set(directory: Path, key: str) -> str:
    """Materialize a tiny PBR set so the post-gen existence checks pass."""
    directory.mkdir(parents=True, exist_ok=True)
    albedo = directory / f"{key}_albedo.png"
    normal = directory / f"{key}_normal.png"
    orm = directory / f"{key}_orm.png"
    for path in (albedo, normal, orm):
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(albedo)


def _fail_status(key: str, message: str) -> JobStatus:
    return JobStatus(
        job_id=f"job-{key}",
        status="failed",
        error_message=message,
    )


@pytest.fixture
def context_factory(tmp_path):
    """Build a minimal context dict + texture_config for the task."""

    def _make(units: list[PrimTextureUnit]) -> dict:
        return {
            "prim_texture_units": units,
            "working_dir": str(tmp_path),
            "usd_path": "/tmp/asset.usd",
            "texture_config": {
                "backend": "simple_image_gen",
                "image_gen": {"backend": "nim"},
                "skip_existing": False,
                "workers": 2,
            },
        }

    return _make


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_passes_endpoint_api_key(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path
):
    """Local OpenAI-compatible image endpoints need an explicit placeholder key."""
    mock_client = mock_client_cls.return_value
    albedo = _make_real_texture_set(tmp_path, "Steel_Carbon")
    mock_client.generate.return_value = _ok_status(albedo, "Steel_Carbon")

    units = [_unit("Steel_Carbon")]
    context = context_factory(units)
    context["texture_config"]["workers"] = 1
    context["texture_config"]["image_gen"] = {
        "backend": "openai",
        "model": "black-forest-labs/flux.2-klein-4b",
        "base_url": "http://localhost:8005/v1",
        "api_key": "not-used",
    }

    GenerateTexturesTask().run(context)

    mock_engine_cls.assert_called_once_with(
        backend="openai",
        model="black-forest-labs/flux.2-klein-4b",
        base_url="http://localhost:8005/v1",
        api_key="not-used",
    )


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_raises_when_every_unit_fails(
    mock_engine_cls, mock_client_cls, context_factory
):
    """All units failing must raise so the pipeline doesn't silently exit 0."""
    mock_client = mock_client_cls.return_value
    mock_client.generate.return_value = _fail_status("any", "HTTP 403 Forbidden")

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        task.run(context_factory(units))


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_continues_on_partial_failure(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path, caplog
):
    """One success + one failure is allowed: returns partial result + warns."""
    import logging

    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        if cfg.variant_name == "Steel_Carbon":
            albedo = _make_real_texture_set(tmp_path, "Steel_Carbon")
            return _ok_status(albedo, "Steel_Carbon")
        return _fail_status(cfg.variant_name, "HTTP 403 Forbidden")

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with caplog.at_level(logging.WARNING):
        result = task.run(context_factory(units))

    generated = result["generated_textures"]
    assert "Steel_Carbon" in generated
    assert "Copper_Polished" not in generated
    assert any("1/2 failures" in rec.message for rec in caplog.records)


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_succeeds_when_all_units_succeed(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path
):
    """All-success path is unchanged."""
    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        albedo = _make_real_texture_set(tmp_path, cfg.variant_name)
        return _ok_status(albedo, cfg.variant_name)

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    result = task.run(context_factory(units))
    generated = result["generated_textures"]
    assert set(generated.keys()) == {"Steel_Carbon", "Copper_Polished"}


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_raises_when_completed_but_albedo_empty(
    mock_engine_cls, mock_client_cls, context_factory
):
    """Schema-drift guard: status='completed' with empty albedo must raise.

    Without this guard a degraded service that returns a parseable
    ``status="completed"`` but empty ``GeneratedTextures(albedo="", ...)``
    would slip past _raise_if_all_failed (because each unit is "successful")
    and silently produce no output.
    """
    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        return _ok_status("", cfg.variant_name)  # empty albedo path

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        task.run(context_factory(units))


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_raises_when_completed_but_albedo_missing_on_disk(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path
):
    """Schema-drift guard: albedo path set but file does not exist."""
    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        # Path looks plausible but no file on disk (mimics failed localization).
        missing_path = str(tmp_path / f"missing_{cfg.variant_name}.png")
        return _ok_status(missing_path, cfg.variant_name)

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        task.run(context_factory(units))


def test_no_units_is_noop(context_factory):
    """Empty unit list short-circuits without invoking the backend."""
    task = GenerateTexturesTask()
    result = task.run(context_factory([]))
    assert result["generated_textures"] == {}


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_rejects_unsupported_scheme_albedo(
    mock_engine_cls, mock_client_cls, context_factory
):
    """Remote URIs (s3://, http://, omni://) are rejected as per-unit failures.

    Although the texture-variation API contract permits storage-agnostic
    URIs, this pipeline's BlendTexturesTask currently only opens local
    file paths -- so trusting an s3:// albedo would resurrect the
    silent-success bug (blend skips with a warning, apply sees nothing,
    CLI prints "Pipeline complete!" with no output). Until downstream
    learns to fetch remote schemes, validate them as failures here.
    """
    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        return _ok_status(f"s3://bucket/{cfg.variant_name}.png", cfg.variant_name)

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        task.run(context_factory(units))


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_cache_does_not_mask_total_fresh_failure(
    mock_engine_cls, mock_client_cls, tmp_path
):
    """Cached entries must not rescue the all-failed signal.

    If every fresh request fails (e.g. expired NIM key returning HTTP
    403), the customer's environment is broken regardless of what
    cache had from a prior run. Surfacing the failure is more important
    than reporting "Pipeline complete!" with stale cache as the only
    output.
    """
    # Pre-seed cached textures for Steel_Carbon so skip_existing keeps it.
    out_dir = tmp_path / "generated"
    _make_real_texture_set(out_dir, "Steel_Carbon")

    mock_client = mock_client_cls.return_value
    mock_client.generate.return_value = _fail_status("any", "HTTP 403 Forbidden")

    context = {
        "prim_texture_units": [_unit("Steel_Carbon"), _unit("Copper_Polished")],
        "working_dir": str(tmp_path),
        "usd_path": "/tmp/asset.usd",
        "texture_config": {
            "backend": "simple_image_gen",
            "image_gen": {"backend": "nim"},
            "skip_existing": True,
            "workers": 2,
        },
    }
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"1/1 texture generation requests failed"):
        task.run(context)


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_skip_existing_with_cache_and_fresh_success_does_not_raise(
    mock_engine_cls, mock_client_cls, tmp_path, caplog
):
    """Resumed run with cache + at least one fresh success completes cleanly."""
    import logging

    # Pre-seed cached textures for Steel_Carbon.
    out_dir = tmp_path / "generated"
    _make_real_texture_set(out_dir, "Steel_Carbon")

    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        albedo = _make_real_texture_set(tmp_path, cfg.variant_name)
        return _ok_status(albedo, cfg.variant_name)

    mock_client.generate.side_effect = side_effect

    context = {
        "prim_texture_units": [_unit("Steel_Carbon"), _unit("Copper_Polished")],
        "working_dir": str(tmp_path),
        "usd_path": "/tmp/asset.usd",
        "texture_config": {
            "backend": "simple_image_gen",
            "image_gen": {"backend": "nim"},
            "skip_existing": True,
            "workers": 2,
        },
    }
    task = GenerateTexturesTask()

    with caplog.at_level(logging.WARNING):
        result = task.run(context)

    generated = result["generated_textures"]
    assert set(generated.keys()) == {"Steel_Carbon", "Copper_Polished"}
    # No failure warnings -- everything succeeded.
    assert not any("failures" in rec.message for rec in caplog.records)


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_resume_reuses_cache_even_when_skip_existing_is_disabled(
    mock_engine_cls, mock_client_cls, tmp_path
):
    """--resume must avoid regenerating complete cached texture sets."""
    out_dir = tmp_path / "generated"
    _make_real_texture_set(out_dir, "Steel_Carbon")

    context = {
        "prim_texture_units": [_unit("Steel_Carbon")],
        "working_dir": str(tmp_path),
        "usd_path": "/tmp/asset.usd",
        "resume": True,
        "texture_config": {
            "backend": "simple_image_gen",
            "image_gen": {"backend": "nim"},
            "skip_existing": False,
            "workers": 2,
        },
    }

    result = GenerateTexturesTask().run(context)

    assert set(result["generated_textures"]) == {"Steel_Carbon"}
    mock_client_cls.return_value.generate.assert_not_called()


def test_cached_texture_set_skips_invalid_candidate(tmp_path, monkeypatch, caplog):
    """Cache lookup validates candidates before returning them.

    This simulates a stale flat-layout cache entry disappearing between the
    initial existence check and validation. The helper should skip it and fall
    through to the nested-layout candidate instead.
    """
    import logging

    out_dir = tmp_path / "generated"
    nested_dir = out_dir / "Steel_Carbon"
    nested_dir.mkdir(parents=True)

    flat_albedo = out_dir / "Steel_Carbon_albedo.png"
    flat_normal = out_dir / "Steel_Carbon_normal.png"
    flat_orm = out_dir / "Steel_Carbon_orm.png"
    nested_albedo = nested_dir / "Steel_Carbon_albedo.png"
    nested_normal = nested_dir / "Steel_Carbon_normal.png"
    nested_orm = nested_dir / "Steel_Carbon_orm.png"
    for path in (
        flat_albedo,
        flat_normal,
        flat_orm,
        nested_albedo,
        nested_normal,
        nested_orm,
    ):
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

    original_exists = Path.exists
    flat_exists_calls = 0

    # fake_exists lets flat_albedo pass _cached_texture_set's initial check,
    # then fail the validation-time exists() call so fallback can be verified.
    def fake_exists(path: Path) -> bool:
        nonlocal flat_exists_calls
        if path == flat_albedo:
            flat_exists_calls += 1
            return flat_exists_calls == 1
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)

    with caplog.at_level(logging.WARNING):
        textures = _cached_texture_set(out_dir, "Steel_Carbon")

    assert textures is not None
    assert textures.albedo == str(nested_albedo)
    assert textures.normal == str(nested_normal)
    assert textures.orm == str(nested_orm)
    assert any(
        "Skipping invalid cached textures" in rec.message for rec in caplog.records
    )


def test_cached_texture_set_rejects_partial_sets(tmp_path, caplog):
    """Albedo-only cache entries are rejected as incomplete PBR sets."""
    import logging

    out_dir = tmp_path / "generated"
    out_dir.mkdir(parents=True)
    (out_dir / "Steel_Carbon_albedo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    with caplog.at_level(logging.WARNING):
        textures = _cached_texture_set(out_dir, "Steel_Carbon")

    assert textures is None
    assert any(
        "Skipping invalid cached textures" in rec.message for rec in caplog.records
    )
