# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD generated-reference image task provisioning."""

from pathlib import Path
from typing import Any

from world_understanding.agentic.usd_tasks.generate_reference_image import (
    GenerateReferenceImageTask,
)


class _FakeImage:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake-image")


class _FakeImageGenModel:
    model_name = "fake-image-gen"
    backend_name = "fake"

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def generate_with_image_prompt_pairs(self, **kwargs: Any) -> _FakeImage:
        self._captured["generate_kwargs"] = kwargs
        return _FakeImage()


def test_generate_reference_image_uses_gemini_default_and_model_kwargs(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")
    output_dir = tmp_path / "generated"

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "model": "gemini-3-pro-image-preview",
            "base_url": "http://image-gen.local/v1",
            "timeout": 12,
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(output_dir),
        "num_images": 1,
    }

    result = GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "gemini"
    assert captured["kwargs"] == {
        "model": "gemini-3-pro-image-preview",
        "base_url": "http://image-gen.local/v1",
        "timeout": 12,
    }
    assert captured["generate_kwargs"]["image_prompt_pairs"] == [
        (
            "This is preview image 1 of a 3D scene rendered from a USD file.",
            str(preview_path),
        )
    ]
    assert result["generated_reference_image_paths"] == [
        str(output_dir / "generated_ref_0.png")
    ]


def test_generate_reference_image_replaces_placeholder_gemini_key_from_env(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "real-gemini-key")
    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "gemini",
            "api_key": "YOUR_GOOGLE_API_KEY",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "gemini"
    assert captured["kwargs"]["api_key"] == "real-gemini-key"


def test_generate_reference_image_uses_explicit_local_openai_dummy_key_before_env(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setenv("OPENAI_API_KEY", "real-hosted-openai-key")
    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "openai",
            "base_url": "http://localhost:8000/v1",
            "api_key": "not-used",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "openai"
    assert captured["kwargs"]["api_key"] == "not-used"


def test_generate_reference_image_forwards_custom_backend_explicit_key(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "internal_image_backend",
            "api_key": "custom-backend-key",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "internal_image_backend"
    assert captured["kwargs"]["api_key"] == "custom-backend-key"


def test_generate_reference_image_does_not_forward_custom_backend_placeholder(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "internal_image_backend",
            "api_key": "YOUR_API_KEY",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "internal_image_backend"
    assert "api_key" not in captured["kwargs"]
