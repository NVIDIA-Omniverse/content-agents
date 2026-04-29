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
