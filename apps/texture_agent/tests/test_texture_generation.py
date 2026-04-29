# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for texture generation abstraction."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from texture_agent.functions.texture_generation import (
    BaseTextureGenerator,
    ImageGenTextureGenerator,
    TextureRequest,
    TextureResult,
    create_texture_generator,
)


class DummyTextureGenerator(BaseTextureGenerator):
    """A simple generator that returns a solid-color image for testing."""

    def __init__(self, color: tuple[int, int, int] = (128, 64, 32)) -> None:
        self._color = color

    @property
    def name(self) -> str:
        return "dummy"

    def generate(self, request: TextureRequest) -> TextureResult:
        image = Image.new("RGB", request.size, self._color)
        return TextureResult(
            image=image,
            prompt_used=request.prompt,
            metadata={"generator": "dummy"},
        )


class TestTextureRequest:
    """Tests for TextureRequest dataclass."""

    def test_default_values(self) -> None:
        req = TextureRequest(
            prompt="rust texture",
            material_name="Steel",
            base_color=(0.3, 0.3, 0.3),
        )
        assert req.size == (1024, 1024)
        assert req.reference_image is None

    def test_custom_size(self) -> None:
        req = TextureRequest(
            prompt="rust",
            material_name="Steel",
            base_color=(0.3, 0.3, 0.3),
            size=(512, 512),
        )
        assert req.size == (512, 512)


class TestBaseTextureGenerator:
    """Tests for the generator abstraction."""

    def test_dummy_generator_returns_correct_size(self) -> None:
        gen = DummyTextureGenerator(color=(255, 0, 0))
        req = TextureRequest(
            prompt="test",
            material_name="Test",
            base_color=(0.5, 0.5, 0.5),
            size=(256, 256),
        )

        result = gen.generate(req)

        assert result.image.size == (256, 256)
        assert result.prompt_used == "test"
        arr = np.array(result.image)
        assert arr[0, 0, 0] == 255  # red

    def test_dummy_generator_name(self) -> None:
        gen = DummyTextureGenerator()
        assert gen.name == "dummy"


class TestImageGenEngine:
    """Tests for ImageGenEngine."""

    def test_name_includes_backend(self) -> None:
        from texture_agent.functions.texture_generation import ImageGenEngine

        engine = ImageGenEngine(backend="gemini")
        assert "gemini" in engine.name

    def test_lazy_initialization(self) -> None:
        """Model is not created until generate() is called."""
        from texture_agent.functions.texture_generation import ImageGenEngine

        engine = ImageGenEngine(backend="nvidia_inference")
        assert engine._model_instance is None

    def test_generate_produces_albedo_normal_orm(self) -> None:
        """Engine produces all three PBR texture files."""
        import tempfile
        from pathlib import Path

        from texture_agent.functions.texture_generation import (
            Conditioning,
            ImageGenEngine,
            TextureVariationConfig,
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = Image.new("RGB", (512, 512), (100, 100, 100))

        engine = ImageGenEngine(backend="test")
        engine._model_instance = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            textures = engine.generate(
                conditioning=Conditioning(text_prompt="rusty metal"),
                config=TextureVariationConfig(variant_name="test"),
                output_dir=Path(tmpdir),
                source_resolution=(512, 512),
            )

            assert Path(textures.albedo).exists()
            assert Path(textures.normal).exists()
            assert Path(textures.orm).exists()

            # Albedo should be the generated image
            albedo = Image.open(textures.albedo)
            assert albedo.size == (512, 512)

            # Prompt was enhanced with PBR instructions
            call_args = mock_model.generate.call_args
            assert "rusty metal" in call_args[0][0]
            assert "PBR" in call_args[0][0]

    def test_generate_passes_albedo_as_conditioning_when_supported(self) -> None:
        """Normal + roughness generations receive the albedo as conditioning
        when the backend declares ``supports_image_conditioning = True``."""
        import tempfile
        from pathlib import Path

        from texture_agent.functions.texture_generation import (
            Conditioning,
            ImageGenEngine,
            TextureVariationConfig,
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = Image.new("RGB", (256, 256), (100, 100, 100))
        mock_model.supports_image_conditioning = True
        mock_model.backend_name = "fake-conditioning-capable"

        engine = ImageGenEngine(backend="test")
        engine._model_instance = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.generate(
                conditioning=Conditioning(text_prompt="rusty metal"),
                config=TextureVariationConfig(variant_name="test"),
                output_dir=Path(tmpdir),
                source_resolution=(256, 256),
            )

        # Three generate() calls: albedo (no ref), normal (ref=[albedo]),
        # roughness (ref=[albedo]).
        calls = mock_model.generate.call_args_list
        assert len(calls) == 3
        albedo_call, normal_call, roughness_call = calls
        # Albedo has no reference image
        assert albedo_call.kwargs.get("images") is None
        # Normal + roughness receive a single-element list (the albedo)
        assert normal_call.kwargs.get("images") is not None
        assert len(normal_call.kwargs["images"]) == 1
        assert roughness_call.kwargs.get("images") is not None
        assert len(roughness_call.kwargs["images"]) == 1

    def test_generate_skips_conditioning_when_unsupported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the backend reports ``supports_image_conditioning = False``
        (e.g. cloud NIM GenAI), the engine does not pass albedo references
        to the normal/roughness passes and logs a single explanatory
        warning so operators can tell the PBR set will be text-conditioned
        only. This is the hot path for the default texture-agent-service
        image-gen backend (``TA_IMAGE_GEN_BACKEND=nim``)."""
        import logging
        import tempfile
        from pathlib import Path

        from texture_agent.functions.texture_generation import (
            Conditioning,
            ImageGenEngine,
            TextureVariationConfig,
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = Image.new("RGB", (256, 256), (90, 90, 90))
        mock_model.supports_image_conditioning = False
        mock_model.backend_name = "nim"

        engine = ImageGenEngine(backend="nim")
        engine._model_instance = mock_model

        with caplog.at_level(logging.WARNING):
            with tempfile.TemporaryDirectory() as tmpdir:
                engine.generate(
                    conditioning=Conditioning(text_prompt="rusty metal"),
                    config=TextureVariationConfig(variant_name="test"),
                    output_dir=Path(tmpdir),
                    source_resolution=(256, 256),
                )

        calls = mock_model.generate.call_args_list
        assert len(calls) == 3
        # All three passes run with no reference image; the engine must not
        # spend a round-trip trying to pass one to a backend that will drop
        # it server-side.
        for call in calls:
            assert call.kwargs.get("images") is None
        # Exactly one warning mentions the backend name and the coherence
        # caveat so the log is self-explanatory in production.
        matched = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "does not support image conditioning" in r.getMessage()
            and "nim" in r.getMessage()
        ]
        assert len(matched) == 1


class TestTextureVariationClient:
    """Tests for the TextureVariationClient (local engine)."""

    def test_generate_returns_completed_status(self) -> None:
        """Successful generation returns completed JobStatus."""
        import tempfile
        from pathlib import Path

        from texture_agent.functions.texture_generation import (
            Conditioning,
            ImageGenEngine,
            TextureVariationClient,
            TextureVariationConfig,
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = Image.new("RGB", (256, 256), (80, 80, 80))

        engine = ImageGenEngine(backend="test")
        engine._model_instance = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            client = TextureVariationClient(engine=engine, output_dir=tmpdir)
            status = client.generate(
                source_asset_uri="file:///test/asset.usd",
                conditioning=Conditioning(text_prompt="weathered wood"),
                config=TextureVariationConfig(strength=0.9),
            )

        assert status.status == "completed"
        assert status.progress == 100
        assert status.result is not None
        assert status.result.generated_textures.albedo
        assert status.result.generated_textures.normal
        assert status.result.generated_textures.orm

    def test_generate_fails_without_conditioning(self) -> None:
        """Empty conditioning raises ValueError."""
        from texture_agent.functions.texture_generation import (
            Conditioning,
            TextureVariationClient,
        )

        client = TextureVariationClient()
        with pytest.raises(ValueError, match="conditioning"):
            client.generate(
                source_asset_uri="file:///test.usd",
                conditioning=Conditioning(),
            )

    def test_conditioning_validation(self) -> None:
        """Conditioning validates that at least one input is provided."""
        from texture_agent.functions.texture_generation import Conditioning

        # Empty — should fail
        c = Conditioning()
        with pytest.raises(ValueError):
            c.validate()

        # Whitespace-only prompt — should fail
        c = Conditioning(text_prompt="   ")
        with pytest.raises(ValueError):
            c.validate()

        # Valid prompt — should pass
        c = Conditioning(text_prompt="rusty metal")
        c.validate()  # no exception

        # Valid ref images — should pass
        c = Conditioning(reference_image_uris=["file:///img.png"])
        c.validate()


class TestCreateTextureGenerator:
    """Tests for the legacy factory function."""

    def test_returns_image_gen_generator(self) -> None:
        gen = create_texture_generator(backend="nvidia_inference")
        assert isinstance(gen, ImageGenTextureGenerator)

    def test_engine_backend(self) -> None:
        gen = create_texture_generator(backend="gemini", model="custom-model")
        assert isinstance(gen, ImageGenTextureGenerator)
        assert gen._engine._model == "custom-model"
