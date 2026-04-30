from __future__ import annotations

import numpy as np
from PIL import Image

from world_understanding.functions.models.image_embedding_models import (
    LocalVisualImageEmbeddingModel,
    create_image_embedding_model,
)


def test_local_visual_image_embedding_is_deterministic() -> None:
    model = LocalVisualImageEmbeddingModel()
    image = Image.new("RGB", (32, 32), color=(20, 80, 140))

    first = model.embed_image(image)
    second = model.embed_image(image)

    assert first.shape == (768,)
    assert np.allclose(first, second)


def test_local_visual_image_embedding_avoids_zero_vectors() -> None:
    model = LocalVisualImageEmbeddingModel()
    black_image = Image.new("RGB", (32, 32), color=(0, 0, 0))
    flat_image = Image.new("RGB", (32, 32), color=(80, 80, 80))

    vectors = model.embed_images([black_image, flat_image])

    assert all(np.linalg.norm(vec) > 0 for vec in vectors)


def test_local_visual_image_embedding_distinguishes_flat_luminance() -> None:
    model = LocalVisualImageEmbeddingModel()
    dark_gray = Image.new("RGB", (32, 32), color=(60, 60, 60))
    light_gray = Image.new("RGB", (32, 32), color=(190, 190, 190))

    dark_vec, light_vec = model.embed_images([dark_gray, light_gray])

    assert float(np.dot(dark_vec, light_vec)) < 0.98


def test_local_visual_image_embedding_distinguishes_texture() -> None:
    model = LocalVisualImageEmbeddingModel()
    flat = Image.new("RGB", (32, 32), color=(128, 128, 128))
    checker = Image.new("RGB", (32, 32), color=(80, 80, 80))
    for y in range(32):
        for x in range(32):
            if (x // 4 + y // 4) % 2 == 0:
                checker.putpixel((x, y), (176, 176, 176))

    flat_vec, checker_vec = model.embed_images([flat, checker])

    assert float(np.dot(flat_vec, checker_vec)) < 0.98


def test_local_visual_image_embedding_distinguishes_copper_and_gold() -> None:
    model = LocalVisualImageEmbeddingModel()
    copper = Image.new("RGB", (32, 32), color=(184, 115, 51))
    gold = Image.new("RGB", (32, 32), color=(190, 140, 40))

    copper_vec, gold_vec = model.embed_images([copper, gold])

    assert float(np.dot(copper_vec, gold_vec)) < 0.98


def test_local_visual_image_embedding_model_list_is_still_list() -> None:
    assert LocalVisualImageEmbeddingModel.AVAILABLE_MODELS == ("local_visual",)
    assert LocalVisualImageEmbeddingModel.list_available_models() == ["local_visual"]


def test_create_image_embedding_model_supports_local_visual() -> None:
    model = create_image_embedding_model("local_visual")

    assert isinstance(model, LocalVisualImageEmbeddingModel)
    assert model.embedding_dimension == 768
