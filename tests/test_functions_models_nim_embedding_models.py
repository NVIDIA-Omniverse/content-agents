# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from world_understanding.functions.models.image_embedding_models import (
    NIMImageEmbeddingModel,
)
from world_understanding.functions.models.multimodal_embedding_models import (
    NIMMultimodalEmbeddingModel,
)
from world_understanding.functions.models.text_embedding_models import (
    NIMTextEmbeddingModel,
)

NEW_VL_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
REMOVED_MODEL = "/".join(("nvidia", "nv" + "clip"))


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


def _image_model_without_init(
    model_name: str,
) -> tuple[NIMImageEmbeddingModel, _FakeClient]:
    client = _FakeClient()
    model = NIMImageEmbeddingModel.__new__(NIMImageEmbeddingModel)
    model.model = model_name
    model.client = client
    return model, client


def _text_model_without_init(
    model_name: str,
) -> tuple[NIMTextEmbeddingModel, _FakeClient]:
    client = _FakeClient()
    model = NIMTextEmbeddingModel.__new__(NIMTextEmbeddingModel)
    model.model = model_name
    model.client = client
    return model, client


def _multimodal_model_without_init(
    model_name: str,
) -> tuple[NIMMultimodalEmbeddingModel, _FakeClient]:
    client = _FakeClient()
    model = NIMMultimodalEmbeddingModel.__new__(NIMMultimodalEmbeddingModel)
    model.model = model_name
    model.client = client
    return model, client


def test_nim_image_embedding_defaults_to_llama_nemotron_embed_vl() -> None:
    assert NIMImageEmbeddingModel.DEFAULT_MODEL == NEW_VL_EMBED_MODEL
    assert NEW_VL_EMBED_MODEL in NIMImageEmbeddingModel.AVAILABLE_MODELS
    assert REMOVED_MODEL not in NIMImageEmbeddingModel.AVAILABLE_MODELS


def test_nim_multimodal_embedding_defaults_to_llama_nemotron_embed_vl() -> None:
    assert NIMMultimodalEmbeddingModel.DEFAULT_MODEL == NEW_VL_EMBED_MODEL
    assert NEW_VL_EMBED_MODEL in NIMMultimodalEmbeddingModel.AVAILABLE_MODELS
    assert REMOVED_MODEL not in NIMMultimodalEmbeddingModel.AVAILABLE_MODELS


def test_llama_nemotron_image_embedding_sends_modality() -> None:
    model, client = _image_model_without_init(NEW_VL_EMBED_MODEL)

    vectors = model.embed_images([Image.new("RGB", (1, 1), color="white")])

    np.testing.assert_allclose(vectors[0], [0.1, 0.2, 0.3])
    assert client.embeddings.calls[0]["model"] == NEW_VL_EMBED_MODEL
    assert client.embeddings.calls[0]["extra_body"] == {
        "modality": ["image"],
        "input_type": "passage",
        "truncate": "NONE",
    }


def test_llama_nemotron_text_embedding_sends_modality() -> None:
    model, client = _text_model_without_init(NEW_VL_EMBED_MODEL)

    vectors = model.embed_texts(["factory floor"])

    np.testing.assert_allclose(vectors[0], [0.1, 0.2, 0.3])
    assert client.embeddings.calls[0]["model"] == NEW_VL_EMBED_MODEL
    assert client.embeddings.calls[0]["extra_body"] == {
        "modality": ["text"],
        "input_type": "passage",
        "truncate": "NONE",
    }


def test_llama_nemotron_multimodal_image_embedding_sends_modality() -> None:
    model, client = _multimodal_model_without_init(NEW_VL_EMBED_MODEL)

    vectors = model.embed_images([Image.new("RGB", (1, 1), color="white")])

    np.testing.assert_allclose(vectors[0], [0.1, 0.2, 0.3])
    assert client.embeddings.calls[0]["model"] == NEW_VL_EMBED_MODEL
    assert client.embeddings.calls[0]["extra_body"] == {
        "modality": ["image"],
        "input_type": "passage",
        "truncate": "NONE",
    }
