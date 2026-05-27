# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from world_understanding.functions.models.image_embedding_models import (
    NIMImageEmbeddingModel,
    create_image_embedding_model,
)


def test_create_image_embedding_model_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="Available backends: nim, openai, mock"):
        create_image_embedding_model("unsupported")


def test_nim_image_embedding_model_defaults_to_live_vlm_embedding_model() -> None:
    assert (
        NIMImageEmbeddingModel.DEFAULT_MODEL == "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
    assert NIMImageEmbeddingModel.list_available_models()[0] == (
        "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
