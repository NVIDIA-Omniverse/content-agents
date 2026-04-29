# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend registry for model factories.

Each backend registers a factory function that takes **kwargs and returns
a model instance. The factory functions are responsible for extracting
and validating their own parameters from kwargs.
"""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.functions.models.image_generation_models import (
    BaseImageGenerationModel,
)
from world_understanding.functions.models.vision_language_models import (
    BaseVisionLanguageModel,
)

# Type aliases for factory functions
ChatFactory = Callable[..., BaseChatModel]
VLMFactory = Callable[..., BaseVisionLanguageModel]
ImageGenFactory = Callable[..., BaseImageGenerationModel]

# Registries: backend name -> factory function
_chat_backends: dict[str, ChatFactory] = {}
_vlm_backends: dict[str, VLMFactory] = {}
_image_gen_backends: dict[str, ImageGenFactory] = {}


def register_chat_backend(name: str, factory: ChatFactory) -> None:
    """Register a chat model backend factory."""
    _chat_backends[name] = factory


def register_vlm_backend(name: str, factory: VLMFactory) -> None:
    """Register a VLM backend factory."""
    _vlm_backends[name] = factory


def register_image_gen_backend(name: str, factory: ImageGenFactory) -> None:
    """Register an image generation backend factory."""
    _image_gen_backends[name] = factory


def get_chat_factory(name: str) -> ChatFactory:
    """Get a registered chat backend factory by name."""
    if name not in _chat_backends:
        available = ", ".join(sorted(_chat_backends.keys()))
        raise ValueError(
            f"Unknown chat backend: {name}. Available backends: {available}"
        )
    return _chat_backends[name]


def get_vlm_factory(name: str) -> VLMFactory:
    """Get a registered VLM backend factory by name."""
    if name not in _vlm_backends:
        available = ", ".join(sorted(_vlm_backends.keys()))
        raise ValueError(
            f"Unknown VLM backend: {name}. Available backends: {available}"
        )
    return _vlm_backends[name]


def get_image_gen_factory(name: str) -> ImageGenFactory:
    """Get a registered image generation backend factory by name."""
    if name not in _image_gen_backends:
        available = ", ".join(sorted(_image_gen_backends.keys()))
        raise ValueError(
            f"Unknown image generation backend: {name}. Available backends: {available}"
        )
    return _image_gen_backends[name]


def list_chat_backends() -> list[str]:
    """List all registered chat backend names."""
    return sorted(_chat_backends.keys())


def list_vlm_backends() -> list[str]:
    """List all registered VLM backend names."""
    return sorted(_vlm_backends.keys())


def list_image_gen_backends() -> list[str]:
    """List all registered image generation backend names."""
    return sorted(_image_gen_backends.keys())
