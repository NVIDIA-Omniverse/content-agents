# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry for image generation models."""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ImageGenerationModelRegistry:
    """Registry for managing image generation model factory functions."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """Register an image generation model factory function.

        Args:
            name: Name to register the model under
            factory: Factory function that creates an image generation model
        """
        if name in self._factories:
            logger.warning(
                f"Image generation model '{name}' already registered, overwriting"
            )
        self._factories[name] = factory
        logger.info(f"Registered image generation model: {name}")

    def get_factory(self, name: str) -> Callable[..., Any] | None:
        """Get an image generation model factory function by name.

        Args:
            name: Name of the image generation model

        Returns:
            Factory function if found, None otherwise
        """
        return self._factories.get(name)

    def list_models(self) -> list[str]:
        """List all registered image generation model names."""
        return list(self._factories.keys())

    def create_model(self, name: str, **kwargs: Any) -> Any | None:
        """Create an image generation model instance.

        Args:
            name: Name of the image generation model
            **kwargs: Model-specific arguments

        Returns:
            Image generation model instance if found, None otherwise
        """
        factory = self.get_factory(name)
        if factory:
            return factory(**kwargs)
        return None


# Global image generation model registry instance
_image_generation_model_registry = ImageGenerationModelRegistry()


def get_image_generation_model_registry() -> ImageGenerationModelRegistry:
    """Get the global image generation model registry."""
    return _image_generation_model_registry
