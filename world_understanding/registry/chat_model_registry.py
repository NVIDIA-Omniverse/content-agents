# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry for chat models."""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ChatModelRegistry:
    """Registry for managing chat model factory functions."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """Register a chat model factory function.

        Args:
            name: Name to register the model under
            factory: Factory function that creates a chat model
        """
        if name in self._factories:
            logger.warning(f"Chat model '{name}' already registered, overwriting")
        self._factories[name] = factory
        logger.info(f"Registered chat model: {name}")

    def get_factory(self, name: str) -> Callable[..., Any] | None:
        """Get a chat model factory function by name.

        Args:
            name: Name of the chat model

        Returns:
            Factory function if found, None otherwise
        """
        return self._factories.get(name)

    def list_models(self) -> list[str]:
        """List all registered chat model names."""
        return list(self._factories.keys())

    def create_model(self, name: str, **kwargs: Any) -> Any | None:
        """Create a chat model instance.

        Args:
            name: Name of the chat model
            **kwargs: Model-specific arguments

        Returns:
            Chat model instance if found, None otherwise
        """
        factory = self.get_factory(name)
        if factory:
            return factory(**kwargs)
        return None


# Global chat model registry instance
_chat_model_registry = ChatModelRegistry()


def get_chat_model_registry() -> ChatModelRegistry:
    """Get the global chat model registry."""
    return _chat_model_registry
