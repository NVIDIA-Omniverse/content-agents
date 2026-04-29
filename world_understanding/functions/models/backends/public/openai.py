# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI backend for chat, VLM, and image generation models."""

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.functions.models.backends.registry import (
    register_chat_backend,
    register_image_gen_backend,
    register_vlm_backend,
)

_DEFAULT_OPENAI_MODEL = "gpt-5.4"
_DEFAULT_TIMEOUT_SECONDS = 120.0


def _resolve_base_url(explicit: str | None = None) -> str | None:
    """Resolve the OpenAI base URL from explicit arg or env.

    langchain-openai's ``ChatOpenAI`` only reads the legacy ``OPENAI_API_BASE``
    env var, while the modern openai SDK (and most of our CI / docker-compose
    envs) use ``OPENAI_BASE_URL``. Accept either so callers that point at an
    OpenAI-compatible endpoint (e.g. NVIDIA's inference-api.nvidia.com) don't
    silently fall back to api.openai.com.
    """
    if explicit:
        return explicit
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")


def create_openai_chat(
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create OpenAI chat model."""
    from langchain_openai import ChatOpenAI

    if not api_key:
        raise ValueError("API key is required for OpenAI backend")

    # Remove kwargs not applicable to OpenAI
    kwargs.pop("api_version", None)

    chat_kwargs: dict[str, Any] = {}
    if temperature is not None:
        chat_kwargs["temperature"] = temperature
    if top_p is not None:
        chat_kwargs["top_p"] = top_p
    if max_tokens is not None:
        chat_kwargs["max_tokens"] = max_tokens
    resolved_base_url = _resolve_base_url(base_url)
    if resolved_base_url:
        chat_kwargs["base_url"] = resolved_base_url
    chat_kwargs.update(kwargs)

    return ChatOpenAI(
        model=model or _DEFAULT_OPENAI_MODEL,
        api_key=api_key,  # type: ignore[arg-type]
        streaming=streaming,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
        **chat_kwargs,
    )


def create_openai_vlm(api_key: str | None = None, **kwargs: Any) -> Any:
    """Create OpenAI VLM."""
    from world_understanding.functions.models.vision_language_models import OpenAIVLM

    if not api_key:
        raise ValueError("API key is required for OpenAI backend")
    # Same OPENAI_BASE_URL / OPENAI_API_BASE resolution as create_openai_chat
    # — OpenAIVLM wraps ChatOpenAI which otherwise ignores OPENAI_BASE_URL.
    if "base_url" not in kwargs:
        resolved = _resolve_base_url()
        if resolved:
            kwargs["base_url"] = resolved
    return OpenAIVLM(api_key=api_key, **kwargs)


def create_openai_image_gen(api_key: str | None = None, **kwargs: Any) -> Any:
    """Create OpenAI image generation model.

    API-key resolution is intentionally delegated to
    :class:`OpenAIImageGenerationModel` so the env fallback and the
    local-NIM ``base_url`` exemption live in one place.
    """
    from world_understanding.functions.models.image_generation_models import (
        OpenAIImageGenerationModel,
    )

    return OpenAIImageGenerationModel(api_key=api_key, **kwargs)


register_chat_backend("openai", create_openai_chat)
register_vlm_backend("openai", create_openai_vlm)
register_image_gen_backend("openai", create_openai_image_gen)
