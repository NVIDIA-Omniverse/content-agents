# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision-Language Model implementations."""

import asyncio
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from world_understanding.functions.models.nim_timeout import _apply_nim_chat_timeout
from world_understanding.telemetry import traced_vlm
from world_understanding.utils.credentials import get_env_api_key_for_backend
from world_understanding.utils.image_utils import image_to_base64
from world_understanding.utils.token_tracking import TokenUsage

# Default configurations
_DEFAULT_NIM_VLM_MODEL = "qwen/qwen3.5-397b-a17b"
_DEFAULT_AZURE_VLM_MODEL = "gpt-5"
_DEFAULT_AWS_VLM_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_DEFAULT_NVIDIA_INFERENCE_MODEL = "gcp/google/gemini-3.1-pro-preview"
_DEFAULT_NVIDIA_INFERENCE_BASE_URL = "https://inference-api.nvidia.com"
_DEFAULT_OPENAI_MODEL = "gpt-5.4"
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
_DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
_DEFAULT_GRADIO_API_NAME = "/process_media"
# No default endpoint — callers must pass `endpoint=...` for the gradio backend.
_DEFAULT_GRADIO_ENDPOINT = ""
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_MAX_TOKENS = None
_DEFAULT_TEMPERATURE = None


class BaseVisionLanguageModel(ABC):
    """Base class for vision-language models.

    Subclasses should set self._last_token_usage after each model invocation
    to enable token tracking.
    """

    def __init__(self):
        """Initialize base VLM with token tracking."""
        self._last_token_usage: TokenUsage | None = None

    @property
    def last_token_usage(self) -> TokenUsage | None:
        """Get token usage from the last model invocation.

        Returns:
            TokenUsage object if available, None otherwise
        """
        return self._last_token_usage

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from text and optional images synchronously.

        Args:
            prompt: User prompt/question
            images: Optional list of images as paths, PIL Images, or arrays
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        pass

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from text and optional images asynchronously.

        Default implementation delegates to sync generate() via asyncio.to_thread.
        Subclasses can override for true async behavior.

        Args:
            prompt: User prompt/question
            images: Optional list of images as paths, PIL Images, or arrays
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        return await asyncio.to_thread(
            self.generate,
            prompt,
            images,
            system_prompt,
            temperature,
            max_tokens,
            **kwargs,
        )

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs followed by a final prompt.

        Args:
            image_caption_pairs: List of tuples (caption, image) where each caption
                                describes or introduces its corresponding image
            final_prompt: The final prompt/question after all images
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        # Default implementation: concatenate all captions with final prompt
        # and provide all images together (for backward compatibility)
        all_captions = []
        all_images = []

        for caption, image in image_caption_pairs:
            all_captions.append(caption)
            all_images.append(image)

        combined_prompt = "\n".join(all_captions) + "\n" + final_prompt
        return self.generate(
            prompt=combined_prompt,
            images=all_images if all_images else None,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def agenerate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs asynchronously.

        Args:
            image_caption_pairs: List of tuples (caption, image) where each caption
                                describes or introduces its corresponding image
            final_prompt: The final prompt/question after all images
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        # Default implementation: concatenate all captions with final prompt
        # and provide all images together (for backward compatibility)
        all_captions = []
        all_images = []

        for caption, image in image_caption_pairs:
            all_captions.append(caption)
            all_images.append(image)

        combined_prompt = "\n".join(all_captions) + "\n" + final_prompt
        return await self.agenerate(
            prompt=combined_prompt,
            images=all_images if all_images else None,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of the backend being used."""
        pass

    def _load_image(
        self, image: str | Path | PILImage.Image | np.ndarray
    ) -> PILImage.Image:
        """Load image from various input formats.

        Args:
            image: Image as file path, PIL Image, or numpy array

        Returns:
            PIL Image object
        """
        if isinstance(image, str | Path):
            return PILImage.open(image).convert("RGB")
        elif isinstance(image, PILImage.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return PILImage.fromarray(image).convert("RGB")
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, PIL Image, or numpy array."
            )

    def _images_to_base64(
        self, images: list[str | Path | PILImage.Image | np.ndarray]
    ) -> list[str]:
        """Convert images to base64 strings.

        Args:
            images: List of images in various formats

        Returns:
            List of base64 encoded image strings
        """
        base64_images = []
        for image in images:
            pil_image = self._load_image(image)
            base64_images.append(image_to_base64(pil_image))
        return base64_images


class GradioVLM(BaseVisionLanguageModel):
    """Vision-Language Model using Gradio client backend.

    Note: Gradio endpoints may require specific network access to function properly.
    The default endpoint is only accessible via NVIDIA San Jose VPN.
    """

    def __init__(
        self,
        endpoint: str = _DEFAULT_GRADIO_ENDPOINT,
        api_name: str = _DEFAULT_GRADIO_API_NAME,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        use_single_image_api: bool | None = None,
        **kwargs: Any,
    ):
        """Initialize Gradio VLM client.

        Args:
            endpoint: Gradio server endpoint URL
            api_name: API endpoint name for the VLM service
            timeout: Request timeout in seconds
            use_single_image_api: If True, call endpoint with a single image
                input; if False, call an endpoint with multi-image. Defaults
                to True for backward compatibility when not provided.
            **kwargs: Additional configuration options
        """

        super().__init__()  # Initialize token tracking

        try:
            from gradio_client import Client
        except ImportError as e:
            raise ImportError(
                "gradio_client is required for GradioVLM. "
                "Install with: pip install gradio_client"
            ) from e

        self.endpoint = endpoint
        self.api_name = api_name
        self.timeout = timeout
        self.client = Client(endpoint, verbose=False)
        self._model_name = kwargs.get("model_name", "gradio-vlm")
        # Configurable pathway for single vs multi image request formatting
        self.use_single_image_api: bool = (
            use_single_image_api if use_single_image_api is not None else True
        )

    @traced_vlm(name="vlm.generate", system="gradio", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Gradio VLM service."""
        try:
            from gradio_client import file as gradio_f
        except ImportError as e:
            raise ImportError("gradio_client is required for file handling") from e

        if self.use_single_image_api:
            # Prepare the request
            tmp_file_path = None
            try:
                if images and len(images) > 0:
                    # For now, support single image (can be extended for multiple)
                    image = images[0]

                    # Handle different image types
                    if isinstance(image, str | Path):
                        image_input = gradio_f(str(image))
                    else:
                        # For PIL Image or numpy array, save to temp file
                        pil_image = self._load_image(image)
                        with tempfile.NamedTemporaryFile(
                            suffix=".png", delete=False
                        ) as tmp_file:
                            pil_image.save(tmp_file.name)
                            tmp_file_path = tmp_file.name
                            image_input = gradio_f(tmp_file.name)
                else:
                    image_input = None

                # Combine system and user prompts
                full_prompt = (
                    f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                )

                predict_kwargs = {
                    "api_name": self.api_name,
                }
                if temperature is not None:
                    predict_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    predict_kwargs["max_tokens"] = max_tokens
                predict_kwargs.update(kwargs)

                # Call Gradio API
                result = self.client.predict(
                    image_input,  # image_input
                    None,  # video_input (not used for images)
                    full_prompt,
                    **predict_kwargs,
                )

                # Extract response (usually first element for text output)
                if isinstance(result, list | tuple) and len(result) > 0:
                    return str(result[0])
                else:
                    return str(result)
            finally:
                # Clean up temporary file if it was created
                if tmp_file_path:
                    try:
                        os.unlink(tmp_file_path)
                    except OSError:
                        pass  # File might already be deleted or inaccessible
        else:
            tmp_file_paths: list[str] = []
            try:
                # Build a list of gradio file handles if images are provided
                images_arg = None
                if images and len(images) > 0:
                    image_inputs: list[Any] = []
                    for img in images:
                        if isinstance(img, str | Path):
                            image_inputs.append(gradio_f(str(img)))
                        else:
                            # For PIL Image or numpy array, save to temp file
                            pil_image = self._load_image(img)
                            with tempfile.NamedTemporaryFile(
                                suffix=".png", delete=False
                            ) as tmp_file:
                                pil_image.save(tmp_file.name)
                                tmp_file_paths.append(tmp_file.name)
                                image_inputs.append(gradio_f(tmp_file.name))
                    images_arg = image_inputs

                # Combine system and user prompts
                full_prompt = (
                    f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                )

                predict_kwargs = {
                    "api_name": self.api_name,
                }
                if temperature is not None:
                    predict_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    predict_kwargs["max_tokens"] = max_tokens
                predict_kwargs.update(kwargs)

                result = self.client.predict(
                    images_arg,
                    None,
                    full_prompt,
                    **predict_kwargs,
                )

                # Extract response (usually first element for text output)
                if isinstance(result, list | tuple) and len(result) > 0:
                    return str(result[0])
                else:
                    return str(result)
            finally:
                # Clean up temporary files if any were created
                if tmp_file_paths:
                    try:
                        for p in tmp_file_paths:
                            try:
                                os.unlink(p)
                            except OSError:
                                pass
                    except Exception:
                        pass

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs using Gradio.

        Note: Gradio endpoints may not natively support true interleaved content.
        This implementation combines captions with the final prompt and
        provides images in order.

        Args:
            image_caption_pairs: List of tuples (caption, image)
            final_prompt: The final prompt after all images
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        # Build combined prompt with all captions
        combined_captions = []
        images = []

        for caption, image in image_caption_pairs:
            combined_captions.append(caption)
            images.append(image)

        # Combine all text parts
        full_user_prompt = "\n".join(combined_captions) + "\n" + final_prompt

        # Use the regular generate method with the combined prompt and ordered images
        return self.generate(
            prompt=full_user_prompt,
            images=images if images else None,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "gradio"


class PerflabAzureOpenAIVLM(BaseVisionLanguageModel):
    """Azure OpenAI VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        api_version: str = "2025-03-01-preview",
        azure_endpoint: str = "",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize Perflab Azure OpenAI VLM.

        Args:
            api_key: Azure OpenAI API key
            model: Model name (defaults to gpt-5)
            api_version: API version
            azure_endpoint: Azure endpoint URL (required; no default)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        if not azure_endpoint:
            raise ValueError(
                "azure_endpoint is required for PerflabAzureOpenAIVLM (no default)."
            )
        super().__init__()  # Initialize token tracking
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for PerflabAzureOpenAIVLM. "
                "Install with: pip install langchain-openai"
            ) from e

        self._model_name = model or _DEFAULT_AZURE_VLM_MODEL
        self.chat_model = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model=self._model_name,
            api_key=api_key,  # type: ignore[arg-type]
            timeout=timeout,
            **kwargs,
        )

    @traced_vlm(name="vlm.generate", system="azure_openai", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Azure OpenAI."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Debug logging for images
        import logging

        logger = logging.getLogger(__name__)
        if images is not None:
            logger.debug(
                f"PerflabAzureOpenAIVLM.generate received {len(images)} images (type: {type(images).__name__})"
            )
        else:
            logger.warning("PerflabAzureOpenAIVLM.generate received images=None")

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        # Set temperature and token limits if provided
        invoke_kwargs = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        # Choose token key based on model family (gpt-5 vs others)
        is_gpt5 = "gpt-5" in (self._model_name or "").lower()
        token_key, alt_key = (
            ("max_completion_tokens", "max_tokens")
            if is_gpt5
            else ("max_tokens", "max_completion_tokens")
        )
        if token_key in kwargs:
            invoke_kwargs[token_key] = kwargs[token_key]
        elif alt_key in kwargs:
            invoke_kwargs[token_key] = kwargs[alt_key]
        elif max_tokens is not None:
            invoke_kwargs[token_key] = max_tokens

        # Add remaining kwargs (excluding max_tokens and max_completion_tokens which we handled)
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        # Remove None values - they shouldn't be passed to the API
        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

        response = self.chat_model.invoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Azure OpenAI asynchronously."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Debug logging for images
        import logging

        logger = logging.getLogger(__name__)
        if images is not None:
            logger.debug(
                f"PerflabAzureOpenAIVLM.agenerate received {len(images)} images (type: {type(images).__name__})"
            )
        else:
            logger.warning("PerflabAzureOpenAIVLM.agenerate received images=None")

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        # Set temperature and token limits if provided
        invoke_kwargs = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        # Choose token key based on model family (gpt-5 vs others)
        is_gpt5 = "gpt-5" in (self._model_name or "").lower()
        token_key, alt_key = (
            ("max_completion_tokens", "max_tokens")
            if is_gpt5
            else ("max_tokens", "max_completion_tokens")
        )
        if token_key in kwargs:
            invoke_kwargs[token_key] = kwargs[token_key]
        elif alt_key in kwargs:
            invoke_kwargs[token_key] = kwargs[alt_key]
        elif max_tokens is not None:
            invoke_kwargs[token_key] = max_tokens

        # Add remaining kwargs (excluding max_tokens and max_completion_tokens which we handled)
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        # Remove None values - they shouldn't be passed to the API
        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        # Set temperature and token limits if provided
        invoke_kwargs = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        # Choose token key based on model family (gpt-5 vs others)
        is_gpt5 = "gpt-5" in (self._model_name or "").lower()
        token_key, alt_key = (
            ("max_completion_tokens", "max_tokens")
            if is_gpt5
            else ("max_tokens", "max_completion_tokens")
        )
        if token_key in kwargs:
            invoke_kwargs[token_key] = kwargs[token_key]
        elif alt_key in kwargs:
            invoke_kwargs[token_key] = kwargs[alt_key]
        elif max_tokens is not None:
            invoke_kwargs[token_key] = max_tokens

        # Add remaining kwargs (excluding max_tokens and max_completion_tokens which we handled)
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        # Remove None values - they shouldn't be passed to the API
        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

        response = self.chat_model.invoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    async def agenerate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs asynchronously with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        # Set temperature and token limits if provided
        invoke_kwargs = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        # Choose token key based on model family (gpt-5 vs others)
        is_gpt5 = "gpt-5" in (self._model_name or "").lower()
        token_key, alt_key = (
            ("max_completion_tokens", "max_tokens")
            if is_gpt5
            else ("max_tokens", "max_completion_tokens")
        )
        if token_key in kwargs:
            invoke_kwargs[token_key] = kwargs[token_key]
        elif alt_key in kwargs:
            invoke_kwargs[token_key] = kwargs[alt_key]
        elif max_tokens is not None:
            invoke_kwargs[token_key] = max_tokens

        # Add remaining kwargs (excluding max_tokens and max_completion_tokens which we handled)
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        # Remove None values - they shouldn't be passed to the API
        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "perflab_azure_openai"


class NvidiaInferenceVLM(BaseVisionLanguageModel):
    """NVIDIA Inference API VLM using OpenAI-compatible endpoint.

    This VLM uses the NVIDIA Inference API (https://inference-api.nvidia.com)
    which provides an OpenAI-compatible interface for various models including
    Azure OpenAI models like GPT-5.1.
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = _DEFAULT_NVIDIA_INFERENCE_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize NVIDIA Inference API VLM.

        Args:
            api_key: NVIDIA Inference API key (INFERENCE_NVIDIA_API_KEY)
            model: Model name (defaults to azure/openai/gpt-5.1)
            base_url: Base URL for the API (defaults to https://inference-api.nvidia.com)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for NvidiaInferenceVLM. "
                "Install with: pip install openai"
            ) from e

        self._model_name = model or _DEFAULT_NVIDIA_INFERENCE_MODEL
        self._base_url = base_url
        self._timeout = timeout
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        from openai import AsyncOpenAI

        self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @traced_vlm(name="vlm.generate", system="nvidia_inference", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using NVIDIA Inference API."""
        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content if images else prompt},
        ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        # Choose token key based on model family (gpt-5 vs others)
        is_gpt5 = "gpt-5" in (self._model_name or "").lower()
        token_key = "max_completion_tokens" if is_gpt5 else "max_tokens"

        if token_key in kwargs:
            request_kwargs[token_key] = kwargs.pop(token_key)
        elif max_tokens is not None:
            request_kwargs[token_key] = max_tokens

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens") and value is not None:
                request_kwargs[key] = value

        response = self.client.chat.completions.create(**request_kwargs)

        # Track token usage
        if response.usage:
            self._last_token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model_name=self._model_name,
                invocation_type="vlm",
            )

        return response.choices[0].message.content or ""

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using NVIDIA Inference API asynchronously."""
        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content if images else prompt},
        ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        # Choose token key based on model family (gpt-5 vs others)
        is_gpt5 = "gpt-5" in (self._model_name or "").lower()
        token_key = "max_completion_tokens" if is_gpt5 else "max_tokens"

        if token_key in kwargs:
            request_kwargs[token_key] = kwargs.pop(token_key)
        elif max_tokens is not None:
            request_kwargs[token_key] = max_tokens

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens") and value is not None:
                request_kwargs[key] = value

        response = await self.aclient.chat.completions.create(**request_kwargs)

        # Track token usage
        if response.usage:
            self._last_token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model_name=self._model_name,
                invocation_type="vlm",
            )

        return response.choices[0].message.content or ""

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with interleaved support."""
        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        # Choose token key based on model family (gpt-5 vs others)
        is_gpt5 = "gpt-5" in (self._model_name or "").lower()
        token_key = "max_completion_tokens" if is_gpt5 else "max_tokens"

        if token_key in kwargs:
            request_kwargs[token_key] = kwargs.pop(token_key)
        elif max_tokens is not None:
            request_kwargs[token_key] = max_tokens

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens") and value is not None:
                request_kwargs[key] = value

        response = self.client.chat.completions.create(**request_kwargs)

        # Track token usage
        if response.usage:
            self._last_token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model_name=self._model_name,
                invocation_type="vlm",
            )

        return response.choices[0].message.content or ""

    async def agenerate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs asynchronously with interleaved support."""
        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        # Choose token key based on model family (gpt-5 vs others)
        is_gpt5 = "gpt-5" in (self._model_name or "").lower()
        token_key = "max_completion_tokens" if is_gpt5 else "max_tokens"

        if token_key in kwargs:
            request_kwargs[token_key] = kwargs.pop(token_key)
        elif max_tokens is not None:
            request_kwargs[token_key] = max_tokens

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens") and value is not None:
                request_kwargs[key] = value

        response = await self.aclient.chat.completions.create(**request_kwargs)

        # Track token usage
        if response.usage:
            self._last_token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model_name=self._model_name,
                invocation_type="vlm",
            )

        return response.choices[0].message.content or ""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "nvidia_inference"


class NvidiaNIMVLM(BaseVisionLanguageModel):
    """NVIDIA NIM VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize NVIDIA NIM VLM.

        Args:
            api_key: NVIDIA API key
            model: Model name (defaults to qwen/qwen3.5-397b-a17b)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()  # Initialize token tracking
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as e:
            raise ImportError(
                "langchain-nvidia-ai-endpoints is required for NvidiaNIMVLM. "
                "Install with: pip install langchain-nvidia-ai-endpoints"
            ) from e

        self._model_name = model or _DEFAULT_NIM_VLM_MODEL
        self.chat_model = ChatNVIDIA(
            model=self._model_name,
            nvidia_api_key=api_key,
            **kwargs,
        )
        # Clear ChatNVIDIA's built-in max_tokens default so it doesn't conflict
        # with per-call max_tokens passed via invoke kwargs.
        self.chat_model.max_tokens = None
        # Cloud NIM rejects `timeout` when ChatNVIDIA serializes constructor
        # fields into the request body. Apply it to the underlying HTTP client
        # instead when the installed SDK exposes one.
        _apply_nim_chat_timeout(self.chat_model, timeout, label="NvidiaNIMVLM")

    @traced_vlm(name="vlm.generate", system="nim", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using NVIDIA NIM."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Debug logging for images
        import logging

        logger = logging.getLogger(__name__)
        if images is not None:
            logger.debug(
                f"NvidiaNIMVLM.generate received {len(images)} images (type: {type(images).__name__})"
            )
        else:
            logger.warning("NvidiaNIMVLM.generate received images=None")

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        # Set temperature and max_tokens if provided
        invoke_kwargs = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        # Handle max_tokens for NIM models:
        # NIM uses max_tokens (not max_completion_tokens). ChatNVIDIA's built-in
        # default is cleared at construction; pass max_tokens per-call for thread safety.
        effective_max_tokens = (
            kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens")
            or max_tokens
        )
        if effective_max_tokens is not None:
            invoke_kwargs["max_tokens"] = effective_max_tokens

        # Add remaining kwargs (excluding max_tokens and max_completion_tokens which we handled)
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        # Remove None values - they shouldn't be passed to the API
        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

        response = self.chat_model.invoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        # Set temperature and max_tokens if provided
        invoke_kwargs = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        # Handle max_tokens for NIM models:
        # NIM uses max_tokens (not max_completion_tokens). ChatNVIDIA's built-in
        # default is cleared at construction; pass max_tokens per-call for thread safety.
        effective_max_tokens = (
            kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens")
            or max_tokens
        )
        if effective_max_tokens is not None:
            invoke_kwargs["max_tokens"] = effective_max_tokens

        # Add remaining kwargs (excluding max_tokens and max_completion_tokens which we handled)
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        # Remove None values - they shouldn't be passed to the API
        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

        response = self.chat_model.invoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "nim"


class OpenAIVLM(BaseVisionLanguageModel):
    """OpenAI VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize OpenAI VLM.

        Args:
            api_key: OpenAI API key
            model: Model name (defaults to gpt-5.4)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAIVLM. "
                "Install with: pip install langchain-openai"
            ) from e

        self._model_name = model or _DEFAULT_OPENAI_MODEL
        self.chat_model = ChatOpenAI(
            model=self._model_name,
            api_key=api_key,  # type: ignore[arg-type]
            timeout=timeout,
            **kwargs,
        )

    @traced_vlm(name="vlm.generate", system="openai", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using OpenAI."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using OpenAI asynchronously."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = []
        for caption, image in image_caption_pairs:
            content.append({"type": "text", "text": caption})
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "openai"


class AnthropicVLM(BaseVisionLanguageModel):
    """Anthropic VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize Anthropic VLM.

        Args:
            api_key: Anthropic API key
            model: Model name (defaults to claude-opus-4-6)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is required for AnthropicVLM. "
                "Install with: pip install langchain-anthropic"
            ) from e

        self._model_name = model or _DEFAULT_ANTHROPIC_MODEL
        self.chat_model = ChatAnthropic(
            model_name=self._model_name,
            api_key=api_key,  # type: ignore[arg-type]
            timeout=timeout,
            **kwargs,
        )

    @traced_vlm(name="vlm.generate", system="anthropic", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Anthropic."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Anthropic asynchronously."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = []
        for caption, image in image_caption_pairs:
            content.append({"type": "text", "text": caption})
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "anthropic"


class GeminiVLM(BaseVisionLanguageModel):
    """Google Gemini VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize Gemini VLM.

        Args:
            api_key: Google API key (loads from GOOGLE_API_KEY or GEMINI_API_KEY
                env var if None)
            model: Model name (defaults to gemini-3-pro-preview)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for GeminiVLM. "
                "Install with: pip install langchain-google-genai"
            ) from e

        api_key = get_env_api_key_for_backend("gemini", api_key)
        if api_key is None:
            raise ValueError(
                "API key is required. Provide via api_key parameter or "
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        self._model_name = model or _DEFAULT_GEMINI_MODEL
        self.chat_model = ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Extract text from response content (handles thinking model list responses)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p["text"]
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(parts) if parts else str(content)
        return str(content)

    @traced_vlm(name="vlm.generate", system="gemini", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Gemini."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_output_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_output_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Gemini asynchronously."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_output_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_output_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = []
        for caption, image in image_caption_pairs:
            content.append({"type": "text", "text": caption})
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_output_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_output_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "gemini"


def create_gradio_vlm(
    endpoint: str | None = None,
    api_name: str = _DEFAULT_GRADIO_API_NAME,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    use_single_image_api: bool = True,
    **kwargs: Any,
) -> GradioVLM:
    """Create a Gradio-based VLM.

    Args:
        endpoint: Gradio server endpoint (uses default if not provided)
        api_name: API endpoint name
        timeout: Request timeout in seconds
        use_single_image_api: If True, force single-image request format; if
            False, try multi-image first.
        **kwargs: Additional configuration

    Returns:
        Configured GradioVLM instance
    """
    return GradioVLM(
        endpoint=endpoint or _DEFAULT_GRADIO_ENDPOINT,
        api_name=api_name,
        timeout=timeout,
        use_single_image_api=use_single_image_api,
        **kwargs,
    )


def create_azure_llmgateway_vlm(
    llmgateway: dict | None = None,
    **kwargs: Any,
):
    from langchain_core.messages import HumanMessage, SystemMessage

    from world_understanding.functions.models.llmgateway import (
        AzureChatOpenAI_LLMGateway,
    )

    class AzureLLMGatewayVLM(BaseVisionLanguageModel):
        def __init__(self):
            super().__init__()  # Initialize token tracking
            self._model_name = kwargs.pop("model", _DEFAULT_AZURE_VLM_MODEL)
            self.chat_model = AzureChatOpenAI_LLMGateway(
                azure_endpoint=(llmgateway or {}).get(
                    "azure_endpoint", "https://prod.api.nvidia.com/llm/v1/azure/"
                ),
                openai_api_version=(llmgateway or {}).get(
                    "api_version", "2025-03-01-preview"
                ),
                deployment_name=self._model_name,
                cred_dict=(llmgateway or {}).get("cred_dict"),
                cred_fields=(llmgateway or {}).get(
                    "cred_fields", ["token_url", "client_id", "client_secret", "scope"]
                ),
                env_prefix=(llmgateway or {}).get("env_prefix", "LLMGATEWAY_CREDS_"),
                cred_file_url=(llmgateway or {}).get("cred_file_url"),
                timeout=_DEFAULT_TIMEOUT_SECONDS,
                **kwargs,
            )

        def generate(
            self,
            prompt: str,
            images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
            system_prompt: str = "You are a helpful AI assistant.",
            temperature: float | None = _DEFAULT_TEMPERATURE,
            max_tokens: int | None = _DEFAULT_MAX_TOKENS,
            **gen_kwargs: Any,
        ) -> str:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            if images:
                for image in images:
                    pil_image = self._load_image(image)
                    base64_image = image_to_base64(pil_image)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        }
                    )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content if images else prompt),
            ]

            invoke_kwargs: dict[str, Any] = {}
            if temperature is not None:
                invoke_kwargs["temperature"] = temperature

            # Pick one token key based on model family and map inputs consistently
            is_gpt5 = "gpt-5" in (self._model_name or "").lower()
            token_key, alt_key = (
                ("max_completion_tokens", "max_tokens")
                if is_gpt5
                else ("max_tokens", "max_completion_tokens")
            )
            if token_key in gen_kwargs:
                invoke_kwargs[token_key] = gen_kwargs[token_key]
            elif alt_key in gen_kwargs:
                invoke_kwargs[token_key] = gen_kwargs[alt_key]
            elif max_tokens is not None:
                invoke_kwargs[token_key] = max_tokens

            # Add remaining gen_kwargs (excluding max_tokens and max_completion_tokens which we handled)
            for key, value in gen_kwargs.items():
                if key not in ("max_tokens", "max_completion_tokens"):
                    invoke_kwargs[key] = value

            # Remove None values - they shouldn't be passed to the API
            invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

            resp = self.chat_model.invoke(messages, **invoke_kwargs)

            # Track token usage
            self._last_token_usage = TokenUsage.from_langchain_response(
                resp, model_name=self._model_name, invocation_type="vlm"
            )

            return resp.content

        def generate_with_image_caption_pairs(
            self,
            image_caption_pairs: list[
                tuple[str, str | Path | PILImage.Image | np.ndarray]
            ],
            final_prompt: str,
            system_prompt: str = "You are a helpful AI assistant.",
            temperature: float | None = _DEFAULT_TEMPERATURE,
            max_tokens: int | None = _DEFAULT_MAX_TOKENS,
            **gen_kwargs: Any,
        ) -> str:
            content: list[dict[str, Any]] = []
            for caption, image in image_caption_pairs:
                content.append({"type": "text", "text": caption})
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )
            content.append({"type": "text", "text": final_prompt})

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ]

            invoke_kwargs: dict[str, Any] = {}
            if temperature is not None:
                invoke_kwargs["temperature"] = temperature

            # Pick one token key based on model family and map inputs consistently
            is_gpt5 = "gpt-5" in (self._model_name or "").lower()
            token_key, alt_key = (
                ("max_completion_tokens", "max_tokens")
                if is_gpt5
                else ("max_tokens", "max_completion_tokens")
            )
            if token_key in gen_kwargs:
                invoke_kwargs[token_key] = gen_kwargs[token_key]
            elif alt_key in gen_kwargs:
                invoke_kwargs[token_key] = gen_kwargs[alt_key]
            elif max_tokens is not None:
                invoke_kwargs[token_key] = max_tokens

            # Add remaining gen_kwargs (excluding max_tokens and max_completion_tokens which we handled)
            for key, value in gen_kwargs.items():
                if key not in ("max_tokens", "max_completion_tokens"):
                    invoke_kwargs[key] = value

            # Remove None values - they shouldn't be passed to the API
            invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

            resp = self.chat_model.invoke(messages, **invoke_kwargs)

            # Track token usage
            self._last_token_usage = TokenUsage.from_langchain_response(
                resp, model_name=self._model_name, invocation_type="vlm"
            )

            return resp.content

        @property
        def model_name(self) -> str:
            return self._model_name

        @property
        def backend_name(self) -> str:
            return "llmgateway_azure_openai"

    return AzureLLMGatewayVLM()


def create_aws_llmgateway_vlm(
    llmgateway: dict | None = None,
    **kwargs: Any,
):
    from langchain_core.messages import HumanMessage, SystemMessage

    from world_understanding.functions.models.llmgateway import (
        ChatConverseAnthropic_LLMGateway,
    )

    class AWSAnthropicLLMGatewayVLM(BaseVisionLanguageModel):
        def __init__(self):
            super().__init__()  # Initialize token tracking
            self._model_name = kwargs.pop("model", _DEFAULT_AWS_VLM_MODEL)
            # Thinking display options
            include_thinking = bool(kwargs.pop("include_thinking", False))
            begin_text = kwargs.pop("thinking_begin_text", "<thinking>\n")
            end_text = kwargs.pop("thinking_end_text", "</thinking>\n")

            temperature = kwargs.pop("temperature", None)
            top_p = kwargs.pop("top_p", None)
            max_tokens = kwargs.pop("max_tokens", None)
            thinking = kwargs.pop("thinking", None)

            # Prepare model kwargs for inference configuration
            model_kwargs: dict[str, Any] = {}
            if temperature is not None:
                model_kwargs["temperature"] = temperature
            if top_p is not None:
                model_kwargs["top_p"] = top_p
            if max_tokens is not None:
                model_kwargs["max_tokens"] = max_tokens
            if thinking is not None:
                model_kwargs["thinking"] = thinking

            self.chat_model = ChatConverseAnthropic_LLMGateway(
                proxy_base_url=(llmgateway or {}).get(
                    "proxy_base_url", "https://prod.api.nvidia.com/llm/v1/aws"
                ),
                aws_region=(llmgateway or {}).get("aws_region", "us-east-2"),
                cred_dict=(llmgateway or {}).get("cred_dict"),
                cred_fields=(llmgateway or {}).get(
                    "cred_fields", ["token_url", "client_id", "client_secret"]
                ),
                env_prefix=(llmgateway or {}).get("env_prefix", "LLMGATEWAY_CREDS_"),
                cred_file_url=(llmgateway or {}).get("cred_file_url"),
                include_thinking=include_thinking,
                thinking_begin_text=begin_text,
                thinking_end_text=end_text,
                model_id=self._model_name,
                model_kwargs=model_kwargs if model_kwargs else None,
                **kwargs,
            )

        def generate(
            self,
            prompt: str,
            images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
            system_prompt: str = "You are a helpful AI assistant.",
            temperature: float | None = _DEFAULT_TEMPERATURE,
            max_tokens: int | None = _DEFAULT_MAX_TOKENS,
            **gen_kwargs: Any,
        ) -> str:
            # Build content list as LangChain multimodal parts
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            if images:
                for image in images:
                    pil_image = self._load_image(image)
                    b64 = image_to_base64(pil_image)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content if images else prompt),
            ]

            inference_config: dict[str, Any] = {}
            if temperature is not None:
                inference_config["temperature"] = float(temperature)
            top_p = gen_kwargs.pop("top_p", None)
            if top_p is not None:
                inference_config["topP"] = float(top_p)

            # Handle max_tokens/max_completion_tokens priority for AWS Anthropic
            # Prefer max_completion_tokens from gen_kwargs, then max_tokens from gen_kwargs, then parameter
            max_completion_tokens = gen_kwargs.pop("max_completion_tokens", None)
            max_tokens_kwarg = gen_kwargs.pop("max_tokens", None)
            if max_completion_tokens is not None:
                inference_config["maxTokens"] = int(max_completion_tokens)
            elif max_tokens_kwarg is not None:
                inference_config["maxTokens"] = int(max_tokens_kwarg)
            elif max_tokens is not None:
                inference_config["maxTokens"] = int(max_tokens)

            stop_sequences = gen_kwargs.pop("stop_sequences", None)
            if stop_sequences is not None:
                inference_config["stopSequences"] = stop_sequences

            # Remove constructor-only keys if present in gen_kwargs
            constructor_only_keys = [
                "include_thinking",
                "thinking",
                "thinking_begin_text",
                "thinking_end_text",
            ]
            for k in constructor_only_keys:
                gen_kwargs.pop(k, None)

            resp = self.chat_model.invoke(
                messages,
                inference_config=inference_config if inference_config else None,
                **gen_kwargs,
            )

            # Track token usage
            self._last_token_usage = TokenUsage.from_langchain_response(
                resp, model_name=self._model_name, invocation_type="vlm"
            )

            return resp.content

        def generate_with_image_caption_pairs(
            self,
            image_caption_pairs: list[
                tuple[str, str | Path | PILImage.Image | np.ndarray]
            ],
            final_prompt: str,
            system_prompt: str = "You are a helpful AI assistant.",
            temperature: float | None = _DEFAULT_TEMPERATURE,
            max_tokens: int | None = _DEFAULT_MAX_TOKENS,
            **gen_kwargs: Any,
        ) -> str:
            content: list[dict[str, Any]] = []
            for caption, image in image_caption_pairs:
                content.append({"type": "text", "text": caption})
                pil_image = self._load_image(image)
                b64 = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            content.append({"type": "text", "text": final_prompt})

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ]

            inference_config: dict[str, Any] = {}
            if temperature is not None:
                inference_config["temperature"] = float(temperature)
            top_p = gen_kwargs.pop("top_p", None)
            if top_p is not None:
                inference_config["topP"] = float(top_p)

            # Handle max_tokens/max_completion_tokens priority for AWS Anthropic
            # Prefer max_completion_tokens from gen_kwargs, then max_tokens from gen_kwargs, then parameter
            max_completion_tokens = gen_kwargs.pop("max_completion_tokens", None)
            max_tokens_kwarg = gen_kwargs.pop("max_tokens", None)
            if max_completion_tokens is not None:
                inference_config["maxTokens"] = int(max_completion_tokens)
            elif max_tokens_kwarg is not None:
                inference_config["maxTokens"] = int(max_tokens_kwarg)
            elif max_tokens is not None:
                inference_config["maxTokens"] = int(max_tokens)

            stop_sequences = gen_kwargs.pop("stop_sequences", None)
            if stop_sequences is not None:
                inference_config["stopSequences"] = stop_sequences

            # Remove constructor-only keys if present in gen_kwargs
            constructor_only_keys = [
                "include_thinking",
                "thinking",
                "thinking_begin_text",
                "thinking_end_text",
            ]
            for k in constructor_only_keys:
                gen_kwargs.pop(k, None)

            resp = self.chat_model.invoke(
                messages,
                inference_config=inference_config if inference_config else None,
                **gen_kwargs,
            )

            # Track token usage
            self._last_token_usage = TokenUsage.from_langchain_response(
                resp, model_name=self._model_name, invocation_type="vlm"
            )

            return resp.content

        @property
        def model_name(self) -> str:
            return self._model_name

        @property
        def backend_name(self) -> str:
            return "llmgateway_aws_anthropic"

    return AWSAnthropicLLMGatewayVLM()


@traced_vlm(name="vlm.create", system="multi", operation="create")
def create_vlm(
    backend: str,
    **kwargs: Any,
) -> BaseVisionLanguageModel:
    """Create a Vision-Language Model for the specified backend.

    Available backends depend on the installation. Public backends (nim) are
    always available. Additional internal backends are available only when
    ``world_understanding_internal`` is installed.

    Args:
        backend: Backend name (use ``list_vlm_backends()`` to see available)
        **kwargs: Backend-specific arguments (api_key, model, llmgateway, etc.)

    Returns:
        Configured VLM instance

    Raises:
        ValueError: If backend is not supported or required parameters missing

    Examples:
        ```python
        vlm = create_vlm("nim", api_key="your-key")
        response = vlm.generate(
            prompt="What is in this image?",
            images=["path/to/image.jpg"]
        )
        ```
    """
    from world_understanding.functions.models.backends.registry import get_vlm_factory

    factory = get_vlm_factory(backend)
    return factory(**kwargs)
