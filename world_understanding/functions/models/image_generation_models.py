# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image generation model implementations.

This module provides interfaces for image generation models that take text prompts
and optional conditioning images (reference, depth, segmentation, etc.) and generate
output images.
"""

import base64
import logging
import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from world_understanding.utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)

# Default configurations
_DEFAULT_GEMINI_MODEL = "gemini-3-pro-image-preview"
_DEFAULT_NVIDIA_INFERENCE_MODEL = "gcp/google/gemini-3-pro-image-preview"
_DEFAULT_NVIDIA_INFERENCE_BASE_URL = "https://inference-api.nvidia.com"
_DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1"
_DEFAULT_NIM_IMAGE_MODEL = "black-forest-labs/flux_2-klein-4b"
_DEFAULT_NIM_IMAGE_BASE_URL = "https://ai.api.nvidia.com/v1/genai"
_DEFAULT_TIMEOUT_SECONDS = 120.0


class BaseImageGenerationModel(ABC):
    """Base class for image generation models.

    Image generation models take text prompts and optional conditioning images
    and generate output images. Unlike VLMs which return text, these models
    return PIL Images.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image from text prompt and optional conditioning images.

        Args:
            prompt: Text prompt describing the desired output
            images: Optional conditioning images (reference, depth, segmentation, etc.)
            **kwargs: Model-specific parameters (temperature, etc.)

        Returns:
            Generated PIL Image
        """
        pass

    @abstractmethod
    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image from interleaved image-prompt pairs.

        This method supports workflows where each conditioning image has an
        associated description (e.g., "This is the target image", "This is the
        depth map for shape retention").

        Args:
            image_prompt_pairs: List of (description, image) tuples where each
                description introduces or describes its corresponding image
            final_prompt: Final generation instruction after all images
            **kwargs: Model-specific parameters

        Returns:
            Generated PIL Image
        """
        pass

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

    @property
    def supports_image_conditioning(self) -> bool:
        """Whether ``generate(images=...)`` respects the conditioning images.

        Most backends either run an img2img model or fold the reference
        images into a multimodal prompt. A handful of endpoints (notably
        the cloud NIM GenAI endpoint) are text-only and silently drop any
        provided images. Callers that build multi-pass pipelines whose
        coherence depends on conditioning (e.g. PBR albedo → normal →
        roughness) can probe this flag to degrade gracefully instead of
        paying for discarded round-trips.

        Defaults to ``True``; override to ``False`` in subclasses whose
        underlying endpoint does not accept reference images.
        """
        return True

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


class GeminiImageGenerationModel(BaseImageGenerationModel):
    """Google Gemini image generation model.

    This model uses Google's Gemini API for image generation tasks including
    style transfer, image editing, and conditional image generation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_GEMINI_MODEL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize Gemini image generation model.

        Args:
            api_key: Google API key (loads from GOOGLE_API_KEY env var if None)
            model: Model name (default: gemini-3-pro-image-preview)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options

        Raises:
            ImportError: If google-genai is not installed
        """
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for GeminiImageGenerationModel. "
                "Install with: pip install google-genai"
            ) from e

        # Load API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key is required. Provide via api_key parameter or "
                    "GOOGLE_API_KEY environment variable."
                )

        self.client = genai.Client(api_key=api_key)
        self._model_name = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image using Gemini.

        Args:
            prompt: Text prompt describing the desired output
            images: Optional list of conditioning images
            **kwargs: Additional arguments to pass to the API

        Returns:
            Generated PIL Image

        Raises:
            ValueError: If no image is generated in the response
        """
        # Build contents list with interleaved text and images
        contents: list[str | PILImage.Image] = [prompt]

        if images:
            for img in images:
                pil_img = self._load_image(img)
                contents.append(pil_img)

        # Call Gemini API
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,  # type: ignore[arg-type]
            **kwargs,
        )

        # Extract generated image from response
        if response.candidates:
            for part in response.candidates[0].content.parts:  # type: ignore[union-attr]
                if part.inline_data is not None:
                    return PILImage.open(BytesIO(part.inline_data.data))  # type: ignore[arg-type]

        raise ValueError("No image generated in response")

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image with interleaved image-prompt pairs.

        This method is particularly useful for multi-conditioning workflows where
        each conditioning image needs a description (e.g., style transfer with
        depth and segmentation guidance).

        Args:
            image_prompt_pairs: List of (description, image) tuples
            final_prompt: Final generation instruction
            **kwargs: Additional arguments to pass to the API

        Returns:
            Generated PIL Image

        Raises:
            ValueError: If no image is generated in the response

        Examples:
            >>> model = GeminiImageGenerationModel()
            >>> generated = model.generate_with_image_prompt_pairs(
            ...     image_prompt_pairs=[
            ...         ("Target image to apply materials to:", target_img),
            ...         ("Depth map for shape retention:", depth_img),
            ...         ("Segmentation for material boundaries:", seg_img),
            ...     ],
            ...     final_prompt="Apply realistic materials matching the style.",
            ... )
        """
        # Build contents with interleaved descriptions and images
        contents: list[str | PILImage.Image] = []

        for description, img in image_prompt_pairs:
            contents.append(description)
            pil_img = self._load_image(img)
            contents.append(pil_img)

        # Add final prompt
        contents.append(final_prompt)

        # Call Gemini API
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,  # type: ignore[arg-type]
            **kwargs,
        )

        # Extract generated image from response
        if response.candidates:
            for part in response.candidates[0].content.parts:  # type: ignore[union-attr]
                if part.inline_data is not None:
                    return PILImage.open(BytesIO(part.inline_data.data))  # type: ignore[arg-type]

        raise ValueError("No image generated in response")

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "gemini"


class NvidiaInferenceImageGenerationModel(BaseImageGenerationModel):
    """Image generation via NVIDIA Inference API (OpenAI-compatible).

    Uses the same ``https://inference-api.nvidia.com`` endpoint as the VLM
    backend but targets image-generation-capable models such as
    ``gcp/google/gemini-3-pro-image-preview``.

    The model is called via OpenAI ``chat.completions.create``.  Input images
    are sent as ``image_url`` content parts (base64 data URIs) and the
    generated image is extracted from the assistant response content parts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = _DEFAULT_NVIDIA_INFERENCE_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> None:
        """Initialise the NVIDIA Inference image generation model.

        Args:
            api_key: NVIDIA Inference API key.  Falls back to the
                ``INFERENCE_NVIDIA_API_KEY`` environment variable.
            model: Model identifier (default: ``gcp/google/gemini-3-pro-image-preview``).
            base_url: API base URL.
            timeout: Request timeout in seconds.
            **kwargs: Reserved for future options.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for NvidiaInferenceImageGenerationModel. "
                "Install with: pip install openai"
            ) from e

        if api_key is None:
            api_key = os.environ.get("INFERENCE_NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is required. Provide via api_key parameter or "
                "INFERENCE_NVIDIA_API_KEY environment variable."
            )

        self._model_name = model or _DEFAULT_NVIDIA_INFERENCE_MODEL
        self._base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate an image from a text prompt and optional conditioning images.

        Args:
            prompt: Text prompt describing the desired output.
            images: Optional list of conditioning / reference images.
            **kwargs: Extra arguments forwarded to the API call
                (e.g. ``temperature``, ``max_tokens``).

        Returns:
            Generated PIL Image.
        """
        content = self._build_content(prompt, images)
        return self._call_and_extract_image(content, **kwargs)

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate an image from interleaved (description, image) pairs.

        Args:
            image_prompt_pairs: List of ``(description, image)`` tuples.
            final_prompt: Final generation instruction appended after all pairs.
            **kwargs: Extra arguments forwarded to the API call.

        Returns:
            Generated PIL Image.
        """
        content: list[dict[str, Any]] = []
        for description, img in image_prompt_pairs:
            content.append({"type": "text", "text": description})
            pil_img = self._load_image(img)
            b64 = image_to_base64(pil_img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})
        return self._call_and_extract_image(content, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return "nvidia_inference"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_content(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None,
    ) -> list[dict[str, Any]]:
        """Build an OpenAI-style content list from prompt + images."""
        content: list[dict[str, Any]] = []
        content.append({"type": "text", "text": prompt})
        if images:
            for img in images:
                pil_img = self._load_image(img)
                b64 = image_to_base64(pil_img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        return content

    def _call_and_extract_image(
        self,
        content: list[dict[str, Any]],
        **kwargs: Any,
    ) -> PILImage.Image:
        """Send chat completion request and extract generated image.

        The assistant response may contain a mix of text and image parts.
        We scan for the first image part and return it.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": content},
        ]

        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        # Forward caller kwargs (temperature, max_tokens, etc.)
        for key, value in kwargs.items():
            if value is not None:
                request_kwargs[key] = value

        logger.info(
            "Calling nvidia_inference image generation: model=%s",
            self._model_name,
        )
        response = self.client.chat.completions.create(**request_kwargs)

        # Extract image from response content parts.
        # The OpenAI-compatible endpoint may return images in several ways:
        #   1. message.content as a string with a data-URI
        #   2. message.content as a list of content parts
        #      - {"type": "image_url", "image_url": {"url": "data:..."}}
        #   3. message.images as a separate list (NVIDIA Inference API)
        #      - [{"type": "image_url", "image_url": {"url": "data:..."}}]
        message = response.choices[0].message
        raw_content = message.content

        # Case 1: content is a list of parts (structured response)
        if isinstance(raw_content, list):
            for part in raw_content:
                img = self._try_extract_image_from_part(part)
                if img is not None:
                    return img

        # Case 2: content is a string – may contain a base64 data URI
        if isinstance(raw_content, str):
            img = self._try_decode_data_uri(raw_content)
            if img is not None:
                return img

        # Case 3: images in a separate "images" field on the message
        # (NVIDIA Inference API returns images here instead of in content)
        raw_msg = response.choices[0].message
        images_list = getattr(raw_msg, "images", None)
        # Also check model_extra for fields the SDK doesn't recognise
        if images_list is None and hasattr(raw_msg, "model_extra"):
            images_list = raw_msg.model_extra.get("images")
        if isinstance(images_list, list):
            for part in images_list:
                img = self._try_extract_image_from_part(part)
                if img is not None:
                    return img

        # Log the raw response for debugging
        logger.warning(
            "No image found in response. content type=%s, "
            "finish_reason=%s, raw_content=%s",
            type(raw_content).__name__,
            getattr(response.choices[0], "finish_reason", "unknown"),
            repr(raw_content)[:500] if raw_content else "None",
        )

        raise ValueError(
            "No image found in response. "
            f"Response content type: {type(raw_content)}, "
            f"model: {self._model_name}"
        )

    @staticmethod
    def _try_extract_image_from_part(part: Any) -> PILImage.Image | None:
        """Try to extract a PIL Image from a single content part."""
        if isinstance(part, dict):
            # {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url:
                    return NvidiaInferenceImageGenerationModel._try_decode_data_uri(url)
        # OpenAI SDK may return typed objects with attributes
        if hasattr(part, "type") and getattr(part, "type", None) == "image_url":
            image_url_obj = getattr(part, "image_url", None)
            if image_url_obj:
                url = getattr(image_url_obj, "url", "")
                if url:
                    return NvidiaInferenceImageGenerationModel._try_decode_data_uri(url)
        return None

    @staticmethod
    def _try_decode_data_uri(text: str) -> PILImage.Image | None:
        """Decode a ``data:image/...;base64,...`` URI to a PIL Image."""
        if "base64," in text:
            b64_data = text.split("base64,", 1)[1].strip()
            try:
                return PILImage.open(BytesIO(base64.b64decode(b64_data)))
            except Exception:
                return None
        return None


class _NamedBytesIO(BytesIO):
    """BytesIO subclass with a name attribute for OpenAI SDK file uploads."""

    def __init__(self, data: bytes, name: str = "image.png") -> None:
        super().__init__(data)
        self.name = name


class OpenAIImageGenerationModel(BaseImageGenerationModel):
    """OpenAI image generation model using the Images API (gpt-image-1).

    Uses ``images.generate`` for text-to-image and ``images.edit`` when
    conditioning images are provided.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_OPENAI_IMAGE_MODEL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI image generation model.

        Args:
            api_key: OpenAI API key (loads from OPENAI_API_KEY env var if None).
                Ignored when ``base_url`` points at a local endpoint that does
                not require auth — a placeholder is used in that case.
            model: Model name (default: gpt-image-1)
            timeout: Request timeout in seconds
            base_url: Override API base URL. Useful for OpenAI-compatible
                servers such as a locally-hosted NIM image generation container
                (e.g. ``http://localhost:8000/v1``).
            **kwargs: Additional configuration options
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIImageGenerationModel. "
                "Install with: pip install openai"
            ) from e

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                if base_url is None:
                    raise ValueError(
                        "API key is required. Provide via api_key parameter or "
                        "OPENAI_API_KEY environment variable."
                    )
                api_key = "not-needed"

        self.client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
        self._model_name = model

    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image using OpenAI Images API.

        Args:
            prompt: Text prompt describing the desired output
            images: Optional conditioning images (uses images.edit when provided)
            **kwargs: Additional arguments forwarded to the API call

        Returns:
            Generated PIL Image
        """
        if images:
            image_files = [self._to_named_bytes_io(img) for img in images]
            response = self.client.images.edit(
                model=self._model_name,
                image=image_files,  # type: ignore[arg-type]
                prompt=prompt,
                n=1,
                **kwargs,
            )
        else:
            response = self.client.images.generate(
                model=self._model_name,
                prompt=prompt,
                n=1,
                **kwargs,
            )
        return self._extract_image(response)

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image from interleaved (description, image) pairs.

        The descriptions are combined into a single prompt since the OpenAI
        Images API does not natively support interleaved text/image inputs.

        Args:
            image_prompt_pairs: List of (description, image) tuples
            final_prompt: Final generation instruction
            **kwargs: Additional arguments forwarded to the API call

        Returns:
            Generated PIL Image
        """
        descriptions = [desc for desc, _ in image_prompt_pairs]
        combined_prompt = "\n".join(descriptions + [final_prompt])
        conditioning_images = [img for _, img in image_prompt_pairs]
        return self.generate(
            prompt=combined_prompt, images=conditioning_images, **kwargs
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return "openai"

    def _to_named_bytes_io(
        self, image: str | Path | PILImage.Image | np.ndarray
    ) -> "_NamedBytesIO":
        """Convert an image to a named BytesIO for OpenAI SDK upload."""
        pil_img = self._load_image(image)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return _NamedBytesIO(buf.getvalue())

    def _extract_image(self, response: Any) -> PILImage.Image:
        """Extract PIL Image from an OpenAI ImagesResponse."""
        if response.data:
            item = response.data[0]
            if item.b64_json:
                return PILImage.open(BytesIO(base64.b64decode(item.b64_json)))
            if item.url:
                import urllib.request

                with urllib.request.urlopen(  # noqa: S310
                    item.url, timeout=_DEFAULT_TIMEOUT_SECONDS
                ) as resp:
                    return PILImage.open(BytesIO(resp.read()))
        raise ValueError("No image found in OpenAI response")


class NIMImageGenerationModel(BaseImageGenerationModel):
    """NVIDIA NIM image generation model using the GenAI REST API.

    Calls ``https://ai.api.nvidia.com/v1/genai/{model}`` with a JSON body and
    returns JPEG images decoded from the ``artifacts[].base64`` response field.

    Image conditioning (ref_images) is not supported by the NIM GenAI endpoint;
    when conditioning images are provided their descriptions are folded into the
    text prompt only — no pixel data is sent to the API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_NIM_IMAGE_MODEL,
        base_url: str = _DEFAULT_NIM_IMAGE_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> None:
        """Initialize NIM image generation model.

        Args:
            api_key: NVIDIA API key (loads from NVIDIA_API_KEY env var if None)
            model: Model name using underscore format, e.g.
                ``black-forest-labs/flux_2-klein-4b`` (default)
            base_url: Base URL for the NIM GenAI endpoint
            timeout: Request timeout in seconds
            **kwargs: Reserved for future options
        """
        if api_key is None:
            api_key = os.getenv("NVIDIA_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key is required. Provide via api_key parameter or "
                    "NVIDIA_API_KEY environment variable."
                )

        self._api_key = api_key
        self._model_name = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @staticmethod
    def _model_to_url_slug(model: str) -> str:
        """Convert internal model name to URL slug.

        The NIM GenAI endpoint uses dots instead of the first underscore in the
        model name portion (after the org prefix).
        e.g. ``black-forest-labs/flux_2-klein-4b`` → ``black-forest-labs/flux.2-klein-4b``
        """
        if "/" in model:
            org, name = model.split("/", 1)
            return f"{org}/{name.replace('_', '.', 1)}"
        return model.replace("_", ".", 1)

    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image using NIM GenAI REST API.

        Args:
            prompt: Text prompt describing the desired output.
            images: Ignored (NIM has no img2img endpoint); present only to
                satisfy the base class interface. If provided, a warning is
                logged and the images are not sent to the API.
            **kwargs: Additional JSON body fields forwarded to the API
                (e.g. ``height``, ``width``).

        Returns:
            Generated PIL Image (JPEG decoded from base64).
        """
        import json
        import urllib.request

        if images:
            logger.warning(
                "NIMImageGenerationModel does not support image conditioning. "
                "The provided images will be ignored."
            )

        slug = self._model_to_url_slug(self._model_name)
        url = f"{self._base_url}/{slug}"

        body: dict[str, Any] = {"prompt": prompt, "height": 1024, "width": 1024}
        body.update(kwargs)

        data = json.dumps(body).encode()
        req = urllib.request.Request(  # noqa: S310
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        logger.info("Calling NIM image generation: model=%s", self._model_name)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            result = json.loads(resp.read())

        artifacts = result.get("artifacts", [])
        if not artifacts:
            raise ValueError("No artifacts in NIM image generation response")

        b64_data = artifacts[0].get("base64", "")
        if not b64_data:
            raise ValueError("Empty base64 in NIM image generation response")

        return PILImage.open(BytesIO(base64.b64decode(b64_data)))

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image from interleaved (description, image) pairs.

        NIM has no interleaved image+text generation endpoint.  The image
        descriptions are concatenated into the text prompt; pixel data is
        discarded.

        Args:
            image_prompt_pairs: List of (description, image) tuples; only the
                descriptions are used.
            final_prompt: Final generation instruction appended after descriptions.
            **kwargs: Extra fields forwarded to the API body.

        Returns:
            Generated PIL Image.
        """
        descriptions = [desc for desc, _ in image_prompt_pairs]
        combined_prompt = "\n".join(descriptions + [final_prompt])
        return self.generate(prompt=combined_prompt, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return "nim"

    @property
    def supports_image_conditioning(self) -> bool:
        # The cloud NIM GenAI endpoint is text-only; any ``images=...`` is
        # dropped (see ``generate``). Downstream PBR pipelines rely on
        # this flag to avoid wasting a round-trip producing an albedo
        # reference that will never reach the model.
        return False


def create_image_generation_model(
    backend: str,
    **kwargs: Any,
) -> BaseImageGenerationModel:
    """Create an image generation model for the specified backend.

    Available backends depend on the installation. Public backends (gemini,
    openai, nim) are always available. Additional internal backends are available
    only when ``world_understanding_internal`` is installed.

    Args:
        backend: Backend name (use ``list_image_gen_backends()`` to see available)
        **kwargs: Backend-specific arguments (api_key, model, base_url, etc.)

    Returns:
        Configured image generation model instance

    Raises:
        ValueError: If backend is not supported
        ImportError: If required packages are not installed
    """
    from world_understanding.functions.models.backends.registry import (
        get_image_gen_factory,
    )

    factory = get_image_gen_factory(backend)
    return factory(**kwargs)
