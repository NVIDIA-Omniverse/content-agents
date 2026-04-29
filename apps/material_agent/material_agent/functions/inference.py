# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference functions for VLM-based material assignment.

BACKWARD COMPATIBILITY: This module maintains the exact same API as before.
Internally, it now delegates to world_understanding.functions.classification
for the core classification logic.

The material_agent API remains unchanged - all functions have the same signatures
and return the same output formats. The only difference is that the implementation
now uses the generic classification core with output_key="material".
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from PIL import Image as PILImage

# Import generic classification functions
from world_understanding.functions.classification.inference import (
    async_batch_classify_objects,
    batch_classify_objects,
    classify_object,
    extract_answer_block,
    get_fibonacci_delay,
)
from world_understanding.functions.models.vision_language_models import (
    BaseVisionLanguageModel,
)
from world_understanding.utils.token_tracking import TokenTracker

logger = logging.getLogger(__name__)


def assign_material(
    vlm: BaseVisionLanguageModel,
    text: str,
    images: list[str | Path | PILImage.Image],
    llm: BaseChatModel,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    image_prompts: list[str] | None = None,
    max_retries: int = 3,
    token_tracker: TokenTracker | None = None,
) -> dict[str, str]:
    """Assign material to an object part using Vision-Language Model.

    ✅ BACKWARD COMPATIBLE: Exact same API as before.

    This is the core inference function for the Material Agent. It analyzes
    images of an object part and selects the most appropriate material from
    a provided list, returning a structured JSON response.

    Internally delegates to generic classify_object() with output_key="material".

    Args:
        vlm: Vision-Language Model instance to use for inference
        text: Context text containing object description and available materials
        images: List of images as file paths (str/Path) or PIL Image objects
        llm: LLM for parsing VLM response into structured format (fallback only)
        system_prompt: Optional custom system prompt (uses default if None)
        invoke_kwargs: Additional kwargs to pass to VLM (e.g., temperature, max_tokens)
        image_prompts: Optional list of prompts/captions for each image
        max_retries: Maximum number of retry attempts for VLM/LLM calls (default: 3)

    Returns:
        Dict with "material" and "original_response" keys

    Example:
        ```python
        from PIL import Image
        from world_understanding.functions.models.vision_language_models import (
            create_vlm,
        )
        from world_understanding.functions.models.chat_models import create_chat_model

        # Create VLM and LLM
        vlm = create_vlm(backend="perflab_azure_openai", api_key="your-key")
        llm = create_chat_model(backend="perflab_azure_openai", api_key="your-key")

        # Prepare input - can use file paths or PIL Images
        text = "This is a car. List of possible materials are silver painted steel, matt black rubber."

        # Option 1: Using file paths
        images = ["car_wheel_view1.png", "car_wheel_view2.png"]

        # Option 2: Using PIL Images
        # images = [Image.open("car_wheel_view1.png"), Image.open("car_wheel_view2.png")]

        # Option 3: With image-specific prompts and custom params
        # image_prompts = ["Front view of the wheel", "Side view of the wheel"]
        # invoke_kwargs = {"temperature": 0.5, "max_tokens": 512}

        # Get structured material assignment
        response = assign_material(vlm, text, images, llm, image_prompts=image_prompts, invoke_kwargs=invoke_kwargs)
        print(response)
        # Output: {
        #     "material": "matt black rubber",
        #     "original_response": "Looking at the images, I can see..."
        # }
        ```
    """
    logger.debug(
        "assign_material() calling classify_object() with output_key='material'"
    )

    # Log material-specific message for backward compatibility with tests
    logger.debug(f"Running material assignment with {len(images)} images")

    # Delegate to generic classification with output_key="material"
    return classify_object(
        vlm=vlm,
        text=text,
        images=images,
        llm=llm,
        system_prompt=system_prompt,
        invoke_kwargs=invoke_kwargs,
        image_prompts=image_prompts,
        max_retries=max_retries,
        output_key="material",  # 🔑 Material-specific output key
        token_tracker=token_tracker,
    )


def batch_assign_materials(
    vlm: BaseVisionLanguageModel,
    entries: list[dict[str, Any]],
    llm: BaseChatModel,
    image_base_dir: Path | None = None,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    on_progress: Any | None = None,
    on_error: Any | None = None,
    processed_ids: set[str] | None = None,
    on_result: Any | None = None,
    on_prediction: Any | None = None,
    max_workers: int | None = None,
    max_retries: int = 3,
    token_tracker: TokenTracker | None = None,
) -> list[dict[str, Any]]:
    """Process multiple material assignment tasks in batch with optional parallel execution.

    ✅ BACKWARD COMPATIBLE: Exact same API as before.

    Internally delegates to generic batch_classify_objects() with output_key="material".

    Args:
        vlm: Vision-Language Model instance to use for inference
        entries: List of dictionaries containing:
            - id: Unique identifier
            - text: Context text with object and materials
            - images: List of images (str/Path filenames or PIL Image objects)
        llm: LLM for parsing VLM responses into structured format
        image_base_dir: Base directory for image files (if not absolute paths)
                       Note: Only applies to string/Path images, not PIL Images
        system_prompt: Optional custom system prompt
        invoke_kwargs: Additional kwargs to pass to VLM (e.g., temperature, max_tokens)
        on_progress: Optional callback function(entry_id, response) called after each item
        on_error: Optional callback function(entry_id, error) called on errors
        processed_ids: Set of entry IDs to skip (for resuming interrupted runs)
        on_result: Optional callback function(result, entry) called after each item
        on_prediction: Optional callback function(entry_id, material_dict) called for each successful prediction
        max_workers: Maximum number of parallel workers (default: None = sequential, 1 = sequential,
                    >1 = parallel). Uses ThreadPoolExecutor for concurrent VLM API calls.
        max_retries: Maximum number of retry attempts for VLM/LLM calls (default: 3)

    Returns:
        List of dictionaries containing:
            - id: Entry identifier
            - vlm_response: Material assignment response (dict with
              "material" and "original_response")
            - status: "success" or "error"
            - error: Error message (if status is "error")

    Example:
        ```python
        from PIL import Image

        # Example with parallel processing
        entries = [
            {
                "id": "car_wheel_001",
                "text": "This is a car. Materials: steel, rubber, plastic.",
                "images": ["wheel1.png", "wheel2.png"]  # File paths
            },
            {
                "id": "car_door_002",
                "text": "This is a car. Materials: steel, glass, plastic.",
                "images": [Image.open("door1.png"), Image.open("door2.png")]  # PIL Images
            }
        ]

        # Sequential processing (default)
        results = batch_assign_materials(vlm=vlm, entries=entries, llm=llm)

        # Parallel processing with 4 workers and custom params
        results = batch_assign_materials(
            vlm=vlm,
            entries=entries,
            llm=llm,
            image_base_dir=Path("data/images"),
            invoke_kwargs={"temperature": 0.5, "max_tokens": 512},
            max_workers=4,
            on_progress=lambda id, resp: print(f"Processed {id}")
        )
        ```
    """
    logger.debug(
        "batch_assign_materials() calling batch_classify_objects() with output_key='material'"
    )

    # Log material-specific message for backward compatibility
    entries_count = len(entries) if entries else 0
    skipped = len(processed_ids) if processed_ids else 0
    to_process = entries_count - skipped
    logger.info(f"Starting batch material assignment for {to_process} entries")

    # Delegate to generic batch classification with output_key="material"
    return batch_classify_objects(
        vlm=vlm,
        entries=entries,
        llm=llm,
        image_base_dir=image_base_dir,
        system_prompt=system_prompt,
        invoke_kwargs=invoke_kwargs,
        on_progress=on_progress,
        on_error=on_error,
        processed_ids=processed_ids,
        on_result=on_result,
        on_prediction=on_prediction,
        max_workers=max_workers,
        max_retries=max_retries,
        output_key="material",  # 🔑 Material-specific output key
        token_tracker=token_tracker,
    )


async def async_batch_assign_materials(
    vlm: BaseVisionLanguageModel,
    entries: list[dict[str, Any]],
    llm: BaseChatModel,
    image_base_dir: Path | None = None,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    on_progress: Any | None = None,
    on_error: Any | None = None,
    processed_ids: set[str] | None = None,
    on_result: Any | None = None,
    on_prediction: Any | None = None,
    max_workers: int | None = None,
    max_retries: int = 3,
    token_tracker: TokenTracker | None = None,
) -> list[dict[str, Any]]:
    """Async version of batch_assign_materials using asyncio.gather.

    Internally delegates to async_batch_classify_objects() with output_key="material".

    Args:
        Same as batch_assign_materials().

    Returns:
        Same as batch_assign_materials().
    """
    logger.debug(
        "async_batch_assign_materials() calling async_batch_classify_objects() "
        "with output_key='material'"
    )

    entries_count = len(entries) if entries else 0
    skipped = len(processed_ids) if processed_ids else 0
    to_process = entries_count - skipped
    logger.info(f"Starting async batch material assignment for {to_process} entries")

    return await async_batch_classify_objects(
        vlm=vlm,
        entries=entries,
        llm=llm,
        image_base_dir=image_base_dir,
        system_prompt=system_prompt,
        invoke_kwargs=invoke_kwargs,
        on_progress=on_progress,
        on_error=on_error,
        processed_ids=processed_ids,
        on_result=on_result,
        on_prediction=on_prediction,
        max_workers=max_workers,
        max_retries=max_retries,
        output_key="material",
        token_tracker=token_tracker,
    )


def assign_materials_multi_prim(
    vlm: BaseVisionLanguageModel,
    prim_ids: list[str],
    text: str,
    images: list[str | Path | PILImage.Image],
    llm: BaseChatModel,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    image_prompts: list[str] | None = None,
    max_retries: int = 3,
    token_tracker: TokenTracker | None = None,
) -> dict[str, dict[str, str]]:
    """Assign materials to multiple prims in a single VLM call.

    Sends one VLM request with images for N prims and expects a JSON response
    mapping each prim_id to its material prediction.

    Args:
        vlm: Vision-Language Model instance
        prim_ids: List of prim path identifiers in this group
        text: Combined user prompt with per-part context and image layout
        images: Combined list of images (reference images + per-prim images)
        llm: LLM for fallback parsing
        system_prompt: Multi-prim system prompt (with materials list)
        invoke_kwargs: Additional kwargs for VLM
        image_prompts: Optional per-image captions
        max_retries: Max retry attempts
        token_tracker: Optional token usage tracker

    Returns:
        Dict mapping prim_id -> {"material": "...", "original_response": "..."}
        Missing prims will not be in the dict (caller handles partial failures).
    """
    from world_understanding.functions.classification.inference import (
        classify_objects_multi_prim,
    )

    logger.debug(
        f"assign_materials_multi_prim() calling classify_objects_multi_prim() "
        f"for {len(prim_ids)} prims"
    )

    return classify_objects_multi_prim(
        vlm=vlm,
        object_ids=prim_ids,
        text=text,
        images=images,
        llm=llm,
        system_prompt=system_prompt,
        invoke_kwargs=invoke_kwargs,
        image_prompts=image_prompts,
        max_retries=max_retries,
        output_key="material",
        token_tracker=token_tracker,
    )


# Export all public functions
__all__ = [
    "assign_material",
    "assign_materials_multi_prim",
    "async_batch_assign_materials",
    "batch_assign_materials",
    "extract_answer_block",
    "get_fibonacci_delay",
]
