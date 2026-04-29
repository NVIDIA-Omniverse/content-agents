# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for preparing dataset with CMF specifications for benchmark or prediction."""

import json
import logging
import os
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.rendering import (
    parse_camera_angle_from_view_name,
)

logger = logging.getLogger(__name__)

# Private global prompt template for VLM system instructions
_VLM_SYSTEM_PROMPT_TEMPLATE = """You are an expert at identifying object parts and their materials.
Provided images are 3D renderings of a part of an object from different angles.

The material properties in the render are irrelevant to the task, you will only consider \
the shape and position of the part.

The part of interest are highlighted in orange contour outline.

When the part is occluded, it may not contain orange contour outline, but just the part itself.

Other parts are rendered in muted colors; again, their color is irrelevant to the task, \
just consider the shape and position of the parts.

When asked to identify the material of the part, you should only focus on the part \
that is highlighted in orange contour outline.

In summary:
- DO NOT judge the material by the material of the rendered image. Only consider the \
shape and position of the part from the rendered images.
- DO judge the color and material by the reference images.

Additional context of the part and materials will be provided with the question.

Available materials:
{materials_list}

Please answer the question with a structured JSON output using the following format:
{{
"material": "material name"
}}

Answer the task requirements in the following format:
<reasoning>your reasoning</reasoning>
<answer>your answer</answer>"""

# Private global prompt template for material selection
_VLM_USER_PROMPT_TEMPLATE = """Please identify the highlighted part and select \
the appropriate material from the predefined list of materials.

You will match the look of the asset exactly to the reference images.

You will think about the best material for it, but if you can't find it in the list \
of materials, you will select the closest match.

Below is the additional context of the part and materials:
{context}"""

# ---------------------------------------------------------------------------
# Multi-prim prompt templates (used when prediction_batch_size > 1)
# ---------------------------------------------------------------------------

_VLM_MULTI_PRIM_SYSTEM_PROMPT_TEMPLATE = """You are an expert at identifying object parts and their materials.
You will be shown 3D renderings of MULTIPLE parts of an object. Each part is identified \
by its prim path (a unique ID).

The material property in the renders is irrelevant to the task — you will only consider \
the shape and position of each part.

Parts of interest are highlighted in orange contour outline in their respective images.

When a part is occluded, it may not contain an orange contour outline, but just the part itself.

Other parts are rendered in muted colors — their color is irrelevant to the task; \
just consider the shape and position.

In summary:
- DO NOT judge the material by the material of the rendered image. Only consider the \
shape and position of the part from the rendered images.
- DO judge the color and material by the reference images.

Available materials:
{materials_list}

You MUST return a JSON object mapping each prim path to its predicted material. \
Use this exact format:
{{
  "<prim_path_1>": {{"material": "material name"}},
  "<prim_path_2>": {{"material": "material name"}}
}}

Answer the task requirements in the following format:
<reasoning>your reasoning for each part</reasoning>
<answer>your JSON answer</answer>"""

_VLM_MULTI_PRIM_USER_PROMPT_TEMPLATE = """Please identify each highlighted part and select \
the appropriate material from the predefined list of materials.

You will match the look of each part exactly to the reference images.

For each part, think about the best material for it, but if you can't find it in the \
list of materials, select the closest match.

The images are organized as follows:
{image_layout}

Below is the additional context for each part:
{per_part_context}"""


def extract_material_name_from_mdl_path(mdl_path: str) -> str | None:
    """Extract material name from MDL path.

    Extracts the material name from the parent directory name in the MDL path,
    removing the "nv" prefix, numeric prefix, and formatting as title case.

    Args:
        mdl_path: The MDL file path (e.g., "../../materials/3D_Library_Material/nv007_tin_plating/tin_plating.mdl")

    Returns:
        The extracted material name formatted as title case (e.g., "Tin Plating"),
        or None if extraction fails

    Example:
        >>> extract_material_name_from_mdl_path("../../materials/3D_Library_Material/nv007_tin_plating/tin_plating.mdl")
        "Tin Plating"
    """
    if not mdl_path:
        return None

    # Split the path and get the parent directory of the .mdl file
    # Path format: "../../materials/3D_Library_Material/nv007_tin_plating/tin_plating.mdl"
    path_parts = mdl_path.split("/")

    # Find the directory containing the .mdl file (second to last part)
    if len(path_parts) >= 2:
        material_dir = path_parts[-2]  # e.g., "nv007_tin_plating"

        # Remove "nv" prefix if present
        if material_dir.startswith("nv"):
            material_name = material_dir[2:]  # Remove "nv" -> "007_tin_plating"

            # Remove leading numeric prefix (e.g., "007_")
            # Find the first non-digit character
            for i, char in enumerate(material_name):
                if not char.isdigit() and char == "_":
                    material_name = material_name[i + 1 :]  # Skip digits and underscore
                    break
                elif not char.isdigit():
                    break

            # Replace underscores with spaces and convert to title case
            material_name = material_name.replace("_", " ").title()
            return material_name

    return None


def match_display_color_to_material(
    display_color: list[float], color_to_material_list: list[dict]
) -> str | None:
    """Match a display color to a material using the color-to-material mapping.

    Args:
        display_color: RGB color as list of floats [r, g, b]
        color_to_material_list: List of dicts with 'color' and 'material' keys

    Returns:
        Material name if match found, None otherwise
    """
    if not display_color or not color_to_material_list:
        return None

    # Round display color to 3 decimals for comparison
    rounded_color = [round(c, 3) for c in display_color]

    # Try to find exact match
    for mapping in color_to_material_list:
        if "color" in mapping and "material" in mapping:
            # Round the key color to 3 decimals for comparison
            key_color = [round(c, 3) for c in mapping["color"]]
            if rounded_color == key_color:
                return mapping["material"]

    return None


class PrepareDatasetTask(Task):
    """Task to prepare dataset with CMF specifications for benchmark or prediction.

    This task optionally extracts CMF specifications for model numbers using the spec_rag
    functionality (when vector store is provided) and creates dataset entries with or without
    ground truth labels.

    Input context keys:
        - vector_store_path: (Optional) Path to vector store directory for spec extraction
        - usd_dir: Path to input USD dataset directory
        - dataset_path: Path to output dataset directory
        - models: List of model numbers to process
        - llm: (Optional) LLM instance for spec extraction (required if vector_store_path provided)
        - config: Configuration dictionary with optional flags:
            * 'include_ground_truth' (bool): Include ground truth labels (default: True)
            * 'include_prim_path_context' (bool): Include prim path in context (default: False)
            * 'include_display_color_context' (bool): Include display color in context (default: False)
            * 'materials_list' (str | list[str]): Available materials - string or list format
            * 'prompts' (dict): Custom prompt templates
            * 'render_mode_filter' (list[str]): Optional filter for render modes (e.g., ["prim_only", "prim_with_stage"])
            * 'include_image_metadata' (bool): Include image metadata in dataset entries (default: False)
            * 'reference_image' (str): Path to reference image for context
            * 'vlm_image_prompts' (dict | list[dict]): Image prompt mappings. Supports
              render modes (e.g., 'prim_with_stage') plus special keys:
                - 'reference_images' / 'reference_image': Prompts for reference photos
                - 'reference_pdfs' / 'reference_pdf': Prompts for PDF-derived pages
            * 'pdf_conversion' (dict): PDF conversion parameters:
                - 'dpi' (int): Resolution in DPI (default: 150)
                - 'format' (str): Image format - png, jpeg, jpg, tiff, ppm (default: "png")
                - 'first_page' (int): First page to convert, 1-indexed (optional)
                - 'last_page' (int): Last page to convert, 1-indexed (optional)
                - 'grayscale' (bool): Convert to grayscale (default: False)

    Output context keys:
        - dataset_entries: List of prepared dataset entries
        - failed_models: List of model numbers that failed to process
        - dataset_jsonl_path: Path where dataset.jsonl was saved
        - vlm_prompt_path: Path where VLM system prompt was saved
    """

    def __init__(self):
        """Initialize the prepare dataset task."""
        self.name = "PrepareDataset"
        self.description = "Prepare dataset with CMF specifications"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Prepare benchmark data for the specified models.

        Args:
            context: Workflow context containing required parameters
            object_store: Optional object store (not used)

        Returns:
            Updated context with prepared dataset entries
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        vector_store_path = context.get("vector_store_path")
        usd_dir = context.get("usd_dir")
        dataset_path = context.get("dataset_path")
        models = context.get("models", [])
        llm = context.get("llm")
        config = context.get("config", {})

        # Vector store is now optional
        if vector_store_path:
            listener.info(f"Using vector store at: {vector_store_path}")
        else:
            listener.info("Vector store not provided - will use default context")

        if not usd_dir:
            raise ValueError("usd_dir not provided in context")
        if not dataset_path:
            raise ValueError("dataset_path not provided in context")
        if not models:
            raise ValueError("models not provided in context")

        # LLM is only required if vector_store_path is provided
        if vector_store_path and not llm:
            raise ValueError(
                "llm not provided in context - required when using vector_store"
            )

        listener.info(f"Preparing dataset for {len(models)} models")

        # Get materials list from config
        # Priority: _materials_formatted (from pipeline) > materials_list
        materials_formatted = config.get("_materials_formatted")
        if materials_formatted:
            # Use formatted materials with descriptions from pipeline
            materials_list = materials_formatted
            listener.info(
                "Using formatted materials from pipeline (includes descriptions)"
            )
        else:
            # Fallback to legacy materials_list format
            materials_list_raw = config.get("materials_list", "")
            if not materials_list_raw:
                listener.warning("No materials_list provided in config")

            # Handle both string and list formats
            if isinstance(materials_list_raw, list):
                # Convert list to comma-separated string
                materials_list = ", ".join(materials_list_raw)
                listener.debug(
                    f"Converted materials_list from list to string: {len(materials_list_raw)} materials"
                )
            else:
                # Already a string
                materials_list = materials_list_raw

        # Check if ground truth should be included (default to True for backward compatibility)
        include_ground_truth = config.get("include_ground_truth", True)
        listener.info(
            f"Dataset mode: {'benchmark' if include_ground_truth else 'prediction'} "
            f"(include_ground_truth={include_ground_truth})"
        )

        # Get display color to material mapping if provided
        display_color_to_material = config.get("display_color_to_material", [])
        if display_color_to_material:
            listener.info(
                f"Using display_color_to_material mapping with "
                f"{len(display_color_to_material)} color entries"
            )

        # Check if prim path should be included in context (default to False)
        include_prim_path_context = config.get("include_prim_path_context", False)
        listener.info(f"Include prim path in context: {include_prim_path_context}")

        # Check if display color should be included in context (default to False)
        include_display_color_context = config.get(
            "include_display_color_context", False
        )
        listener.info(
            f"Include display color in context: {include_display_color_context}"
        )

        # Check if geometric information should be included in context (default to True)
        include_geometric_context = config.get("include_geometric_context", True)
        listener.info(
            f"Include geometric information in context: {include_geometric_context}"
        )

        # Get custom prompt templates from config if provided
        prompt_config = config.get("prompts", {})
        vlm_system_prompt_template = prompt_config.get(
            "vlm_system", _VLM_SYSTEM_PROMPT_TEMPLATE
        )
        vlm_user_prompt_template = prompt_config.get(
            "vlm_user", _VLM_USER_PROMPT_TEMPLATE
        )

        # Get VLM image prompts if provided (support dict or list[dict] formats)
        vlm_image_prompts_raw = prompt_config.get("vlm_image_prompts")
        vlm_image_prompts: dict[str, Any] = {}
        reference_image_prompts: list[str] = []
        reference_pdf_prompts: list[str] = []
        if isinstance(vlm_image_prompts_raw, dict):
            vlm_image_prompts = vlm_image_prompts_raw
            # Extract reference image prompts (list) or single fallback
            if isinstance(vlm_image_prompts.get("reference_images"), list):
                reference_image_prompts = [
                    str(p) for p in vlm_image_prompts.get("reference_images", [])
                ]
            elif isinstance(vlm_image_prompts.get("reference_image"), str):
                reference_image_prompts = [vlm_image_prompts["reference_image"]]
            # Extract reference PDF prompts (list) or single fallback
            if isinstance(vlm_image_prompts.get("reference_pdfs"), list):
                reference_pdf_prompts = [
                    str(p) for p in vlm_image_prompts.get("reference_pdfs", [])
                ]
            elif isinstance(vlm_image_prompts.get("reference_pdf"), str):
                reference_pdf_prompts = [vlm_image_prompts["reference_pdf"]]
            listener.info(
                f"Loaded VLM image prompts for: {list(vlm_image_prompts.keys())}"
            )
        elif isinstance(vlm_image_prompts_raw, list):
            for item in vlm_image_prompts_raw:
                if isinstance(item, dict):
                    vlm_image_prompts.update(item)
            # Attempt to extract reference prompts if present in merged map
            if isinstance(vlm_image_prompts.get("reference_images"), list):
                reference_image_prompts = [
                    str(p) for p in vlm_image_prompts.get("reference_images", [])
                ]
            elif isinstance(vlm_image_prompts.get("reference_image"), str):
                reference_image_prompts = [vlm_image_prompts["reference_image"]]
            if isinstance(vlm_image_prompts.get("reference_pdfs"), list):
                reference_pdf_prompts = [
                    str(p) for p in vlm_image_prompts.get("reference_pdfs", [])
                ]
            elif isinstance(vlm_image_prompts.get("reference_pdf"), str):
                reference_pdf_prompts = [vlm_image_prompts["reference_pdf"]]
            if vlm_image_prompts:
                listener.info(
                    f"Loaded VLM image prompts for: {list(vlm_image_prompts.keys())}"
                )

        # Get PDF conversion parameters
        pdf_conversion_config = config.get("pdf_conversion", {})
        pdf_dpi = pdf_conversion_config.get("dpi", 150)
        pdf_format = pdf_conversion_config.get("format", "png")
        pdf_first_page = pdf_conversion_config.get("first_page")
        pdf_last_page = pdf_conversion_config.get("last_page")
        pdf_grayscale = pdf_conversion_config.get("grayscale", False)

        # Get reference images if provided (supports multiple)
        # Note: Paths are already resolved in config_prepare_dataset.py
        from pathlib import Path

        from PIL import Image as PILImage
        from world_understanding.functions.graphics import convert_pdf_to_images

        reference_images_raw = config.get("reference_images", [])
        legacy_reference_image = config.get("reference_image")
        reference_pdfs = config.get("reference_pdfs", [])

        # Max pixel dimension for reference images written to dataset dir.
        # Large original photos (e.g. 20 MB PNGs) would otherwise be base64-
        # encoded per VLM call, blocking the async event loop and serialising
        # concurrent workers.  1024 px is sufficient for VLM material matching.
        ref_image_max_size: int = config.get("reference_image_max_size", 1024)

        def _copy_ref_image(src: Path, dst: Path) -> None:
            """Copy a reference image to dst, downscaling if larger than ref_image_max_size."""
            original_size_kb = src.stat().st_size // 1024
            img = PILImage.open(src).convert("RGBA")
            if max(img.size) > ref_image_max_size:
                orig_w, orig_h = img.size
                img.thumbnail(
                    (ref_image_max_size, ref_image_max_size), PILImage.LANCZOS
                )
                listener.info(
                    f"Resized reference image {src.name} "
                    f"({orig_w}x{orig_h}, {original_size_kb} KB) "
                    f"→ {img.size[0]}x{img.size[1]} "
                    f"(max {ref_image_max_size}px)"
                )
            img.save(dst)

        # List of reference image relative paths (as stored in dataset entries)
        reference_rel_paths: list[str] = []
        reference_source_types: list[str] = []

        if isinstance(reference_images_raw, list) and reference_images_raw:
            listener.info(f"Using {len(reference_images_raw)} reference image(s)")
            for idx, ref in enumerate(reference_images_raw):
                ref_path = Path(ref)
                if not ref_path.exists():
                    listener.warning(f"Reference image not found: {ref_path}")
                    continue
                # Try to copy (and resize) to dataset directory with stable indexed name
                copied_name = f"reference_image_{idx}.png"
                copied_path = dataset_path / copied_name
                try:
                    _copy_ref_image(ref_path, copied_path)
                    listener.info(f"Copied reference image[{idx}] to: {copied_path}")
                    reference_rel_paths.append(copied_name)
                    reference_source_types.append("image")
                except Exception as e:
                    listener.error(
                        f"Failed to copy reference image[{idx}] ({ref_path}): {e}"
                    )
                    # Fallback to original relative path from dataset directory
                    try:
                        rel_fallback = ref_path.relative_to(dataset_path)
                    except ValueError:
                        rel_fallback = Path(os.path.relpath(ref_path, dataset_path))
                    reference_rel_paths.append(str(rel_fallback))
                    reference_source_types.append("image")
        elif legacy_reference_image:
            ref_path = Path(legacy_reference_image)
            if not ref_path.exists():
                listener.warning(f"Reference image not found: {ref_path}")
            else:
                listener.info(f"Using reference image: {ref_path}")
                copied_name = "reference_image.png"
                copied_path = dataset_path / copied_name
                try:
                    _copy_ref_image(ref_path, copied_path)
                    listener.info(f"Copied reference image to: {copied_path}")
                    reference_rel_paths.append(copied_name)
                    reference_source_types.append("image")
                except Exception as e:
                    listener.error(f"Failed to copy reference image: {e}")
                    # Fallback to original relative path
                    try:
                        rel_fallback = ref_path.relative_to(dataset_path)
                    except ValueError:
                        rel_fallback = Path(os.path.relpath(ref_path, dataset_path))
                    reference_rel_paths.append(str(rel_fallback))
                    reference_source_types.append("image")

        # Process reference PDFs (convert to images)
        if reference_pdfs:
            listener.info(f"Processing {len(reference_pdfs)} reference PDF(s)")
            listener.info(
                f"PDF conversion settings: dpi={pdf_dpi}, format={pdf_format}, "
                f"first_page={pdf_first_page or 'all'}, last_page={pdf_last_page or 'all'}, "
                f"grayscale={pdf_grayscale}"
            )
            for pdf_idx, pdf_path_str in enumerate(reference_pdfs):
                pdf_path = Path(pdf_path_str)
                if not pdf_path.exists():
                    listener.warning(f"Reference PDF not found: {pdf_path}")
                    continue

                try:
                    # Convert PDF to images in dataset directory
                    pdf_image_dir = dataset_path / f"pdf_{pdf_idx}"
                    results = convert_pdf_to_images(
                        pdf_path=pdf_path,
                        output_dir=pdf_image_dir,
                        dpi=pdf_dpi,
                        fmt=pdf_format,
                        first_page=pdf_first_page,
                        last_page=pdf_last_page,
                        grayscale=pdf_grayscale,
                    )

                    # Add converted images to reference list
                    for result in results:
                        img_path_str = result.get("image_path")
                        if img_path_str:
                            img_path = Path(img_path_str)
                            try:
                                rel_path = img_path.relative_to(dataset_path)
                                reference_rel_paths.append(str(rel_path))
                                reference_source_types.append("pdf")
                                listener.debug(
                                    f"Added PDF page as reference: {rel_path}"
                                )
                            except ValueError:
                                # Fallback to relative path
                                rel_path = Path(os.path.relpath(img_path, dataset_path))
                                reference_rel_paths.append(str(rel_path))
                                reference_source_types.append("pdf")

                    listener.info(
                        f"Converted PDF[{pdf_idx}] to {len(results)} image(s)"
                    )

                except RuntimeError as e:
                    listener.error(
                        f"Cannot process PDF {pdf_path.name}: {e}. "
                        "Install pypdfium2 with: pip install pypdfium2"
                    )
                    raise
                except Exception as e:
                    listener.error(f"Failed to convert PDF {pdf_path.name}: {e}")
                    # Continue with other PDFs
                    continue

        dataset_entries = []
        failed_models = []

        for model_number in models:
            try:
                listener.info(f"Processing model: {model_number}")

                # Check for USD dataset structure in input directory
                usd_input_dir = usd_dir / model_number
                dataset_json_path = usd_input_dir / "dataset.json"
                prims_jsonl_path = usd_input_dir / "prims.jsonl"
                usd_model_json_path = usd_input_dir / "usd_model.json"

                # Create output directory for this model
                output_dir = dataset_path / model_number
                output_dir.mkdir(parents=True, exist_ok=True)

                if not dataset_json_path.exists():
                    raise ValueError(f"Dataset JSON not found for {model_number}")
                if not prims_jsonl_path.exists():
                    raise ValueError(f"Prims JSONL not found for {model_number}")
                if not usd_model_json_path.exists():
                    raise ValueError(f"USD model JSON not found for {model_number}")

                # Load dataset metadata
                with open(dataset_json_path, encoding="utf-8") as f:
                    dataset_metadata = json.load(f)
                total_prims = dataset_metadata["statistics"]["total_prims"]
                listener.info(f"Loaded dataset metadata with {total_prims} prims")

                # Extract stage-level metrics if available
                dataset_metadata.get("meters_per_unit")
                dataset_metadata.get("stage_world_bbox")
                dataset_metadata.get("stage_world_bbox_meters")

                # Load prims data
                prims_data = []
                with open(prims_jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        prims_data.append(json.loads(line))
                listener.info(f"Loaded {len(prims_data)} prims from prims.jsonl")

                # Extract spec context if vector store is available
                spec_path = output_dir / "spec.txt"

                if vector_store_path:
                    # Import spec_rag only when needed
                    from material_agent.pcba.spec_rag import (
                        extract_spec_text_by_model_number,
                    )

                    model_parts = model_number.split("_")
                    if not model_parts:
                        raise ValueError(f"Invalid model_number format: {model_number}")

                    context_snippet = extract_spec_text_by_model_number(
                        model_number=model_parts[
                            0
                        ],  # get the model number without the subidentifier
                        llm=llm,
                        vector_store_dir=vector_store_path,
                    )

                    # Save spec even if it contains "No information"
                    with open(spec_path, "w", encoding="utf-8") as f:
                        f.write(context_snippet)

                    # Log warning instead of raising error if no information found
                    if "No information" in context_snippet:
                        listener.warning(
                            f"No specification information found for {model_number} in vector store"
                        )
                        # Use a default context instead
                        context_snippet = (
                            "No additional specification context available."
                        )
                    else:
                        listener.debug(f"Extracted specs: {context_snippet}")
                else:
                    # Use default context when vector store is not available
                    context_snippet = "No additional specification context available."
                    with open(spec_path, "w", encoding="utf-8") as f:
                        f.write(context_snippet)
                    listener.debug("Using default context (no vector store provided)")

                # Process each prim
                for prim_idx, prim_data in enumerate(prims_data):
                    prim_path = prim_data["prim_path"]
                    listener.debug(f"Processing prim {prim_idx}: {prim_path}")

                    # Build context for this prim
                    prim_context = context_snippet

                    # Add prim path to context if enabled
                    if include_prim_path_context:
                        # Add prim path to the context
                        prim_path_context = (
                            f"For the context, the prim path of the 3D USD "
                            f"stage for this part is {prim_path}."
                        )
                        no_spec_context = (
                            "No additional specification context available."
                        )
                        if prim_context and prim_context != no_spec_context:
                            prim_context = f"{prim_context}\n\n{prim_path_context}"
                        else:
                            prim_context = prim_path_context

                    # Add display color to context if enabled and available
                    if include_display_color_context:
                        display_color = prim_data.get("display_color")
                        if display_color:
                            # Format display color as RGB values
                            color_str = f"[{display_color[0]:.3f}, {display_color[1]:.3f}, {display_color[2]:.3f}]"
                            display_color_context = f"The display color of this part in the 3D model is: {color_str}"
                            no_spec_context = (
                                "No additional specification context available."
                            )
                            if prim_context and prim_context != no_spec_context:
                                prim_context = (
                                    f"{prim_context}\n\n{display_color_context}"
                                )
                            else:
                                prim_context = display_color_context
                            listener.debug(
                                f"Added display color to context: {color_str}"
                            )
                        else:
                            listener.debug(
                                f"No display color available for {prim_path}"
                            )

                    # Add geometric context if enabled and available
                    if include_geometric_context:
                        geometric_context_parts = []

                        # Add world bbox in meters if available
                        world_bbox_meters = prim_data.get("world_bbox_meters")
                        if world_bbox_meters:
                            size_m = world_bbox_meters["size"]
                            geometric_context_parts.append(
                                f"Bounding box dimensions (meters): "
                                f"width={size_m[0]:.3f}m, "
                                f"height={size_m[1]:.3f}m, "
                                f"depth={size_m[2]:.3f}m"
                            )

                        # Add relative metrics if available
                        relative_metrics = prim_data.get("relative_metrics")
                        if relative_metrics:
                            rel_size = relative_metrics["relative_size"]

                            # Format relative size as percentages
                            geometric_context_parts.append(
                                f"Relative size (% of whole object): "
                                f"width={rel_size[0] * 100:.1f}%, "
                                f"height={rel_size[1] * 100:.1f}%, "
                                f"depth={rel_size[2] * 100:.1f}%"
                            )

                            # Add relative position as numeric values
                            relative_center = relative_metrics["relative_center"]
                            geometric_context_parts.append(
                                f"Relative position (meters from object center): "
                                f"x={relative_center[0]:.3f}, "
                                f"y={relative_center[1]:.3f}, "
                                f"z={relative_center[2]:.3f}"
                            )

                        # Combine geometric context
                        if geometric_context_parts:
                            geometric_context = (
                                "Bounding box information:\n"
                                + "\n".join(
                                    [f"  - {part}" for part in geometric_context_parts]
                                )
                            )

                            no_spec_context = (
                                "No additional specification context available."
                            )
                            if prim_context and prim_context != no_spec_context:
                                prim_context = f"{prim_context}\n\n{geometric_context}"
                            else:
                                prim_context = geometric_context
                            listener.debug(f"Added geometric context for {prim_path}")
                        else:
                            listener.debug(
                                f"No geometric information available for {prim_path}"
                            )

                    # Add metadata context if available
                    metadata = prim_data.get("metadata")
                    if metadata:
                        metadata_context_parts = []

                        # Add annotation if available
                        custom_data = metadata.get("custom_data", {})
                        annotation = custom_data.get("annotation")
                        if annotation:
                            metadata_context_parts.append(
                                f"Part annotation: {annotation}"
                            )

                        # Add key HOOPS metadata fields (selective to avoid overwhelming the prompt)
                        hoops_metadata = metadata.get("hoops_metadata", {})
                        if hoops_metadata:
                            # Include only the most relevant fields for material identification
                            relevant_fields = {
                                "PTC_COMMON_NAME": "Part name",
                                "PTC_WM_NAME": "Windchill name",
                                "PTC_WM_NUMBER": "Part number",
                                "PART_TYPE": "Part type",
                            }
                            for key, label in relevant_fields.items():
                                if key in hoops_metadata and hoops_metadata[key]:
                                    value = hoops_metadata[key]
                                    # Skip default/empty values
                                    if value not in ["-", "N", "", 0, "0"]:
                                        metadata_context_parts.append(
                                            f"{label}: {value}"
                                        )

                        # Add reference information if available
                        references = metadata.get("references")
                        if references and len(references) > 0:
                            # Just mention that references exist, don't list all
                            metadata_context_parts.append(
                                f"This part references {len(references)} other component(s)"
                            )

                        # Combine metadata context
                        if metadata_context_parts:
                            metadata_context = "Part metadata:\n" + "\n".join(
                                [f"  - {part}" for part in metadata_context_parts]
                            )

                            no_spec_context = (
                                "No additional specification context available."
                            )
                            if prim_context and prim_context != no_spec_context:
                                prim_context = f"{prim_context}\n\n{metadata_context}"
                            else:
                                prim_context = metadata_context
                            listener.debug(f"Added metadata context for {prim_path}")
                        else:
                            listener.debug(
                                f"No relevant metadata available for {prim_path}"
                            )

                    # Add reference images indexing info to context (if present)
                    if reference_rel_paths:
                        ref_context_lines = []
                        # Build a compact description list with indices
                        image_prompt_idx = 0
                        pdf_prompt_idx = 0
                        for i in range(len(reference_rel_paths)):
                            source = (
                                reference_source_types[i]
                                if i < len(reference_source_types)
                                else "image"
                            )
                            if source == "pdf":
                                prompts_list = reference_pdf_prompts
                                prompt_idx = pdf_prompt_idx
                                pdf_prompt_idx += 1
                                default_desc = "Reference PDF page"
                            else:
                                prompts_list = reference_image_prompts
                                prompt_idx = image_prompt_idx
                                image_prompt_idx += 1
                                default_desc = "Reference image"

                            if prompt_idx < len(prompts_list):
                                ref_desc = prompts_list[prompt_idx]
                            elif prompts_list:
                                ref_desc = prompts_list[0]
                            else:
                                ref_desc = default_desc

                            ref_context_lines.append(f"[{i}] {ref_desc}")

                        reference_context = (
                            "Reference images precede rendered images. Indexing:\n"
                            + "\n".join([f"  - {line}" for line in ref_context_lines])
                        )

                        no_spec_context = (
                            "No additional specification context available."
                        )
                        if prim_context and prim_context != no_spec_context:
                            prim_context = f"{prim_context}\n\n{reference_context}"
                        else:
                            prim_context = reference_context

                    # Format the prompt with the context
                    prompt = vlm_user_prompt_template.format(context=prim_context)
                    listener.debug(f"Prompt: {prompt}")

                    # Extract material name (only if including ground truth)
                    material_name = None
                    if include_ground_truth:
                        # Try display color mapping first if available
                        if display_color_to_material:
                            display_color = prim_data.get("display_color")
                            if display_color:
                                material_name = match_display_color_to_material(
                                    display_color, display_color_to_material
                                )
                                if material_name:
                                    listener.debug(
                                        f"Matched display color {display_color} to "
                                        f"material: {material_name}"
                                    )
                                else:
                                    listener.warning(
                                        f"No material match found for display color "
                                        f"{display_color} in prim {prim_path}"
                                    )
                            else:
                                listener.warning(
                                    f"display_color_to_material mapping provided but "
                                    f"prim {prim_path} has no display_color"
                                )

                        # Fall back to MDL path extraction if no color mapping or match
                        if not material_name and "material_bindings" in prim_data:
                            mdl_path = prim_data["material_bindings"]["mdl_path"]
                            material_name = extract_material_name_from_mdl_path(
                                mdl_path
                            )
                            if material_name:
                                listener.debug(
                                    f"Extracted material from MDL path: {material_name}"
                                )

                        if not material_name:
                            listener.warning(
                                f"Could not determine material for {prim_path}, skipping"
                            )
                            continue

                        listener.debug(f"Ground truth material: {material_name}")

                    # Extract all image paths from renders
                    image_paths = []
                    image_metadata = []  # Store render metadata for potential future use
                    for render in prim_data["renders"]:
                        # The render paths in prims.jsonl are relative to the USD model directory
                        # We need to make them relative to where dataset.jsonl will be saved
                        render_path = usd_input_dir / render["path"]
                        # Create relative path from dataset directory to the render file
                        try:
                            relative_path = render_path.relative_to(dataset_path)
                        except ValueError:
                            # If not in same tree, use os.path.relpath
                            relative_path = os.path.relpath(render_path, dataset_path)
                        image_paths.append(str(relative_path))

                        # Store metadata about the render (view, camera, render_mode)
                        view_name = render.get("view", "unknown")
                        metadata_entry = {
                            "path": str(relative_path),
                            "view": view_name,
                            "camera": render.get("camera", "default"),
                            "render_mode": render.get("render_mode", "unknown"),
                        }

                        # Add VLM prompt for this render mode if available
                        render_mode = render.get("render_mode", "unknown")
                        if render_mode in vlm_image_prompts:
                            base_prompt = vlm_image_prompts[render_mode]
                            # Append camera position context to the prompt
                            camera_angle = parse_camera_angle_from_view_name(view_name)
                            metadata_entry["vlm_prompt"] = (
                                f"{base_prompt}\n\n"
                                f"Camera Position: Looking from {camera_angle} towards the center"
                            )

                        image_metadata.append(metadata_entry)

                    # Filter images by render_mode if specified
                    render_mode_filter = config.get("render_mode_filter", None)
                    if render_mode_filter:
                        # Filter image_paths and metadata by render_mode
                        filtered_pairs = [
                            (path, meta)
                            for path, meta in zip(
                                image_paths, image_metadata, strict=False
                            )
                            if meta.get("render_mode") in render_mode_filter
                        ]
                        if filtered_pairs:
                            image_paths, image_metadata = zip(
                                *filtered_pairs, strict=False
                            )
                            image_paths = list(image_paths)
                            image_metadata = list(image_metadata)
                        else:
                            image_paths = []
                            image_metadata = []
                        listener.debug(
                            f"Filtered to {len(image_paths)} renders with modes: {render_mode_filter}"
                        )

                    # Add reference images (if any), preserving order and adding metadata
                    if reference_rel_paths:
                        # Insert all reference images at the beginning, in order
                        image_prompt_idx = 0
                        pdf_prompt_idx = 0
                        for i, ref_path in enumerate(reference_rel_paths):
                            source = (
                                reference_source_types[i]
                                if i < len(reference_source_types)
                                else "image"
                            )
                            if source == "pdf":
                                prompts_list = reference_pdf_prompts
                                prompt_idx = pdf_prompt_idx
                                pdf_prompt_idx += 1
                            else:
                                prompts_list = reference_image_prompts
                                prompt_idx = image_prompt_idx
                                image_prompt_idx += 1

                            image_paths.insert(i, ref_path)
                            ref_metadata = {
                                "path": ref_path,
                                "view": "reference",
                                "camera": "reference",
                                "render_mode": "reference_image",
                                "reference_index": i,
                                "reference_type": source,
                            }
                            # Attach per-reference prompts if provided
                            if prompt_idx < len(prompts_list):
                                ref_metadata["vlm_prompt"] = prompts_list[prompt_idx]
                            elif prompts_list:
                                # Use single fallback prompt for all references of that type
                                ref_metadata["vlm_prompt"] = prompts_list[0]

                            image_metadata.insert(i, ref_metadata)

                    # Sort images (except references) to ensure consistent ordering by view name
                    # Keep references at the beginning if present
                    num_refs = len(reference_rel_paths)
                    if num_refs > 0 and len(image_paths) > num_refs:
                        # Split into refs and renders
                        ref_paths = image_paths[:num_refs]
                        ref_meta = image_metadata[:num_refs]
                        render_pairs = list(
                            zip(
                                image_paths[num_refs:],
                                image_metadata[num_refs:],
                                strict=False,
                            )
                        )
                        # Sort by path
                        render_pairs.sort(key=lambda p: p[0])
                        # Recombine
                        image_paths = ref_paths + [p[0] for p in render_pairs]
                        image_metadata = ref_meta + [p[1] for p in render_pairs]
                    else:
                        image_paths.sort()

                    if len(image_paths) == 0:
                        listener.warning(
                            f"No image paths found for {prim_path}, skipping"
                        )
                        continue

                    listener.debug(f"Using {len(image_paths)} renders for {prim_path}")

                    # Build data item in v0.2 format
                    # Convert image_metadata to v0.2 media structure
                    media_images: list[dict[str, Any]] = []
                    for img_path, img_meta in zip(
                        image_paths, image_metadata, strict=False
                    ):
                        image_obj: dict[str, Any] = {
                            "path": img_path,
                            "type": (
                                "reference"
                                if img_meta.get("render_mode") == "reference_image"
                                else "render"
                            ),
                        }
                        # Add metadata if present
                        if img_meta:
                            image_obj["metadata"] = {
                                k: v
                                for k, v in img_meta.items()
                                if k != "path"  # Don't duplicate path
                            }
                        media_images.append(image_obj)

                    data_item = {
                        "id": prim_path,
                        "source": {
                            "usd_path": prim_path,
                            "prim_type": "Mesh",  # Default, could be extracted
                        },
                        "user_prompt": prompt,
                        "media": {"images": media_images},
                    }

                    # Add ground truth in v0.2 format
                    if include_ground_truth:
                        data_item["ground_truth"] = {"material": material_name}

                    output_path = (
                        output_dir / f"{model_number}_prim_{prim_idx:04d}.json"
                    )
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(data_item, f, indent=4)
                    listener.info(f"Saved data item to {output_path}")

                    dataset_entries.append(data_item)

                listener.info(
                    f"Prepared {len(dataset_entries)} dataset entries "
                    f"for {model_number}"
                )

            except Exception as e:
                failed_models.append(model_number)
                listener.warning(f"Failed to prepare data for {model_number}: {e}")

        # Save dataset entries (v0.2 format)
        dataset_jsonl_path = dataset_path / "dataset.jsonl"
        with open(dataset_jsonl_path, "w", encoding="utf-8") as f:
            for entry in dataset_entries:
                f.write(json.dumps(entry) + "\n")
        listener.info(f"Saved dataset to {dataset_jsonl_path}")

        # Create unified dataset.json (v0.2 format)
        from datetime import datetime

        # Format the VLM system prompt template with materials_list
        formatted_vlm_system_prompt = vlm_system_prompt_template.format(
            materials_list=materials_list
        )

        dataset_config = {
            "schema_version": "0.2",
            "metadata": {
                "created": datetime.now().isoformat(),
                "creator": "material-agent",
                "description": "Material assignment dataset prepared with CMF specifications",
                "num_entries": len(dataset_entries),
            },
            "task": {
                "type": "material_assignment",
                "description": "Identify parts and assign materials from predefined list",
            },
            "inference": {
                "prompts": [
                    {
                        "step_name": "material_selection",
                        "step_index": 0,
                        "system_prompt": formatted_vlm_system_prompt,
                        "output_format": {"material": "material name"},
                    }
                ]
            },
            "prims_file": "dataset.jsonl",
        }

        dataset_config_path = dataset_path / "dataset.json"
        with open(dataset_config_path, "w", encoding="utf-8") as f:
            json.dump(dataset_config, f, indent=2)
        listener.info(f"Saved dataset config to {dataset_config_path}")
        listener.info("  System prompt stored in dataset.json (v0.2 format)")

        # Check if any models failed to process
        if failed_models:
            error_msg = (
                f"Failed to prepare data for {len(failed_models)} model(s): "
                f"{', '.join(failed_models)}. "
                f"Check that Phase 1 files (dataset.json, prims.jsonl) exist in the USD directory."
            )
            listener.error(error_msg)
            raise ValueError(error_msg)

        # Update context with results
        context["dataset_entries"] = dataset_entries
        context["failed_models"] = failed_models
        context["dataset_jsonl_path"] = dataset_jsonl_path
        context["dataset_config_path"] = dataset_config_path

        return context
