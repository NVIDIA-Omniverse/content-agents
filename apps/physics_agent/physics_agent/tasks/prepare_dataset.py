# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for preparing dataset for asset classification."""

import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.rendering import (
    parse_camera_angle_from_view_name,
)

from physics_agent.api.defaults import (
    DEFAULT_REFERENCE_IMAGE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    PREPARE_DATASET_PROMPTS_DEFAULTS,
)

logger = logging.getLogger(__name__)


def _merged_vlm_image_prompts(value: Any) -> dict[str, Any]:
    prompts = dict(PREPARE_DATASET_PROMPTS_DEFAULTS["vlm_image_prompts"])
    if value is None:
        return prompts
    if isinstance(value, Mapping):
        prompt_mappings = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        prompt_mappings = tuple(value)
    else:
        raise ValueError(
            "prompts.vlm_image_prompts must be a mapping or a sequence of mappings"
        )

    for index, item in enumerate(prompt_mappings):
        if not isinstance(item, Mapping):
            raise ValueError(
                "prompts.vlm_image_prompts entries must be mappings, got "
                f"{type(item).__name__} at index {index}"
            )
        prompts.update(
            {
                str(key): _normalize_vlm_image_prompt_value(str(key), prompt)
                for key, prompt in item.items()
            }
        )
    return prompts


def _normalize_vlm_image_prompt_value(key: str, value: Any) -> str | list[str]:
    if isinstance(value, str):
        return value
    if (
        key == "reference_images"
        and isinstance(value, Sequence)
        and not isinstance(
            value,
            str | bytes | bytearray,
        )
    ):
        prompts: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(
                    "prompts.vlm_image_prompts.reference_images entries must be "
                    f"strings, got {type(item).__name__} at index {index}"
                )
            prompts.append(item)
        return prompts
    raise ValueError(
        f"prompts.vlm_image_prompts.{key} must be a string"
        + (" or sequence of strings" if key == "reference_images" else "")
    )


class PrepareDatasetTask(Task):
    """Task to prepare dataset for asset classification.

    This task creates dataset entries from USD renderings, combining images
    with configurable prompts for VLM classification.

    Input context keys:
        - usd_dir: Path to input USD dataset directory
        - dataset_path: Path to output dataset directory
        - models: List of model numbers to process
        - config: Configuration dictionary with optional flags:
            * 'include_prim_path_context' (bool): Include prim path in context
            * 'include_geometric_context' (bool): Include geometric info
            * 'prompts' (dict): Custom prompt templates
            * 'render_mode_filter' (list[str]): Optional filter for render modes

    Output context keys:
        - dataset_entries: List of prepared dataset entries
        - failed_models: List of model numbers that failed to process
        - dataset_jsonl_path: Path where dataset.jsonl was saved
    """

    def __init__(self):
        """Initialize the prepare dataset task."""
        self.name = "PrepareDataset"
        self.description = "Prepare dataset for asset classification"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        """Prepare dataset entries for the specified models.

        Args:
            context: Workflow context containing required parameters
            object_store: Optional object store (not used)

        Returns:
            Updated context with prepared dataset entries
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        usd_dir = context.get("usd_dir")
        dataset_path = context.get("dataset_path")
        models = context.get("models", [])
        config = context.get("config", {})

        if not usd_dir:
            raise ValueError("usd_dir not provided in context")
        if not dataset_path:
            raise ValueError("dataset_path not provided in context")
        if not models:
            raise ValueError("models not provided in context")

        usd_dir = Path(usd_dir)
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        listener.info(f"Preparing dataset for {len(models)} models")

        # Get configuration options
        include_prim_path_context = config.get("include_prim_path_context", False)
        include_geometric_context = config.get("include_geometric_context", True)

        # Load structure assignments if available
        structure_assignments: dict[str, str] = {}
        structure_assignments_path = config.get("structure_assignments_path")
        if structure_assignments_path:
            try:
                with open(structure_assignments_path, encoding="utf-8") as f:
                    sa_data = json.load(f)
                for prim_path, info in sa_data.get("assignments", {}).items():
                    if isinstance(info, dict):
                        structure_assignments[prim_path] = info.get(
                            "component_name", ""
                        )
                    elif isinstance(info, str):
                        structure_assignments[prim_path] = info
                listener.info(
                    f"Loaded {len(structure_assignments)} structure assignments from {structure_assignments_path}"
                )
            except Exception as e:
                listener.warning(f"Failed to load structure assignments: {e}")

        # Get prompt templates from config, falling back to the Physics Agent
        # schema prompt so predictions remain consumable by apply_physics.
        prompt_config = config.get("prompts", {})
        if not isinstance(prompt_config, Mapping):
            prompt_config = {}
        system_prompt = prompt_config.get("system") or DEFAULT_SYSTEM_PROMPT
        user_prompt_template = prompt_config.get("user") or DEFAULT_USER_PROMPT

        vlm_image_prompts = _merged_vlm_image_prompts(
            prompt_config.get("vlm_image_prompts")
        )

        # Get reference images from context
        reference_images = context.get("reference_images", [])
        # Support per-image prompts: string (shared) or list (per-image)
        reference_image_prompts_config = vlm_image_prompts["reference_images"]
        if isinstance(reference_image_prompts_config, str):
            reference_image_prompts_list = [reference_image_prompts_config] * len(
                reference_images
            )
        else:
            reference_image_prompts_list = list(reference_image_prompts_config)

        dataset_entries = []
        failed_models = []

        for model_number in models:
            try:
                listener.info(f"Processing model: {model_number}")

                # Check for USD dataset structure in input directory
                usd_input_dir = usd_dir / model_number
                dataset_json_path = usd_input_dir / "dataset.json"
                prims_jsonl_path = usd_input_dir / "prims.jsonl"

                # Create output directory for this model
                output_dir = dataset_path / model_number
                output_dir.mkdir(parents=True, exist_ok=True)

                if not dataset_json_path.exists():
                    raise ValueError(f"Dataset JSON not found for {model_number}")
                if not prims_jsonl_path.exists():
                    raise ValueError(f"Prims JSONL not found for {model_number}")

                # Load dataset metadata
                with open(dataset_json_path, encoding="utf-8") as f:
                    dataset_metadata = json.load(f)
                total_prims = dataset_metadata["statistics"]["total_prims"]
                listener.info(f"Loaded dataset metadata with {total_prims} prims")

                # Load prims data
                prims_data = []
                with open(prims_jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        prims_data.append(json.loads(line))
                listener.info(f"Loaded {len(prims_data)} prims from prims.jsonl")

                # Process each prim
                for prim_idx, prim_data in enumerate(prims_data):
                    prim_path = prim_data["prim_path"]
                    listener.debug(f"Processing prim {prim_idx}: {prim_path}")

                    # Build context for this prim
                    prim_context = ""

                    # Add prim path to context if enabled
                    if include_prim_path_context:
                        prim_path_context = (
                            f"The prim path of this 3D asset is: {prim_path}"
                        )
                        prim_context = prim_path_context

                    # Add geometric context if enabled and available
                    if include_geometric_context:
                        geometric_parts = []

                        # Add world bbox in meters if available
                        world_bbox_meters = prim_data.get("world_bbox_meters")
                        if world_bbox_meters:
                            size_m = world_bbox_meters["size"]
                            geometric_parts.append(
                                f"Dimensions (meters): "
                                f"width={size_m[0]:.3f}m, "
                                f"height={size_m[1]:.3f}m, "
                                f"depth={size_m[2]:.3f}m"
                            )
                            bbox_volume = size_m[0] * size_m[1] * size_m[2]
                            geometric_parts.append(
                                f"Bounding box volume: {bbox_volume:.6f} m³"
                            )

                        # Add relative metrics if available
                        relative_metrics = prim_data.get("relative_metrics")
                        if relative_metrics:
                            rel_size = relative_metrics["relative_size"]
                            geometric_parts.append(
                                f"Relative size (% of whole): "
                                f"width={rel_size[0] * 100:.1f}%, "
                                f"height={rel_size[1] * 100:.1f}%, "
                                f"depth={rel_size[2] * 100:.1f}%"
                            )

                        if geometric_parts:
                            geometric_context = "Geometric info:\n" + "\n".join(
                                [f"  - {part}" for part in geometric_parts]
                            )
                            if prim_context:
                                prim_context = f"{prim_context}\n\n{geometric_context}"
                            else:
                                prim_context = geometric_context

                    # Inject structure assignment if available
                    if prim_path in structure_assignments:
                        segment_name = structure_assignments[prim_path]
                        structure_context = (
                            f"Structure analysis: This component has been "
                            f"identified as part of the **{segment_name}** "
                            f"segment. Use this as the component_name."
                        )
                        if prim_context:
                            prim_context = f"{prim_context}\n\n{structure_context}"
                        else:
                            prim_context = structure_context

                    # Format the user prompt with context
                    if prim_context:
                        prompt = f"{user_prompt_template}\n\nContext:\n{prim_context}"
                    else:
                        prompt = user_prompt_template

                    # Extract all image paths from renders
                    image_paths = []
                    image_metadata = []
                    render_mode_filter = config.get("render_mode_filter")

                    for render in prim_data.get("renders", []):
                        # Filter by render mode if specified
                        render_mode = render.get("render_mode", "unknown")
                        if render_mode_filter and render_mode not in render_mode_filter:
                            continue

                        render_path = usd_input_dir / render["path"]
                        try:
                            relative_path = render_path.relative_to(dataset_path)
                        except ValueError:
                            relative_path = os.path.relpath(render_path, dataset_path)
                        image_paths.append(str(relative_path))

                        # Store metadata
                        view_name = render.get("view", "unknown")
                        metadata_entry = {
                            "path": str(relative_path),
                            "view": view_name,
                            "camera": render.get("camera", "default"),
                            "render_mode": render_mode,
                        }

                        # Add VLM prompt for this render mode if available
                        if render_mode in vlm_image_prompts:
                            base_prompt = vlm_image_prompts[render_mode]
                            camera_angle = parse_camera_angle_from_view_name(view_name)
                            metadata_entry["vlm_prompt"] = (
                                f"{base_prompt}\n\n"
                                f"Camera Position: Looking from {camera_angle}"
                            )

                        image_metadata.append(metadata_entry)

                    if not image_paths:
                        listener.warning(
                            f"No image paths found for {prim_path}, skipping"
                        )
                        continue

                    # Sort images for consistent ordering (keep metadata aligned)
                    paired = list(zip(image_paths, image_metadata, strict=True))
                    paired.sort(key=lambda x: x[0])
                    listener.debug(f"Using {len(paired)} renders for {prim_path}")

                    # Build data item in v0.2 format
                    # Prepend reference images so VLM sees them first
                    media_images = []
                    for ref_idx, ref_img in enumerate(reference_images):
                        ref_path = Path(ref_img)
                        try:
                            rel_ref = ref_path.relative_to(dataset_path)
                        except ValueError:
                            rel_ref = os.path.relpath(ref_path, dataset_path)
                        ref_prompt = (
                            reference_image_prompts_list[ref_idx]
                            if ref_idx < len(reference_image_prompts_list)
                            else DEFAULT_REFERENCE_IMAGE_PROMPT
                        )
                        media_images.append(
                            {
                                "path": str(rel_ref),
                                "type": "reference",
                                "metadata": {
                                    "view": "reference",
                                    "camera": "reference",
                                    "render_mode": "reference_image",
                                    "reference_index": ref_idx,
                                    "vlm_prompt": ref_prompt,
                                },
                            }
                        )

                    for img_path, img_meta in paired:
                        image_obj: dict[str, Any] = {
                            "path": img_path,
                            "type": "render",
                        }
                        if img_meta:
                            image_obj["metadata"] = {
                                k: v for k, v in img_meta.items() if k != "path"
                            }
                        media_images.append(image_obj)

                    data_item = {
                        "id": prim_path,
                        "source": {
                            "usd_path": prim_path,
                            "prim_type": "Mesh",
                        },
                        "user_prompt": prompt,
                        "media": {"images": media_images},
                    }
                    entry_metadata = {
                        key: prim_data[key]
                        for key in (
                            "world_bbox",
                            "world_bbox_meters",
                            "relative_metrics",
                        )
                        if key in prim_data
                    }
                    if entry_metadata:
                        data_item["metadata"] = entry_metadata

                    # Save individual entry
                    output_file = (
                        output_dir / f"{model_number}_prim_{prim_idx:04d}.json"
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data_item, f, indent=4)

                    dataset_entries.append(data_item)

                listener.info(
                    f"Prepared {len(dataset_entries)} entries for {model_number}"
                )

            except Exception as e:
                failed_models.append(model_number)
                listener.warning(f"Failed to prepare data for {model_number}: {e}")

        # Save dataset entries
        dataset_jsonl_path = dataset_path / "dataset.jsonl"
        with open(dataset_jsonl_path, "w", encoding="utf-8") as f:
            for entry in dataset_entries:
                f.write(json.dumps(entry) + "\n")
        listener.info(f"Saved dataset to {dataset_jsonl_path}")

        # Create dataset.json (v0.2 format)
        dataset_config = {
            "schema_version": "0.2",
            "metadata": {
                "created": datetime.now().isoformat(),
                "creator": "physics-agent",
                "description": "Asset classification dataset",
                "num_entries": len(dataset_entries),
            },
            "task": {
                "type": "asset_classification",
                "description": "Classify assets based on visual analysis",
            },
            "inference": {
                "prompts": [
                    {
                        "step_name": "classification",
                        "step_index": 0,
                        "system_prompt": system_prompt,
                    }
                ]
            },
            "prims_file": "dataset.jsonl",
        }

        dataset_config_path = dataset_path / "dataset.json"
        with open(dataset_config_path, "w", encoding="utf-8") as f:
            json.dump(dataset_config, f, indent=2)
        listener.info(f"Saved dataset config to {dataset_config_path}")

        # Update context with results
        context["dataset_entries"] = dataset_entries
        context["failed_models"] = failed_models
        context["dataset_jsonl_path"] = str(dataset_jsonl_path)
        context["dataset_config_path"] = str(dataset_config_path)

        return context
