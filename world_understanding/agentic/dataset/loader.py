# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset loader with auto-detection of v0.1 vs v0.2 formats.

This module provides functions to load USD agent datasets with automatic format
detection and conversion.
"""

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

from world_understanding.agentic.dataset.schema import (
    DatasetConfig,
    DatasetEntry,
    DatasetMetadata,
    GroundTruth,
    GroundTruthMetadata,
    ImageMetadata,
    ImageObject,
    InferenceConfig,
    MediaConfig,
    PromptConfig,
    SourceInfo,
    TaskConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Format Detection
# =============================================================================


def detect_dataset_version(dataset_dir: Path) -> Literal["0.1", "0.2"]:
    """Auto-detect dataset format version.

    Detection logic:
    - v0.2: Has dataset.json with schema_version="0.2"
    - v0.1: Has vlm_system_prompt.txt or usd/dataset.json

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        "0.1" or "0.2"

    Raises:
        ValueError: If format cannot be determined
    """
    dataset_json = dataset_dir / "dataset.json"
    vlm_prompt_txt = dataset_dir / "vlm_system_prompt.txt"
    usd_dataset_json = dataset_dir / "usd" / "dataset.json"

    # Check for v0.2 format
    if dataset_json.exists():
        try:
            with open(dataset_json, encoding="utf-8") as f:
                data = json.load(f)
                if data.get("schema_version") == "0.2":
                    return "0.2"
        except (json.JSONDecodeError, OSError):
            pass

    # Check for v0.1 format indicators
    if vlm_prompt_txt.exists() or usd_dataset_json.exists():
        return "0.1"

    # If dataset.json exists but no schema_version, assume v0.1 (intermediate format)
    if dataset_json.exists():
        return "0.1"

    raise ValueError(
        f"Cannot determine dataset format version for directory: {dataset_dir}\n"
        f"Expected either:\n"
        f"  - v0.2: dataset.json with schema_version='0.2'\n"
        f"  - v0.1: vlm_system_prompt.txt or usd/dataset.json"
    )


# =============================================================================
# v0.2 Loading
# =============================================================================


def load_dataset_config_v02(dataset_dir: Path) -> DatasetConfig:
    """Load dataset.json (v0.2 format).

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        DatasetConfig instance

    Raises:
        FileNotFoundError: If dataset.json doesn't exist
        ValueError: If dataset.json is invalid
    """
    dataset_json = dataset_dir / "dataset.json"

    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json}")

    with open(dataset_json, encoding="utf-8") as f:
        data = json.load(f)

    return DatasetConfig(**data)


def load_dataset_entries_v02(
    dataset_dir: Path,
    config: DatasetConfig,
    entry_filter: Callable[[DatasetEntry], bool] | None = None,
) -> Iterator[DatasetEntry]:
    """Load dataset.jsonl entries (v0.2 format).

    Args:
        dataset_dir: Path to dataset directory
        config: Dataset configuration
        entry_filter: Optional filter function to apply to entries

    Yields:
        DatasetEntry instances

    Raises:
        FileNotFoundError: If dataset.jsonl doesn't exist
    """
    prims_file = dataset_dir / config.prims_file

    if not prims_file.exists():
        raise FileNotFoundError(f"Dataset entries file not found: {prims_file}")

    with open(prims_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                entry = DatasetEntry(**data)

                # Apply filter if provided
                if entry_filter is None or entry_filter(entry):
                    yield entry

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to validate entry at line {line_num}: {e}")
                continue


# =============================================================================
# v0.1 Loading with Conversion
# =============================================================================


def load_dataset_config_v01(dataset_dir: Path) -> DatasetConfig:
    """Load v0.1 dataset and convert to DatasetConfig (v0.2).

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        DatasetConfig instance (converted from v0.1)

    Raises:
        FileNotFoundError: If required v0.1 files don't exist
    """
    # Load system prompt
    vlm_prompt_file = dataset_dir / "vlm_system_prompt.txt"
    if not vlm_prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {vlm_prompt_file}")

    with open(vlm_prompt_file, encoding="utf-8") as f:
        system_prompt = f.read()

    # Try to load metadata from usd/dataset.json if it exists
    usd_dataset_json = dataset_dir / "usd" / "dataset.json"
    source_usd = None
    created = None
    num_entries = 0

    if usd_dataset_json.exists():
        with open(usd_dataset_json, encoding="utf-8") as f:
            usd_data = json.load(f)
            source_usd = usd_data.get("metadata", {}).get("source_usd")
            created = usd_data.get("metadata", {}).get("created")
            num_entries = usd_data.get("statistics", {}).get("total_prims", 0)

    # Count entries from dataset.jsonl if available
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    if dataset_jsonl.exists() and num_entries == 0:
        with open(dataset_jsonl, encoding="utf-8") as f:
            num_entries = sum(1 for line in f if line.strip())

    # Create v0.2 config with guessed values
    metadata = DatasetMetadata(
        created=created or "unknown",
        creator="unknown",  # v0.1 doesn't track this
        source_usd=source_usd,
        description="Converted from v0.1 format",
        num_entries=num_entries,
    )

    # Determine task type based on available files/context
    # Default to material_assignment as it's most common in v0.1
    task = TaskConfig(
        type="material_assignment",
        description="Material assignment task (converted from v0.1)",
    )

    # Create single prompt config
    prompt = PromptConfig(
        step_name="main",
        step_index=0,
        system_prompt=system_prompt,
    )

    inference = InferenceConfig(prompts=[prompt])

    return DatasetConfig(
        schema_version="0.2",
        metadata=metadata,
        task=task,
        inference=inference,
        prims_file="dataset.jsonl",
        usd_model_file="usd_model.json"
        if (dataset_dir / "usd_model.json").exists()
        or (dataset_dir / "usd" / "usd_model.json").exists()
        else None,
    )


def load_dataset_entries_v01(
    dataset_dir: Path,
    entry_filter: Callable[[DatasetEntry], bool] | None = None,
) -> Iterator[DatasetEntry]:
    """Load v0.1 dataset entries and convert to DatasetEntry (v0.2).

    Args:
        dataset_dir: Path to dataset directory
        entry_filter: Optional filter function to apply to entries

    Yields:
        DatasetEntry instances (converted from v0.1)

    Raises:
        FileNotFoundError: If dataset.jsonl doesn't exist
    """
    dataset_jsonl = dataset_dir / "dataset.jsonl"

    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_jsonl}")

    with open(dataset_jsonl, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                entry = convert_v01_entry_to_v02(data, dataset_dir)

                # Apply filter if provided
                if entry_filter is None or entry_filter(entry):
                    yield entry

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to convert entry at line {line_num}: {e}")
                continue


def convert_v01_entry_to_v02(
    v01_data: dict[str, Any], dataset_dir: Path
) -> DatasetEntry:
    """Convert a v0.1 dataset entry to v0.2 format.

    v0.1 format:
    {
        "id": "/prim/path",
        "text": "user prompt text",
        "images": ["path1.png", "path2.png"],
        "image_metadata": [
            {"path": "...", "view": "...", "camera": "...", "render_mode": "...", "vlm_prompt": "..."}
        ],
        "ground_truth": "material_name"
    }

    Args:
        v01_data: v0.1 entry dictionary
        dataset_dir: Dataset directory for path resolution

    Returns:
        DatasetEntry (v0.2 format)
    """
    # Extract ID
    entry_id = v01_data.get("id", "unknown")

    # Create source info
    source = SourceInfo(
        type="usd_prim",
        prim_path=entry_id if entry_id.startswith("/") else None,
    )

    # Extract user prompt (text field in v0.1)
    user_prompt = v01_data.get("text", "")

    # Convert images
    images: list[ImageObject] = []
    image_paths = v01_data.get("images", [])
    image_metadata_list = v01_data.get("image_metadata", [])

    # Create a lookup map for metadata
    metadata_by_path: dict[str, dict[str, Any]] = {}
    for meta in image_metadata_list:
        if "path" in meta:
            metadata_by_path[meta["path"]] = meta

    for img_path in image_paths:
        # Get metadata for this image
        meta_dict = metadata_by_path.get(img_path, {})

        # Extract vlm_prompt and other metadata
        metadata = ImageMetadata(
            view=meta_dict.get("view"),
            camera=meta_dict.get("camera"),
            render_mode=meta_dict.get("render_mode"),
            vlm_prompt=meta_dict.get("vlm_prompt"),
            width=meta_dict.get("width"),
            height=meta_dict.get("height"),
        )

        # Determine image type (render vs reference)
        # v0.1 doesn't distinguish, assume render for primary images
        img_type: Literal["render", "reference", "photo"] = "render"

        images.append(
            ImageObject(
                path=img_path,
                type=img_type,
                metadata=metadata,
            )
        )

    media = MediaConfig(images=images, reference_images=None)

    # Convert ground truth
    ground_truth = None
    if "ground_truth" in v01_data:
        gt_value = v01_data["ground_truth"]
        if gt_value:
            # v0.1 stores ground truth as string (material name)
            ground_truth = GroundTruth(
                material=str(gt_value),
                metadata=GroundTruthMetadata(source="v0.1_conversion"),
            )

    return DatasetEntry(
        id=entry_id,
        source=source,
        user_prompt=user_prompt,
        media=media,
        ground_truth=ground_truth,
    )


# =============================================================================
# Unified Loading Interface
# =============================================================================


def load_dataset_config(dataset_dir: Path | str) -> DatasetConfig:
    """Load dataset configuration with auto-detection.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        DatasetConfig instance (v0.2 format)

    Raises:
        FileNotFoundError: If dataset files don't exist
        ValueError: If dataset format is invalid
    """
    dataset_dir = Path(dataset_dir)

    version = detect_dataset_version(dataset_dir)

    if version == "0.2":
        return load_dataset_config_v02(dataset_dir)
    else:
        logger.warning(
            f"Loading v0.1 dataset from {dataset_dir}. "
            f"Consider migrating to v0.2 for better performance and features."
        )
        return load_dataset_config_v01(dataset_dir)


def load_dataset_entries(
    dataset_dir: Path | str,
    config: DatasetConfig | None = None,
    entry_filter: Callable[[DatasetEntry], bool] | None = None,
) -> Iterator[DatasetEntry]:
    """Load dataset entries with auto-detection.

    Args:
        dataset_dir: Path to dataset directory
        config: Optional pre-loaded config (will load if not provided)
        entry_filter: Optional filter function to apply to entries

    Yields:
        DatasetEntry instances (v0.2 format)

    Raises:
        FileNotFoundError: If dataset files don't exist
        ValueError: If dataset format is invalid
    """
    dataset_dir = Path(dataset_dir)

    # Load config if not provided
    if config is None:
        config = load_dataset_config(dataset_dir)

    version = detect_dataset_version(dataset_dir)

    if version == "0.2":
        yield from load_dataset_entries_v02(dataset_dir, config, entry_filter)
    else:
        yield from load_dataset_entries_v01(dataset_dir, entry_filter)


def load_dataset(
    dataset_dir: Path | str,
    entry_filter: Callable[[DatasetEntry], bool] | None = None,
) -> tuple[DatasetConfig, Iterator[DatasetEntry]]:
    """Load complete dataset with auto-detection.

    This is the main entry point for loading datasets.

    Args:
        dataset_dir: Path to dataset directory
        entry_filter: Optional filter function to apply to entries

    Returns:
        Tuple of (config, entries_iterator)

    Example:
        ```python
        config, entries = load_dataset("path/to/dataset")

        print(f"Task: {config.task.type}")
        print(f"Entries: {config.metadata.num_entries}")

        for entry in entries:
            print(f"Processing {entry.id}")
        ```
    """
    dataset_dir = Path(dataset_dir)

    config = load_dataset_config(dataset_dir)
    entries = load_dataset_entries(dataset_dir, config, entry_filter)

    return config, entries
