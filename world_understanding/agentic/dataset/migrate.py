# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Migration utilities for converting v0.1 datasets to v0.2 format.

This module provides functions to migrate existing v0.1 datasets to the unified
v0.2 schema with flat directory structure.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from world_understanding.agentic.dataset.loader import (
    detect_dataset_version,
    load_dataset_config,
    load_dataset_entries,
)
from world_understanding.agentic.dataset.schema import DatasetConfig, DatasetEntry

logger = logging.getLogger(__name__)


def migrate_dataset(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    in_place: bool = False,
    keep_intermediate: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Migrate a v0.1 dataset to v0.2 format.

    Args:
        input_dir: Path to v0.1 dataset directory
        output_dir: Path to output directory (if None and not in_place, uses input_dir_v02)
        in_place: Migrate in place (destructive, modifies original)
        keep_intermediate: Keep intermediate v0.1 files after migration
        dry_run: Don't actually write files, just validate and report

    Returns:
        Migration statistics dictionary

    Raises:
        ValueError: If dataset is already v0.2 or format is invalid
        FileNotFoundError: If input directory doesn't exist

    Example:
        ```python
        # Migrate to new directory
        stats = migrate_dataset("old_dataset/", "new_dataset_v02/")

        # Migrate in place
        stats = migrate_dataset("dataset/", in_place=True)

        # Dry run to check what would happen
        stats = migrate_dataset("dataset/", dry_run=True)
        ```
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Detect version
    try:
        version = detect_dataset_version(input_dir)
    except ValueError as e:
        raise ValueError(f"Cannot detect dataset version: {e}") from e

    if version == "0.2":
        raise ValueError(
            f"Dataset is already v0.2 format: {input_dir}\nNo migration needed."
        )

    logger.info(f"Migrating v{version} dataset: {input_dir}")

    # Determine output directory
    if in_place:
        output_dir = input_dir
        logger.warning(
            "⚠️  Migrating in place - original files will be modified/removed!"
        )
    elif output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_v02"
        logger.info(f"Output directory: {output_dir}")
    else:
        output_dir = Path(output_dir)
        logger.info(f"Output directory: {output_dir}")

    if dry_run:
        logger.info("🔍 DRY RUN - No files will be modified")

    # Load v0.1 dataset
    logger.info("Loading v0.1 dataset...")
    config = load_dataset_config(input_dir)
    entries = list(load_dataset_entries(input_dir, config))

    logger.info(f"Loaded {len(entries)} entries")

    # Create output directory
    if not dry_run and not in_place:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save v0.2 dataset.json
    stats = _save_dataset_config(config, output_dir, dry_run)

    # Save v0.2 dataset.jsonl
    entry_stats = _save_dataset_entries(entries, output_dir, dry_run)
    stats.update(entry_stats)

    # Copy/flatten renders directory
    render_stats = _migrate_renders(input_dir, output_dir, in_place, dry_run)
    stats.update(render_stats)

    # Copy usd_model.json if exists
    model_stats = _migrate_usd_model(input_dir, output_dir, in_place, dry_run)
    stats.update(model_stats)

    # Copy reference images if they exist
    ref_stats = _migrate_reference_images(input_dir, output_dir, in_place, dry_run)
    stats.update(ref_stats)

    # Clean up v0.1 files if not keeping intermediate
    if not keep_intermediate and not dry_run:
        cleanup_stats = _cleanup_v01_files(output_dir, in_place)
        stats.update(cleanup_stats)

    logger.info("✅ Migration complete!")
    logger.info("  Version: v0.1 → v0.2")
    logger.info(f"  Entries: {stats['num_entries']}")
    logger.info(f"  Images migrated: {stats['num_images_migrated']}")
    logger.info(f"  Output: {output_dir}")

    return stats


def _save_dataset_config(
    config: DatasetConfig, output_dir: Path, dry_run: bool
) -> dict[str, Any]:
    """Save dataset.json in v0.2 format.

    Args:
        config: DatasetConfig instance
        output_dir: Output directory
        dry_run: Don't actually write

    Returns:
        Statistics dict
    """
    dataset_json_path = output_dir / "dataset.json"

    if not dry_run:
        with open(dataset_json_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)
        logger.info(f"✓ Saved dataset.json: {dataset_json_path}")
    else:
        logger.info(f"[DRY RUN] Would save dataset.json: {dataset_json_path}")

    return {
        "config_path": str(dataset_json_path),
        "system_prompt_length": len(config.inference.prompts[0].system_prompt),
    }


def _save_dataset_entries(
    entries: list[DatasetEntry], output_dir: Path, dry_run: bool
) -> dict[str, Any]:
    """Save dataset.jsonl in v0.2 format.

    Args:
        entries: List of DatasetEntry instances
        output_dir: Output directory
        dry_run: Don't actually write

    Returns:
        Statistics dict
    """
    dataset_jsonl_path = output_dir / "dataset.jsonl"

    if not dry_run:
        with open(dataset_jsonl_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(entry.model_dump_json() + "\n")
        logger.info(f"✓ Saved dataset.jsonl: {dataset_jsonl_path}")
    else:
        logger.info(f"[DRY RUN] Would save dataset.jsonl: {dataset_jsonl_path}")

    # Calculate statistics
    total_images = sum(len(entry.media.images) for entry in entries)
    total_ref_images = sum(len(entry.media.reference_images or []) for entry in entries)

    return {
        "entries_path": str(dataset_jsonl_path),
        "num_entries": len(entries),
        "num_images": total_images,
        "num_reference_images": total_ref_images,
    }


def _migrate_renders(
    input_dir: Path, output_dir: Path, in_place: bool, dry_run: bool
) -> dict[str, Any]:
    """Migrate and flatten renders directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        in_place: Whether migrating in place
        dry_run: Don't actually copy

    Returns:
        Statistics dict
    """
    # Check for nested usd/renders/ structure
    usd_renders = input_dir / "usd" / "renders"
    top_renders = input_dir / "renders"
    output_renders = output_dir / "renders"

    images_migrated = 0

    # If usd/renders/ exists, flatten it
    if usd_renders.exists():
        image_files = list(usd_renders.rglob("*.png")) + list(
            usd_renders.rglob("*.jpg")
        )

        if not dry_run:
            output_renders.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            dest = output_renders / img_file.name

            if not dry_run:
                if in_place:
                    # Move file
                    shutil.move(str(img_file), str(dest))
                else:
                    # Copy file
                    shutil.copy2(img_file, dest)
            else:
                logger.debug(f"[DRY RUN] Would copy: {img_file} → {dest}")

            images_migrated += 1

        logger.info(f"✓ Flattened {images_migrated} images from usd/renders/")

    # If top-level renders/ exists and not in_place, copy it
    elif top_renders.exists() and not in_place:
        image_files = list(top_renders.glob("*.png")) + list(top_renders.glob("*.jpg"))

        if not dry_run:
            output_renders.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            dest = output_renders / img_file.name

            if not dry_run:
                shutil.copy2(img_file, dest)

            images_migrated += 1

        logger.info(f"✓ Copied {images_migrated} images from renders/")

    return {"num_images_migrated": images_migrated}


def _migrate_usd_model(
    input_dir: Path, output_dir: Path, in_place: bool, dry_run: bool
) -> dict[str, Any]:
    """Migrate usd_model.json file.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        in_place: Whether migrating in place
        dry_run: Don't actually copy

    Returns:
        Statistics dict
    """
    # Check both locations
    usd_model_nested = input_dir / "usd" / "usd_model.json"
    usd_model_top = input_dir / "usd_model.json"
    output_model = output_dir / "usd_model.json"

    model_migrated = False

    if usd_model_nested.exists():
        if not dry_run:
            if in_place:
                shutil.move(str(usd_model_nested), str(output_model))
            else:
                shutil.copy2(usd_model_nested, output_model)
        logger.info("✓ Migrated usd_model.json from usd/")
        model_migrated = True

    elif usd_model_top.exists() and not in_place:
        if not dry_run:
            shutil.copy2(usd_model_top, output_model)
        logger.info("✓ Copied usd_model.json")
        model_migrated = True

    return {"usd_model_migrated": model_migrated}


def _migrate_reference_images(
    input_dir: Path, output_dir: Path, in_place: bool, dry_run: bool
) -> dict[str, Any]:
    """Migrate reference images if they exist.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        in_place: Whether migrating in place
        dry_run: Don't actually copy

    Returns:
        Statistics dict
    """
    # Look for reference_image* files
    ref_images = list(input_dir.glob("reference_image*"))

    if not ref_images:
        return {"num_reference_images_migrated": 0}

    if not in_place and not dry_run:
        for ref_img in ref_images:
            dest = output_dir / ref_img.name
            shutil.copy2(ref_img, dest)

    logger.info(f"✓ Migrated {len(ref_images)} reference image(s)")

    return {"num_reference_images_migrated": len(ref_images)}


def _cleanup_v01_files(output_dir: Path, in_place: bool) -> dict[str, Any]:
    """Clean up v0.1 intermediate files.

    Args:
        output_dir: Output directory
        in_place: Whether migrating in place

    Returns:
        Statistics dict
    """
    files_to_remove = [
        output_dir / "vlm_system_prompt.txt",
        output_dir / "spec.txt",
        output_dir / "usd" / "prims.jsonl",
        output_dir / "usd" / "dataset.json",
    ]

    files_removed = 0

    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Removed: {file_path}")
            files_removed += 1

    # Remove empty usd/renders/ directory
    usd_renders = output_dir / "usd" / "renders"
    if usd_renders.exists():
        try:
            # Remove empty subdirectories recursively
            for item in sorted(usd_renders.rglob("*"), reverse=True):
                if item.is_dir() and not any(item.iterdir()):
                    item.rmdir()

            # Remove renders/ itself if empty
            if not any(usd_renders.iterdir()):
                usd_renders.rmdir()
                logger.debug(f"Removed empty: {usd_renders}")
        except OSError:
            pass

    # Remove empty usd/ directory
    usd_dir = output_dir / "usd"
    if usd_dir.exists():
        try:
            if not any(usd_dir.iterdir()):
                usd_dir.rmdir()
                logger.info("✓ Removed empty usd/ directory")
        except OSError:
            pass

    return {"files_cleaned_up": files_removed}


def migrate_datasets_batch(
    input_dirs: list[Path | str],
    output_parent_dir: Path | str | None = None,
    keep_intermediate: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Migrate multiple datasets in batch.

    Args:
        input_dirs: List of input dataset directories
        output_parent_dir: Parent directory for outputs (uses input parent if None)
        keep_intermediate: Keep intermediate files
        dry_run: Don't actually write

    Returns:
        Aggregated statistics dict

    Example:
        ```python
        datasets = [
            "data/examples/pcba/",
            "data/examples/ladder/",
        ]
        stats = migrate_datasets_batch(datasets)
        ```
    """
    results = []
    total_entries = 0
    total_images = 0
    failed = []

    for input_dir in input_dirs:
        input_path = Path(input_dir)

        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Migrating: {input_path}")
            logger.info(f"{'=' * 60}")

            # Determine output directory
            if output_parent_dir:
                output_dir = Path(output_parent_dir) / f"{input_path.name}_v02"
            else:
                output_dir = input_path.parent / f"{input_path.name}_v02"

            stats = migrate_dataset(
                input_dir=input_path,
                output_dir=output_dir,
                in_place=False,
                keep_intermediate=keep_intermediate,
                dry_run=dry_run,
            )

            results.append(
                {
                    "input": str(input_path),
                    "output": str(output_dir),
                    "status": "success",
                    "stats": stats,
                }
            )

            total_entries += stats.get("num_entries", 0)
            total_images += stats.get("num_images_migrated", 0)

        except Exception as e:
            logger.error(f"❌ Failed to migrate {input_path}: {e}")
            failed.append(str(input_path))
            results.append(
                {
                    "input": str(input_path),
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Batch Migration Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total datasets: {len(input_dirs)}")
    logger.info(f"Successful: {len(input_dirs) - len(failed)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total entries migrated: {total_entries}")
    logger.info(f"Total images migrated: {total_images}")

    if failed:
        logger.warning(f"Failed datasets: {failed}")

    return {
        "total_datasets": len(input_dirs),
        "successful": len(input_dirs) - len(failed),
        "failed": len(failed),
        "failed_datasets": failed,
        "total_entries": total_entries,
        "total_images": total_images,
        "results": results,
    }
