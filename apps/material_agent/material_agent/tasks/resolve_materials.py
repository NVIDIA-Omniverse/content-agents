# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for resolving material file paths for USD assignment."""

import logging
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.config.s3 import WU_S3_BUCKET, WU_S3_REGION

logger = logging.getLogger(__name__)


class ResolveMaterialFilesTask(Task):
    """Task to resolve material file paths for USD assignment.

    This task takes the matched materials and extracts their paths (S3, HTTPS, or local)
    to be used directly in USD without downloading. USD/Omniverse can handle remote URLs natively.

    Input context keys:
        - matched_materials: Dictionary mapping material names to path info lists

    Output context keys:
        - resolved_materials: Dictionary mapping material names to file paths/URLs
        - download_stats: Statistics about path resolution
    """

    def __init__(self):
        """Initialize the resolve material files task."""
        self.name = "ResolveMaterialFiles"
        self.description = "Extract material paths for direct USD assignment"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Extract material paths from matched materials.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with resolved material paths
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        matched_materials = context.get("matched_materials", {})

        if not matched_materials:
            listener.warning("No matched materials to resolve")
            context["resolved_materials"] = {}
            context["download_stats"] = {
                "resolved": 0,
                "failed": 0,
                "skipped": 0,
            }
            return context

        # Check if this is library-based mapping
        is_library_based = context.get("is_library_based_mapping", False)
        material_library_path = context.get("material_library_path")

        if is_library_based:
            listener.info(
                f"Resolving {len(matched_materials)} library-based material prim paths"
            )
            return self._resolve_library_materials(
                context, matched_materials, material_library_path, listener
            )

        listener.info(f"Resolving {len(matched_materials)} material paths")

        resolved_materials = {}
        resolved_count = 0
        failed_count = 0
        skipped_count = 0

        for material_name, path_infos in matched_materials.items():
            if not path_infos:
                listener.warning(f"No paths found for material '{material_name}'")
                skipped_count += 1
                continue

            # Use the first match (highest scoring)
            path_info = path_infos[0]

            if not isinstance(path_info, dict):
                listener.warning(
                    f"Unexpected path format for material '{material_name}': {path_info}"
                )
                skipped_count += 1
                continue

            # Get the path to use (prefer s3_path for full URIs, fallback to source_path)
            s3_path = path_info.get("s3_path")
            source_path = path_info.get("source_path")

            # Use whichever path is available
            material_path = s3_path or source_path

            if not material_path:
                listener.warning(f"No path available for material '{material_name}'")
                failed_count += 1
                continue

            # Convert S3 URIs to HTTPS URLs for USD/Omniverse compatibility
            if material_path.startswith("s3://"):
                # Parse: s3://bucket-name/path -> https://bucket-name.s3.region.amazonaws.com/path
                uri_parts = material_path[5:]  # Remove 's3://'
                if "/" in uri_parts:
                    bucket_name, path = uri_parts.split("/", 1)
                    # Use standard S3 HTTPS format
                    if WU_S3_BUCKET and WU_S3_BUCKET in bucket_name:
                        https_url = f"https://{bucket_name}.s3.{WU_S3_REGION}.amazonaws.com/{path}"
                    else:
                        # Generic format - USD might need region, but try generic first
                        https_url = f"https://{bucket_name}.s3.amazonaws.com/{path}"
                    material_path = https_url
                    listener.debug(f"Converted S3 URI to HTTPS: {material_path}")

            # Store the path (HTTPS URL, local path, or original path)
            resolved_materials[material_name] = material_path
            resolved_count += 1
            listener.info(f"Resolved '{material_name}' -> {material_path}")

        download_stats = {
            "resolved": resolved_count,
            "failed": failed_count,
            "skipped": skipped_count,
        }

        listener.info(
            f"Material path resolution complete: {resolved_count} resolved, "
            f"{failed_count} failed, {skipped_count} skipped"
        )

        # Update context
        context["resolved_materials"] = resolved_materials
        context["download_stats"] = download_stats

        return context

    def _resolve_library_materials(
        self,
        context: dict[str, Any],
        matched_materials: dict[str, list],
        material_library_path: str,
        listener,
    ) -> dict[str, Any]:
        """Resolve library-based material prim paths.

        For library materials, we just extract and pass through the prim paths
        since materials will be referenced from the library USD file.

        Args:
            context: Workflow context
            matched_materials: Dictionary mapping material names to path info lists
            material_library_path: Path to the material library USD file
            listener: Event listener for progress reporting

        Returns:
            Updated context with resolved library material prim paths
        """
        resolved_materials = {}
        resolved_count = 0
        failed_count = 0
        skipped_count = 0

        for material_name, path_infos in matched_materials.items():
            if not path_infos:
                listener.warning(f"No prim path found for material '{material_name}'")
                skipped_count += 1
                continue

            # Use the first match (should only be one for library materials)
            path_info = path_infos[0]

            if not isinstance(path_info, dict):
                listener.warning(
                    f"Unexpected path format for material '{material_name}': {path_info}"
                )
                skipped_count += 1
                continue

            # Get the prim path within the library
            prim_path = path_info.get("source_path")

            if not prim_path:
                listener.warning(
                    f"No prim path available for material '{material_name}'"
                )
                failed_count += 1
                continue

            # For library materials, we store the prim path as-is
            # The library path is stored separately in context
            resolved_materials[material_name] = prim_path
            resolved_count += 1
            listener.info(f"Resolved '{material_name}' -> prim path: {prim_path}")

        download_stats = {
            "resolved": resolved_count,
            "failed": failed_count,
            "skipped": skipped_count,
        }

        listener.info(
            f"Library material resolution complete: {resolved_count} resolved, "
            f"{failed_count} failed, {skipped_count} skipped"
        )

        # Keep library path in context for downstream tasks
        context["resolved_materials"] = resolved_materials
        context["download_stats"] = download_stats

        return context
