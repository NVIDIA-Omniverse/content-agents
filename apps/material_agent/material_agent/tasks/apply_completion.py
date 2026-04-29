# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for completing the material application workflow."""

import logging
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class ApplyCompletionTask(Task):
    """Task to mark the material application workflow as complete.

    This task serves as the final step in the apply workflow, marking
    the process as complete and providing summary information.

    Input context keys:
        - unique_materials: List of unique materials that were searched
        - matched_materials: Dictionary of materials and their matched paths
        - unresolved_materials: List of materials that could not be resolved
        - search_stats: Statistics from the USD Search process
        - resolved_materials: Dictionary of resolved material file paths
        - download_stats: Statistics from material resolution
        - materials_applied: Dictionary of materials applied to USD
        - assignment_stats: Statistics from USD material assignment
        - output_usd_path: Path where the USD file was saved
        - layer_only: Whether only a layer was created
        - rendered_image_path: Path to rendered image (if rendering was enabled)
        - rendering_skipped: Boolean indicating if rendering was skipped

    Output context keys:
        - application_complete: Boolean flag indicating completion
        - summary: Summary information about the application process
    """

    def __init__(self):
        """Initialize the apply completion task."""
        self.name = "ApplyCompletion"
        self.description = "Mark material application workflow as complete"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Complete the material application workflow.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with completion status
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        unique_materials = context.get("unique_materials", [])
        matched_materials = context.get("matched_materials", {})
        unresolved_materials = context.get("unresolved_materials", [])
        search_stats = context.get("search_stats", {})
        resolved_materials = context.get("resolved_materials", {})
        download_stats = context.get("download_stats", {})
        materials_applied = context.get("materials_applied", {})
        assignment_stats = context.get("assignment_stats", {})
        output_usd_path = context.get("output_usd_path")
        layer_only = context.get("layer_only", False)
        rendered_image_path = context.get("rendered_image_path")
        rendered_image_paths = context.get("rendered_image_paths", [])
        rendering_skipped = context.get("rendering_skipped", True)

        listener.info("Completing material application workflow")

        # Create summary
        summary = {
            "materials_identified": len(unique_materials),
            "materials_with_matches": len(
                [m for m, paths in matched_materials.items() if paths]
            ),
            "materials_unresolved": len(unresolved_materials),
            "total_matches_found": sum(
                len(paths) for paths in matched_materials.values()
            ),
            "search_success_rate": (
                search_stats.get("successful_queries", 0)
                / max(search_stats.get("total_queries", 1), 1)
                * 100
            ),
            "materials_resolved": len(resolved_materials),
            "paths_resolved": download_stats.get("resolved", 0),
            "materials_applied_to_usd": len(materials_applied),
            "prims_with_materials": assignment_stats.get("total_prims", 0),
            "output_mode": "Layer only" if layer_only else "Full stage",
            "output_path": str(output_usd_path) if output_usd_path else None,
            "rendered_image_path": str(rendered_image_path)
            if rendered_image_path
            else None,
            "rendered_image_paths": [str(p) for p in rendered_image_paths]
            if rendered_image_paths
            else [],
            "rendering_skipped": rendering_skipped,
        }

        # Log completion details
        listener.info("Material application workflow completed:")
        listener.info(f"  • Materials identified: {summary['materials_identified']}")
        listener.info(
            f"  • Materials with matches: {summary['materials_with_matches']}"
        )

        if unresolved_materials:
            listener.warning(
                f"  • Materials unresolved: {summary['materials_unresolved']}"
            )
            listener.warning("    Unresolved materials:")
            for material in unresolved_materials:
                listener.warning(f"      - {material}")

        listener.info(f"  • Total matches found: {summary['total_matches_found']}")
        listener.info(f"  • Search success rate: {summary['search_success_rate']:.1f}%")
        listener.info(f"  • Material paths resolved: {summary['paths_resolved']}")
        listener.info(
            f"  • Materials applied to USD: {summary['materials_applied_to_usd']}"
        )
        listener.info(f"  • Prims with materials: {summary['prims_with_materials']}")
        listener.info(f"  • Output mode: {summary['output_mode']}")

        if output_usd_path:
            listener.info(f"  • Output saved to: {output_usd_path}")

        # Log rendering information if enabled
        if not rendering_skipped and rendered_image_paths:
            if len(rendered_image_paths) == 1:
                listener.info(f"  • Rendered image: {rendered_image_paths[0]}")
            else:
                listener.info(
                    f"  • Rendered images ({len(rendered_image_paths)} views):"
                )
                for img_path in rendered_image_paths:
                    listener.info(f"    - {img_path}")

        # Update context
        context["application_complete"] = True
        context["summary"] = summary

        return context
