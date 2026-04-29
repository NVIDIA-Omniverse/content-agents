# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion task for iterative apply workflow."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from pxr import Sdf
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


class IterativeApplyCompletionTask(Task):
    """Complete the iterative apply workflow and copy final output.

    This task finalizes the iterative apply workflow by:
    1. Copying the final iteration's output to the specified output path
    2. Copying related files (flattened version, renders) if they exist
    3. Providing summary information

    Input context keys:
        - final_iteration: Results from the last iteration
        - iteration_count: Total number of iterations executed
        - termination_reason: Why iteration stopped
        - final_output_usd_path: Optional path to copy final output to
        - all_iteration_outputs: List of all iteration output paths

    Output context keys:
        - iterative_apply_complete: Boolean flag indicating completion
        - final_output_path: Path to the final output USD file
        - summary: Summary information about the iterative process
    """

    def __init__(self):
        """Initialize the iterative apply completion task."""
        self.name = "IterativeApplyCompletion"
        self.description = "Finalize iterative apply workflow"

    def _copy_usd_with_updated_paths(
        self, source_usd: Path, dest_usd: Path, listener
    ) -> None:
        """Copy USD file and update relative sublayer paths for new location.

        When copying a USD file with relative sublayer references to a new location,
        the relative paths need to be updated to remain valid from the new location.

        Args:
            source_usd: Source USD file path
            dest_usd: Destination USD file path
            listener: Event listener for logging
        """
        try:
            # Open the source layer (not stage, to avoid composition)
            source_layer = Sdf.Layer.FindOrOpen(str(source_usd))
            if not source_layer:
                raise RuntimeError(f"Failed to open source layer: {source_usd}")

            # Get sublayer paths from source
            sublayer_paths = list(source_layer.subLayerPaths)

            # Export (copy) the layer to the destination
            if not source_layer.Export(str(dest_usd)):
                raise RuntimeError(f"Failed to export layer to {dest_usd}")

            # Now open the destination layer and update sublayer paths
            dest_layer = Sdf.Layer.FindOrOpen(str(dest_usd))
            if not dest_layer:
                raise RuntimeError(f"Failed to open destination layer: {dest_usd}")

            # Update each sublayer path to be relative to new destination
            updated_paths = []
            for sublayer_path in sublayer_paths:
                # Skip if it's an absolute path or URL
                if (
                    sublayer_path.startswith(("/", "http://", "https://"))
                    or ":" in sublayer_path[:10]
                ):  # Windows absolute paths
                    updated_paths.append(sublayer_path)
                    listener.debug(f"Keeping absolute/URL path: {sublayer_path}")
                    continue

                # It's a relative path - resolve it from source, then relativize to dest
                source_dir = source_usd.parent
                # Resolve the sublayer path from source location
                abs_sublayer = (source_dir / sublayer_path).resolve()

                # Make it relative to destination
                dest_dir = dest_usd.parent
                rel_path = os.path.relpath(abs_sublayer, dest_dir)
                # Convert to forward slashes for USD
                rel_path = rel_path.replace("\\", "/")

                listener.info(
                    f"Updated sublayer path: {sublayer_path} -> {rel_path} "
                    f"(from {source_usd.name} to {dest_usd.name})"
                )
                updated_paths.append(rel_path)

            # Set updated sublayer paths in destination layer
            dest_layer.subLayerPaths.clear()
            for path in updated_paths:
                dest_layer.subLayerPaths.append(path)

            # Save the updated destination layer
            if not dest_layer.Save():
                raise RuntimeError(f"Failed to save updated layer: {dest_usd}")

            listener.info(
                f"✓ Copied USD with updated paths from {source_usd} to {dest_usd}"
            )

        except Exception as e:
            listener.warning(
                f"Failed to copy USD with path updates, falling back to simple copy: {e}"
            )
            import traceback

            listener.debug(traceback.format_exc())
            # Fallback to simple copy if something goes wrong
            shutil.copy2(source_usd, dest_usd)

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Complete the iterative apply workflow.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with completion status
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        iteration_count = context.get("iteration_count", 0)
        final_iteration = context.get("final_iteration") or {}
        termination_reason = context.get("termination_reason", "unknown")
        all_iteration_outputs = context.get("all_iteration_outputs", [])

        listener.info("Completing iterative apply workflow...")
        listener.info(f"  Total iterations: {iteration_count}")
        listener.info(f"  Termination reason: {termination_reason}")

        # Get the final iteration's output path
        final_iteration_output = None
        if all_iteration_outputs:
            final_iteration_output = Path(all_iteration_outputs[-1])
            listener.info(f"  Final iteration output: {final_iteration_output}")

        # Check if we should copy to a final output location
        final_output_usd_path = context.get("final_output_usd_path")

        if final_output_usd_path and final_iteration_output:
            final_output_usd_path = Path(final_output_usd_path)

            if final_iteration_output.exists():
                listener.info(f"Copying final output to: {final_output_usd_path}")

                # Ensure parent directory exists
                final_output_usd_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the main USD file with updated sublayer paths
                self._copy_usd_with_updated_paths(
                    final_iteration_output, final_output_usd_path, listener
                )

                # Also copy the flattened version if it exists
                final_iter_dir = final_iteration_output.parent
                flattened_file = (
                    final_iter_dir / f"{final_iteration_output.stem}_render_flat.usd"
                )
                if flattened_file.exists():
                    flattened_dest = (
                        final_output_usd_path.parent
                        / f"{final_output_usd_path.stem}_render_flat.usd"
                    )
                    shutil.copy2(flattened_file, flattened_dest)
                    listener.info(f"✓ Copied flattened USD to {flattened_dest}")

                # Copy renders directory if it exists
                renders_dir = final_iter_dir / "renders"
                if renders_dir.exists() and renders_dir.is_dir():
                    renders_dest = final_output_usd_path.parent / "renders"
                    if renders_dest.exists():
                        shutil.rmtree(renders_dest)
                    shutil.copytree(renders_dir, renders_dest)
                    listener.info(f"✓ Copied renders to {renders_dest}")

                # Set this as the final output path
                context["final_output_path"] = str(final_output_usd_path)
            else:
                listener.warning(
                    f"Final iteration output not found: {final_iteration_output}"
                )
                context["final_output_path"] = str(final_iteration_output)
        else:
            # No final output path specified, use the last iteration's output
            if final_iteration_output:
                context["final_output_path"] = str(final_iteration_output)
                listener.info(
                    f"No final output path specified, using last iteration output: "
                    f"{final_iteration_output}"
                )

        # Build summary
        summary = {
            "iteration_count": iteration_count,
            "termination_reason": termination_reason,
            "final_score": final_iteration.get("judge_score"),
            "final_materials_applied": final_iteration.get(
                "materials_applied_count", 0
            ),
            "final_prims_with_materials": final_iteration.get(
                "prims_with_materials", 0
            ),
            "all_iteration_outputs": all_iteration_outputs,
            "final_output_path": context.get("final_output_path"),
        }

        context["iterative_apply_complete"] = True
        context["summary"] = summary

        return context
