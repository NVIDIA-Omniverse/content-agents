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
from world_understanding.utils.usd.asset_paths import (
    is_absolute_asset_path,
    is_relative_to,
    is_uri_asset_path,
    resolve_relative_asset_path_under_base,
)

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

    def _sanitize_layer_asset_path(
        self,
        asset_path: str,
        source_dir: Path,
        dest_dir: Path,
        listener,
        label: str,
    ) -> str | None:
        """Return a remapped local asset path, or None when it is unsafe."""
        if not asset_path:
            return asset_path
        try:
            if is_uri_asset_path(asset_path):
                raise ValueError(f"resolver URI schemes are not allowed: {asset_path}")
            if is_absolute_asset_path(asset_path):
                resolved = Path(asset_path).resolve()
                if not is_relative_to(resolved, source_dir.resolve()):
                    raise ValueError(
                        f"absolute asset path is outside source directory: {asset_path}"
                    )
            else:
                resolved = resolve_relative_asset_path_under_base(
                    asset_path, source_dir
                )
            return os.path.relpath(resolved, dest_dir).replace("\\", "/")
        except ValueError as e:
            listener.warning(f"Dropping unsafe {label}: {e}")
            return None

    def _sanitize_sublayer_paths(
        self,
        layer: Sdf.Layer,
        source_dir: Path,
        dest_dir: Path,
        listener,
    ) -> None:
        """Sanitize and remap root-layer sublayer paths."""
        updated_paths: list[str] = []
        for sublayer_path in list(layer.subLayerPaths):
            rel_path = self._sanitize_layer_asset_path(
                sublayer_path,
                source_dir,
                dest_dir,
                listener,
                "sublayer path",
            )
            if rel_path is None:
                continue
            if rel_path != sublayer_path:
                listener.info(f"Updated sublayer path: {sublayer_path} -> {rel_path}")
            updated_paths.append(rel_path)

        layer.subLayerPaths.clear()
        for path in updated_paths:
            layer.subLayerPaths.append(path)

    def _sanitize_composition_list(
        self,
        list_editor,
        source_dir: Path,
        dest_dir: Path,
        listener,
        kind: str,
    ) -> None:
        """Sanitize Sdf reference/payload list-editor item lists."""
        for field_name in (
            "explicitItems",
            "prependedItems",
            "appendedItems",
            "addedItems",
            "orderedItems",
        ):
            items = list(getattr(list_editor, field_name))
            if not items:
                continue
            sanitized = []
            for item in items:
                remapped = self._sanitize_layer_asset_path(
                    item.assetPath,
                    source_dir,
                    dest_dir,
                    listener,
                    f"{kind} asset path",
                )
                if remapped is None:
                    continue
                if kind == "reference":
                    sanitized.append(
                        Sdf.Reference(
                            remapped,
                            item.primPath,
                            item.layerOffset,
                            item.customData,
                        )
                    )
                else:
                    sanitized.append(
                        Sdf.Payload(remapped, item.primPath, item.layerOffset)
                    )
            setattr(list_editor, field_name, sanitized)

    def _sanitize_native_resolver_asset_paths(
        self,
        layer: Sdf.Layer,
        source_dir: Path,
        dest_dir: Path,
        listener,
    ) -> None:
        """Sanitize USD paths that native renderers could resolve directly."""
        self._sanitize_sublayer_paths(layer, source_dir, dest_dir, listener)

        prim_specs: list[Sdf.PrimSpec] = []

        def _collect_prim(path: Sdf.Path) -> None:
            obj = layer.GetObjectAtPath(path)
            if isinstance(obj, Sdf.PrimSpec):
                prim_specs.append(obj)

        layer.Traverse("/", _collect_prim)

        for prim_spec in prim_specs:
            self._sanitize_composition_list(
                prim_spec.referenceList,
                source_dir,
                dest_dir,
                listener,
                "reference",
            )
            self._sanitize_composition_list(
                prim_spec.payloadList,
                source_dir,
                dest_dir,
                listener,
                "payload",
            )

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

            # Export (copy) the layer to the destination
            if not source_layer.Export(str(dest_usd)):
                raise RuntimeError(f"Failed to export layer to {dest_usd}")

            # Now open the destination layer and update sublayer paths
            dest_layer = Sdf.Layer.FindOrOpen(str(dest_usd))
            if not dest_layer:
                raise RuntimeError(f"Failed to open destination layer: {dest_usd}")

            self._sanitize_native_resolver_asset_paths(
                dest_layer,
                source_usd.parent,
                dest_usd.parent,
                listener,
            )

            # Save the updated destination layer
            if not dest_layer.Save():
                raise RuntimeError(f"Failed to save updated layer: {dest_usd}")

            listener.info(
                f"✓ Copied USD with updated paths from {source_usd} to {dest_usd}"
            )

        except Exception as e:
            listener.warning(f"Failed to copy USD with path updates: {e}")
            import traceback

            listener.debug(traceback.format_exc())
            raise

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
