# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for optimizing USD files via REST API."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.config.s3 import WU_S3_BUCKET, WU_S3_PROFILE, WU_S3_REGION
from world_understanding.functions.graphics.scene_optimizer_nvcf import (
    optimize_usd_from_path,
)

logger = logging.getLogger(__name__)


class OptimizeUSDTask(Task):
    """Task to optimize USD file via REST API.

    This task calls an optimization REST API with the input USD and produces
    an optimized USD file along with metadata about the optimization.

    Input context keys:
        - input_usd_path: Path to the USD file to optimize
        - output_usd_path: Path where optimized USD will be saved
        - optimization_config: Optional dict with API-specific parameters:
            - scene_optimizer_settings: Dict with operation settings:
                - enable_deinstance: bool (default True)
                - enable_split_meshes: bool (default True)
                - enable_deduplicate: bool (default True)
                - deinstance: Dict with deinstance settings
                - split_meshes: Dict with split settings
                - deduplicate: Dict with deduplicate settings
                - generate_report, capture_stats, verbose, etc.
            - flatten_prototypes: bool (default True) - Fully flatten the USD stage
                before optimization. This converts abstract prototypes (over/class)
                to def, inlines all referenced geometry, removes prototype prims,
                and preserves stage metadata (upAxis, metersPerUnit) and shader
                connections.
            - poll_seconds: Optional int for NVCF polling timeout
            - api_key, base_url, s3_bucket, s3_region, s3_profile, timeout

    Output context keys:
        - optimized_usd_path: Path to the optimized USD file
        - optimization_metadata: Dict with optimization statistics/info
        - optimization_success: Boolean indicating success
        - original_usd_path: Path to the original (pre-optimization) USD file
    """

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Execute USD optimization synchronously.

        This is a wrapper that calls the async implementation.

        Args:
            context: Workflow context with input parameters
            object_store: Optional object store (not used)

        Returns:
            Updated context with optimization results

        Raises:
            ValueError: If required parameters are missing
            Exception: If optimization API call fails
        """
        import asyncio

        return asyncio.run(self.arun(context, object_store))

    async def arun(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Execute USD optimization asynchronously.

        This overrides the base Task.arun() to provide true async execution
        instead of running sync code in a thread pool.

        Args:
            context: Workflow context with input parameters
            object_store: Optional object store (not used)

        Returns:
            Updated context with optimization results

        Raises:
            ValueError: If required parameters are missing
            Exception: If optimization API call fails
        """
        listener = get_listener(context)

        # Get input parameters
        input_usd = context.get("input_usd_path")
        output_usd = context.get("output_usd_path")
        optimization_config = context.get("optimization_config", {})

        if not input_usd:
            raise ValueError("input_usd_path is required in context")
        if not output_usd:
            raise ValueError("output_usd_path is required in context")

        input_usd = Path(input_usd)
        output_usd = Path(output_usd)

        listener.info(f"Optimizing USD: {input_usd}")
        listener.info(f"Output will be saved to: {output_usd}")

        if optimization_config:
            if "scene_optimizer_settings" in optimization_config:
                settings = optimization_config["scene_optimizer_settings"]

                # Build enabled operations list (matches client)
                enabled_ops = self._get_enabled_operations(settings)

                # Log operations (matches client line 179)
                listener.info("Using scene optimizer settings:")
                listener.info(f"  Operations: {' -> '.join(enabled_ops)}")
                listener.info(
                    f"  Generate report: {settings.get('generate_report', True)}"
                )
                listener.info(f"  Capture stats: {settings.get('capture_stats', True)}")
                listener.info(f"  Verbose: {settings.get('verbose', False)}")
                listener.info(
                    f"  Wait for assets: {settings.get('wait_for_assets', False)}"
                )
                listener.info(
                    f"  Stage timeout: {settings.get('stage_timeout', 180.0)}s"
                )
                listener.info(
                    f"  Extract geom subset indices: {settings.get('extract_geom_subset_indices', True)}"
                )
            else:
                listener.info(f"Optimization config: {optimization_config}")

        try:
            # Flatten prototypes BEFORE optimization
            # This converts over/class to def, resolves all references, and removes prototypes
            # Default is True since optimize_usd is typically used with pre-flattened scenes
            flatten_prototypes = optimization_config.get("flatten_prototypes", True)

            # Track if we need to use a flattened input file
            actual_input = input_usd
            temp_flattened_input = None
            pre_converted_count = 0

            # Count original prims BEFORE any optimization
            from pxr import Usd, UsdGeom

            original_stage = Usd.Stage.Open(str(input_usd))
            original_prim_count = len(
                [p for p in original_stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            )
            listener.info(
                f"Original prim count (before optimization): {original_prim_count}"
            )

            if flatten_prototypes:
                from world_understanding.utils.usd.prim import (
                    convert_abstract_prototypes_to_def,
                    flatten_prototype_references,
                )

                listener.info("Flattening prototypes before optimization...")
                listener.info("  - Converting abstract prototypes (over/class) to def")
                listener.info("  - Resolving all references (inlining geometry)")
                listener.info("  - Removing prototype prims")

                stage = original_stage  # Reuse the already opened stage

                # Step 1: Convert over/class to def (so they become traversable)
                converted_count = convert_abstract_prototypes_to_def(stage)
                if converted_count > 0:
                    listener.info(
                        f"  Converted {converted_count} abstract prototype(s) to def"
                    )

                # Step 2: Flatten - resolve references and remove prototypes
                flattened_layer = flatten_prototype_references(stage)

                # Save to a temp file to avoid modifying original (.usd for consistency)
                temp_flattened_input = (
                    input_usd.parent / f"_flattened_{input_usd.stem}.usd"
                )
                flattened_layer.Export(str(temp_flattened_input))
                actual_input = temp_flattened_input
                pre_converted_count = converted_count

                listener.info(f"  Flattened USD saved to: {temp_flattened_input}")
            else:
                listener.info(
                    "Skipping prototype flattening (flatten_prototypes=False)"
                )

            # Determine backend: "local" (default) or "remote"
            backend = optimization_config.get("backend", "local")

            async def _run_nvcf() -> dict[str, Any]:
                """Run NVCF cloud backend."""
                listener.info("Calling NVCF optimization API...")
                return await optimize_usd_from_path(
                    input_path=actual_input,
                    output_path=output_usd,
                    api_key=optimization_config.get("api_key"),
                    base_url=optimization_config.get("base_url"),
                    s3_bucket=optimization_config.get("s3_bucket", WU_S3_BUCKET),
                    s3_region=optimization_config.get("s3_region", WU_S3_REGION),
                    s3_profile=optimization_config.get("s3_profile", WU_S3_PROFILE),
                    timeout=optimization_config.get("timeout", 3600),
                    max_retries=optimization_config.get("max_retries", 3),
                    optimization_config=optimization_config,
                )

            try:
                if backend == "local":
                    import asyncio

                    try:
                        from world_understanding.functions.graphics.scene_optimizer_local import (
                            optimize_usd_local,
                        )

                        listener.info("Running local Scene Optimizer backend...")
                        result = await asyncio.to_thread(
                            optimize_usd_local,
                            input_path=actual_input,
                            output_path=output_usd,
                            optimization_config=optimization_config,
                        )
                    except (RuntimeError, FileNotFoundError) as local_err:
                        # Auto-fallback to NVCF if local backend is unavailable.
                        # Covers: macOS (.so missing → RuntimeError), and
                        # environments where WU_SO_PYTHON binary doesn't exist
                        # (e.g. Python 3.13 distroless image has no python3.12 →
                        # subprocess.run raises FileNotFoundError).
                        err_str = str(local_err)
                        if isinstance(local_err, FileNotFoundError) or any(
                            m in err_str
                            for m in (
                                "WU_SO_PACKAGE_DIR",
                                "Scene Optimizer package directory missing",
                                "Scene Optimizer Core package not found",
                                "Scene Optimizer subprocess failed",
                            )
                        ):
                            if not (
                                os.getenv("NVCF_OPTIMIZER_FUNCTION_ID")
                                or os.getenv("OPTIMIZER_ENDPOINT")
                            ):
                                raise RuntimeError(
                                    "Scene optimization failed: local backend "
                                    f"unavailable ({local_err}) and no remote "
                                    "backend is configured. Fix one of: "
                                    "(a) run `./scripts/fetch_build_resources.sh` "
                                    "to fetch the public Scene Optimizer Core "
                                    "package, (b) set NVCF_OPTIMIZER_FUNCTION_ID "
                                    "(or OPTIMIZER_ENDPOINT) for the remote "
                                    "backend, or (c) set `optimize_usd.enabled: "
                                    "false` in your config."
                                ) from local_err
                            listener.warning(
                                f"Local SO backend unavailable ({local_err}), "
                                "falling back to NVCF"
                            )
                            result = await _run_nvcf()
                        else:
                            raise
                elif backend == "remote":
                    result = await _run_nvcf()
                else:
                    raise ValueError(
                        f"Invalid optimization backend: '{backend}'. "
                        "Must be 'local' or 'remote'."
                    )
            finally:
                # Clean up temp file if created (even on failure)
                if temp_flattened_input and temp_flattened_input.exists():
                    temp_flattened_input.unlink()
                    listener.debug(
                        f"Cleaned up temp flattened input: {temp_flattened_input}"
                    )

            if result.get("status") != "success":
                error_msg = result.get("error", "Unknown optimization error")
                raise RuntimeError(f"Optimization failed: {error_msg}")

            # Extract metadata from result
            metadata = {
                "optimization_time": result.get("optimization_time"),
                "stage_size_bytes": result.get("stage_size_bytes"),
                "operations_executed": result.get("operations_executed", []),
                "report": result.get("report", ""),
                "correspondence_map": result.get("correspondence_map", {}),
                "prototypes_converted_pre": pre_converted_count,
                "original_prim_count": original_prim_count,
            }

            # Update context with results
            context["optimized_usd_path"] = str(output_usd)
            context["optimization_metadata"] = metadata
            context["optimization_success"] = True
            # Save original path for restore_usd step
            context["original_usd_path"] = str(input_usd)
            # Save original prim count for stats reporting
            context["original_prim_count"] = original_prim_count

            listener.info("✓ USD optimization completed")
            listener.info(f"Optimized USD saved to: {output_usd}")

            # Log metadata
            if metadata:
                for key, value in metadata.items():
                    listener.debug(f"  {key}: {value}")

                # Add optimization config to metadata for reproducibility
                metadata["optimization_config"] = optimization_config

                metadata_path = output_usd.with_suffix(".metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                listener.info(f"Saved metadata to: {metadata_path}")

        except Exception as e:
            listener.error(f"USD optimization failed: {e}")
            context["optimization_success"] = False
            context["optimization_error"] = str(e)
            raise

        return context

    def _get_enabled_operations(self, settings: dict[str, Any]) -> list[str]:
        """Build list of enabled operations.

        Matches client_scene_optimizer.py lines 744-750.

        Args:
            settings: Scene optimizer settings dict with snake_case keys
                (enable_deinstance, enable_split_meshes, enable_deduplicate)

        Returns:
            List of enabled operation names
        """
        enabled_ops = []
        if settings.get("enable_deinstance", True):
            enabled_ops.append("deinstance")
        if settings.get("enable_split_meshes", True):
            enabled_ops.append("split")
        if settings.get("enable_deduplicate", True):
            enabled_ops.append("deduplicate")
        return enabled_ops
