# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for applying resolved materials to USD prims."""

import logging
import os
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom, UsdShade
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.usd.material import (
    add_mdl_material,
    bind_material_to_prim,
)
from world_understanding.utils.usd.prim import nullify_material

logger = logging.getLogger(__name__)


class ApplyMaterialsToUSDTask(Task):
    """Task to apply resolved materials to USD prims.

    This task takes the resolved material files and applies them to USD prims
    based on the predictions. It supports two modes:
    1. Default: Create a new USD stage with everything (geometry + materials)
    2. Layer mode: Create only a sublayer with material bindings

    Input context keys:
        - input_usd_path: Path to the input USD file
        - output_usd_path: Path to save the output USD file
        - predictions_path: Path to the predictions file
        - resolved_materials: Dictionary mapping material names to local file paths
        - layer_only: Boolean flag to create only a material binding layer (default: False)
        - flatten_output: Boolean flag to flatten output (default: False)
                         When False, preserves references to material libraries
                         When True, creates self-contained flattened USD
        - skip_instance_check: Boolean flag to skip instance material traversal
                              (default: False). Set True for payload pipelines
                              where instances inherit materials via composition.

    Output context keys:
        - output_usd_path: Path where the USD file was saved
        - materials_applied: Dictionary of materials that were applied
        - assignment_stats: Statistics about material assignments
    """

    def __init__(self):
        """Initialize the apply materials to USD task."""
        self.name = "ApplyMaterialsToUSD"
        self.description = "Apply resolved materials to USD prims"

    def _create_material_on_stage(
        self,
        stage: Usd.Stage,
        material_name: str,
        material_path: str,
        output_usd_path: Path,
        path_prefix: str | None = None,
    ) -> tuple[str | None, bool]:
        """Create a material on the USD stage.

        Args:
            stage: USD stage to add material to
            material_name: Name of the material
            material_path: Path to the material file
            output_usd_path: Path to the output USD file (for relative path calculation)
            path_prefix: Optional path prefix for material (None for default, "" for root)

        Returns:
            Tuple of (material_prim_path, success)
        """
        try:
            # Sanitize material name for USD path
            sanitized_name = self._sanitize_material_name(material_name)

            # For MDL materials, use the proven add_mdl_material function
            if material_path.endswith(".mdl"):
                # Extract subIdentifier from the MDL filename (without .mdl extension)
                mdl_filename = Path(material_path).stem

                # Convert material path to be relative to the output USD file
                relative_material_path = self._make_path_relative_to_usd(
                    material_path, output_usd_path
                )

                # Add MDL material using proven utility
                stage, material_prim_path = add_mdl_material(
                    stage=stage,
                    material_name=sanitized_name,
                    source_asset_path=relative_material_path,
                    sub_identifier=mdl_filename,
                    path_prefix=path_prefix,
                    color=None,
                )

                self.listener.info(
                    f"Created material '{material_name}' at {material_prim_path}"
                )
                return material_prim_path, True
            else:
                # For non-MDL materials, create a basic UsdPreviewSurface
                # This is a fallback and should rarely be used
                self.listener.warning(
                    f"Material '{material_name}' is not an MDL file. "
                    f"Creating fallback UsdPreviewSurface material."
                )
                # Create materials scope if it doesn't exist
                materials_scope_path = "/Materials"
                if not stage.GetPrimAtPath(materials_scope_path):
                    UsdGeom.Scope.Define(stage, materials_scope_path)

                material_prim_path = f"{materials_scope_path}/{sanitized_name}"
                material = UsdShade.Material.Define(stage, material_prim_path)

                shader_path = f"{material_prim_path}/PreviewShader"
                shader = UsdShade.Shader.Define(stage, shader_path)
                shader.CreateIdAttr().Set("UsdPreviewSurface")

                # Set a default color based on material name
                color = self._get_material_color(material_name)
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    color
                )
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

                # Connect shader to material
                material.CreateSurfaceOutput().ConnectToSource(
                    shader.ConnectableAPI(), "surface"
                )

                self.listener.info(
                    f"Created fallback material '{material_name}' at {material_prim_path}"
                )
                return material_prim_path, True

        except Exception as e:
            self.listener.error(f"Failed to create material '{material_name}': {e}")
            return None, False

    def _sanitize_material_name(self, material_name: str) -> str:
        """Sanitize material name for use as USD prim name.

        Replaces problematic characters (spaces, slashes, dashes) with underscores
        to ensure the name is valid for USD material/shader names.

        Args:
            material_name: Original material name from predictions

        Returns:
            Sanitized material name safe for USD
        """
        # Replace spaces, forward slashes, backslashes, and dashes with underscores
        sanitized = material_name.replace(" ", "_")
        sanitized = sanitized.replace("/", "_")
        sanitized = sanitized.replace("\\", "_")
        sanitized = sanitized.replace("-", "_")
        return sanitized

    def _load_prim_material_mapping(self, predictions_path: Path) -> dict[str, str]:
        """Load prim-to-material mapping from predictions file.

        Args:
            predictions_path: Path to the predictions JSONL file

        Returns:
            Dictionary mapping prim paths to material names
        """
        import json

        prim_to_material = {}

        if not predictions_path or not Path(predictions_path).exists():
            self.listener.warning(f"Predictions file not found: {predictions_path}")
            return prim_to_material

        try:
            with open(predictions_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        prediction = json.loads(line)
                        prim_id = prediction.get("id")

                        # Extract material from various possible fields
                        material = None
                        if "materials" in prediction:
                            mat_data = prediction["materials"]
                            if isinstance(mat_data, dict):
                                material = mat_data.get("material")
                            elif isinstance(mat_data, str):
                                material = mat_data
                        elif "material" in prediction:
                            material = prediction["material"]

                        if prim_id and material:
                            prim_to_material[prim_id] = material
                            self.listener.debug(f"Mapped {prim_id} -> {material}")

                    except json.JSONDecodeError as e:
                        self.listener.warning(f"Failed to parse prediction line: {e}")
                        continue

        except Exception as e:
            self.listener.error(f"Failed to load predictions file: {e}")

        return prim_to_material

    def _make_path_relative_to_usd(
        self, material_path: str, output_usd_path: Path
    ) -> str:
        """Convert material path to be relative to the output USD file or use as-is for URLs.

        Args:
            material_path: Absolute/relative path to the material file, or remote URL
            output_usd_path: Path to the output USD file

        Returns:
            Path to material relative to the USD file (for local files) or absolute URL (for remote files)
        """
        # If it's a remote URL (http/https), return as-is
        if material_path.startswith(("http://", "https://")):
            self.listener.debug(f"Using remote URL as-is: {material_path}")
            return material_path

        try:
            # Convert both to absolute paths first
            material_abs = Path(material_path).resolve()
            usd_abs = Path(output_usd_path).resolve()

            # Get the directory containing the USD file
            usd_dir = usd_abs.parent

            # Compute relative path from USD directory to material file
            rel_path = os.path.relpath(material_abs, usd_dir)

            # Convert to forward slashes for USD
            rel_path = rel_path.replace("\\", "/")

            self.listener.info(
                f"Path relativization: material={material_path} -> abs={material_abs}, "
                f"usd={output_usd_path} -> abs={usd_abs}, usd_dir={usd_dir}, relative={rel_path}"
            )
            return rel_path
        except Exception as e:
            self.listener.warning(
                f"Failed to make path relative, using original: {material_path}. Error: {e}"
            )
            # Fall back to original path with forward slashes
            return material_path.replace("\\", "/")

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Apply materials to USD prims.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with applied materials
        """
        # Get event listener (or logger fallback) and store as instance variable
        # so helper methods can access it via self.listener
        listener = get_listener(context, logger_name=__name__)
        self.listener = listener

        input_usd_path = context.get("input_usd_path")
        output_usd_path = context.get("output_usd_path")
        resolved_materials = context.get("resolved_materials", {})
        layer_only = context.get("layer_only", False)
        flatten_output = context.get("flatten_output", True)
        predictions_path = context.get("predictions_path")
        is_library_based = context.get("is_library_based_mapping", False)
        material_library_path = context.get("material_library_path")
        skip_instance_check = context.get("skip_instance_check", False)

        if not input_usd_path:
            raise ValueError("No input USD path provided")

        if not output_usd_path:
            raise ValueError("No output USD path provided")

        if not resolved_materials:
            # Check if we have predictions but no resolved materials (material resolution failure)
            predictions_exist = (
                predictions_path
                and Path(predictions_path).exists()
                and Path(predictions_path).stat().st_size > 0
            )
            if predictions_exist:
                # This is a critical failure - we have predictions but couldn't resolve any materials
                error_msg = (
                    "Critical error: Material resolution failed. "
                    "VLM predicted materials but none could be resolved from the material library. "
                    "This usually means:\n"
                    "  1. VLM returned material names that don't match the library (check system prompt)\n"
                    "  2. Material library is missing or incorrectly configured\n"
                    "  3. Material names in predictions don't match available materials\n"
                    "Check the MaterialRetrieval task logs for details."
                )
                self.listener.error(error_msg)
                raise ValueError(error_msg)
            else:
                # No predictions at all - might be expected in some workflows
                self.listener.warning(
                    "No resolved materials to apply (no predictions found)"
                )
                context["materials_applied"] = {}
                context["assignment_stats"] = {
                    "total_prims": 0,
                    "materials_applied": 0,
                    "materials_created": 0,
                    "failed": 0,
                }
                return context

        self.listener.info(f"Applying {len(resolved_materials)} materials to USD")
        self.listener.info(f"Mode: {'Layer only' if layer_only else 'Full stage'}")
        if not layer_only:
            self.listener.info(
                f"Flatten: {'Yes (self-contained)' if flatten_output else 'No (preserves references)'}"
            )

        # Load prim-to-material mapping from predictions
        prim_to_material = self._load_prim_material_mapping(predictions_path)
        self.listener.info(f"Loaded {len(prim_to_material)} prim-to-material mappings")

        # Statistics tracking
        materials_applied = {}
        materials_created_count = 0
        prims_with_materials = 0
        failed_count = 0

        try:
            if layer_only:
                # Create only a material binding layer
                stage, materials_applied, stats = self._create_material_layer(
                    input_usd_path,
                    output_usd_path,
                    resolved_materials,
                    prim_to_material,
                    is_library_based,
                    material_library_path,
                    skip_instance_check=skip_instance_check,
                )
            else:
                # Create a complete new stage with materials
                stage, materials_applied, stats = self._create_full_stage(
                    input_usd_path,
                    output_usd_path,
                    resolved_materials,
                    prim_to_material,
                    is_library_based,
                    material_library_path,
                    flatten_output,
                    skip_instance_check=skip_instance_check,
                )

            materials_created_count = stats["materials_created"]
            prims_with_materials = stats["prims_with_materials"]
            failed_count = stats.get("failed", 0)

            # Save the stage (skip if already saved during flattening)
            if not (not layer_only and flatten_output):
                self.listener.info(f"Saving USD to {output_usd_path}")
                stage.GetRootLayer().Export(str(output_usd_path))
            else:
                self.listener.info(
                    f"USD already saved during flattening: {output_usd_path}"
                )

        except Exception as e:
            self.listener.error(f"Failed to apply materials to USD: {e}")
            failed_count += 1

        # Calculate statistics
        assignment_stats = {
            "total_prims": prims_with_materials,
            "materials_applied": len(materials_applied),
            "materials_created": materials_created_count,
            "failed": failed_count,
        }

        self.listener.info(
            f"Material application completed: "
            f"{prims_with_materials} prims updated, "
            f"{materials_created_count} materials created, "
            f"{failed_count} failed"
        )

        # Update context
        context["output_usd_path"] = output_usd_path
        context["materials_applied"] = materials_applied
        context["assignment_stats"] = assignment_stats

        return context

    def _apply_materials_to_instances(
        self,
        stage: Usd.Stage,
        prim_to_material: dict[str, str],
        materials_applied: dict[str, str],
    ) -> dict[str, int]:
        """Apply materials to instance prims by looking up their master's material.

        This handles instances that don't have direct predictions by finding
        their master/prototype prim and applying the master's predicted material.

        Args:
            stage: USD stage
            prim_to_material: Dictionary mapping prim paths to material names from predictions
            materials_applied: Dictionary mapping material names to material prim paths

        Returns:
            Statistics dictionary with counts
        """
        self.listener.info("Checking for instances without predictions...")
        instances_found = 0
        instances_applied = 0
        instances_skipped = 0

        # Traverse ALL prims in the stage to find instances
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())

            # Skip if already has a material assignment from predictions
            if prim_path in prim_to_material:
                continue

            # Check if this is an instance
            if not prim.IsInstance():
                continue

            instances_found += 1

            # Get the instance's master/prototype
            master = prim.GetPrototype()
            if not master or not master.IsValid():
                self.listener.debug(f"Instance {prim_path} has no valid prototype")
                instances_skipped += 1
                continue

            master_path = str(master.GetPath())

            # Check if master has a material prediction
            master_material = prim_to_material.get(master_path)
            if not master_material:
                self.listener.debug(
                    f"Master {master_path} has no prediction for instance {prim_path}"
                )
                instances_skipped += 1
                continue

            # Get material prim path
            material_prim_path = materials_applied.get(master_material)
            if not material_prim_path:
                self.listener.warning(
                    f"Material '{master_material}' not available for instance {prim_path}"
                )
                instances_skipped += 1
                continue

            # Nullify existing material and apply master's material
            try:
                nullify_material(prim)
                bind_material_to_prim(
                    stage=stage,
                    material_path=material_prim_path,
                    prim_path=prim_path,
                    binding_strength=UsdShade.Tokens.weakerThanDescendants,
                )
                instances_applied += 1
                self.listener.debug(
                    f"Applied {master_material} to instance {prim_path} "
                    f"(from master {master_path})"
                )
            except Exception as e:
                self.listener.warning(
                    f"Failed to apply material to instance {prim_path}: {e}"
                )
                instances_skipped += 1
                continue

        if instances_found > 0:
            self.listener.info(
                f"Instance materials: {instances_applied} applied, "
                f"{instances_skipped} skipped ({instances_found} total instances)"
            )

        return {
            "instances_found": instances_found,
            "instances_applied": instances_applied,
            "instances_skipped": instances_skipped,
        }

    def _copy_library_materials(
        self,
        stage: Usd.Stage,
        library_path: str,
        output_usd_path: Path,
        resolved_materials: dict[str, str],
        default_prim_name: str = "",
    ) -> tuple[Usd.Stage, dict]:
        """Copy only the used materials from a library into the output stage.

        Instead of sublayering the entire library (which includes all materials
        and breaks texture paths when flattened), this copies only the materials
        that are actually used and remaps their asset paths (textures, MDL files)
        to be relative to the output USD.

        Material paths from the library (e.g. /World/Looks/Iron) are remapped
        to sit under the asset's default prim (e.g. /MyAsset/Looks/Iron) so
        the output doesn't introduce extra root prims like /World.

        Args:
            stage: Output USD stage to copy materials into
            library_path: Path to the material library USD file
            output_usd_path: Path to the output USD file
            resolved_materials: Dict mapping material name -> prim path in library
            default_prim_name: Name of the default prim to place materials under

        Returns:
            Tuple of (stage, materials_applied dict)
        """
        materials_applied = {}

        try:
            # Validate library path exists before attempting to open
            if not Path(library_path).exists():
                self.listener.error(f"Material library file not found: {library_path}")
                return stage, materials_applied

            # Open library layer (read-only, no sublayering)
            library_layer = Sdf.Layer.FindOrOpen(str(library_path))
            if not library_layer:
                self.listener.error(f"Failed to open material library: {library_path}")
                return stage, materials_applied

            output_layer = stage.GetRootLayer()

            # Compute directory paths for asset path remapping
            library_dir = Path(library_path).resolve().parent
            output_dir = Path(output_usd_path).resolve().parent

            # Remap library material paths to sit under the asset's
            # default prim instead of the library's own root (e.g.
            # /World/Looks/Iron -> /MyAsset/Looks/Iron).  This avoids
            # creating extra root prims like /World in the output.
            def _remap_target(lib_path: str) -> str:
                if not default_prim_name:
                    return lib_path
                parts = lib_path.strip("/").split("/")
                # Library paths typically look like /World/Looks/MatName
                # or /RootPrim/Looks/MatName.  Replace the first component
                # (the library's root) with the asset's default prim.
                if len(parts) >= 2:
                    parts[0] = default_prim_name
                return "/" + "/".join(parts)

            # Build remapped target paths
            target_materials: dict[str, tuple[str, str]] = {}
            for material_name, lib_path in resolved_materials.items():
                target_path = _remap_target(lib_path)
                target_materials[material_name] = (lib_path, target_path)
                if target_path != lib_path:
                    self.listener.debug(
                        f"Remapped material path: {lib_path} -> {target_path}"
                    )

            # Ensure parent prim hierarchy exists for remapped targets
            parent_paths: set[str] = set()
            for _, target_path in target_materials.values():
                path = Sdf.Path(target_path)
                parent = path.GetParentPath()
                while parent != Sdf.Path.absoluteRootPath:
                    parent_paths.add(str(parent))
                    parent = parent.GetParentPath()

            for parent_path in sorted(parent_paths):
                if not output_layer.GetPrimAtPath(parent_path):
                    Sdf.CreatePrimInLayer(output_layer, parent_path)
                # Ensure parent prims are 'def' (not 'over') so they are
                # traversable by stage.Traverse() and visible to renderers
                prim_spec = output_layer.GetPrimAtPath(parent_path)
                if prim_spec and prim_spec.specifier != Sdf.SpecifierDef:
                    prim_spec.specifier = Sdf.SpecifierDef
                    self.listener.debug(f"Created parent prim: {parent_path}")

            # Copy each used material from library to output
            for material_name, (lib_path, target_path) in target_materials.items():
                source_spec = library_layer.GetPrimAtPath(lib_path)
                if not source_spec:
                    self.listener.error(
                        f"Material prim not found in library for "
                        f"'{material_name}': {lib_path}"
                    )
                    continue

                success = Sdf.CopySpec(
                    library_layer,
                    Sdf.Path(lib_path),
                    output_layer,
                    Sdf.Path(target_path),
                )

                if success:
                    self._remap_asset_paths_in_prim(
                        output_layer,
                        Sdf.Path(target_path),
                        library_dir,
                        output_dir,
                    )
                    materials_applied[material_name] = target_path
                    self.listener.info(
                        f"Copied material '{material_name}' to {target_path}"
                    )
                else:
                    self.listener.error(
                        f"Failed to copy material '{material_name}' from library"
                    )

            # Save and reopen: Sdf.CopySpec operates at the layer level,
            # so the Usd.Stage's composition cache is stale. Reopening
            # ensures material bindings and prim traversal reflect the
            # newly copied specs.
            stage.Save()
            stage = Usd.Stage.Open(str(output_usd_path))
            if not stage:
                raise RuntimeError(
                    f"Failed to reopen stage after saving: {output_usd_path}"
                )

            self.listener.info(
                f"✓ Copied {len(materials_applied)} materials from library "
                f"(out of {len(resolved_materials)} requested)"
            )

        except Exception as e:
            self.listener.error(f"Failed to copy library materials: {e}")
            import traceback

            self.listener.debug(traceback.format_exc())

        return stage, materials_applied

    def _fix_stale_default_prim(self, stage: Usd.Stage, original_name: str) -> None:
        """Detect and correct a stale defaultPrim after composition.

        The NVCF optimizer may wrap content under a new root (e.g. /World),
        making the original defaultPrim name stale. This detects the mismatch
        and updates defaultPrim to the actual root prim.

        Args:
            stage: USD stage to check
            original_name: The original defaultPrim name from the input
        """
        dp = stage.GetDefaultPrim()
        if not dp.IsValid():
            root_children = list(stage.GetPseudoRoot().GetChildren())
            if root_children:
                actual_root_name = root_children[0].GetName()
                stage.GetRootLayer().defaultPrim = actual_root_name
                self.listener.warning(
                    f"Default prim '{original_name}' not found in composed "
                    f"stage. Updated to actual root prim: /{actual_root_name}"
                )

    def _remap_asset_paths_in_prim(
        self,
        layer: Sdf.Layer,
        prim_path: Sdf.Path,
        source_dir: Path,
        target_dir: Path,
    ) -> None:
        """Remap all SdfAssetPath values in a prim and its descendants.

        Converts relative paths that were relative to source_dir to be
        relative to target_dir instead.

        Args:
            layer: The layer containing the prim specs
            prim_path: Path to the prim to process
            source_dir: Directory the original paths were relative to
            target_dir: Directory the new paths should be relative to
        """
        prim_spec = layer.GetPrimAtPath(prim_path)
        if not prim_spec:
            return

        # Process attributes on this prim
        for attr_name in list(prim_spec.attributes.keys()):
            attr_spec = prim_spec.attributes[attr_name]
            value = attr_spec.default
            if isinstance(value, Sdf.AssetPath):
                new_path = self._remap_single_asset_path(
                    value.path, source_dir, target_dir
                )
                if new_path != value.path:
                    attr_spec.default = Sdf.AssetPath(new_path)
            elif isinstance(value, Sdf.AssetPathArray):
                new_arr = Sdf.AssetPathArray(
                    [
                        Sdf.AssetPath(
                            self._remap_single_asset_path(
                                ap.path, source_dir, target_dir
                            )
                        )
                        for ap in value
                    ]
                )
                if new_arr != value:
                    attr_spec.default = new_arr

        # Recurse into children
        for child_spec in prim_spec.nameChildren:
            self._remap_asset_paths_in_prim(
                layer,
                prim_path.AppendChild(child_spec.name),
                source_dir,
                target_dir,
            )

    def _remap_single_asset_path(
        self,
        path_str: str,
        source_dir: Path,
        target_dir: Path,
    ) -> str:
        """Remap a single asset path from source_dir-relative to target_dir-relative.

        Args:
            path_str: The original path string
            source_dir: Directory the path was relative to
            target_dir: Directory the path should be relative to

        Returns:
            The remapped path string
        """
        if not path_str:
            return path_str

        # Skip any URI with a scheme (http, https, s3, omniverse, file, etc.)
        if "://" in path_str:
            return path_str

        # Skip already absolute paths
        if os.path.isabs(path_str):
            return path_str

        # Resolve the relative path against the source directory
        abs_path = (source_dir / path_str).resolve()

        # Compute new relative path from target directory
        try:
            new_rel = os.path.relpath(abs_path, target_dir)
        except ValueError:
            # Cross-drive paths on Windows can't be made relative
            return abs_path.as_posix()

        # Use forward slashes for USD compatibility
        return new_rel.replace("\\", "/")

    def _create_full_stage(
        self,
        input_usd_path: Path,
        output_usd_path: Path,
        resolved_materials: dict,
        prim_to_material: dict,
        is_library_based: bool = False,
        material_library_path: str | None = None,
        flatten_output: bool = False,
        skip_instance_check: bool = False,
    ) -> tuple[Usd.Stage, dict, dict]:
        """Create a complete new USD stage with materials applied.

        Args:
            input_usd_path: Path to input USD file
            output_usd_path: Path for output USD file
            resolved_materials: Dictionary of resolved material paths
            prim_to_material: Dictionary mapping prim paths to material names
            is_library_based: Whether materials are from a library (default: False)
            material_library_path: Path to material library USD file (optional)
            flatten_output: Whether to flatten the output stage (default: False)
                           When False, preserves references to material libraries
                           When True, creates a self-contained flattened USD

        Returns:
            Tuple of (stage, materials_applied, statistics)
        """
        self.listener.info(f"Creating full stage from {input_usd_path}")

        # Always start by creating a new stage that references the input
        # We'll flatten at the END if requested, after all materials are applied
        self.listener.info(
            "Creating new stage with reference to input"
            + (
                " (will flatten after materials applied)"
                if flatten_output
                else " (preserving composition)"
            )
        )

        # Open input stage to get up axis and default prim before creating output
        input_stage = Usd.Stage.Open(str(input_usd_path))
        if not input_stage:
            raise RuntimeError(f"Failed to open input USD: {input_usd_path}")
        original_up_axis = UsdGeom.GetStageUpAxis(input_stage)
        original_meters_per_unit = UsdGeom.GetStageMetersPerUnit(input_stage)
        self.listener.info(f"Original USD up axis: {original_up_axis}")
        self.listener.info(f"Original USD metersPerUnit: {original_meters_per_unit}")

        # Read defaultPrim from the input's root layer (non-composable metadata)
        input_default_prim = input_stage.GetRootLayer().defaultPrim
        self.listener.info(
            f"Original USD default prim: {input_default_prim or '(none)'}"
        )

        output_stage = Usd.Stage.CreateNew(str(output_usd_path))

        # Set stage metrics to match the input before adding sublayers
        UsdGeom.SetStageUpAxis(output_stage, original_up_axis)
        UsdGeom.SetStageMetersPerUnit(output_stage, original_meters_per_unit)
        self.listener.info(
            f"Set output USD up axis to: {original_up_axis}, "
            f"metersPerUnit to: {original_meters_per_unit}"
        )

        # Preserve defaultPrim from input (non-composable, must be on root layer)
        if input_default_prim:
            output_stage.GetRootLayer().defaultPrim = input_default_prim
            self.listener.info(f"Set output USD default prim to: {input_default_prim}")

        # Add the input USD as a sublayer to preserve all content
        output_stage.GetRootLayer().subLayerPaths.append(str(input_usd_path))

        # Save and reload to ensure composition is complete
        output_stage.Save()
        output_stage = Usd.Stage.Open(str(output_usd_path))

        # After composition, verify the default prim is valid.
        self._fix_stale_default_prim(output_stage, input_default_prim)

        # Apply materials to the new stage
        materials_applied = {}
        materials_created_count = 0
        prims_with_materials = 0

        if is_library_based and material_library_path:
            # Library-based: Copy only used materials (not the entire library)
            self.listener.info(
                "Using library-based materials - copying used materials only"
            )
            output_stage, materials_applied = self._copy_library_materials(
                output_stage,
                material_library_path,
                output_usd_path,
                resolved_materials,
                default_prim_name=input_default_prim or "",
            )
            materials_created_count = len(materials_applied)
        else:
            # File-based: Create materials for each resolved material using proven utility functions
            for material_name, material_path in resolved_materials.items():
                material_prim_path, success = self._create_material_on_stage(
                    stage=output_stage,
                    material_name=material_name,
                    material_path=material_path,
                    output_usd_path=output_usd_path,
                    path_prefix=None,  # Will use DefaultPrim/Looks
                )

                if success:
                    materials_applied[material_name] = material_prim_path
                    materials_created_count += 1

        # Apply materials to prims based on predictions mapping
        instance_proxies_skipped = 0
        for prim_path, material_name in prim_to_material.items():
            # Get the prim from the stage
            prim = output_stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                self.listener.warning(f"Prim not found in stage: {prim_path}")
                continue

            # Instance proxies are READ-ONLY in USD - cannot author properties to them
            # Skip them here; they may be handled later via prototype material propagation
            if prim.IsInstanceProxy():
                instance_proxies_skipped += 1
                self.listener.debug(
                    f"Skipping instance proxy {prim_path} - will inherit material from prototype"
                )
                continue

            # Find the corresponding material in our applied materials
            material_prim_path = materials_applied.get(material_name)
            if not material_prim_path:
                self.listener.warning(
                    f"Material '{material_name}' not found in applied materials for prim {prim_path}"
                )
                continue

            # Nullify existing material bindings and display colors
            try:
                nullify_material(prim)
            except Exception as e:
                self.listener.warning(
                    f"Failed to nullify material on prim {prim_path}: {e}"
                )

            # Clear material bindings on GeomSubset children so they don't override
            # the parent binding (USD uses weakerThanDescendants by default which
            # means descendant GeomSubset bindings would otherwise win).
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Subset):
                    UsdShade.MaterialBindingAPI(child).UnbindAllBindings()

            # Bind the new material
            try:
                bind_material_to_prim(
                    stage=output_stage,
                    material_path=material_prim_path,
                    prim_path=prim_path,
                )
                prims_with_materials += 1
                self.listener.info(
                    f"Bound material '{material_name}' to prim {prim_path}"
                )
            except Exception as e:
                self.listener.warning(
                    f"Failed to bind material '{material_name}' to prim {prim_path}: {e}"
                )

        if instance_proxies_skipped > 0:
            self.listener.info(
                f"Skipped {instance_proxies_skipped} instance proxy prims (read-only)"
            )

        # Apply materials to instances by looking up their master's material
        if skip_instance_check:
            self.listener.info("Skipping instance material check (payload mode)")
            instance_stats = {
                "instances_found": 0,
                "instances_applied": 0,
                "instances_skipped": 0,
            }
        else:
            instance_stats = self._apply_materials_to_instances(
                output_stage, prim_to_material, materials_applied
            )

        # If flatten_output is requested, flatten the entire composed stage now
        # This happens AFTER all materials and libraries are composed
        if flatten_output:
            self.listener.info(
                "Flattening composed stage (resolving all sublayers and references)"
            )
            # Preserve stage metrics before flattening (Flatten() doesn't keep these)
            original_up_axis = UsdGeom.GetStageUpAxis(output_stage)
            original_meters_per_unit = UsdGeom.GetStageMetersPerUnit(output_stage)

            # Save the stage first to ensure all edits are written
            output_stage.Save()

            # Flatten the fully composed stage
            flattened_layer = output_stage.Flatten()

            # Export the flattened layer, overwriting the output file
            flattened_layer.Export(str(output_usd_path))

            # Reload and restore stage metrics
            output_stage = Usd.Stage.Open(str(output_usd_path))
            UsdGeom.SetStageUpAxis(output_stage, original_up_axis)
            UsdGeom.SetStageMetersPerUnit(output_stage, original_meters_per_unit)
            output_stage.Save()
            self.listener.info(
                "✓ Stage flattened - output is now self-contained with no external references"
            )

        stats = {
            "materials_created": materials_created_count,
            "prims_with_materials": prims_with_materials
            + instance_stats["instances_applied"],
            "instances_applied": instance_stats["instances_applied"],
            "instances_skipped": instance_stats["instances_skipped"],
        }

        return output_stage, materials_applied, stats

    def _create_material_layer(
        self,
        input_usd_path: Path,
        output_usd_path: Path,
        resolved_materials: dict,
        prim_to_material: dict,
        is_library_based: bool = False,
        material_library_path: str | None = None,
        skip_instance_check: bool = False,
    ) -> tuple[Usd.Stage, dict, dict]:
        """Create only a material binding layer that can be composed over the input.

        Args:
            input_usd_path: Path to input USD file
            output_usd_path: Path for output USD layer file
            resolved_materials: Dictionary of resolved material paths
            prim_to_material: Dictionary mapping prim paths to material names

        Returns:
            Tuple of (stage, materials_applied, statistics)
        """
        self.listener.info(f"Creating material binding layer for {input_usd_path}")

        # Open input stage to get up axis and default prim before creating output
        input_stage = Usd.Stage.Open(str(input_usd_path))
        if not input_stage:
            raise RuntimeError(f"Failed to open input USD: {input_usd_path}")
        original_up_axis = UsdGeom.GetStageUpAxis(input_stage)
        original_meters_per_unit = UsdGeom.GetStageMetersPerUnit(input_stage)
        self.listener.info(f"Original USD up axis: {original_up_axis}")
        self.listener.info(f"Original USD metersPerUnit: {original_meters_per_unit}")

        # Read defaultPrim from the input's root layer (non-composable metadata)
        input_default_prim = input_stage.GetRootLayer().defaultPrim
        self.listener.info(
            f"Original USD default prim: {input_default_prim or '(none)'}"
        )

        # Create a new stage for the material layer
        stage = Usd.Stage.CreateNew(str(output_usd_path))

        # Set stage metrics to match the input before adding sublayers
        UsdGeom.SetStageUpAxis(stage, original_up_axis)
        UsdGeom.SetStageMetersPerUnit(stage, original_meters_per_unit)
        self.listener.info(
            f"Set output USD up axis to: {original_up_axis}, "
            f"metersPerUnit to: {original_meters_per_unit}"
        )

        # Preserve defaultPrim from input (non-composable, must be on root layer)
        if input_default_prim:
            stage.GetRootLayer().defaultPrim = input_default_prim
            self.listener.info(f"Set output USD default prim to: {input_default_prim}")

        # Use sublayer to compose over the input USD file
        # This creates a non-destructive layer that can be composed over the original
        stage.GetRootLayer().subLayerPaths.append(str(input_usd_path))

        # After adding sublayer, verify the default prim is valid.
        self._fix_stale_default_prim(stage, input_default_prim)

        # Create materials and bindings in the overlay layer
        materials_applied = {}
        materials_created_count = 0
        prims_with_materials = 0

        if is_library_based and material_library_path:
            # Library-based: Copy only used materials (not the entire library)
            self.listener.info(
                "Using library-based materials - copying used materials only"
            )
            stage, materials_applied = self._copy_library_materials(
                stage,
                material_library_path,
                output_usd_path,
                resolved_materials,
                default_prim_name=input_default_prim or "",
            )
            materials_created_count = len(materials_applied)
        else:
            # File-based: Create materials using proven utility functions
            # For layer-only mode, we'll use "/Materials" as the path prefix
            for material_name, material_path in resolved_materials.items():
                material_prim_path, success = self._create_material_on_stage(
                    stage=stage,
                    material_name=material_name,
                    material_path=material_path,
                    output_usd_path=output_usd_path,
                    path_prefix="",  # Root level for layer mode
                )

                if success:
                    materials_applied[material_name] = material_prim_path
                    materials_created_count += 1

        # Apply material bindings as "over" opinions in the layer based on
        # predictions.  We use the Sdf API directly because:
        #   - stage.OverridePrim() silently skips spec creation when the prim
        #     already exists via a sublayer.
        #   - Stage-level binding APIs refuse to author on instance proxies.
        #
        # Instance handling: we only bind to non-instance prims (prototypes /
        # reference sources).  USD instances inherit material bindings from
        # their prototype via composition — a stronger sublayer opinion on
        # the reference source overrides the referenced content's local
        # bindings, and instances see the override through the shared
        # prototype.  No de-instancing is needed.
        root_layer = stage.GetRootLayer()

        # Build instance root → local referenced prim path mapping.
        # For USD instances that reference a local prim (same-file, empty
        # assetPath), we write material overrides at the referenced prim paths
        # in our output layer.  The instance prototype inherits the stronger
        # sublayer opinion, so all instances sharing that prototype get the
        # material override.  Instances referencing external files (non-empty
        # assetPath) cannot be overridden this way and are skipped.
        instance_root_to_ref_prim: dict[str, str | None] = {}
        for prim in stage.Traverse():
            if prim.IsInstance():
                ir_path = str(prim.GetPath())
                ref_path: str | None = None
                for spec in prim.GetPrimStack():
                    added = spec.referenceList.GetAddedOrExplicitItems()
                    if added:
                        ref = added[0]
                        if not ref.assetPath and ref.primPath:
                            ref_path = str(ref.primPath)
                        break
                instance_root_to_ref_prim[ir_path] = ref_path

        instance_roots: set[str] = set(instance_root_to_ref_prim.keys())
        remapped_instance_prims = 0
        skipped_instance_prims = 0
        for prim_path, material_name in prim_to_material.items():
            material_prim_path = materials_applied.get(material_name)
            if not material_prim_path:
                self.listener.warning(
                    f"Material '{material_name}' not found in applied materials for prim {prim_path}"
                )
                continue

            # For prims under an instance root, remap the prediction path to
            # the referenced prototype path so the binding is written to the
            # shared prototype source — all instances sharing that prototype
            # will then see the override via USD composition.
            binding_target_path = prim_path
            skip = False
            for ir in instance_roots:
                if prim_path == ir or prim_path.startswith(ir + "/"):
                    ref_prim = instance_root_to_ref_prim[ir]
                    if ref_prim:
                        suffix = prim_path[len(ir) :]
                        binding_target_path = ref_prim + suffix
                        remapped_instance_prims += 1
                    else:
                        # External-asset reference — cannot override in this layer
                        self.listener.debug(
                            f"Skipping {prim_path}: instance references external asset"
                        )
                        skip = True
                        skipped_instance_prims += 1
                    break
            if skip:
                continue

            # Create over spec and write binding at the Sdf level
            prim_spec = Sdf.CreatePrimInLayer(root_layer, binding_target_path)
            prim_spec.specifier = Sdf.SpecifierOver

            # Ensure MaterialBindingAPI is applied so ComputeBoundMaterial works
            api_name = "MaterialBindingAPI"
            api_schemas = prim_spec.GetInfo("apiSchemas")
            if not api_schemas or api_name not in api_schemas.prependedItems:
                prim_spec.SetInfo(
                    "apiSchemas",
                    Sdf.TokenListOp.Create(prependedItems=[api_name]),
                )

            # Author material:binding relationship directly on the layer
            binding_rel = prim_spec.relationships.get(
                "material:binding"
            ) or Sdf.RelationshipSpec(prim_spec, "material:binding")
            binding_rel.targetPathList.explicitItems = [Sdf.Path(material_prim_path)]

            prims_with_materials += 1
            self.listener.info(
                f"Bound material '{material_name}' to prim {binding_target_path}"
                + (
                    f" (remapped from instance proxy {prim_path})"
                    if binding_target_path != prim_path
                    else ""
                )
            )

        if remapped_instance_prims:
            self.listener.info(
                f"Remapped {remapped_instance_prims} instance prim paths "
                f"to prototype paths"
            )
        if skipped_instance_prims:
            self.listener.info(
                f"Skipped {skipped_instance_prims} instance prims "
                f"(external-asset references cannot be overridden)"
            )

        # Apply materials to instances by looking up their master's material
        if skip_instance_check:
            self.listener.info("Skipping instance material check (payload mode)")
            instance_stats = {
                "instances_found": 0,
                "instances_applied": 0,
                "instances_skipped": 0,
            }
        else:
            instance_stats = self._apply_materials_to_instances(
                stage, prim_to_material, materials_applied
            )

        stats = {
            "materials_created": materials_created_count,
            "prims_with_materials": prims_with_materials
            + instance_stats["instances_applied"],
            "instances_applied": instance_stats["instances_applied"],
            "instances_skipped": instance_stats["instances_skipped"],
        }

        return stage, materials_applied, stats

    def _get_material_color(self, material_name: str) -> tuple[float, float, float]:
        """Get a default color based on material name.

        Args:
            material_name: Name of the material

        Returns:
            RGB color tuple
        """
        # Simple color mapping based on common material names
        color_map = {
            "metal": (0.7, 0.7, 0.8),
            "aluminum": (0.8, 0.8, 0.85),
            "steel": (0.6, 0.6, 0.65),
            "iron": (0.5, 0.5, 0.5),
            "plastic": (0.9, 0.9, 0.9),
            "rubber": (0.2, 0.2, 0.2),
            "glass": (0.95, 0.95, 1.0),
            "wood": (0.5, 0.3, 0.2),
            "concrete": (0.5, 0.5, 0.5),
            "brick": (0.6, 0.3, 0.2),
            "fabric": (0.7, 0.7, 0.6),
            "leather": (0.4, 0.2, 0.1),
        }

        # Check if any key is in the material name
        material_lower = material_name.lower()
        for key, color in color_map.items():
            if key in material_lower:
                return color

        # Default gray color
        return (0.5, 0.5, 0.5)
