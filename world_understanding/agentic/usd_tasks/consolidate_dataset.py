# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consolidate USD dataset from Phase 1 intermediate format to v0.2 final format.

This task transforms the Phase 1 intermediate structure into the flat v0.2 format:
- Loads Phase 1 files from output_dir (e.g., dataset/usd/)
- Creates v0.2 dataset.json and dataset.jsonl
- Saves v0.2 files to parent directory (e.g., dataset/)
- Preserves Phase 1 files in output_dir for downstream steps
- Maintains nested renders/ structure for image path uniqueness
- Optionally cleans up intermediate files (keep_intermediate=False)

Directory structure after consolidation:
  dataset/
    dataset.json          # v0.2 config
    dataset.jsonl         # v0.2 entries
    usd/
      dataset.json        # Phase 1 metadata (preserved)
      prims.jsonl         # Phase 1 entries (preserved)
      usd_model.json      # USD model structure
      renders/            # Nested image structure (preserved)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

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
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class ConsolidateDatasetTask(Task):
    """Consolidate Phase 1 intermediate data into v0.2 format.

    This task performs the final transformation from intermediate USD dataset
    format to the unified v0.2 schema with flat directory structure.

    Expected context inputs:
        - output_dir: Base output directory for dataset
        - system_prompt: VLM system prompt (shared across entries)
        - user_prompt_template: Template for generating user prompts (optional)
        - task_type: Task type ("material_assignment" or "iterative_classification")
        - task_description: Human-readable task description
        - creator_name: Name of creating agent (e.g., "material-agent")
        - materials_list: List of available materials/classes (optional)
        - keep_intermediate: Keep intermediate files for debugging (default: False)
        - reference_images: List of reference image paths (optional)
        - vlm_image_prompts: Dict mapping render modes to prompts (optional)

    Expected from object_store:
        - prim_entries: List of prim entries from Phase 1 (optional, will load from file)
        - usd_model: Optional USDModel for metadata

    Updates context with:
        - dataset_config_path: Path to dataset.json
        - dataset_entries_path: Path to dataset.jsonl
        - num_entries: Number of entries in dataset
        - renders_flattened: Whether renders were flattened
    """

    def __init__(self) -> None:
        self.name = "ConsolidateDataset"
        self.description = "Consolidate intermediate data into v0.2 format"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Consolidate intermediate dataset into v0.2 format.

        Args:
            context: Workflow context
            object_store: Object store

        Returns:
            Updated context with dataset paths
        """
        listener = get_listener(context, logger_name=__name__)

        # Get inputs
        output_dir = Path(context.get("output_dir", "output"))
        # For unified pipeline: usd_dir specifies where Phase 1 files are located
        # For standalone (material-agent): usd_dir defaults to output_dir
        usd_dir = Path(context.get("usd_dir", output_dir))
        system_prompt = context.get("system_prompt", "")
        task_type = context.get("task_type", "material_assignment")
        task_description = context.get(
            "task_description", f"{task_type.replace('_', ' ').title()} task"
        )
        creator_name = context.get("creator_name", "unknown")
        # Default to keeping intermediate files for downstream pipeline steps
        keep_intermediate = context.get("keep_intermediate", True)

        listener.info("Consolidating dataset to v0.2 format")
        listener.info(f"  Input directory (Phase 1): {usd_dir}")
        listener.info(f"  Output directory (Phase 2): {output_dir}")
        listener.info(f"  Task type: {task_type}")
        listener.info(f"  Creator: {creator_name}")

        # Load Phase 1 intermediate data
        # For unified pipeline: read from usd_dir (e.g., dataset/usd/)
        # For material-agent: usd_dir == output_dir (same directory)
        prims_file = usd_dir / "prims.jsonl"
        usd_dataset_json = usd_dir / "dataset.json"

        if not prims_file.exists():
            raise FileNotFoundError(
                f"Phase 1 intermediate file not found: {prims_file}\n"
                f"Expected Phase 1 (build USD dataset) to create this file."
            )

        # Load prims from Phase 1
        prim_entries = []
        with open(prims_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prim_entries.append(json.loads(line))

        listener.info(f"Loaded {len(prim_entries)} prim entries from Phase 1")

        # Load USD dataset metadata if available
        source_usd = context.get("usd_path", "")

        if usd_dataset_json.exists():
            with open(usd_dataset_json, encoding="utf-8") as f:
                usd_data = json.load(f)
                if not source_usd:
                    source_usd = usd_data.get("metadata", {}).get("source_usd", "")

        # Ensure source_usd is a string (may be Path object from context)
        source_usd_str = str(source_usd) if source_usd else ""

        # Create dataset.json (v0.2 config)
        dataset_config = self._create_dataset_config(
            num_entries=len(prim_entries),
            system_prompt=system_prompt,
            task_type=task_type,
            task_description=task_description,
            creator_name=creator_name,
            source_usd=source_usd_str,
            context=context,
        )

        # Save v0.2 dataset files to output_dir
        # Phase 1 files are in usd_dir (e.g., dataset/usd/)
        # Phase 2 files go to output_dir (e.g., dataset/)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_config_path = output_dir / "dataset.json"
        with open(dataset_config_path, "w", encoding="utf-8") as f:
            json.dump(dataset_config.model_dump(mode="json"), f, indent=2)
        listener.info(f"Created dataset.json: {dataset_config_path}")

        # Create dataset.jsonl entries (v0.2)
        dataset_entries = self._create_dataset_entries(
            prim_entries=prim_entries,
            context=context,
            listener=listener,
        )

        # Save v0.2 dataset.jsonl to output_dir
        dataset_entries_path = output_dir / "dataset.jsonl"
        with open(dataset_entries_path, "w", encoding="utf-8") as f:
            for entry in dataset_entries:
                f.write(entry.model_dump_json() + "\n")
        listener.info(
            f"Created dataset.jsonl with {len(dataset_entries)} entries: {dataset_entries_path}"
        )

        # Check renders directory structure (no flattening needed - nested paths preserved)
        # Renders are in usd_dir (Phase 1 directory)
        renders_flattened = self._flatten_renders_directory(
            usd_dir, listener, dry_run=False
        )

        # USD model is already in usd_dir from Phase 1, nothing to copy
        # (kept here for reference in case structure changes in future)

        # Keep intermediate files by default for compatibility with other pipeline steps
        # Only clean up if explicitly requested (keep_intermediate=False)
        # NOTE: In unified pipelines, prepare_dataset needs these Phase 1 files
        if keep_intermediate is False:  # Explicit False, not just falsy
            self._cleanup_intermediate_files(usd_dir, listener)
            listener.info("Cleaned up intermediate files (keep_intermediate=False)")
        else:
            listener.info("Keeping Phase 1 intermediate files for downstream steps")

        # Update context
        context["dataset_config_path"] = str(dataset_config_path)
        context["dataset_entries_path"] = str(dataset_entries_path)
        context["num_entries"] = len(dataset_entries)
        context["renders_flattened"] = renders_flattened

        listener.info("✓ Dataset consolidation complete")
        listener.info("  Format: v0.2")
        listener.info(f"  Entries: {len(dataset_entries)}")
        listener.info(f"  Config: {dataset_config_path}")
        listener.info(f"  Data: {dataset_entries_path}")

        return context

    def _create_dataset_config(
        self,
        num_entries: int,
        system_prompt: str,
        task_type: str,
        task_description: str,
        creator_name: str,
        source_usd: str,
        context: dict[str, Any],
    ) -> DatasetConfig:
        """Create DatasetConfig (dataset.json).

        Args:
            num_entries: Number of entries in dataset
            system_prompt: System prompt for VLM
            task_type: Task type
            task_description: Task description
            creator_name: Creating agent name
            source_usd: Source USD file path
            context: Workflow context

        Returns:
            DatasetConfig instance
        """
        # Create metadata
        metadata = DatasetMetadata(
            created=datetime.now().isoformat(),
            creator=creator_name,
            source_usd=source_usd or None,
            description=context.get("dataset_description"),
            num_entries=num_entries,
        )

        # Create task config
        # Validate task_type is one of the allowed values
        valid_task_types = {
            "material_assignment",
            "iterative_classification",
            "detection",
        }
        if task_type not in valid_task_types:
            raise ValueError(
                f"Invalid task_type: {task_type}. Must be one of {valid_task_types}"
            )
        task = TaskConfig(
            type=task_type,  # type: ignore[arg-type]
            description=task_description,
        )

        # Create prompt config(s)
        # For iterative_classification, create multiple prompts from classification_steps
        classification_steps = context.get("classification_steps")

        if task_type == "iterative_classification" and classification_steps:
            # Create one prompt per classification step
            prompts = []
            for step_idx, step_config in enumerate(classification_steps):
                step_system_prompt = step_config.get("system_prompt") or system_prompt
                prompt = PromptConfig(
                    step_name=step_config["name"],
                    step_index=step_idx,
                    system_prompt=step_system_prompt,
                    classes=step_config.get("classes"),
                    temperature=context.get("temperature"),
                    max_tokens=context.get("max_tokens"),
                )
                prompts.append(prompt)
        else:
            # Single-step task (material_assignment, detection)
            prompt = PromptConfig(
                step_name=context.get("step_name", "main"),
                step_index=0,
                system_prompt=system_prompt,
                output_format=context.get("output_format"),
                classes=context.get("materials_list") or context.get("classes"),
                temperature=context.get("temperature"),
                max_tokens=context.get("max_tokens"),
            )
            prompts = [prompt]

        inference = InferenceConfig(prompts=prompts)

        return DatasetConfig(
            schema_version="0.2",
            metadata=metadata,
            task=task,
            inference=inference,
            prims_file="dataset.jsonl",
            usd_model_file="usd/usd_model.json",
        )

    def _create_dataset_entries(
        self,
        prim_entries: list[dict[str, Any]],
        context: dict[str, Any],
        listener: Any,
    ) -> list[DatasetEntry]:
        """Create DatasetEntry instances from Phase 1 prim entries.

        Args:
            prim_entries: List of prim entries from Phase 1
            context: Workflow context
            listener: Event listener

        Returns:
            List of DatasetEntry instances
        """
        dataset_entries = []
        vlm_image_prompts = context.get("vlm_image_prompts", {})
        reference_images = context.get("reference_images", [])

        for prim_entry in prim_entries:
            try:
                entry = self._convert_prim_to_dataset_entry(
                    prim_entry=prim_entry,
                    vlm_image_prompts=vlm_image_prompts,
                    reference_images=reference_images,
                    context=context,
                )
                dataset_entries.append(entry)
            except Exception as e:
                prim_path = prim_entry.get("prim_path", "unknown")
                listener.warning(f"Failed to convert prim {prim_path}: {e}")
                continue

        return dataset_entries

    def _convert_prim_to_dataset_entry(
        self,
        prim_entry: dict[str, Any],
        vlm_image_prompts: dict[str, Any],
        reference_images: list[str],
        context: dict[str, Any],
    ) -> DatasetEntry:
        """Convert Phase 1 prim entry to v0.2 DatasetEntry.

        Args:
            prim_entry: Phase 1 prim entry
            vlm_image_prompts: Mapping of render modes to prompts
            reference_images: List of reference image paths
            context: Workflow context

        Returns:
            DatasetEntry instance
        """
        prim_path = prim_entry["prim_path"]

        # Create source info
        source = SourceInfo(
            type="usd_prim",
            prim_path=prim_path,
            model_number=context.get("model_number"),
        )

        # Generate user prompt(s)
        # For iterative_classification, generate multiple prompts (one per step)
        # For detection with classification_steps, use the step's prompt and classes
        task_type = context.get("task_type", "detection")
        classification_steps = context.get("classification_steps")

        if task_type == "iterative_classification" and classification_steps:
            user_prompts = self._generate_user_prompts_multi_step(
                prim_entry, classification_steps, context
            )
            user_prompt = None  # Use user_prompts instead
        elif (
            task_type == "detection"
            and classification_steps
            and len(classification_steps) > 0
        ):
            # Single-step classification - use template if available, otherwise fallback
            step = classification_steps[0]
            classes = step.get("classes", [])
            class_list = " or ".join([f"'{c}'" for c in classes])
            step_prompt = step.get("prompt", "")

            # Get user prompt template from context (from vlm_user in config)
            prompts_config = context.get("prompts", {})
            user_template = prompts_config.get("vlm_user")

            if user_template and "{context}" in user_template:
                # Build context information for this prim
                prompt_context = self._build_prompt_context(prim_entry, context)
                context_str = self._format_context_string(prompt_context)

                # Build full step prompt with classes
                step_prompt_with_classes = (
                    f"{step_prompt}\n\nIs this part a {class_list}?"
                )

                # Replace placeholders in template
                user_prompt = user_template.replace("{context}", context_str)
                if "{step_prompt}" in user_prompt:
                    user_prompt = user_prompt.replace(
                        "{step_prompt}", step_prompt_with_classes
                    )
                else:
                    # If no {step_prompt} placeholder, append it
                    user_prompt = f"{step_prompt_with_classes}\n\n{user_prompt}"
            else:
                # Fallback to simple prompt
                user_prompt = f"Is this part a {class_list}?"

            user_prompts = None
        else:
            user_prompt = self._generate_user_prompt(prim_entry, context)
            user_prompts = None

        # Convert renders to images
        images = []
        for render in prim_entry.get("renders", []):
            # Update path to flat structure (will be flattened later)
            original_path = render["path"]
            flat_path = self._flatten_image_path(original_path)

            # Get vlm_prompt for this render mode
            render_mode = render.get("render_mode", "unknown")
            vlm_prompt = vlm_image_prompts.get(render_mode)

            metadata = ImageMetadata(
                view=render.get("view"),
                camera=render.get("camera"),
                render_mode=render_mode,
                vlm_prompt=vlm_prompt,
            )

            images.append(
                ImageObject(
                    path=flat_path,
                    type="render",
                    metadata=metadata,
                )
            )

        # Add reference images if provided
        ref_images = None
        if reference_images:
            ref_images = [
                ImageObject(
                    path=ref_path,
                    type="reference",
                )
                for ref_path in reference_images
            ]

        media = MediaConfig(images=images, reference_images=ref_images)

        # Extract ground truth if available
        ground_truth = None
        if "material_bindings" in prim_entry:
            material_binding = prim_entry["material_bindings"].get("resolved")
            if material_binding:
                # Extract material name from binding
                material_name = self._extract_material_name(material_binding)
                if material_name:
                    ground_truth = GroundTruth(
                        material=material_name,
                        metadata=GroundTruthMetadata(
                            source="oracle", annotator="usd_material_binding"
                        ),
                    )

        # Preserve USD metadata
        usd_metadata = {}
        if "metadata" in prim_entry:
            usd_metadata["geometry"] = prim_entry["metadata"]
        if "hierarchy" in prim_entry:
            usd_metadata["hierarchy"] = prim_entry["hierarchy"]
        if "display_color" in prim_entry:
            usd_metadata["display_color"] = prim_entry["display_color"]
        if "world_bbox" in prim_entry:
            usd_metadata["world_bbox"] = prim_entry["world_bbox"]

        return DatasetEntry(
            id=prim_path,
            source=source,
            user_prompt=user_prompt,
            user_prompts=user_prompts,
            media=media,
            ground_truth=ground_truth,
            usd_metadata=usd_metadata if usd_metadata else None,
        )

    def _generate_user_prompt(
        self, prim_entry: dict[str, Any], context: dict[str, Any]
    ) -> str:
        """Generate user prompt for a prim entry.

        Args:
            prim_entry: Phase 1 prim entry
            context: Workflow context with prompt template

        Returns:
            User prompt string
        """
        # Check if context already has a generated prompt for this prim
        # (e.g., from PrepareDatasetTask)
        if "user_prompts" in context and isinstance(context["user_prompts"], dict):
            prim_path = str(prim_entry.get("prim_path", ""))
            if prim_path in context["user_prompts"]:
                return str(context["user_prompts"][prim_path])

        # Fall back to template-based generation
        template = str(context.get("user_prompt_template", "Analyze this component."))

        # Build context dict for template
        prompt_context: dict[str, Any] = {}

        # Add geometric information
        if "world_bbox" in prim_entry:
            bbox = prim_entry["world_bbox"]
            prompt_context["bbox"] = bbox

        # Add prim path if requested
        if context.get("include_prim_path_context", False):
            prompt_context["prim_path"] = prim_entry.get("prim_path", "")

        # Format template with context
        if prompt_context and "{context}" in template:
            context_str = "\n".join(f"{k}: {v}" for k, v in prompt_context.items())
            return template.replace("{context}", context_str)

        return template

    def _generate_user_prompts_multi_step(
        self,
        prim_entry: dict[str, Any],
        classification_steps: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[str]:
        """Generate user prompts for multi-step iterative classification.

        Args:
            prim_entry: Phase 1 prim entry
            classification_steps: List of classification step configurations
            context: Workflow context

        Returns:
            List of user prompts (one per step)
        """
        user_prompts = []

        # Get user prompt template from context (from prompts.vlm_user in config)
        prompts_config = context.get("prompts", {})
        user_template = prompts_config.get("vlm_user", "{step_prompt}")

        # Build context information for this prim
        prompt_context = self._build_prompt_context(prim_entry, context)

        for _step_idx, step_config in enumerate(classification_steps):
            step_prompt = step_config.get("prompt", "")
            step_classes = step_config.get("classes", [])

            # Build class list string
            class_list = " or ".join([f"'{c}'" for c in step_classes])

            # Add class selection to step prompt
            step_prompt_with_classes = f"{step_prompt}\n\nIs this part a {class_list}?"

            # Use template if available, otherwise fall back to basic prompt
            if "{context}" in user_template:
                # Replace {context} placeholder with actual context data
                context_str = self._format_context_string(prompt_context)
                full_prompt = user_template.replace("{context}", context_str)
                # Replace step-specific prompt if template has that placeholder
                if "{step_prompt}" in full_prompt:
                    full_prompt = full_prompt.replace(
                        "{step_prompt}", step_prompt_with_classes
                    )
                else:
                    # If no step_prompt placeholder, append the step prompt
                    full_prompt = f"{step_prompt_with_classes}\n\n{full_prompt}"
            else:
                # No context template, use basic format
                full_prompt = step_prompt_with_classes

            user_prompts.append(full_prompt)

        return user_prompts

    def _build_prompt_context(
        self, prim_entry: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Build context dictionary for prompt generation.

        Args:
            prim_entry: Phase 1 prim entry
            context: Workflow context with inclusion flags

        Returns:
            Dictionary of context information
        """
        prompt_context: dict[str, Any] = {}

        # Add prim path if requested
        if context.get("include_prim_path_context", False):
            prompt_context["prim_path"] = prim_entry.get("prim_path", "")

        # Add hierarchy information if available and requested
        if (
            context.get("include_prim_path_context", False)
            and "hierarchy" in prim_entry
        ):
            hierarchy = prim_entry["hierarchy"]
            if "parent_path" in hierarchy:
                prompt_context["parent_path"] = hierarchy["parent_path"]
            if "ancestors" in hierarchy:
                prompt_context["ancestors"] = hierarchy["ancestors"]

        # Add geometric information if requested
        if context.get("include_geometric_context", False):
            # Prefer world_bbox_meters (real-world units) over world_bbox (USD units)
            if "world_bbox_meters" in prim_entry:
                bbox_meters = prim_entry["world_bbox_meters"]
                size_m = bbox_meters.get("size", [])
                if len(size_m) == 3:
                    prompt_context["bounding_box_meters"] = {
                        "width": size_m[0],
                        "height": size_m[1],
                        "depth": size_m[2],
                    }
            elif "world_bbox" in prim_entry:
                # Fallback to world_bbox if world_bbox_meters not available
                bbox = prim_entry["world_bbox"]
                prompt_context["bounding_box"] = {
                    "min": bbox.get("min", []),
                    "max": bbox.get("max", []),
                    "center": bbox.get("center", []),
                    "size": bbox.get("size", []),
                }

            # Add extent from metadata if available
            if "metadata" in prim_entry and "extent" in prim_entry["metadata"]:
                prompt_context["extent"] = prim_entry["metadata"]["extent"]

        # Add material information if available
        if "metadata" in prim_entry and "material" in prim_entry["metadata"]:
            prompt_context["material_binding"] = prim_entry["metadata"]["material"]

        return prompt_context

    def _format_context_string(self, prompt_context: dict[str, Any]) -> str:
        """Format context dictionary as a readable string.

        Args:
            prompt_context: Context information dictionary

        Returns:
            Formatted context string
        """
        if not prompt_context:
            return "No additional context available."

        context_lines = []

        for key, value in prompt_context.items():
            if key == "bounding_box_meters" and isinstance(value, dict):
                # Format in meters (matching material-agent)
                width = value.get("width", 0)
                height = value.get("height", 0)
                depth = value.get("depth", 0)
                context_lines.append(
                    f"Bounding box dimensions (meters): "
                    f"width={width:.3f}m, height={height:.3f}m, depth={depth:.3f}m"
                )
            elif key == "bounding_box" and isinstance(value, dict):
                # Fallback format for USD units
                bbox_info = []
                if "center" in value:
                    center = value["center"]
                    if len(center) == 3:
                        bbox_info.append(
                            f"center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
                        )
                if "size" in value:
                    size = value["size"]
                    if len(size) == 3:
                        bbox_info.append(
                            f"size: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})"
                        )
                if bbox_info:
                    context_lines.append(f"bounding_box: {', '.join(bbox_info)}")
            elif key == "ancestors" and isinstance(value, list):
                context_lines.append(f"hierarchy: {' -> '.join(value)}")
            else:
                context_lines.append(f"{key}: {value}")

        return "\n".join(context_lines)

    def _extract_material_name(self, material_binding: str) -> str | None:
        """Extract material name from USD material binding path.

        Args:
            material_binding: USD material binding path

        Returns:
            Material name or None
        """
        if not material_binding:
            return None

        # Material binding format: /RootNode/Materials/material_name
        # or: /Looks/material_name
        parts = material_binding.split("/")
        if len(parts) > 1:
            return parts[-1]  # Last component

        return None

    def _flatten_image_path(self, original_path: str) -> str:
        """Preserve image path structure.

        NOTE: Image paths MUST maintain their nested structure to ensure uniqueness.
        Multiple prims may have the same mesh type name (e.g., "mesh"), so flattening
        to just the filename would cause collisions.

        Args:
            original_path: Original path (e.g., renders/Root/030R01/k5/shape/mesh.png)

        Returns:
            Original path unchanged to preserve uniqueness
        """
        # DO NOT flatten - nested structure is essential for unique image paths
        return original_path

    def _flatten_renders_directory(
        self, output_dir: Path, listener: Any, dry_run: bool = False
    ) -> bool:
        """Check renders directory structure (no flattening needed).

        NOTE: Renders MUST maintain their nested structure for image path uniqueness.
        This method only validates the structure exists.

        Args:
            output_dir: Output directory
            listener: Event listener
            dry_run: Unused, kept for compatibility

        Returns:
            False (no flattening performed)
        """
        renders_dir = output_dir / "renders"

        if not renders_dir.exists():
            listener.info("No renders/ directory found")
            return False

        # Count images for logging
        image_files = list(renders_dir.rglob("*.png")) + list(
            renders_dir.rglob("*.jpg")
        )

        if image_files:
            listener.info(
                f"Renders directory contains {len(image_files)} images in nested structure"
            )
        else:
            listener.info("Renders directory is empty")

        # No flattening performed - nested structure is essential
        return False

    def _cleanup_intermediate_files(self, usd_dir: Path, listener: Any) -> None:
        """Clean up intermediate files after consolidation.

        Removes:
        - prims.jsonl (Phase 1 intermediate)
        - dataset.json (Phase 1 intermediate)
        - vlm_system_prompt.txt (v0.1 legacy)
        - spec.txt (v0.1 legacy)

        Args:
            usd_dir: Phase 1 directory containing intermediate files
            listener: Event listener
        """
        # Phase 1 intermediate files are in usd_dir
        files_to_remove = [
            usd_dir / "prims.jsonl",
            usd_dir / "dataset.json",
            usd_dir / "vlm_system_prompt.txt",
            usd_dir / "spec.txt",
        ]

        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
                listener.debug(f"Removed intermediate file: {file_path}")

        # Clean up empty subdirectories in renders/ if any
        renders_dir = usd_dir / "renders"
        if renders_dir.exists():
            try:
                for item in renders_dir.iterdir():
                    if item.is_dir() and not any(item.iterdir()):
                        item.rmdir()
                        listener.debug(f"Removed empty directory: {item}")
            except OSError:
                pass  # Directory not empty or permission issue, leave it
