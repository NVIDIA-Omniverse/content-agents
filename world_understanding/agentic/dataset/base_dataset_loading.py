# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base dataset loading task for JSONL datasets.

This module provides a base class for loading JSONL datasets with optional
validation and metadata extraction. material-agent, physics-agent, and joint-agent
inherit from this base class.
"""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class BaseDatasetLoadingTask(Task):
    """Base class for loading JSONL datasets with optional validation.

    This base class provides common functionality for loading JSONL datasets
    with companion metadata files. Subclasses can extend with agent-specific
    validation and processing logic.

    Features:
    - JSONL file loading with line-by-line parsing
    - Companion dataset.json metadata loading
    - System prompt extraction from inference config
    - Object store and context management
    - Flexible path resolution (from constructor or context)
    - Graceful error handling with detailed logging

    Subclasses should override:
    - `_validate_dataset()`: Implement agent-specific validation
    - `_post_process_entries()`: Normalize or transform entries
    - `_update_context()`: Add agent-specific context fields

    Example:
        >>> class MyDatasetLoader(BaseDatasetLoadingTask):
        ...     def _validate_dataset(self, dataset, dataset_path):
        ...         # Custom validation logic
        ...         return [e for e in dataset if self._is_valid(e)]
        ...
        >>> loader = MyDatasetLoader(dataset_path=Path("data.jsonl"))
        >>> context = loader.run(context={}, object_store=store)
    """

    def __init__(
        self,
        dataset_path: Path | None = None,
        validate: bool = True,
        name: str = "DatasetLoading",
        description: str = "Load and validate dataset from JSONL file",
    ):
        """Initialize the dataset loading task.

        Args:
            dataset_path: Path to dataset file (None to use from context)
            validate: Whether to validate dataset entries (subclass-specific)
            name: Task name for logging
            description: Task description
        """
        self.dataset_path = dataset_path
        self.validate = validate
        self.name = name
        self.description = description

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Load dataset from JSONL file.

        Args:
            context: Workflow context
            object_store: Storage for large objects

        Returns:
            Updated context with dataset metadata
        """
        # 1. Resolve and validate path
        dataset_path = self._resolve_dataset_path(context)

        # 2. Load JSONL entries
        dataset = self._load_jsonl(dataset_path)
        logger.info(f"Loaded {len(dataset)} entries from {dataset_path.name}")

        # 3. Load metadata (optional companion file)
        metadata = self._load_metadata(dataset_path, context)

        # 4. Validate entries (subclass-specific)
        if self.validate:
            original_count = len(dataset)
            dataset = self._validate_dataset(dataset, dataset_path)
            if len(dataset) < original_count:
                logger.warning(
                    f"Validation filtered {original_count - len(dataset)} entries"
                )
            logger.info(f"Validated {len(dataset)} entries")

        # 5. Post-process entries (hook for subclasses)
        dataset = self._post_process_entries(dataset, dataset_path, metadata)

        # 6. Store in object store
        self._store_in_object_store(object_store, dataset, metadata)

        # 7. Update context
        self._update_context(context, dataset, dataset_path, metadata)

        return context

    def _resolve_dataset_path(self, context: dict[str, Any]) -> Path:
        """Resolve dataset path from constructor or context.

        Args:
            context: Workflow context

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path not provided
            FileNotFoundError: If path doesn't exist
        """
        dataset_path = (
            self.dataset_path
            if self.dataset_path is not None
            else context.get("dataset_path")
        )
        if dataset_path is None:
            raise ValueError("dataset_path not provided in constructor or context")

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        logger.info(f"Loading dataset from {dataset_path}")
        return dataset_path

    def _load_jsonl(self, dataset_path: Path) -> list[dict]:
        """Load JSONL file into list of entries.

        Args:
            dataset_path: Path to JSONL file

        Returns:
            List of parsed JSON entries
        """
        with open(dataset_path, encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        return dataset

    def _load_metadata(
        self, dataset_path: Path, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Load companion dataset.json metadata file.

        Extracts system prompts from inference.prompts[0].system_prompt
        and stores them in context if not already present.

        Args:
            dataset_path: Path to dataset JSONL file
            context: Workflow context to update with system prompt

        Returns:
            Metadata dictionary (empty if file not found)
        """
        dataset_json_path = dataset_path.parent / "dataset.json"
        if not dataset_json_path.exists():
            logger.debug(f"No metadata file found at {dataset_json_path}")
            return {}

        try:
            with open(dataset_json_path, encoding="utf-8") as f:
                loaded_metadata: dict[str, Any] = json.load(f)

            # Extract system prompt if available
            self._extract_system_prompt(loaded_metadata, context)

            logger.debug(f"Loaded dataset metadata from {dataset_json_path}")
            return loaded_metadata

        except Exception as e:
            logger.warning(f"Failed to load metadata from {dataset_json_path}: {e}")
            return {}

    def _extract_system_prompt(
        self, metadata: dict[str, Any], context: dict[str, Any]
    ) -> None:
        """Extract and store system prompt from metadata.

        Args:
            metadata: Dataset metadata containing inference config
            context: Context to update with system prompt
        """
        if "inference" not in metadata:
            return

        inference_config = metadata["inference"]
        if "prompts" not in inference_config or not inference_config["prompts"]:
            return

        first_prompt = inference_config["prompts"][0]
        if "system_prompt" not in first_prompt:
            return

        # Only set if not already in context
        if not context.get("system_prompt"):
            system_prompt = first_prompt["system_prompt"]
            context["system_prompt"] = system_prompt

            # Also store in config for compatibility
            if "config" not in context:
                context["config"] = {}
            context["config"]["system_prompt"] = system_prompt

            logger.info(
                f"Loaded system prompt from metadata ({len(system_prompt)} chars)"
            )

    def _validate_dataset(self, dataset: list[dict], dataset_path: Path) -> list[dict]:
        """Validate dataset entries (hook for subclasses).

        Default implementation does no validation. Subclasses should override
        to implement agent-specific validation logic.

        Args:
            dataset: List of dataset entries
            dataset_path: Path to dataset for resolving relative paths

        Returns:
            Filtered list of valid entries
        """
        return dataset

    def _post_process_entries(
        self,
        dataset: list[dict],
        dataset_path: Path,
        metadata: dict[str, Any],
    ) -> list[dict]:
        """Post-process entries after loading (hook for subclasses).

        Default implementation does nothing. Subclasses can override to:
        - Normalize entry format
        - Add computed fields
        - Resolve relative paths

        Args:
            dataset: List of dataset entries
            dataset_path: Path to dataset file
            metadata: Loaded metadata

        Returns:
            Processed dataset entries
        """
        return dataset

    def _store_in_object_store(
        self,
        object_store: ObjectStore | None,
        dataset: list[dict],
        metadata: dict[str, Any],
    ) -> None:
        """Store dataset and metadata in object store.

        Args:
            object_store: Object store instance
            dataset: Dataset entries
            metadata: Dataset metadata
        """
        if not object_store:
            return

        object_store.set("dataset", dataset)
        if metadata:
            object_store.set("dataset_metadata", metadata)
        logger.debug("Stored dataset in object store")

    def _update_context(
        self,
        context: dict[str, Any],
        dataset: list[dict],
        dataset_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        """Update context with dataset information.

        Args:
            context: Workflow context to update
            dataset: Loaded dataset entries
            dataset_path: Path to dataset file
            metadata: Dataset metadata
        """
        # Store dataset in context (for reports and downstream tasks)
        context["dataset"] = dataset

        # Store metadata
        context["dataset_size"] = len(dataset)
        context["dataset_path"] = str(dataset_path)
        context["dataset_dir"] = str(dataset_path.parent)

        if metadata:
            context["dataset_metadata"] = metadata
