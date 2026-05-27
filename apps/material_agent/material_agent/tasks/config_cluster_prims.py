# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for cluster_prims and expand_cluster_predictions steps."""

import logging
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)

_SENSITIVE_CONFIG_KEY_PARTS = ("api_key", "apikey", "token", "secret", "password")


def _redact_sensitive_config(value: Any) -> Any:
    """Return a logging-safe copy of a config value."""
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in _SENSITIVE_CONFIG_KEY_PARTS):
                redacted[key] = "<redacted>" if item else item
            else:
                redacted[key] = _redact_sensitive_config(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_config(item) for item in value]
    return value


class ClusterPrimsConfigTask(Task):
    """Load config for the cluster_prims step.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - dataset_path: Path to dataset.jsonl
        - working_dir: Pipeline working directory
        - cluster_prims_config: Full cluster_prims config dict
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        config_path = Path(context["config_path"])
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError(f"Empty config: {config_path}")

        if "dataset_path" not in config:
            raise ValueError("dataset_path is required in cluster_prims config")
        context["dataset_path"] = config["dataset_path"]

        if "working_dir" not in config:
            raise ValueError("working_dir is required in cluster_prims config")
        context["working_dir"] = config["working_dir"]

        # Pass through the full config as cluster_prims_config.
        # The temp YAML is flat (all cluster_prims settings at top level),
        # so use it directly rather than looking for a nested key.
        cluster_config = {
            k: v
            for k, v in config.items()
            if k not in ("dataset_path", "working_dir", "enabled")
        }

        context["cluster_prims_config"] = cluster_config

        listener.info(f"[cluster_prims] dataset: {config['dataset_path']}")
        listener.info(
            f"[cluster_prims] config: "
            f"{_redact_sensitive_config(context['cluster_prims_config'])}"
        )
        return context


class ExpandClusterPredictionsConfigTask(Task):
    """Load config for the expand_cluster_predictions step.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - predictions_path: Path to predictions.jsonl (representatives only)
        - cluster_map_path: Path to clusters/cluster_map.jsonl
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        config_path = Path(context["config_path"])
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError(f"Empty config: {config_path}")

        # Propagate cluster_prims_ran so ExpandClusterPredictionsTask can skip itself
        cluster_prims_ran = config.get("cluster_prims_ran", False)
        context["cluster_prims_ran"] = cluster_prims_ran

        if not cluster_prims_ran:
            listener.info(
                "[expand_cluster_predictions] cluster_prims did not run — skipping config load"
            )
            return context

        if "predictions_path" not in config:
            raise ValueError(
                "predictions_path is required in expand_cluster_predictions config"
            )
        context["predictions_path"] = config["predictions_path"]

        if "cluster_map_path" not in config:
            raise ValueError(
                "cluster_map_path is required in expand_cluster_predictions config"
            )
        context["cluster_map_path"] = config["cluster_map_path"]

        listener.info(
            f"[expand_cluster_predictions] predictions: {config['predictions_path']}"
        )
        listener.info(
            f"[expand_cluster_predictions] cluster_map: {config['cluster_map_path']}"
        )
        return context
