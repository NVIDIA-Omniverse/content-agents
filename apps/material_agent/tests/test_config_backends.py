# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that all backend references in pipeline configs are valid.

Catches mismatches between config files and installed backend packages
(e.g., configs referencing nvidia_inference without world_understanding_internal).
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _collect_yaml_configs() -> list[Path]:
    """Find all unified pipeline config files."""
    return sorted(CONFIGS_DIR.glob("unified_*.yaml"))


def _extract_backends(config: dict[str, Any]) -> set[tuple[str, str]]:
    """Extract all (backend_value, context) pairs from a config dict.

    Walks the config tree looking for 'backend' keys inside vlm/llm/llm_judge
    sections. Returns tuples of (backend_value, "vlm"|"llm"|"render") for
    context in error messages.
    """
    backends: set[tuple[str, str]] = set()

    def _walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            # Check if this dict has a 'backend' key and is a model config
            if "backend" in obj:
                parent = path.rsplit(".", 1)[-1] if path else ""
                # Only flag VLM/LLM model backends, not render backends
                if parent in ("vlm", "llm", "llm_judge", "vlm_judge"):
                    backends.add((obj["backend"], parent))
            for key, value in obj.items():
                _walk(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item, path)

    _walk(obj=config)
    return backends


def _get_available_vlm_backends() -> set[str]:
    from world_understanding.functions.models.backends.registry import (
        list_vlm_backends,
    )

    return set(list_vlm_backends())


def _get_available_chat_backends() -> set[str]:
    from world_understanding.functions.models.backends.registry import (
        list_chat_backends,
    )

    return set(list_chat_backends())


# Collect configs at module level for parametrize
_yaml_configs = _collect_yaml_configs()


@pytest.mark.parametrize(
    "config_path",
    _yaml_configs,
    ids=[p.stem for p in _yaml_configs],
)
def test_config_backends_are_registered(config_path: Path) -> None:
    """Every backend referenced in a config file must be registered.

    This catches the case where a config references 'nvidia_inference' but
    world_understanding_internal is not installed.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "steps" not in config:
        pytest.skip("No steps section in config")

    backends = _extract_backends(config.get("steps", {}))
    if not backends:
        pytest.skip("No model backends found in config")

    available_vlm = _get_available_vlm_backends()
    available_chat = _get_available_chat_backends()

    missing = []
    for backend_name, context in sorted(backends):
        if context in ("vlm", "vlm_judge"):
            if backend_name not in available_vlm:
                missing.append(
                    f"  {context}.backend: {backend_name} "
                    f"(available VLM: {', '.join(sorted(available_vlm))})"
                )
        elif context in ("llm", "llm_judge"):
            if backend_name not in available_chat:
                missing.append(
                    f"  {context}.backend: {backend_name} "
                    f"(available chat: {', '.join(sorted(available_chat))})"
                )

    if missing:
        hint = (
            "If the backend requires world_understanding_internal, install it:\n"
            "  uv pip install -e packages/world_understanding_internal"
        )
        pytest.fail(
            f"{config_path.name} references unregistered backends:\n"
            + "\n".join(missing)
            + f"\n\n{hint}"
        )


def test_default_backends_are_registered() -> None:
    """Default VLM/LLM backends in api/defaults.py must be registered."""
    import material_agent.api  # force full load
    from material_agent.api.defaults import DEFAULT_LLM_BACKEND, DEFAULT_VLM_BACKEND

    available_vlm = _get_available_vlm_backends()
    available_chat = _get_available_chat_backends()

    if DEFAULT_VLM_BACKEND not in available_vlm:
        pytest.fail(
            f"DEFAULT_VLM_BACKEND={DEFAULT_VLM_BACKEND!r} is not registered. "
            f"Available: {', '.join(sorted(available_vlm))}.\n"
            f"Install: uv pip install -e packages/world_understanding_internal"
        )

    if DEFAULT_LLM_BACKEND not in available_chat:
        pytest.fail(
            f"DEFAULT_LLM_BACKEND={DEFAULT_LLM_BACKEND!r} is not registered. "
            f"Available: {', '.join(sorted(available_chat))}.\n"
            f"Install: uv pip install -e packages/world_understanding_internal"
        )
