# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for agent-service Compose GPU pinning."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class ComposeLoader(yaml.SafeLoader):
    """YAML loader that treats Compose merge tags as their underlying value."""


def _construct_override(loader: ComposeLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


ComposeLoader.add_constructor("!override", _construct_override)


def _load_compose(path: str) -> dict[str, Any]:
    with (REPO_ROOT / path).open(encoding="utf-8") as f:
        return yaml.load(f, Loader=ComposeLoader)


def _load_merged_compose(*paths: str, profile: str | None = None) -> dict[str, Any]:
    if shutil.which("docker") is None:
        pytest.skip("docker is required to validate the merged Compose config")

    cmd = ["docker", "compose"]
    for path in paths:
        cmd.extend(["-f", path])
    if profile:
        cmd.extend(["--profile", profile])
    cmd.extend(["config", "--no-interpolate", "--format", "json"])

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _reservation_devices(service: dict[str, Any]) -> list[dict[str, Any]]:
    return (
        service.get("deploy", {})
        .get("resources", {})
        .get("reservations", {})
        .get("devices", [])
    )


def test_physics_multi_gpu_overlay_routes_to_local_vlm_nim() -> None:
    compose = _load_compose("apps/physics_agent_service/docker-compose.multi-gpu.yml")

    environment = compose["services"]["physics-agent-service"]["environment"]

    assert "PA_VLM_NIM_BASE_URL=http://vlm-nim:8000/v1" in environment
    assert "PA_VLM_MODEL=nvidia/cosmos-reason2-8b" in environment
    assert "PA_LLM_NIM_BASE_URL=http://vlm-nim:8000/v1" in environment
    assert "PA_NIM_API_KEY=not-used" in environment


def test_physics_multi_gpu_overlay_pins_sidecars_to_separate_gpus() -> None:
    compose = _load_compose("apps/physics_agent_service/docker-compose.multi-gpu.yml")

    ovrtx_devices = _reservation_devices(compose["services"]["ovrtx-rendering-api"])
    vlm_devices = _reservation_devices(compose["services"]["vlm-nim"])

    assert ovrtx_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["0"],
            "capabilities": ["gpu"],
        }
    ]
    assert vlm_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["1"],
            "capabilities": ["gpu"],
        }
    ]


def test_material_vlm_nim_base_file_does_not_pregrant_gpu() -> None:
    """Keep GPU selection out of the base service to avoid Compose list append."""
    compose = _load_compose("apps/material_agent_service/docker-compose.yml")

    assert _reservation_devices(compose["services"]["vlm-nim"]) == []


def test_material_multi_gpu_overlay_pins_vlm_nim_to_gpu_1() -> None:
    compose = _load_compose("apps/material_agent_service/docker-compose.multi-gpu.yml")

    devices = _reservation_devices(compose["services"]["vlm-nim"])

    assert devices == [
        {
            "driver": "nvidia",
            "device_ids": ["1"],
            "capabilities": ["gpu"],
        }
    ]


def test_material_multi_gpu_merged_config_replaces_base_gpu_reservations() -> None:
    compose = _load_merged_compose(
        "apps/material_agent_service/docker-compose.yml",
        "apps/material_agent_service/docker-compose.multi-gpu.yml",
        profile="vlm",
    )

    ovrtx_devices = _reservation_devices(compose["services"]["ovrtx-rendering-api"])
    vlm_devices = _reservation_devices(compose["services"]["vlm-nim"])

    assert ovrtx_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["0"],
            "capabilities": ["gpu"],
        }
    ]
    assert vlm_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["1"],
            "capabilities": ["gpu"],
        }
    ]


def test_texture_multi_gpu_overlay_uses_local_image_gen_placeholder_key() -> None:
    compose = _load_compose("apps/texture_agent_service/docker-compose.multi-gpu.yml")

    environment = compose["services"]["texture-agent-service"]["environment"]

    assert "TA_IMAGE_GEN_BACKEND=openai" in environment
    assert "TA_IMAGE_GEN_BASE_URL=http://image-gen-nim:8000/v1" in environment
    assert "TA_IMAGE_GEN_API_KEY=not-used" in environment
